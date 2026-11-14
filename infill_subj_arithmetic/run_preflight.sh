#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
unset STOP_RUNPOD_ON_EXIT RUNPOD_CREDENTIALS_FILE

if [[ "${1:-}" == "--pod" ]]; then
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "Error: GPU processes are active; wait for other training to finish." >&2
    exit 1
  fi
  command -v uv >/dev/null || { echo "Error: install uv on the pod first." >&2; exit 1; }
  if [[ ! -x .venv-validated-py311/bin/python ]]; then
    uv python install 3.11
    uv venv --seed --python 3.11 .venv-validated-py311
  fi
  source .venv-validated-py311/bin/activate
  mkdir -p results
  result_dir="$(mktemp -d "$PWD/results/preflight-$(date +%Y%m%d-%H%M%S).XXXXXX")"
  rm -f .venv-validated-py311/preflight-passed.json
  exec > >(tee "$result_dir/preflight.log") 2>&1
  python -c 'import sys; assert sys.version_info[:2] == (3, 11)'
  python -m pip install -r requirements.txt
  python -m pip check
  STORY_DATA_URL="${STORY_DATA_URL:-https://goo.gl/0OYkPK}" bash run_sft.sh --prepare-only
  python -u preflight.py "$result_dir"
  python -m pip freeze --all > "$result_dir/requirements.lock.txt"
  python - "$result_dir" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path
import torch

result = Path(sys.argv[1])
record = {
    "result_directory": str(result),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "requirements_sha256": hashlib.sha256(Path("requirements.txt").read_bytes()).hexdigest(),
    "scope": "GPT-2 Large SFT batch 1; GFN batch 1, short synthetic infill; not full-batch memory or paper benchmarks",
}
for path in (result / "preflight-passed.json",
             Path(".venv-validated-py311/preflight-passed.json")):
    path.write_text(json.dumps(record, indent=2))
PY
  echo "PASS: training environment validated. Results and version freeze: $result_dir"
  echo "Pod is still running. No automatic shutdown or full training was requested."
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage locally: bash run_preflight.sh USER@HOST [SSH_PORT]" >&2
  echo "Usage on pod: bash run_preflight.sh --pod" >&2
  exit 1
fi
host="$1"
port="${2:-22}"
[[ "$host" != -* && "$port" =~ ^[0-9]+$ ]] || { echo "Error: invalid SSH target." >&2; exit 1; }
remote_dir="/workspace/gfns-llm/infill_subj_arithmetic"
ssh -p "$port" "$host" "mkdir -p '$remote_dir'; if command -v nvidia-smi >/dev/null; then nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv; fi"
scp -P "$port" run_preflight.sh preflight.py run_sft.sh sft.py main.py data.py \
  rewards.py replay_buffer.py utils.py requirements.txt runpod_util.py "$host:$remote_dir/"
scp -P "$port" -r configs "$host:$remote_dir/"
status=0
ssh -p "$port" "$host" "cd '$remote_dir' && bash run_preflight.sh --pod" || status=$?
mkdir -p results
local_dir="$(mktemp -d "$PWD/results/preflight-download-$(date +%Y%m%d-%H%M%S).XXXXXX")"
scp -P "$port" -r "$host:$remote_dir/results" "$local_dir/" ||
  { echo "Error: could not collect preflight logs; inspect results on the pod." >&2; exit 1; }
echo "Collected pod preflight results: $local_dir"
exit "$status"
