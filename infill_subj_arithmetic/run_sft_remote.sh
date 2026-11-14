#!/usr/bin/env bash
set -euo pipefail

# Keep edits during a long installation from changing the running shell program.
if [[ "${1:-}" != "--snapshot" ]]; then
  exec bash -c "$(cat -- "$0")" "$0" --snapshot "$@"
fi
shift

cd "$(dirname "$0")"

stage="${TRAINING_STAGE:-sft}"
case "$stage" in
  sft) session=stories-sft; worker=run_sft.sh ;;
  gfn) session=stories-gfn; worker=run_gfn.sh ;;
  *) echo "Error: unsupported training stage." >&2; exit 1 ;;
esac
output_name="${session//-/_}"
output_path="output/$output_name"

download_only=false
tail_lines=+1
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --monitor) download_only=true; tail_lines=50; shift ;;
    --download-only) download_only=true; shift ;;
    --pod-id)
      [[ -n "${2:-}" ]] || { echo "Error: --pod-id requires a value." >&2; exit 1; }
      export RUNPOD_POD_ID="$2"; shift 2 ;;
    *) echo "Error: unknown option $1" >&2; exit 1 ;;
  esac
done
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash run_sft_remote.sh [--monitor|--download-only] [--pod-id POD_ID] USER@HOST [SSH_PORT]" >&2
  echo "Use RunPod's direct TCP SSH endpoint (supports SCP), or an SSH config alias." >&2
  exit 1
fi
host="$1"
port="${2:-22}"
[[ "$host" != -* && "$port" =~ ^[0-9]+$ ]] || { echo "Error: invalid SSH host or port." >&2; exit 1; }
tools=(ssh)
if [[ "$download_only" != true ]]; then
  tools+=(scp python3)
fi
for tool in "${tools[@]}"; do
  command -v "$tool" >/dev/null || { echo "Error: $tool is required locally." >&2; exit 1; }
done
export RUNPOD_ENV_FILE="${RUNPOD_ENV_FILE:-$PWD/../.env}"

remote_dir="/workspace/gfns-llm/infill_subj_arithmetic"
ssh_args=(-p "$port")
scp_args=(-P "$port")
download_results() {
  echo "Waiting for training completion; keep this local terminal running."
  echo "Disconnecting does not stop training; the updated pod worker requests shutdown on training exit."
  completion_status="$(ssh "${ssh_args[@]}" "$host" \
    "cd '$remote_dir' || exit 1
     tail_pid=''
     trap 'if [ -n \"\$tail_pid\" ]; then kill \"\$tail_pid\" 2>/dev/null; wait \"\$tail_pid\" 2>/dev/null; fi' EXIT
     while [ ! -f '$output_path/exit-status.txt' ]; do
       if [ -z \"\$tail_pid\" ] && [ -f '$output_path/active-log.txt' ]; then
         IFS= read -r log_path < '$output_path/active-log.txt'
         tail -n '$tail_lines' -F \"\$log_path\" >&2 &
         tail_pid=\$!
       fi
       if ! tmux has-session -t '=$session' 2>/dev/null; then
         echo 'Error: no completion marker or running tmux session; inspect pod logs.' >&2
         exit 1
       fi
       sleep 2
     done
     if [ -n \"\$tail_pid\" ]; then
       sleep 2
     elif [ -f '$output_path/active-log.txt' ]; then
       IFS= read -r log_path < '$output_path/active-log.txt'
       tail -n '$tail_lines' \"\$log_path\" >&2
     fi
     cat '$remote_dir/$output_path/exit-status.txt'")"
  [[ "$completion_status" =~ ^[0-9]+$ ]] || { echo "Error: invalid remote completion status." >&2; exit 1; }
  # Downloads are handled separately by download_results.sh.
  echo "Results download disabled. Outputs remain at $remote_dir/$output_path on the pod."
  if [[ "$completion_status" -ne 0 ]]; then
    echo "Error: training failed with status $completion_status; inspect outputs on the pod." >&2
    exit 1
  fi
}
if [[ "$download_only" == true ]]; then
  download_results
  exit 0
fi
python3 -c 'from runpod_util import validate_runpod_config; validate_runpod_config()'

dataset_path="${STORY_DATA_PATH:-}"
dataset_url="${STORY_DATA_URL:-https://goo.gl/0OYkPK}"
if [[ -n "$dataset_path" ]]; then
  [[ -f "$dataset_path" ]] || { echo "Error: STORY_DATA_PATH is not an existing CSV file." >&2; exit 1; }
elif [[ -z "$dataset_url" ]]; then
  echo "ROCStories access: https://cs.rochester.edu/nlp/rocstories/"
  echo "Submit the dataset access form; check your email (including spam) for the download link."
  echo "Alternatively set STORY_DATA_PATH to an already downloaded ROCStories CSV."
  read -r -p "Paste your authorized direct CSV URL (blank to stop): " dataset_url
  [[ -n "$dataset_url" ]] || { echo "No dataset supplied; training was not started." >&2; exit 1; }
fi

if [[ "$stage" == gfn ]]; then
  steps="${GFN_TRAIN_STEPS:-1000}"
  [[ "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "Error: GFN_TRAIN_STEPS must be positive." >&2; exit 1; }
  if [[ -n "${SFT_MODEL_PATH:-}" ]]; then
    [[ -f "$SFT_MODEL_PATH/config.json" ]] || { echo "Error: SFT_MODEL_PATH must contain a full SFT model." >&2; exit 1; }
  fi
fi

ssh "${ssh_args[@]}" "$host" \
  "mkdir -p '$remote_dir/data'; if command -v tmux >/dev/null; then
     for session in stories-sft stories-gfn; do
       if tmux has-session -t \"=\$session\" 2>/dev/null; then
         echo \"Error: \$session already exists; close its completed tmux session or collect its outputs first.\" >&2
         exit 1
       fi
     done
   fi"
scp "${scp_args[@]}" run_sft.sh run_sft_remote.sh sft.py requirements.txt validate_training_env.py runpod_util.py "$host:$remote_dir/"
if [[ "$stage" == gfn ]]; then
  scp "${scp_args[@]}" run_gfn.sh run_gfn_remote.sh main.py data.py rewards.py replay_buffer.py utils.py "$host:$remote_dir/"
  scp "${scp_args[@]}" -r configs "$host:$remote_dir/"
  if [[ -n "${SFT_MODEL_PATH:-}" ]]; then
    ssh "${ssh_args[@]}" "$host" "mkdir -p '$remote_dir/input'"
    scp "${scp_args[@]}" -r "$SFT_MODEL_PATH" "$host:$remote_dir/input/"
    model_name="$(basename "$SFT_MODEL_PATH")"
    remote_model="$remote_dir/input/$model_name"
  else
    remote_model="$remote_dir/output/stories_sft"
  fi
fi
if [[ -n "$dataset_path" ]]; then
  scp "${scp_args[@]}" "$dataset_path" "$host:$remote_dir/data/stories.csv"
fi

ssh "${ssh_args[@]}" "$host" "bash -s" <<'REMOTE'
set -euo pipefail
cd /workspace/gfns-llm/infill_subj_arithmetic
missing=()
command -v tmux >/dev/null || missing+=(tmux)
command -v curl >/dev/null || missing+=(curl)
command -v python3 >/dev/null || missing+=(python3)
if [[ ${#missing[@]} -gt 0 ]]; then
  if [[ "$(id -u)" != 0 ]] || ! command -v apt-get >/dev/null; then
    echo "Error: install these tools on the pod first: ${missing[*]}" >&2
    exit 1
  fi
  apt-get update
  apt-get install -y "${missing[@]}"
fi
if [[ ! -x .venv-validated-py311/bin/python ]]; then
  if ! command -v python3.11 >/dev/null && ! command -v uv >/dev/null; then
    python3 -m pip install uv
  fi
  if command -v uv >/dev/null; then
    uv python install 3.11
    uv venv --seed --python 3.11 .venv-validated-py311
  else
    python3.11 -m venv .venv-validated-py311
  fi
fi
if ! cmp -s requirements.txt .venv-validated-py311/installed-requirements.txt; then
  .venv-validated-py311/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv-validated-py311/installed-requirements.txt
fi
.venv-validated-py311/bin/python -m pip check
.venv-validated-py311/bin/python -c 'import sys, torch; assert sys.version_info[:2] == (3, 11); assert torch.cuda.is_available(), "CUDA GPU unavailable"; print("CUDA GPU:", torch.cuda.get_device_name(0))'
if [[ ! -x .venv-runpod-sdk/bin/python ]]; then
  if command -v uv >/dev/null; then
    uv venv --seed --python .venv-validated-py311/bin/python .venv-runpod-sdk
  else
    .venv-validated-py311/bin/python -m venv .venv-runpod-sdk
  fi
fi
if ! .venv-runpod-sdk/bin/python -c 'import runpod' >/dev/null 2>&1; then
  .venv-runpod-sdk/bin/python -m pip install 'runpod>=2.0,<3'
fi
.venv-runpod-sdk/bin/python -m pip check
.venv-runpod-sdk/bin/python -c 'import runpod; from importlib.metadata import version; assert version("runpod").split(".")[0] == "2"; assert callable(runpod.stop_pod)'
REMOTE

credentials_file="$(python3 runpod_util.py --credentials |
  ssh "${ssh_args[@]}" "$host" "cd '$remote_dir' && .venv-validated-py311/bin/python runpod_util.py --receive")"
[[ "$credentials_file" =~ ^/tmp/gfn-runpod-[A-Za-z0-9_-]+$ ]] ||
  { echo "Error: invalid remote credentials path; training was not started." >&2; exit 1; }
printf -v start_command '%q ' bash "$remote_dir/$worker"
if [[ "$stage" == gfn ]]; then
  printf -v start_command '%q ' env "GFN_BASE_MODEL=$remote_model" "GFN_TRAIN_STEPS=$steps" bash "$remote_dir/$worker"
fi
printf -v remote_command '%q ' bash -c "cd \"$remote_dir\" && source .venv-validated-py311/bin/activate && IFS= read -r STORY_DATA_URL && export STORY_DATA_URL RUNPOD_CREDENTIALS_FILE='$credentials_file' && $start_command"
if ! printf '%s\n' "$dataset_url" | ssh "${ssh_args[@]}" "$host" "$remote_command"; then
  ssh "${ssh_args[@]}" "$host" "if ! tmux has-session -t '=$session' 2>/dev/null; then rm -f -- '$credentials_file'; fi" ||
    echo "Error: remove $credentials_file on the pod manually." >&2
  echo "Error: launch failed; verify pod training/status and stop it manually if idle." >&2
  exit 1
fi
echo "Uploaded files and launched $stage on the pod."
echo "The training process requests pod shutdown on exit, even with your laptop disconnected."
echo "Shutdown is best effort; verify pod status. Results download is disabled."
download_results
