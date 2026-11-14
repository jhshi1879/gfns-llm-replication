#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "--worker" ]]; then
  python_bin="$2"
  if [[ -n "${3:-}" ]]; then
    export STOP_RUNPOD_ON_EXIT=1 RUNPOD_CREDENTIALS_FILE="$3"
  fi
  mkdir -p output/stories_sft
  log_path="output/stories_sft/train-$(date +%Y%m%d-%H%M%S).log"
  train_command=("$python_bin" -u sft.py
    --model_name gpt2-large \
    --use_auth_token False \
    --output_dir output/stories_sft \
    --learning_rate 0.0005 \
    --batch_size 64 \
    --gradient_accumulation_steps 32 \
    --save_total_limit 1 \
    --num_train_epochs 20)
  {
    printf 'Started: %s\nWorking directory: %s\nPython: %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PWD" "$python_bin"
    printf 'Training command: '
    printf '%q ' "${train_command[@]}"
    printf '\n'
    for script in run_sft.sh sft.py requirements.txt run_sft_remote.sh validate_training_env.py \
                  output/stories_sft/preflight-passed.json output/stories_sft/requirements.lock.txt; do
      if [[ -f "$script" ]]; then
        printf '\n===== %s =====\n' "$script"
        cat "$script"
      fi
    done
    printf '\n===== Training output =====\n'
  } > "$log_path"
  printf '%s\n' "$log_path" > output/stories_sft/active-log.txt
  set +e
  "${train_command[@]}" 2>&1 | tee -a "$log_path"
  status=$?
  set -e
  {
    printf '\n===== Completion =====\nFinished: %s\nExit status: %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status"
    printf 'Model output directory: %s/output/stories_sft\n' "$PWD"
    if [[ "$status" -eq 0 && -f output/stories_sft/train_results.json ]]; then
      printf '\n===== Final training results =====\n'
      cat output/stories_sft/train_results.json
    fi
    if [[ -f output/stories_sft/config.json ]]; then
      printf '\n===== Saved model configuration =====\n'
      cat output/stories_sft/config.json
    fi
  } | tee -a "$log_path"
  printf '\nSFT exited with status %s. Log: %s\n' "$status" "$log_path"
  printf '%s\n' "$status" > output/stories_sft/.exit-status.tmp
  mv output/stories_sft/.exit-status.tmp output/stories_sft/exit-status.txt
  exec bash
fi

command -v tmux >/dev/null || { echo "Error: install tmux first." >&2; exit 1; }
if [[ "${1:-}" == "--prepare-only" ]]; then
  python_bin="$(command -v python)" || { echo "Error: activate the training Python environment first." >&2; exit 1; }
else
  python_bin="$PWD/.venv-validated-py311/bin/python"
  if [[ ! -x "$python_bin" ]]; then
    python_bin="$(command -v python)" || { echo "Error: activate a training environment or use run_sft_remote.sh for automatic setup." >&2; exit 1; }
  fi
fi
if [[ "${1:-}" != "--prepare-only" ]] && tmux has-session -t '=stories-sft' 2>/dev/null; then
  echo "Error: tmux session stories-sft already exists; attach with: tmux attach -t stories-sft" >&2
  exit 1
fi

if [[ -f data/stories.csv || -n "${STORY_DATA_URL:-}" ]]; then
  mkdir -p data
  source_csv="data/stories.csv"
  download_path=""
  if [[ ! -f "$source_csv" ]]; then
    command -v curl >/dev/null || { echo "Error: install curl to download ROCStories." >&2; exit 1; }
    download_path="$(mktemp data/stories-download.XXXXXX)"
    trap 'rm -f -- "$download_path"' EXIT
    curl --fail --location --retry 3 --connect-timeout 30 --max-time 600 \
      --output "$download_path" "$STORY_DATA_URL"
    source_csv="$download_path"
  fi
  "$python_bin" - "$source_csv" <<'PY'
import csv
import os
import sys
import tempfile
from pathlib import Path

fields = [f"sentence{i}" for i in range(1, 6)]
with open(sys.argv[1], newline="", encoding="utf-8-sig") as source:
    reader = csv.DictReader(source)
    if not set(fields).issubset(reader.fieldnames or []):
        raise ValueError("Download is not a ROCStories CSV with sentence1 through sentence5.")
    stories = []
    for row_number, row in enumerate(reader, start=2):
        if any(not row.get(field) or not row[field].strip() for field in fields):
            raise ValueError(f"Missing story sentence at CSV row {row_number}.")
        stories.append(" ".join(row[field].strip() for field in fields))
if not stories:
    raise ValueError("ROCStories CSV is empty.")
target = Path("data/stories_sft.csv")
with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                 dir=target.parent, delete=False) as output:
    temporary_path = Path(output.name)
    try:
        writer = csv.writer(output)
        writer.writerow(["text"])
        writer.writerows([story] for story in stories)
    except BaseException:
        temporary_path.unlink()
        raise
try:
    os.replace(temporary_path, target)
finally:
    temporary_path.unlink(missing_ok=True)
print(f"Prepared {len(stories)} stories from the full supplied CSV: {target}")
PY
  if [[ -n "$download_path" ]]; then
    mv "$download_path" data/stories.csv
    trap - EXIT
  fi
elif [[ ! -f data/stories_sft.csv ]]; then
  echo "Error: request ROCStories access at https://cs.rochester.edu/nlp/rocstories/." >&2
  echo "Then set STORY_DATA_URL to the emailed CSV download link, or place the CSV at data/stories.csv." >&2
  exit 1
else
  echo "Using existing data/stories_sft.csv; full-corpus preparation is not verified."
fi

if [[ "${1:-}" == "--prepare-only" ]]; then
  exit 0
fi

mkdir -p output/stories_sft
"$python_bin" - <<'PY'
from pathlib import Path

Path("output/stories_sft/preflight-passed.json").unlink(missing_ok=True)
PY
"$python_bin" -m pip freeze --all > output/stories_sft/requirements.lock.txt
rm -f output/stories_sft/exit-status.txt output/stories_sft/active-log.txt
printf -v worker_command '%q ' bash "$PWD/run_sft.sh" --worker "$python_bin" "${RUNPOD_CREDENTIALS_FILE:-}"
tmux new-session -d -s stories-sft -c "$PWD" "$worker_command"
tmux has-session -t '=stories-sft'
echo "Started SFT. Attach with: tmux attach -t stories-sft"
echo "Detach with Ctrl-b, then d. The session stays open after training exits."
