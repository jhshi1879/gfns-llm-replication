#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
base_model="${GFN_BASE_MODEL:-$PWD/output/stories_sft}"
steps="${GFN_TRAIN_STEPS:-1000}"
[[ "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "Error: GFN_TRAIN_STEPS must be positive." >&2; exit 1; }
[[ -f "$base_model/config.json" ]] || { echo "Error: completed story-SFT model required at $base_model." >&2; exit 1; }
[[ -f "$base_model/model.safetensors" || -f "$base_model/pytorch_model.bin" ||
   -f "$base_model/model.safetensors.index.json" || -f "$base_model/pytorch_model.bin.index.json" ]] ||
  { echo "Error: no full SFT model weights found at $base_model." >&2; exit 1; }
base_model="$(cd "$base_model" && pwd)"

if [[ "${1:-}" == "--worker" ]]; then
  python_bin="$2"
  if [[ -n "${3:-}" ]]; then
    export STOP_RUNPOD_ON_EXIT=1 RUNPOD_CREDENTIALS_FILE="$3"
  fi
  output_dir="$PWD/output/stories_gfn"
  mkdir -p "$output_dir"
  log_path="$output_dir/train-$(date +%Y%m%d-%H%M%S).log"
  train_command=("$python_bin" -u runpod_util.py --supervise "$python_bin" -u main.py +experiment=infilling
    # "train_steps=$steps" removed: YAML config now controls train_steps
    "model.name=$base_model" replay_buffer.size=25
    max_eval_len=25 eval_temp=0.9 "+save_final=true" "hydra.run.dir=$output_dir")
  {
    printf 'Started: %s\nBase model: %s\nCommand: ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$base_model"
    printf '%q ' "${train_command[@]}"
    printf '\n'
    for script in run_gfn.sh main.py data.py rewards.py replay_buffer.py utils.py requirements.txt; do
      printf '\n===== %s =====\n' "$script"
      cat "$script"
    done
    find configs -type f -name '*.yaml' -exec sh -c 'printf "\n===== %s =====\n" "$1"; cat "$1"' sh {} \;
    printf '\n===== Training output =====\n'
  } > "$log_path"
  printf '%s\n' "$log_path" > "$output_dir/active-log.txt"
  set +e
  "${train_command[@]}" 2>&1 | tee -a "$log_path"
  status=$?
  set -e
  {
    printf '\n===== Completion =====\nFinished: %s\nExit status: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status"
    if [[ "$status" -eq 0 && -f "$output_dir/train_results.json" ]]; then
      cat "$output_dir/train_results.json"
    fi
  } | tee -a "$log_path"
  printf '%s\n' "$status" > "$output_dir/.exit-status.tmp"
  mv "$output_dir/.exit-status.tmp" "$output_dir/exit-status.txt"
  exec bash
fi

command -v tmux >/dev/null || { echo "Error: install tmux first." >&2; exit 1; }
if tmux has-session -t '=stories-gfn' 2>/dev/null; then
  echo "Error: stories-gfn session already exists." >&2
  exit 1
fi
python_bin="$(command -v python)"
bash run_sft.sh --prepare-only
[[ -f data/stories.csv ]] || { echo "Error: original stories.csv required for infilling." >&2; exit 1; }
mkdir -p output/stories_gfn
rm -f output/stories_gfn/exit-status.txt output/stories_gfn/active-log.txt
export GFN_BASE_MODEL="$base_model" GFN_TRAIN_STEPS="$steps"
printf -v worker_command '%q ' bash "$PWD/run_gfn.sh" --worker "$python_bin" "${RUNPOD_CREDENTIALS_FILE:-}"
tmux new-session -d -s stories-gfn -c "$PWD" "$worker_command"
tmux has-session -t '=stories-gfn'
echo "Started GFlowNet infilling: $steps steps, base model $base_model"
