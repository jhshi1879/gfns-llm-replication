#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
script_dir="$PWD"
project_dir="$(cd ../.. && pwd)"

if [[ "${1:-}" == "--worker" ]]; then
  python_bin="$2"
  run_dir="$3"
  shift 3
  log_path="$run_dir/eval.log"
  eval_command=("$python_bin" -u "$script_dir/eval.py" --output-dir "$run_dir/results" "$@")
  {
    printf 'Started: %s\nWorking directory: %s\nPython: %s\nCommand: ' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$script_dir" "$python_bin"
    printf '%q ' "${eval_command[@]}"
    printf '\n\n===== Evaluation output =====\n'
  } > "$log_path"
  set +e
  "${eval_command[@]}" 2>&1 | tee -a "$log_path"
  status=$?
  set -e
  printf '\nFinished: %s\nExit status: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" | tee -a "$log_path"
  printf '%s\n' "$status" > "$run_dir/.exit-status.tmp"
  mv "$run_dir/.exit-status.tmp" "$run_dir/exit-status.txt"
  exit "$status"
fi

ssh_host=""
ssh_port=""
stop_pod=true
credentials_file=""
preflight_only=false
eval_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-host|--ssh-port)
      [[ -n "${2:-}" && "$2" != --* ]] || { echo "Error: $1 requires a value." >&2; exit 1; }
      if [[ "$1" == "--ssh-host" ]]; then ssh_host="$2"; else ssh_port="$2"; fi
      shift 2 ;;
    --ssh-host=*) ssh_host="${1#*=}"; [[ -n "$ssh_host" ]] || { echo "Error: --ssh-host requires a value." >&2; exit 1; }; shift ;;
    --ssh-port=*) ssh_port="${1#*=}"; [[ -n "$ssh_port" ]] || { echo "Error: --ssh-port requires a value." >&2; exit 1; }; shift ;;
    --stop-pod-on-exit) stop_pod=true; shift ;;
    --no-stop-pod-on-exit) stop_pod=false; shift ;;
    --shutdown-credentials-file)
      [[ -n "${2:-}" ]] || { echo "Error: --shutdown-credentials-file requires a value." >&2; exit 1; }
      credentials_file="$2"; shift 2 ;;
    --check-only|--help|-h) preflight_only=true; eval_args+=("$1"); shift ;;
    *) eval_args+=("$1"); shift ;;
  esac
done
set -- ${eval_args[@]+"${eval_args[@]}"}
if [[ -n "$ssh_host" ]]; then
  [[ -z "$credentials_file" ]] || { echo "Error: shutdown credentials must be prepared locally." >&2; exit 1; }
  [[ "$ssh_host" != -* && "$ssh_host" != *[[:space:]]* ]] ||
    { echo "Error: invalid SSH host." >&2; exit 1; }
  ssh_port="${ssh_port:-22}"
  [[ "$ssh_port" =~ ^[0-9]{1,5}$ ]] && (( 10#$ssh_port >= 1 && 10#$ssh_port <= 65535 )) ||
    { echo "Error: invalid SSH port." >&2; exit 1; }
  command -v ssh >/dev/null || { echo "Error: install ssh first." >&2; exit 1; }
  remote_command="bash -c 'args=(); IFS= read -r -d \"\" marker; while IFS= read -r -d \"\" arg; do args+=(\"\$arg\"); done; exec bash /workspace/gfns-llm/infill_subj_arithmetic/results/gfn-final/run_eval.sh \"\${args[@]}\"'"
  if [[ "$stop_pod" == true && "$preflight_only" == false ]]; then
    command -v python3 >/dev/null || { echo "Error: python3 is required for shutdown credentials." >&2; exit 1; }
    pod_id="$(ssh -p "$ssh_port" "$ssh_host" \
      'python3 /workspace/gfns-llm/infill_subj_arithmetic/runpod_util.py --current-pod-id')"
    [[ "$pod_id" =~ ^[A-Za-z0-9_-]+$ ]] ||
      { echo "Error: invalid current pod ID; evaluation was not started." >&2; exit 1; }
    credentials_file="$(python3 "$project_dir/runpod_util.py" --credentials-for-pod "$pod_id" |
      ssh -p "$ssh_port" "$ssh_host" \
        'python3 /workspace/gfns-llm/infill_subj_arithmetic/runpod_util.py --receive-current')"
    [[ "$credentials_file" =~ ^/tmp/gfn-runpod-[A-Za-z0-9_-]+$ ]] ||
      { echo "Error: invalid remote credentials path; evaluation was not started." >&2; exit 1; }
    if printf '%s\0' launch --shutdown-credentials-file "$credentials_file" "$@" |
      ssh -p "$ssh_port" "$ssh_host" "$remote_command"; then
      exit 0
    else
      status=$?
      ssh -p "$ssh_port" "$ssh_host" \
        "if ! tmux has-session -t '=stories-eval' 2>/dev/null; then rm -f -- '$credentials_file'; fi" ||
        echo "Error: check for an unused credentials file on the pod." >&2
      exit "$status"
    fi
  fi
  if [[ "$stop_pod" == false ]]; then
    printf '%s\0' launch --no-stop-pod-on-exit "$@" | ssh -p "$ssh_port" "$ssh_host" "$remote_command"
  else
    printf '%s\0' launch "$@" | ssh -p "$ssh_port" "$ssh_host" "$remote_command"
  fi
  exit $?
elif [[ -n "$ssh_port" ]]; then
  echo "Error: --ssh-port requires --ssh-host." >&2
  exit 1
fi

if [[ -n "$credentials_file" ]]; then
  [[ "$stop_pod" == true && "$credentials_file" =~ ^/tmp/gfn-runpod-[A-Za-z0-9_-]+$ &&
     -f "$credentials_file" && ! -L "$credentials_file" ]] ||
    { echo "Error: invalid shutdown credentials file; evaluation was not started." >&2; exit 1; }
  trap 'rm -f -- "$credentials_file"' EXIT
fi

if [[ -n "${EVAL_PYTHON:-}" ]]; then
  python_bin="$(command -v "$EVAL_PYTHON")" || { echo "Error: EVAL_PYTHON not found." >&2; exit 1; }
elif [[ -x "$project_dir/.venv-eval-py311/bin/python" ]]; then
  python_bin="$project_dir/.venv-eval-py311/bin/python"
elif [[ -x "$project_dir/.venv-validated-py311/bin/python" ]]; then
  python_bin="$project_dir/.venv-validated-py311/bin/python"
else
  python_bin="$(command -v python)" || { echo "Error: activate a Python environment first." >&2; exit 1; }
fi

check_only=false
validation_supplied=false
training_supplied=false
adapter_supplied=false
bert_model_supplied=false
bert_layers_supplied=false
for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
    printf 'SSH launch: --ssh-host USER@HOST [--ssh-port PORT] (default port: 22)\n'
    printf 'Pod shutdown is enabled by default; use --no-stop-pod-on-exit to disable it.\n\n'
    exec "$python_bin" "$script_dir/eval.py" --help
  fi
  if [[ "$argument" == "--check-only" ]]; then
    check_only=true
  fi
  case "$argument" in
    --validation-csv|--validation-csv=*) validation_supplied=true ;;
    --training-csv|--training-csv=*) training_supplied=true ;;
    --adapter-dir|--adapter-dir=*) adapter_supplied=true ;;
    --bert-model-dir|--bert-model-dir=*) bert_model_supplied=true ;;
    --bert-num-layers|--bert-num-layers=*) bert_layers_supplied=true ;;
  esac
done

if [[ "$check_only" == false ]]; then
  command -v tmux >/dev/null || { echo "Error: install tmux first." >&2; exit 1; }
  if tmux has-session -t '=stories-eval' 2>/dev/null; then
    echo "Error: stories-eval session already exists; attach with: tmux attach -t stories-eval" >&2
    exit 1
  fi
fi

download_dataset() {
  local target="$1" url="$2" kind="$3"
  command -v curl >/dev/null || { echo "Error: install curl to download datasets." >&2; exit 1; }
  mkdir -p "$(dirname "$target")"
  download_path="$(mktemp "$(dirname "$target")/.eval-download.XXXXXX")"
  trap 'rm -f -- "$download_path"' EXIT
  echo "Downloading $kind dataset to $target"
  curl --fail --location --proto '=https' --proto-redir '=https' \
    --retry 3 --connect-timeout 30 --max-time 600 --output "$download_path" "$url"
  "$python_bin" - "$download_path" "$kind" <<'PY'
import csv
import sys

path, kind = sys.argv[1:]
with open(path, newline='', encoding='utf-8-sig') as source:
    reader = csv.DictReader(source)
    fields = set(reader.fieldnames or [])
    roc = {f'sentence{i}' for i in range(1, 6)}
    cloze = {f'InputSentence{i}' for i in range(1, 5)} | {
        'RandomFifthSentenceQuiz1', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding'}
    required = roc if kind == 'training' or roc <= fields else cloze
    if not required <= fields:
        raise ValueError(f'Downloaded {kind} dataset has missing CSV columns.')
    count = 0
    for row in reader:
        count += 1
        if any(not (row.get(field) or '').strip() for field in required):
            raise ValueError(f'Downloaded {kind} dataset has empty values at CSV row {count + 1}.')
        if required == cloze and row['AnswerRightEnding'].strip() not in ('1', '2'):
            raise ValueError(f'Downloaded validation dataset has an invalid label at CSV row {count + 1}.')
    if not count:
        raise ValueError(f'Downloaded {kind} dataset is empty.')
PY
  ln "$download_path" "$target" || {
    echo "Error: could not install dataset at $target; an existing file was not overwritten." >&2
    exit 1
  }
  rm -f -- "$download_path"
  trap - EXIT
}

default_args=()
if [[ "$validation_supplied" == false ]]; then
  validation_csv="$project_dir/data/story-cloze-winter2018-val.csv"
  if [[ ! -f "$validation_csv" ]]; then
    download_dataset "$validation_csv" 'https://goo.gl/XWjas1' validation
  fi
  echo "Using validation dataset: $validation_csv"
  default_args+=(--validation-csv "$validation_csv")
fi
if [[ "$training_supplied" == false ]]; then
  training_csv="$project_dir/data/rocstories-eval-winter2017.csv"
  if [[ -f "$project_dir/data/stories.csv" ]]; then
    training_csv="$project_dir/data/stories.csv"
  elif [[ ! -f "$training_csv" ]]; then
    download_dataset "$training_csv" 'https://goo.gl/0OYkPK' training
  fi
  echo "Using training dataset: $training_csv"
  default_args+=(--training-csv "$training_csv")
fi
if [[ "$adapter_supplied" == false && -z "${GFN_ADAPTER_DIR:-}" &&
      ! -f "$script_dir/final_adapter/adapter_config.json" &&
      -f "$project_dir/output/stories_gfn/final_adapter/adapter_config.json" ]]; then
  default_args+=(--adapter-dir "$project_dir/output/stories_gfn/final_adapter")
fi
bert_model="$project_dir/models/deberta-xlarge-mnli"
if [[ "$bert_model_supplied" == false && -z "${EVAL_BERT_MODEL_DIR:-}" &&
      -f "$bert_model/config.json" ]]; then
  default_args+=(--bert-model-dir "$bert_model")
  if [[ "$bert_layers_supplied" == false && -z "${EVAL_BERT_NUM_LAYERS:-}" ]]; then
    default_args+=(--bert-num-layers 40)
  fi
fi

if [[ "$check_only" == true ]]; then
  exec "$python_bin" -u "$script_dir/eval.py" "$@" ${default_args[@]+"${default_args[@]}"}
fi

"$python_bin" -u "$script_dir/eval.py" --check-only "$@" ${default_args[@]+"${default_args[@]}"}
if [[ "$stop_pod" == true ]]; then
  [[ -f "$project_dir/runpod_util.py" && -f "$project_dir/.venv-runpod-sdk/bin/python" ]] ||
    { echo "Error: RunPod shutdown helper or SDK environment is missing." >&2; exit 1; }
  if [[ -z "$credentials_file" ]]; then
    pod_id="$("$python_bin" "$project_dir/runpod_util.py" --current-pod-id)"
    credentials_file="$("$python_bin" "$project_dir/runpod_util.py" --credentials-for-pod "$pod_id" |
      "$python_bin" "$project_dir/runpod_util.py" --receive-current)"
  fi
  [[ "$credentials_file" =~ ^/tmp/gfn-runpod-[A-Za-z0-9_-]+$ && -f "$credentials_file" && ! -L "$credentials_file" ]] ||
    { echo "Error: invalid shutdown credentials file; evaluation was not started." >&2; exit 1; }
  trap 'rm -f -- "$credentials_file"' EXIT
fi
mkdir -p "$script_dir/evaluations"
run_dir="$(mktemp -d "$script_dir/evaluations/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
worker_env=("GFN_BASE_MODEL=${GFN_BASE_MODEL:-}" "GFN_ADAPTER_DIR=${GFN_ADAPTER_DIR:-}"
            "EVAL_BERT_MODEL_DIR=${EVAL_BERT_MODEL_DIR:-}" "EVAL_BERT_NUM_LAYERS=${EVAL_BERT_NUM_LAYERS:-}")
if [[ "$stop_pod" == true ]]; then
  worker_env+=("STOP_RUNPOD_ON_EXIT=1" "RUNPOD_CREDENTIALS_FILE=$credentials_file" "RANK=0")
  printf -v worker_command '%q ' env "${worker_env[@]}" "$python_bin" -u "$project_dir/runpod_util.py" \
    --supervise bash "$script_dir/run_eval.sh" --worker "$python_bin" "$run_dir" \
    "$@" ${default_args[@]+"${default_args[@]}"}
  printf -v error_redirect ' 2>>%q' "$run_dir/shutdown.log"
  worker_command+="$error_redirect"
else
  printf -v worker_command '%q ' env "${worker_env[@]}" bash "$script_dir/run_eval.sh" \
    --worker "$python_bin" "$run_dir" "$@" ${default_args[@]+"${default_args[@]}"}
fi
tmux new-session -d -s stories-eval -c "$script_dir" "$worker_command"
if [[ "$stop_pod" == true ]]; then
  trap - EXIT
fi
printf '%s\n' "$run_dir" > "$script_dir/evaluations/active-run.txt"
if ! tmux has-session -t '=stories-eval' 2>/dev/null; then
  if [[ "$stop_pod" == true && -f "$credentials_file" ]]; then
    rm -f -- "$credentials_file"
  fi
  echo "Evaluation exited before the session could be attached. Log: $run_dir/eval.log" >&2
  if [[ -f "$run_dir/exit-status.txt" ]]; then
    status="$(cat "$run_dir/exit-status.txt")"
    [[ "$status" =~ ^[0-9]+$ ]] || exit 1
    exit "$status"
  fi
  exit 1
fi
echo "Started selected evaluation(s) in tmux session stories-eval."
echo "Log: $run_dir/eval.log"
echo "Default results: $run_dir/results (unless --output-dir was supplied)."
echo "Attach: tmux attach -t stories-eval (detach with Ctrl-b, then d)."
if [[ "$stop_pod" == true ]]; then
  echo "Pod shutdown is requested after evaluation succeeds or fails. Verify pod status; results are not downloaded."
else
  echo "Automatic pod shutdown is disabled; results are not downloaded."
fi