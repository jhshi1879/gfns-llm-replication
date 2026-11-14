#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --pod-id)
      [[ -n "${2:-}" ]] || { echo "Error: --pod-id requires a value." >&2; exit 1; }
      export RUNPOD_POD_ID="$2"; shift 2 ;;
    *) echo "Error: unknown option $1" >&2; exit 1 ;;
  esac
done
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash download_results.sh [--pod-id POD_ID] USER@HOST [SSH_PORT]" >&2
  exit 1
fi
host="$1"
port="${2:-22}"
[[ "$host" != -* && "$port" =~ ^[0-9]+$ ]] ||
  { echo "Error: invalid SSH host or port." >&2; exit 1; }
for tool in ssh scp python3; do
  command -v "$tool" >/dev/null || { echo "Error: $tool is required locally." >&2; exit 1; }
done
case "${TRAINING_STAGE:-sft}" in
  sft) output_name=stories_sft ;;
  gfn) output_name=stories_gfn ;;
  *) echo "Error: unsupported training stage." >&2; exit 1 ;;
esac
remote_dir="/workspace/gfns-llm/infill_subj_arithmetic"
ssh_args=(-p "$port")
scp_args=(-P "$port")
remote_output="$remote_dir/output/$output_name"
runpod_credentials="$(python3 runpod_util.py --credentials)"
shutdown_pod() {
  local status=$?
  trap - EXIT
  if printf '%s\n' "$runpod_credentials" |
    ssh "${ssh_args[@]}" "$host" "cd '$remote_dir' && .venv-runpod-sdk/bin/python runpod_util.py --stop"; then
    echo "RunPod accepted the stop request; verify pod status in the console."
  else
    echo "Error: automatic pod shutdown failed; stop the pod manually and verify billing." >&2
    [[ "$status" -ne 0 ]] || status=1
  fi
  exit "$status"
}
trap shutdown_pod EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

entries="$(ssh "${ssh_args[@]}" "$host" \
  "find '$remote_output' -mindepth 1 -maxdepth 1 ! -name 'checkpoint-*' -printf '%f\n'")"
[[ -n "$entries" ]] || { echo "Error: no downloadable outputs found." >&2; exit 1; }
sources=()
while IFS= read -r entry; do
  [[ "$entry" =~ ^[A-Za-z0-9._-]+$ && "$entry" != "." && "$entry" != ".." ]] ||
    { echo "Error: unsupported output filename." >&2; exit 1; }
  sources+=("$host:$remote_output/$entry")
done <<< "$entries"

mkdir -p "$PWD/results"
local_dir="$(mktemp -d "$PWD/results/output-download-$(date +%Y%m%d-%H%M%S).XXXXXX")"
echo "Downloading $output_name outputs excluding checkpoint-* folders. Ensure sufficient local disk space."
mkdir -p "$local_dir/$output_name"
if ! scp "${scp_args[@]}" -r "${sources[@]}" "$local_dir/$output_name/"; then
  echo "Error: download failed; incomplete files remain at $local_dir. Requesting pod shutdown." >&2
  exit 1
fi
echo "Downloaded results to: $local_dir/$output_name"
