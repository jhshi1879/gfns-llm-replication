#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export TRAINING_STAGE=gfn
exec bash run_sft_remote.sh "$@"
