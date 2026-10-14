#!/usr/bin/env bash
set -euo pipefail
diagnostic_run="$1"
trap 'diagnostic_code=$?; printf "%s\n" "$diagnostic_code" > "$diagnostic_run/openloop-exit-code.txt"' EXIT
source /data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/train-env.sh
export CUDA_VISIBLE_DEVICES=0
cd /data/lmy/lmy_home/Robbyant/lingbot-vla-v2
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$diagnostic_run/started-at.txt"
for diagnostic_model in smoke100 pretrained official_robotwin; do
    printf '%s\n' "$diagnostic_model" > "$diagnostic_run/current-stage.txt"
    python -u /data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/diagnose-actions.py \
        "$diagnostic_run" "$diagnostic_model" > "$diagnostic_run/$diagnostic_model-openloop.log" 2>&1
done
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$diagnostic_run/openloop-finished-at.txt"
printf 'completed\n' > "$diagnostic_run/current-stage.txt"
