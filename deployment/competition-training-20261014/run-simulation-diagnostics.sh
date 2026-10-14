#!/usr/bin/env bash
set -euo pipefail
diagnostic_run="$1"
training_dir=/data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014
source "$training_dir/train-env.sh"
export CUDA_VISIBLE_DEVICES=1
cd /data/lmy/lmy_home/Robbyant/lingbot-vla-v2
diagnostic_model="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"]["smoke100"])' "$diagnostic_run/manifest.json")"
diagnostic_port="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
export QWEN3VL_PATH=/data/lmy/lmy_home/Robbyant/models/Qwen3-VL-4B-Instruct
python -u -m deploy.lingbot_vla_v2_policy --model_path "$diagnostic_model" \
    --port "$diagnostic_port" --use_length 50 --use_bf16 False --use_fp32 True \
    --use_compile True > "$diagnostic_run/simulation-inference.log" 2>&1 &
diagnostic_server_pid=$!
cleanup_diagnostics() {
    diagnostic_code=$?
    kill "$diagnostic_server_pid" 2>/dev/null || true
    wait "$diagnostic_server_pid" 2>/dev/null || true
    printf '%s\n' "$diagnostic_code" > "$diagnostic_run/simulation-exit-code.txt"
}
trap cleanup_diagnostics EXIT
python - "$diagnostic_port" "$diagnostic_server_pid" <<'PY'
import os, socket, sys, time
port, pid = map(int, sys.argv[1:]); deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            break
    except OSError:
        time.sleep(2)
else:
    raise TimeoutError('Inference server startup timed out')
PY
source /data/lmy/lmy_home/Robbyant/deployment/robotwin-baseline-20261014/conda-scoped.sh
conda activate RoboTwin
cd /data/lmy/lmy_home/Robbyant/RoboTwin
export PYTHONPATH="$(python -c 'import site;print(site.getsitepackages()[0])'):${PYTHONPATH}"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
timeout --signal=TERM --kill-after=30s 70m python -u "$training_dir/diagnose-simulation.py" \
    "$diagnostic_run" "$diagnostic_port" > "$diagnostic_run/simulation.log" 2>&1
