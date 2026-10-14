#!/usr/bin/env bash
set -euo pipefail
benchmark_run="$1"
source /data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/train-env.sh
cd "$benchmark_run"
exec 9>/data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/smoke.lock
flock -n 9 || { echo 'Another six-GPU training run is active'; exit 1; }
python - <<'PY'
import subprocess, torch
assert torch.cuda.device_count() == 6 and torch.cuda.is_available()
usage = subprocess.check_output(['nvidia-smi', '-i', '0,1,2,3,4,5', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
assert all(int(value) < 512 for value in usage.splitlines()), 'A GPU is occupied; refusing benchmark'
PY
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -i 0,1,2,3,4,5 -l 2 > gpu-metrics.csv &
benchmark_monitor_pid=$!
trap 'kill "$benchmark_monitor_pid" 2>/dev/null || true' EXIT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
date -u '+%Y-%m-%dT%H:%M:%SZ' > started-at.txt
set +e
bash -o pipefail /data/lmy/lmy_home/Robbyant/lingbot-vla-v2/train.sh \
    /data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/benchmark-entry.py "$benchmark_run/config.yaml"
benchmark_code=$?
set -e
printf '%s\n' "$benchmark_code" > exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > finished-at.txt
exit "$benchmark_code"
