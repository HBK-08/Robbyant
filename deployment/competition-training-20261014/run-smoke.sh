#!/usr/bin/env bash
set -euo pipefail
training_run="$1"
source /data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/train-env.sh
cd "$training_run"
exec 9>/data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/smoke.lock
flock -n 9 || { echo 'A smoke training launcher is already active'; exit 1; }
python - <<'PY'
import json, shutil, torch
from pathlib import Path
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 6, 'Expected exactly six visible GPUs'
assert shutil.disk_usage('.').free > 150 * 1024**3, 'Insufficient checkpoint disk space'
print(json.dumps({'cuda_count':torch.cuda.device_count(),'torch':torch.__version__,'devices':[torch.cuda.get_device_properties(i).name for i in range(6)]}),flush=True)
PY
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -i 0,1,2,3,4,5 -l 5 > gpu-metrics.csv &
training_monitor_pid=$!
trap 'kill "$training_monitor_pid" 2>/dev/null || true' EXIT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
date -u '+%Y-%m-%dT%H:%M:%SZ' > started-at.txt
set +e
bash -o pipefail /data/lmy/lmy_home/Robbyant/lingbot-vla-v2/train.sh \
    /data/lmy/lmy_home/Robbyant/lingbot-vla-v2/tasks/vla/train_lingbotvla.py \
    "$training_run/config.yaml"
training_exit=$?
set -e
date -u '+%Y-%m-%dT%H:%M:%SZ' > finished-at.txt
printf '%s\n' "$training_exit" > exit-code.txt
exit "$training_exit"
