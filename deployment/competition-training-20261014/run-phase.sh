#!/usr/bin/env bash
set -euo pipefail
phase_run="$1"
source /data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/train-env.sh
cd "$phase_run"
exec 9>/data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/smoke.lock
flock -n 9 || { echo 'Another six-GPU training run is active'; exit 1; }
python - <<'PY'
import shutil,subprocess,torch
assert torch.cuda.is_available() and torch.cuda.device_count()==6
assert shutil.disk_usage('.').free > 150*1024**3
memory=subprocess.check_output(['nvidia-smi','-i','0,1,2,3,4,5','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert all(int(v)<512 for v in memory.splitlines()),'A GPU is occupied; refusing training start'
PY
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -i 0,1,2,3,4,5 -l 5 > gpu-metrics.csv &
phase_monitor_pid=$!
trap 'kill "$phase_monitor_pid" 2>/dev/null || true' EXIT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
date -u '+%Y-%m-%dT%H:%M:%SZ' > started-at.txt
set +e
bash -o pipefail /data/lmy/lmy_home/Robbyant/lingbot-vla-v2/train.sh \
    /data/lmy/lmy_home/Robbyant/deployment/competition-training-20261014/phase-entry.py "$phase_run/config.yaml"
phase_code=$?
set -e
printf '%s\n' "$phase_code" > exit-code.txt
date -u '+%Y-%m-%dT%H:%M:%SZ' > finished-at.txt
exit "$phase_code"
