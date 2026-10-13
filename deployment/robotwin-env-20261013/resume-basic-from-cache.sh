#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
[[ "${CONDA_PREFIX}" == /root/miniconda3/envs/RoboTwin ]]
artifact_dir=/home/lmy/Robbyant/deployment/robotwin-env-20261013
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INPUT=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_CACHE_DIR="${artifact_dir}/pip-cache"
export PIP_CONSTRAINT="${artifact_dir}/constraints.txt"
export TMPDIR="${artifact_dir}/tmp"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python - "${1:-install-basic-resume.log}" <<'PY'
from pathlib import Path
import os, signal, sys, time
root = Path('/home/lmy/Robbyant/deployment/robotwin-env-20261013')
wheel = root / 'wheels/open3d-0.18.0-cp310-cp310-manylinux_2_27_x86_64.whl'
assert wheel.exists() and wheel.stat().st_size > 399_000_000
source_log = root / sys.argv[1]
assert source_log.resolve().parent == root.resolve()
log_text = source_log.read_text()
assert 'Collecting torch==2.4.1' in log_text
assert 'Installing collected packages:' not in log_text.rsplit('Collecting torch==2.4.1', 1)[1]
targets = []
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        argv = (directory / 'cmdline').read_bytes().split(b'\0')
        if argv[1:6] != [b'-m', b'pip', b'install', b'-r', b'script/requirements.txt']:
            continue
        status = (directory / 'status').read_text()
        ppid = next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:'))
        parent_argv = (Path('/proc') / ppid / 'cmdline').read_bytes().split(b'\0')
        allowed_parents = [str(root / name).encode() for name in
                           ['install-basic.sh', 'resume-basic-from-cache.sh']]
        if not any(parent in parent_argv for parent in allowed_parents):
            continue
        targets.append(int(directory.name))
    except (OSError, StopIteration):
        continue
assert len(targets) == 1, targets
for pid in targets:
    print('Restarting only our dependency downloader to reuse the completed Open3D cache:', pid, flush=True)
    os.kill(pid, signal.SIGINT)
    for _ in range(100):
        if not Path('/proc', str(pid)).exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError('Previous downloader did not exit; refusing concurrent environment writes')
PY
cd /home/lmy/Robbyant/RoboTwin
python -m pip install -r script/requirements.txt
