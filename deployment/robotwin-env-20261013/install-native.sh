#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
[[ "${CONDA_PREFIX}" == /root/miniconda3/envs/RoboTwin ]]
artifact_dir=/home/lmy/Robbyant/deployment/robotwin-env-20261013
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INPUT=1
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_CACHE_DIR="${artifact_dir}/pip-cache"
export PIP_CONSTRAINT="${artifact_dir}/constraints.txt"
export PIP_FIND_LINKS="${artifact_dir}/wheels"
export TMPDIR="${artifact_dir}/tmp"
export MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=8.9
cd /home/lmy/Robbyant/RoboTwin
python - <<'PY'
import torch, numpy
assert torch.__version__.split('+')[0] == '2.4.1'
assert numpy.__version__ == '1.26.4'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print('Validated official simulation stack:', torch.__version__, numpy.__version__)
PY
python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
out = Path('/home/lmy/Robbyant/deployment/robotwin-env-20261013')
for package, filename in [('sapien', 'sapien/wrapper/urdf_loader.py'), ('mplib', 'mplib/planner.py')]:
    path = metadata.distribution(package).locate_file(filename)
    (out / (package + '-before-official-patch.py')).write_bytes(path.read_bytes())
PY
# Execute the fixed revision's official script. -e prevents its final message
# from falsely reporting success after an earlier installation failure.
bash -e script/_install.sh
python -m pip check > "${artifact_dir}/pip-check.txt"
