#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
[[ "${CONDA_PREFIX}" == /root/miniconda3/envs/RoboTwin ]]
artifact_dir=/home/lmy/Robbyant/deployment/robotwin-env-20261013
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INPUT=1
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_CACHE_DIR="${artifact_dir}/pip-cache" TMPDIR="${artifact_dir}/tmp"
export MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9
export PATH="${CUDA_HOME}/bin:${PATH}"
cd /home/lmy/Robbyant/RoboTwin/envs/curobo
python - <<'PY'
import importlib.metadata as metadata
import setuptools_scm
import subprocess
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == 'd64c4b005459db10c5dd867d8b30a87d5bda9bdb'
assert setuptools_scm.get_version(root='.', version_scheme='no-guess-dev', local_scheme='dirty-tag') == '0.7.8'
print('Before rebuild:', metadata.version('nvidia-curobo'))
PY
# Rebuild against the declared build dependencies now installed. Never replace
# package metadata manually or set a pretend version.
python -m pip install -e . --no-deps --no-build-isolation \
    --report "${artifact_dir}/curobo-version-repair.json"
python - <<'PY'
import importlib.metadata as metadata
assert metadata.version('nvidia-curobo') == '0.7.8', metadata.version('nvidia-curobo')
print('CuRobo metadata now matches the official source tag:', metadata.version('nvidia-curobo'))
PY
