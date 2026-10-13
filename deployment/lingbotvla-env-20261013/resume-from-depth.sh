#!/usr/bin/env bash
# Continue the unchanged official script at its depth-dependency stage.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONNOUSERSITE=1 PIP_NO_INPUT=1
export PIP_CACHE_DIR=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/pip-cache
export PIP_FIND_LINKS=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/wheels
export PIP_CONSTRAINT=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/constraints.txt
export TMPDIR=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/tmp
source /root/miniconda3/etc/profile.d/conda.sh
ENV_NAME=lingbotvla
conda activate "${ENV_NAME}"
[[ "${CONDA_PREFIX}" == /root/miniconda3/envs/lingbotvla ]]
REPO_ROOT=/home/lmy/Robbyant/lingbot-vla-v2
cd "${REPO_ROOT}"
assert_torch_stack() {
    python - <<'PY'
import torch
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
PY
}
source <(awk '/^python -m pip install -r .*requirements-depth.txt/ {copy=1} copy {print}' tools/create_train_env.sh)
