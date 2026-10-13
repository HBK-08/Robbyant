#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
[[ "${CONDA_PREFIX}" == /root/miniconda3/envs/lingbotvla ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1 PIP_NO_INPUT=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
export PIP_CACHE_DIR=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/pip-cache
export TMPDIR=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/tmp
python -m pip install --no-deps \
  'utils3d @ https://github.com/EasternJournalist/utils3d/archive/3fab839f0be9931dac7c8488eb0e1600c236e183.tar.gz'
python -m pip install --no-deps -e /home/lmy/Robbyant/lingbot-vla-v2/lingbotvla/models/vla/vision_models/MoGe
# LeRobot 0.4.2 declares setuptools >=71,<81; this also retains pkg_resources
# for older training dependencies without changing the Python/PyTorch stack.
python -m pip install 'setuptools==80.9.0'
# MLflow allows newer SDK releases whose protobuf requirement conflicts with
# the repository's protobuf==3.20.3. This SDK release only requires requests
# and google-auth and satisfies MLflow's databricks-sdk>=0.20.0 constraint.
python -m pip install 'databricks-sdk==0.57.0'
python - <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
for name, expected in {'torch': '2.8.0', 'numpy': '1.26.4', 'pyarrow': '21.0.0',
                       'huggingface-hub': '0.34.3', 'accelerate': '1.7.0'}.items():
    actual = metadata.version(name).split('+')[0]
    assert actual == expected, (name, actual, expected)
    print(name, actual)
# The official script adds a historical source path absent from this checkout.
# utils3d is now installed from the exact commit specified by MoGe instead.
pth = Path(sys.prefix) / 'lib/python3.12/site-packages/stablevla_local_depth.pth'
if pth.exists():
    stale_path = pth.read_text().strip()
    if 'morgbd_clean/3rd/utils3d' in stale_path and not Path(stale_path).exists():
        pth.unlink()
        print('Removed unused .pth for absent historical utils3d checkout')
PY
