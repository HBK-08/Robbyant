#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
[[ "${CONDA_PREFIX}" == /root/miniconda3/envs/lingbotvla ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export XDG_CACHE_HOME=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/runtime-cache
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
cd /home/lmy/Robbyant/lingbot-vla-v2
exec python /home/lmy/Robbyant/deployment/lingbotvla-env-20261013/verify_environment.py
