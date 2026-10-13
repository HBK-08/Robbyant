#!/usr/bin/env bash
set -euo pipefail
cd /home/lmy/Robbyant/lingbot-vla-v2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1
export PIP_CACHE_DIR=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/pip-cache
export PIP_FIND_LINKS=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/wheels
export PIP_CONSTRAINT=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/constraints.txt
export CONDA_PKGS_DIRS=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/conda-pkgs
export TMPDIR=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/tmp
exec bash tools/create_train_env.sh "$@"
