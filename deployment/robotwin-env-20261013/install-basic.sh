#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
artifact_dir=/home/lmy/Robbyant/deployment/robotwin-env-20261013
export CONDA_PKGS_DIRS=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013/conda-pkgs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INPUT=1
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_CACHE_DIR="${artifact_dir}/pip-cache"
export PIP_CONSTRAINT="${artifact_dir}/constraints.txt"
export TMPDIR="${artifact_dir}/tmp"
export MAX_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=8.9
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"
if [[ ! -d /root/miniconda3/envs/RoboTwin ]]; then
    conda create -n RoboTwin python=3.10 pip -y
fi
conda activate RoboTwin
[[ "${CONDA_PREFIX}" == /root/miniconda3/envs/RoboTwin ]]
python -c 'import sys; assert sys.version_info[:2] == (3, 10)'
conda install -n RoboTwin -c conda-forge vulkan-tools 'ffmpeg<8' --freeze-installed -y
python -m pip install 'setuptools==69.5.1' wheel ninja
cd /home/lmy/Robbyant/RoboTwin
python -m pip install -r script/requirements.txt
