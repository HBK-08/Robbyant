#!/usr/bin/env bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
training_root=/data/lmy/lmy_home/Robbyant
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9
export LD_LIBRARY_PATH="${training_root}/deployment/robotwin-env-20261013/nvidia-graphics-580.178.04/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${training_root}/lingbot-vla-v2"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
export XDG_CACHE_HOME="${training_root}/deployment/competition-training-20261014/runtime-cache"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export NO_PROXY="127.0.0.1,localhost,0.0.0.0,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
mkdir -p "$XDG_CACHE_HOME"
