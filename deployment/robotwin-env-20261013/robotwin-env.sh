#!/usr/bin/env bash
# Source this file only in a dedicated RoboTwin shell, or use run-robotwin.sh.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
[[ "${CONDA_PREFIX}" == /root/miniconda3/envs/RoboTwin ]] || return 1
robotwin_artifact_dir=/home/lmy/Robbyant/deployment/robotwin-env-20261013
robotwin_graphics_dir="${robotwin_artifact_dir}/nvidia-graphics-580.178.04"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${robotwin_graphics_dir}/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${robotwin_graphics_dir}/usr/share/vulkan/icd.d/nvidia_icd.json"
export VK_DRIVER_FILES="${VK_ICD_FILENAMES}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${robotwin_graphics_dir}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.9
export XDG_CACHE_HOME="${robotwin_artifact_dir}/runtime-cache"
export XDG_RUNTIME_DIR="${robotwin_artifact_dir}/runtime"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled
mkdir -p "${XDG_RUNTIME_DIR}" "${XDG_CACHE_HOME}"
chmod 700 "${XDG_RUNTIME_DIR}"
