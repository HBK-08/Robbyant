#!/usr/bin/env bash
source /root/miniconda3/etc/profile.d/conda.sh
baseline_original_ld="${BASELINE_ORIGINAL_LD:-${LD_LIBRARY_PATH:-}}"
baseline_conda_function="$(declare -f conda)"
eval "${baseline_conda_function/conda ()/_baseline_original_conda ()}"
conda() {
    _baseline_original_conda "$@" || return $?
    if [[ "$1" == activate ]]; then
        baseline_dir=/data/lmy/lmy_home/Robbyant/deployment/robotwin-baseline-20261014
        graphics_dir=/data/lmy/lmy_home/Robbyant/deployment/robotwin-env-20261013/nvidia-graphics-580.178.04
        export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
        export LD_LIBRARY_PATH="${graphics_dir}/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib:${baseline_original_ld}"
        export VK_DRIVER_FILES="${graphics_dir}/usr/share/vulkan/icd.d/nvidia_icd.json"
        export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
        export __EGL_VENDOR_LIBRARY_FILENAMES="${graphics_dir}/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        export PYTHONPATH="${baseline_dir}/runtime:/data/lmy/lmy_home/Robbyant/lingbot-vla-v2:/data/lmy/lmy_home/Robbyant/RoboTwin"
        export LINGBOT_FORCE_GPU0=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
        export XDG_CACHE_HOME="${baseline_dir}/runtime-cache/${CONDA_DEFAULT_ENV}"
        export XDG_RUNTIME_DIR="${baseline_dir}/runtime-state"
        export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
        export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9
        export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
        export TORCHINDUCTOR_COMPILE_THREADS=2
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
        export NO_PROXY="127.0.0.1,localhost,0.0.0.0,::1,${NO_PROXY:-}"
        export no_proxy="$NO_PROXY"
        mkdir -p "$XDG_RUNTIME_DIR" "$XDG_CACHE_HOME"
        chmod 700 "$XDG_RUNTIME_DIR"
    fi
}
