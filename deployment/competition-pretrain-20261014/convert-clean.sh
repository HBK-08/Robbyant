#!/usr/bin/env bash
set -euo pipefail
prep_root=/data/lmy/lmy_home/Robbyant
prep_dir="$prep_root/deployment/competition-pretrain-20261014"
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$prep_dir/cache/huggingface" XDG_CACHE_HOME="$prep_dir/cache"
export HF_LEROBOT_HOME="$prep_root/datasets/competition-clean50/lerobot"
export LD_LIBRARY_PATH="/root/miniconda3/envs/lingbotvla/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$prep_dir/conversion-source/lerobot-a445d9c9da6bea99a8972daa4fe1fdd053d711d2"
prep_task="${1:-lift_pot}"
"$prep_dir/conversion-venv/bin/python" "$prep_dir/convert_clean.py" --task "$prep_task"
unset PYTHONPATH
# Explicitly disable Hub writes. The utility keeps the v2.1 source as <task>_old.
/root/miniconda3/envs/lingbotvla/bin/python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id "$prep_task" --root "$prep_root/datasets/competition-clean50/lerobot" --push-to-hub false
