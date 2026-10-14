#!/usr/bin/env bash
set -euo pipefail
prep_root=/data/lmy/lmy_home/Robbyant
prep_dir="$prep_root/deployment/competition-pretrain-20261014"
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$prep_dir/cache/huggingface" XDG_CACHE_HOME="$prep_dir/cache"
export MPLCONFIGDIR="$prep_dir/cache/matplotlib"
export PYTHONPATH="$prep_root/lingbot-vla-v2"
export LD_LIBRARY_PATH="/root/miniconda3/envs/lingbotvla/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1
cd "$prep_root/lingbot-vla-v2"
case "${1:-}" in
  configs) /root/miniconda3/envs/lingbotvla/bin/python "$prep_dir/prepare_configs.py" ;;
  v30-single) /root/miniconda3/envs/lingbotvla/bin/python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id lift_pot --root "$prep_root/datasets/competition-clean50/lerobot" --push-to-hub false ;;
  norm-single) /root/miniconda3/envs/lingbotvla/bin/python "$prep_dir/cpu_arguments.py" "$prep_dir/norm-lift-pot.yaml" ;;
  norm-all) /root/miniconda3/envs/lingbotvla/bin/python "$prep_dir/cpu_arguments.py" "$prep_dir/norm-clean50.yaml" ;;
  verify) /root/miniconda3/envs/lingbotvla/bin/python "$prep_dir/verify_pretrain.py" ;;
  checkpoints) /root/miniconda3/envs/lingbotvla/bin/python "$prep_dir/verify_checkpoints.py" ;;
  *) echo 'Only preparation modes: configs, v30-single, norm-single, norm-all, checkpoints, verify. No training mode.' >&2; exit 2 ;;
esac
