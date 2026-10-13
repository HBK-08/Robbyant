#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbotvla
[[ "${CONDA_PREFIX}" == /root/miniconda3/envs/lingbotvla ]]
artifact_dir=/home/lmy/Robbyant/deployment/lingbotvla-env-20261013
repo_dir=/home/lmy/Robbyant/lingbot-vla-v2
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INPUT=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_CACHE_DIR="${artifact_dir}/pip-cache"
export TMPDIR="${artifact_dir}/tmp"
export GIT_TERMINAL_PROMPT=0
unset PIP_CONSTRAINT PIP_FIND_LINKS
cd "${repo_dir}"
# Run the official editable MoGe command with its dependencies enabled. Apply
# only explicit repository pins, with line 175's Hub override taking precedence.
# This constraint addition is a documented repair, not an unchanged script run.
python -m pip install \
  -c "${artifact_dir}/official-final-pins.txt" \
  --report "${artifact_dir}/official-alignment-install.json" \
  -e "${repo_dir}/lingbotvla/models/vla/vision_models/MoGe" \
  huggingface-hub==0.34.0
python "${artifact_dir}/audit_official_alignment.py"
cp "${artifact_dir}/verification.json" "${artifact_dir}/verification-before-official-alignment.json"
cp "${artifact_dir}/verification.log" "${artifact_dir}/verification-before-official-alignment.log"
bash "${artifact_dir}/verify-environment.sh" > "${artifact_dir}/verification.log" 2>&1
python "${artifact_dir}/audit_official_alignment.py"
