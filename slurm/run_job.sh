#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source slurm/settings.sh
source "${FACENET_VENV}/bin/activate"
export FACENET_DATA_DIRECTORY

echo "job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-none} start=$(date --iso-8601=seconds)"
python -m "$@"
echo "end=$(date --iso-8601=seconds)"
