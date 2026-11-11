#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source slurm/settings.sh
mkdir -p "${SLURM_LOG_DIR}"

submit() {
    local name="$1"
    shift
    sbatch --parsable \
        --job-name="${name}" \
        --output="${SLURM_LOG_DIR}/%x-%j.log" \
        --gres=gpu:1 \
        --cpus-per-task="${GPU_JOB_CPUS}" \
        --mem="${GPU_JOB_MEMORY}" \
        slurm/run_job.sh "$@"
}

submit facenet-tests pytest -q
submit facenet-benchmark scripts.benchmark_l2_pool
for backend in eager triton; do
    submit "profile-${backend}" scripts.profile_training_step \
        --pool-backend "${backend}" --run-name "${backend}"
    submit "profile-${backend}-compiled" scripts.profile_training_step \
        --pool-backend "${backend}" --compile --run-name "${backend}_compiled"
done
submit train-nn2-eager scripts.train_model \
    --model nn2 --pool-backend eager --run-name nn2_eager
submit train-nn2-triton-compiled scripts.train_model \
    --model nn2 --pool-backend triton --compile --run-name nn2_triton_compiled
squeue --format="%.8i %.28j %.10T %.20R"
