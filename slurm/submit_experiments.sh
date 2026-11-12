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

all_optimizations="--pool-backend triton --compile --transfer-uint8 --cudnn-benchmark --compile-optimizer --merge-pointwise"

profile() {
    local run_name="$1"
    shift
    submit "profile-${run_name}" scripts.profile_training_step "$@" --run-name "${run_name}"
}

submit facenet-tests pytest -q
submit facenet-benchmark scripts.benchmark_l2_pool
profile eager --pool-backend eager
profile eager_triton --pool-backend triton
profile compiled --pool-backend eager --compile
profile compiled_uint8 --pool-backend eager --compile --transfer-uint8
profile compiled_cudnn_benchmark --pool-backend eager --compile --cudnn-benchmark
profile compiled_optimizer --pool-backend eager --compile --compile-optimizer
profile compiled_triton --pool-backend triton --compile
profile compiled_merged_pointwise --pool-backend eager --compile --merge-pointwise
profile all_optimizations ${all_optimizations}
submit train-nn2-eager scripts.train_model --model nn2 --pool-backend eager \
    --run-name nn2_eager
submit train-nn2-all-optimizations scripts.train_model --model nn2 \
    ${all_optimizations} --run-name nn2_all_optimizations
squeue --format="%.8i %.32j %.10T %.20R"
