"""Profile training steps of FaceNet NN2 on real Pins batches.

Every step is timed in two parts: the wait for the next batch from the data
loader and the GPU work of the step (synchronized at its end). Then
torch.profiler records a few steps and the operators are ranked by their own
GPU time. Writes runs/profiles/<run name>/summary.json and operators.txt.

Usage (from the repository root):
  python -m scripts.profile_training_step --pool-backend eager --run-name eager
  python -m scripts.profile_training_step --pool-backend triton --compile \
      --run-name triton_compiled
"""
import argparse
from collections.abc import Callable, Iterable, Iterator
import statistics
import time
from typing import Protocol, cast

from pydantic import BaseModel
import torch
from torch import nn
from torch.profiler import (
    ProfilerAction,
    ProfilerActivity,
    profile,
    schedule,  # pyright: ignore[reportUnknownVariableType]
)

from face_verification.identity_split import load_split
from face_verification.layers import PoolBackend
from face_verification.loaders import build_training_loader
from face_verification.networks import FaceNetNN2
from face_verification.training_loop import (
    TrainingHyperparameters,
    TrainingParts,
    build_training_parts,
    train_step,
)
from face_verification.training_settings import (
    BYTES_PER_GIBIBYTE,
    COMPILE_WARMUP_STEPS,
    MICROSECONDS_PER_MILLISECOND,
    NN2_LEARNING_RATE,
    NN2_WARMUP_EPOCHS,
    NN2_WEIGHT_DECAY,
    PROFILE_ACTIVE_STEPS,
    PROFILE_TABLE_ROWS,
    PROFILE_TIMED_STEPS,
    PROFILE_WAIT_STEPS,
    PROFILE_WARMUP_STEPS,
    RUNS_DIRECTORY,
    SPLIT_PATH,
)

Batch = tuple[torch.Tensor, torch.Tensor]
ProfilerPlan = Callable[[int], ProfilerAction]


class AveragedEvent(Protocol):
    """The fields of a torch.profiler operator average that are reported."""

    key: str
    count: int
    self_device_time_total: float


class OperatorTime(BaseModel):
    """Own GPU time of one operator over the profiled steps."""

    name: str
    calls: int
    gpu_milliseconds_per_step: float


class ProfileSummary(BaseModel):
    """Step timing, memory and the most expensive operators of one setup."""

    pool_backend: str
    compiled: bool
    timed_steps: int
    data_wait_seconds: float
    gpu_step_seconds: float
    peak_gpu_gibibytes: float
    top_operators: list[OperatorTime]


def parse_arguments() -> argparse.Namespace:
    """Read the pooling backend, compilation and the run name."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-backend", choices=["eager", "triton"], required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def prepare(arguments: argparse.Namespace) -> tuple[nn.Module, TrainingParts]:
    """Build the network, compile it when asked, and its optimizer parts."""
    model: nn.Module = FaceNetNN2(cast(PoolBackend, arguments.pool_backend)).cuda()
    hyperparameters = TrainingHyperparameters(
        learning_rate=NN2_LEARNING_RATE,
        weight_decay=NN2_WEIGHT_DECAY,
        warmup_epochs=NN2_WARMUP_EPOCHS,
        epochs=1,
    )
    parts = build_training_parts(model, hyperparameters)
    if arguments.compile:
        compiled = torch.compile(model)  # pyright: ignore[reportUnknownMemberType]
        model = cast(nn.Module, compiled)
    return model.train(), parts


def run_steps(
    model: nn.Module,
    parts: TrainingParts,
    batches: Iterator[Batch],
    count: int,
) -> list[tuple[float, float]]:
    """Run steps and return the data wait and the GPU time of each."""
    timings: list[tuple[float, float]] = []
    for _ in range(count):
        started = time.perf_counter()
        images, labels = next(batches)
        loaded = time.perf_counter()
        train_step(model, images.cuda(non_blocking=True), labels.cuda(), parts)
        torch.cuda.synchronize()
        timings.append((loaded - started, time.perf_counter() - loaded))
    return timings


def profile_steps(
    model: nn.Module,
    parts: TrainingParts,
    batches: Iterator[Batch],
) -> list[OperatorTime]:
    """Record steps with torch.profiler and rank operators by own GPU time."""
    plan = cast(
        ProfilerPlan,
        schedule(
            wait=PROFILE_WAIT_STEPS,
            warmup=PROFILE_WARMUP_STEPS,
            active=PROFILE_ACTIVE_STEPS,
        ),
    )
    activities: list[ProfilerActivity] = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    steps = PROFILE_WAIT_STEPS + PROFILE_WARMUP_STEPS + PROFILE_ACTIVE_STEPS
    with profile(activities=activities, schedule=plan) as profiler:
        for _ in range(steps):
            run_steps(model, parts, batches, 1)
            profiler.step()
    return rank_operators(profiler)


def rank_operators(profiler: profile) -> list[OperatorTime]:
    """Return the operators with the most own GPU time per profiled step."""
    averages: object = profiler.key_averages()
    events = cast(Iterable[AveragedEvent], averages)
    ranked = sorted(events, key=lambda event: event.self_device_time_total)
    scale = 1 / (MICROSECONDS_PER_MILLISECOND * PROFILE_ACTIVE_STEPS)
    return [
        OperatorTime(
            name=event.key,
            calls=event.count,
            gpu_milliseconds_per_step=event.self_device_time_total * scale,
        )
        for event in reversed(ranked[-PROFILE_TABLE_ROWS:])
    ]


def summarize(
    arguments: argparse.Namespace,
    timings: list[tuple[float, float]],
    operators: list[OperatorTime],
) -> ProfileSummary:
    """Collect the mean timings, the peak memory and the ranked operators."""
    return ProfileSummary(
        pool_backend=arguments.pool_backend,
        compiled=arguments.compile,
        timed_steps=len(timings),
        data_wait_seconds=statistics.mean(wait for wait, _ in timings),
        gpu_step_seconds=statistics.mean(step for _, step in timings),
        peak_gpu_gibibytes=torch.cuda.max_memory_allocated() / BYTES_PER_GIBIBYTE,
        top_operators=operators,
    )


def write_summary(arguments: argparse.Namespace, summary: ProfileSummary) -> None:
    """Write the summary as JSON and the operator ranking as a text table."""
    directory = RUNS_DIRECTORY / "profiles" / arguments.run_name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    lines: list[str] = [
        f"{item.gpu_milliseconds_per_step:9.3f} ms {item.calls:6d}  {item.name}"
        for item in summary.top_operators
    ]
    (directory / "operators.txt").write_text("\n".join(lines) + "\n")
    print(summary.model_dump_json(indent=2))


def main() -> None:
    """Warm up, time steps, profile steps and write the results."""
    arguments = parse_arguments()
    model, parts = prepare(arguments)
    batches = iter(build_training_loader(load_split(SPLIT_PATH)))
    run_steps(model, parts, batches, COMPILE_WARMUP_STEPS)
    torch.cuda.reset_peak_memory_stats()
    timings = run_steps(model, parts, batches, PROFILE_TIMED_STEPS)
    operators = profile_steps(model, parts, batches)
    write_summary(arguments, summarize(arguments, timings, operators))


if __name__ == "__main__":
    main()
