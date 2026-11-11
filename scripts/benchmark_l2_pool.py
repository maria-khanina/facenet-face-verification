"""Time L2 pooling alone on the input shapes it has inside FaceNet NN2.

Three implementations are compared on a forward and backward pass under fp16
autocast, as in training: the eager PyTorch operations, the same operations
compiled by torch.compile, and the fused Triton kernels. For each shape the
script also reports the peak extra GPU memory of one pass and the largest
difference of the output and the input gradient from a float32 reference.
Writes runs/l2_pool_benchmark.json.

Usage (from the repository root):
  python -m scripts.benchmark_l2_pool
"""
from collections.abc import Callable
import json
from typing import cast

from pydantic import BaseModel
import torch
from torch import nn
import torch.nn.functional as functional
from triton import testing

from face_verification.architecture_settings import IMAGE_CHANNELS
from face_verification.l2_pool_triton import l2_pool_triton
from face_verification.layers import L2Pool, PoolFunction, l2_pool_eager
from face_verification.networks import FaceNetNN2
from face_verification.training_settings import (
    BENCHMARK_BATCH_SIZE,
    BENCHMARK_MEASURE_MILLISECONDS,
    BENCHMARK_WARMUP_MILLISECONDS,
    BYTES_PER_MEBIBYTE,
    INPUT_SIZE,
    L2_POOL_KERNEL_SIZE,
    L2_POOL_PADDING,
    L2_POOL_STRIDE,
    RUNS_DIRECTORY,
)

Shape = tuple[int, int, int, int]


class VariantTiming(BaseModel):
    """Speed, memory and error of one implementation on one shape."""

    variant: str
    milliseconds: float
    peak_extra_mebibytes: float
    output_error: float
    gradient_error: float


class ShapeResult(BaseModel):
    """All implementations on one input shape and how many layers have it."""

    shape: list[int]
    layers: int
    variants: list[VariantTiming]


def attach_shape_recorder(model: nn.Module, shapes: dict[Shape, int]) -> None:
    """Count the input shape of every L2 pooling call of the model."""

    def remember(module: nn.Module, inputs: tuple[torch.Tensor]) -> None:
        shape = cast(Shape, tuple(inputs[0].shape))
        shapes[shape] = shapes.get(shape, 0) + 1

    for module in model.modules():
        if isinstance(module, L2Pool):
            module.register_forward_pre_hook(remember)


def collect_pool_shapes() -> dict[Shape, int]:
    """Run NN2 once and count the input shapes of its L2 pooling layers."""
    model = FaceNetNN2("eager").cuda()
    shapes: dict[Shape, int] = {}
    attach_shape_recorder(model, shapes)
    images = torch.zeros(BENCHMARK_BATCH_SIZE, IMAGE_CHANNELS, *INPUT_SIZE)
    with torch.no_grad():
        model(images.cuda())
    return shapes


def pool(function: PoolFunction, features: torch.Tensor) -> torch.Tensor:
    """Apply an implementation with the window of the network."""
    return function(features, L2_POOL_KERNEL_SIZE, L2_POOL_STRIDE, L2_POOL_PADDING)


def make_pass(
    function: PoolFunction,
    features: torch.Tensor,
    gradient: torch.Tensor,
) -> Callable[[], None]:
    """Return a forward and backward pass under fp16 autocast."""

    def run_pass() -> None:
        features.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = pool(function, features)
        upstream = gradient.to(output.dtype)
        output.backward(upstream)  # pyright: ignore[reportUnknownMemberType]

    return run_pass


def reference(
    features: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the float32 eager output and input gradient."""
    exact = features.detach().float().requires_grad_()
    output = pool(l2_pool_eager, exact)
    output.backward(gradient.float())  # pyright: ignore[reportUnknownMemberType]
    return output.detach(), cast(torch.Tensor, exact.grad)


def measure_memory(run_pass: Callable[[], None]) -> float:
    """Return the peak GPU memory of one pass above what was allocated before."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    run_pass()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / BYTES_PER_MEBIBYTE


def measure_errors(
    function: PoolFunction,
    features: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[float, float]:
    """Return the largest output and gradient differences from float32."""
    exact_output, exact_gradient = reference(features, gradient)
    make_pass(function, features, gradient)()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = pool(function, features.detach())
    output_error = (output.float() - exact_output).abs().max().item()
    gradient_error = (cast(torch.Tensor, features.grad).float() - exact_gradient).abs()
    return output_error, gradient_error.max().item()


def time_variant(
    name: str,
    function: PoolFunction,
    features: torch.Tensor,
    gradient: torch.Tensor,
) -> VariantTiming:
    """Measure speed, memory and error of one implementation."""
    run_pass = make_pass(function, features, gradient)
    milliseconds = testing.do_bench(
        run_pass,
        warmup=BENCHMARK_WARMUP_MILLISECONDS,
        rep=BENCHMARK_MEASURE_MILLISECONDS,
    )
    output_error, gradient_error = measure_errors(function, features, gradient)
    return VariantTiming(
        variant=name,
        milliseconds=milliseconds,
        peak_extra_mebibytes=measure_memory(run_pass),
        output_error=output_error,
        gradient_error=gradient_error,
    )


def build_variants() -> dict[str, PoolFunction]:
    """Return the three implementations under their report names."""
    compiled = torch.compile(l2_pool_eager)  # pyright: ignore[reportUnknownMemberType]
    return {
        "eager": l2_pool_eager,
        "torch.compile": cast(PoolFunction, compiled),
        "triton": l2_pool_triton,
    }


def benchmark_shape(shape: Shape, layers: int) -> ShapeResult:
    """Time every implementation on fp16 post-ReLU features of one shape."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    features = functional.relu(torch.randn(shape, device="cuda", generator=generator))
    features = features.half().requires_grad_()
    gradient = torch.randn(shape, device="cuda", generator=generator).half()
    timings: list[VariantTiming] = [
        time_variant(name, function, features, gradient)
        for name, function in build_variants().items()
    ]
    return ShapeResult(shape=list(shape), layers=layers, variants=timings)


def main() -> None:
    """Benchmark every L2 pooling shape of NN2 and write the results."""
    results: list[ShapeResult] = [
        benchmark_shape(shape, layers)
        for shape, layers in collect_pool_shapes().items()
    ]
    payload: list[dict[str, object]] = [result.model_dump() for result in results]
    RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    (RUNS_DIRECTORY / "l2_pool_benchmark.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
