"""Training-speed options that the training and profiling scripts switch on."""
import argparse
from typing import cast

from pydantic import BaseModel
import torch
from torch import nn

from face_verification.layers import PoolBackend


class SpeedOptions(BaseModel):
    """Which optimizations of the training step are on."""

    pool_backend: PoolBackend
    merge_pointwise: bool
    compile_model: bool
    compile_optimizer: bool
    transfer_uint8: bool
    cudnn_benchmark: bool


def add_speed_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one command-line switch per optimization."""
    parser.add_argument("--pool-backend", choices=["eager", "triton"], required=True)
    parser.add_argument("--merge-pointwise", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-optimizer", action="store_true")
    parser.add_argument("--transfer-uint8", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")


def read_speed_options(arguments: argparse.Namespace) -> SpeedOptions:
    """Collect the switches of the command line."""
    return SpeedOptions(
        pool_backend=cast(PoolBackend, arguments.pool_backend),
        merge_pointwise=arguments.merge_pointwise,
        compile_model=arguments.compile,
        compile_optimizer=arguments.compile_optimizer,
        transfer_uint8=arguments.transfer_uint8,
        cudnn_benchmark=arguments.cudnn_benchmark,
    )


def apply_global_options(options: SpeedOptions) -> None:
    """Let cuDNN time its convolution algorithms when asked."""
    torch.backends.cudnn.benchmark = options.cudnn_benchmark


def compile_when_asked(model: nn.Module, options: SpeedOptions) -> nn.Module:
    """Return the model compiled with torch.compile, or the model itself."""
    if not options.compile_model:
        return model
    compiled = torch.compile(model)  # pyright: ignore[reportUnknownMemberType]
    return cast(nn.Module, compiled)
