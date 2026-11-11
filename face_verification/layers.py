"""Building blocks of the FaceNet NN2 network."""
from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as functional

from face_verification.architecture_settings import (
    REDUCTION_STRIDE,
    SMALL_KERNEL,
    SMALL_PADDING,
    InceptionLayout,
    PoolType,
    ReductionLayout,
)
from face_verification.training_settings import (
    L2_POOL_EPSILON,
    L2_POOL_KERNEL_SIZE,
    L2_POOL_PADDING,
    L2_POOL_STRIDE,
)

PoolBackend = Literal["eager", "triton"]
PoolFunction = Callable[[torch.Tensor, int, int, int], torch.Tensor]


def l2_pool_eager(
    features: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """L2-pool with separate PyTorch operations, as in the original notebook."""
    sum_squared = functional.avg_pool2d(
        features.pow(2),
        kernel_size,
        stride,
        padding,
        divisor_override=1,
    )
    return torch.sqrt(sum_squared + L2_POOL_EPSILON)


def select_pool_function(backend: PoolBackend) -> PoolFunction:
    """Return the L2 pooling implementation of the backend."""
    if backend == "triton":
        from face_verification.l2_pool_triton import l2_pool_triton

        return l2_pool_triton
    return l2_pool_eager


class BasicConv2d(nn.Module):
    """Convolution without bias, batch normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        """Create the convolution and its batch normalization."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Convolve, normalize and rectify."""
        return functional.relu(self.batch_norm(self.conv(features)), inplace=True)


def pointwise_convolution(in_channels: int, out_channels: int) -> BasicConv2d:
    """Return a 1x1 convolution block."""
    return BasicConv2d(in_channels, out_channels, 1, 1, 0)


def reduced_convolution(
    in_channels: int,
    reduced_channels: int,
    out_channels: int,
    stride: int,
) -> nn.Sequential:
    """Return a 1x1 reduction followed by a 3x3 convolution."""
    return nn.Sequential(
        pointwise_convolution(in_channels, reduced_channels),
        BasicConv2d(
            reduced_channels,
            out_channels,
            SMALL_KERNEL,
            stride,
            SMALL_PADDING,
        ),
    )


class L2Pool(nn.Module):
    """Square root of the sum of squares over each pooling window."""

    def __init__(self, backend: PoolBackend) -> None:
        """Select the implementation of the pooling."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.pool_function = select_pool_function(backend)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Pool with a 3x3 window, stride 1 and padding 1."""
        return self.pool_function(
            features,
            L2_POOL_KERNEL_SIZE,
            L2_POOL_STRIDE,
            L2_POOL_PADDING,
        )


def build_pool_layer(pool_type: PoolType, backend: PoolBackend) -> nn.Module:
    """Return the pooling of an inception module's pool branch."""
    if pool_type == "l2_pool":
        return L2Pool(backend)
    return nn.MaxPool2d(SMALL_KERNEL, stride=1, padding=SMALL_PADDING)


class NN2Inception(nn.Module):
    """Inception module with 1x1, 3x3, 5x5 and pool branches."""

    def __init__(self, layout: InceptionLayout, backend: PoolBackend) -> None:
        """Create the four branches from the module layout."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        channels, single, reduce3, out3, reduce5, out5, projection, pool = layout
        self.branch1x1 = pointwise_convolution(channels, single)
        self.branch3x3 = reduced_convolution(channels, reduce3, out3, 1)
        self.branch5x5 = reduced_convolution(channels, reduce5, out5, 1)
        self.branch_pool = nn.Sequential(
            build_pool_layer(pool, backend),
            pointwise_convolution(channels, projection),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Concatenate the outputs of the branches along the channels."""
        branches: list[torch.Tensor] = [
            self.branch1x1(features),
            self.branch3x3(features),
            self.branch5x5(features),
            self.branch_pool(features),
        ]
        return torch.cat(branches, dim=1)


class InceptionReduction(nn.Module):
    """Inception module that halves the spatial size."""

    def __init__(self, layout: ReductionLayout) -> None:
        """Create the strided 3x3, 5x5 and max pool branches."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        channels, reduce3, out3, reduce5, out5 = layout
        self.branch3x3 = reduced_convolution(channels, reduce3, out3, REDUCTION_STRIDE)
        self.branch5x5 = reduced_convolution(channels, reduce5, out5, REDUCTION_STRIDE)
        self.branch_pool = nn.MaxPool2d(
            SMALL_KERNEL,
            stride=REDUCTION_STRIDE,
            padding=SMALL_PADDING,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Concatenate the outputs of the branches along the channels."""
        branches: list[torch.Tensor] = [
            self.branch3x3(features),
            self.branch5x5(features),
            self.branch_pool(features),
        ]
        return torch.cat(branches, dim=1)
