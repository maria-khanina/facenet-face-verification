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


class PointwiseBranches(nn.Module):
    """The 1x1 convolutions that start several branches from the same input.

    Merged, they are one convolution with all output channels, split after
    batch normalization and ReLU; both act per channel, so the result is the
    same as separate convolutions while the input is read once.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: tuple[int, ...],
        merged: bool,
    ) -> None:
        """Create one merged convolution or one convolution per branch."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.split_sizes: list[int] = list(out_channels)
        self.merged = merged
        widths: list[int] = [sum(out_channels)] if merged else list(out_channels)
        self.convolutions = nn.ModuleList(
            [pointwise_convolution(in_channels, width) for width in widths]
        )

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Return the output of every branch."""
        if self.merged:
            merged = self.convolutions[0](features)
            return list(torch.split(merged, self.split_sizes, dim=1))
        return [convolution(features) for convolution in self.convolutions]


def spatial_convolution(
    in_channels: int,
    out_channels: int,
    stride: int,
) -> BasicConv2d:
    """Return the 3x3 convolution that follows a 1x1 reduction."""
    return BasicConv2d(in_channels, out_channels, SMALL_KERNEL, stride, SMALL_PADDING)


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

    def __init__(
        self,
        layout: InceptionLayout,
        backend: PoolBackend,
        merge_pointwise: bool,
    ) -> None:
        """Create the four branches from the module layout."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        channels, single, reduce3, out3, reduce5, out5, projection, pool = layout
        self.pointwise = PointwiseBranches(
            channels, (single, reduce3, reduce5), merge_pointwise
        )
        self.convolution3x3 = spatial_convolution(reduce3, out3, 1)
        self.convolution5x5 = spatial_convolution(reduce5, out5, 1)
        self.branch_pool = nn.Sequential(
            build_pool_layer(pool, backend),
            pointwise_convolution(channels, projection),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Concatenate the outputs of the branches along the channels."""
        single, reduced3, reduced5 = self.pointwise(features)
        branches: list[torch.Tensor] = [
            single,
            self.convolution3x3(reduced3),
            self.convolution5x5(reduced5),
            self.branch_pool(features),
        ]
        return torch.cat(branches, dim=1)


class InceptionReduction(nn.Module):
    """Inception module that halves the spatial size."""

    def __init__(self, layout: ReductionLayout, merge_pointwise: bool) -> None:
        """Create the strided 3x3, 5x5 and max pool branches."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        channels, reduce3, out3, reduce5, out5 = layout
        self.pointwise = PointwiseBranches(
            channels, (reduce3, reduce5), merge_pointwise
        )
        self.convolution3x3 = spatial_convolution(reduce3, out3, REDUCTION_STRIDE)
        self.convolution5x5 = spatial_convolution(reduce5, out5, REDUCTION_STRIDE)
        self.branch_pool = nn.MaxPool2d(
            SMALL_KERNEL,
            stride=REDUCTION_STRIDE,
            padding=SMALL_PADDING,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Concatenate the outputs of the branches along the channels."""
        reduced3, reduced5 = self.pointwise(features)
        branches: list[torch.Tensor] = [
            self.convolution3x3(reduced3),
            self.convolution5x5(reduced5),
            self.branch_pool(features),
        ]
        return torch.cat(branches, dim=1)
