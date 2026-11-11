"""Fused Triton kernels of L2 pooling with a 3x3 window, stride 1 and padding 1.

The eager version runs four kernels (square, window sum, add epsilon, square
root) and writes three intermediate tensors to GPU memory; its backward pass
runs four more. Here the forward kernel loads the window, sums the squares in
float32 registers and writes only the result, and the backward kernel computes
the input gradient in one pass. For an input x at a position the gradient is
x times the sum of grad_output / output over the windows that contain the
position; with this window those are the outputs at its 3x3 neighbours, so no
atomic additions are needed. Only the input and the output are saved for the
backward pass, as in eager mode.
"""
from collections.abc import Callable
from typing import cast

import torch
from torch.autograd.function import FunctionCtx
import triton

from face_verification.training_settings import (
    L2_POOL_EPSILON,
    L2_POOL_KERNEL_SIZE,
    L2_POOL_PADDING,
    L2_POOL_STRIDE,
    TRITON_BLOCK_SIZE,
    TRITON_WINDOW_RADIUS,
)


@triton.jit
def block_offsets(BLOCK_SIZE: triton.language.constexpr) -> triton.language.tensor:
    """Return the flat offsets of the elements of this block."""
    block_start = triton.language.program_id(0) * BLOCK_SIZE
    return block_start + triton.language.arange(0, BLOCK_SIZE)


@triton.jit
def neighbour_valid(
    offsets: triton.language.tensor,
    total: int,
    height: int,
    width: int,
    row_shift: triton.language.constexpr,
    column_shift: triton.language.constexpr,
) -> triton.language.tensor:
    """Return whether the shifted position of every element lies in its map."""
    neighbour_row = (offsets // width) % height + row_shift
    neighbour_column = offsets % width + column_shift
    valid = (offsets < total) & (neighbour_row >= 0) & (neighbour_row < height)
    return valid & (neighbour_column >= 0) & (neighbour_column < width)


@triton.jit
def sum_window_squares(
    input_pointer: triton.language.tensor,
    offsets: triton.language.tensor,
    total: int,
    height: int,
    width: int,
    BLOCK_SIZE: triton.language.constexpr,
    RADIUS: triton.language.constexpr,
) -> triton.language.tensor:
    """Sum the squares of the window around every position in float32."""
    squares = triton.language.zeros([BLOCK_SIZE], dtype=triton.language.float32)
    for row_shift in triton.language.static_range(-RADIUS, RADIUS + 1):
        for column_shift in triton.language.static_range(-RADIUS, RADIUS + 1):
            valid = neighbour_valid(
                offsets, total, height, width, row_shift, column_shift
            )
            shifted = offsets + row_shift * width + column_shift
            value = triton.language.load(input_pointer + shifted, mask=valid, other=0.0)
            widened = value.to(triton.language.float32)
            squares += widened * widened
    return squares


@triton.jit
def l2_pool_forward_kernel(
    input_pointer: triton.language.tensor,
    output_pointer: triton.language.tensor,
    height: int,
    width: int,
    total: int,
    BLOCK_SIZE: triton.language.constexpr,
    RADIUS: triton.language.constexpr,
    EPSILON: triton.language.constexpr,
) -> None:
    """Write sqrt(sum of squares over the window + epsilon) for one block."""
    offsets = block_offsets(BLOCK_SIZE)
    squares = sum_window_squares(
        input_pointer, offsets, total, height, width, BLOCK_SIZE, RADIUS
    )
    result = triton.language.sqrt(squares + EPSILON)
    triton.language.store(
        output_pointer + offsets,
        result.to(output_pointer.dtype.element_ty),
        mask=offsets < total,
    )


@triton.jit
def load_gradient_ratio(
    gradient_pointer: triton.language.tensor,
    output_pointer: triton.language.tensor,
    shifted: triton.language.tensor,
    valid: triton.language.tensor,
) -> triton.language.tensor:
    """Load grad_output / output at the given positions, zero outside the map."""
    gradient = triton.language.load(gradient_pointer + shifted, mask=valid, other=0.0)
    output = triton.language.load(output_pointer + shifted, mask=valid, other=1.0)
    return gradient.to(triton.language.float32) / output.to(triton.language.float32)


@triton.jit
def sum_gradient_ratios(
    gradient_pointer: triton.language.tensor,
    output_pointer: triton.language.tensor,
    offsets: triton.language.tensor,
    total: int,
    height: int,
    width: int,
    BLOCK_SIZE: triton.language.constexpr,
    RADIUS: triton.language.constexpr,
) -> triton.language.tensor:
    """Sum grad_output / output over the windows that contain every position."""
    ratios = triton.language.zeros([BLOCK_SIZE], dtype=triton.language.float32)
    for row_shift in triton.language.static_range(-RADIUS, RADIUS + 1):
        for column_shift in triton.language.static_range(-RADIUS, RADIUS + 1):
            valid = neighbour_valid(
                offsets, total, height, width, row_shift, column_shift
            )
            shifted = offsets + row_shift * width + column_shift
            ratios += load_gradient_ratio(
                gradient_pointer, output_pointer, shifted, valid
            )
    return ratios


@triton.jit
def l2_pool_backward_kernel(
    input_pointer: triton.language.tensor,
    output_pointer: triton.language.tensor,
    gradient_output_pointer: triton.language.tensor,
    gradient_input_pointer: triton.language.tensor,
    height: int,
    width: int,
    total: int,
    BLOCK_SIZE: triton.language.constexpr,
    RADIUS: triton.language.constexpr,
) -> None:
    """Write x * sum(grad_output / output) over the neighbour windows."""
    offsets = block_offsets(BLOCK_SIZE)
    ratios = sum_gradient_ratios(
        gradient_output_pointer,
        output_pointer,
        offsets,
        total,
        height,
        width,
        BLOCK_SIZE,
        RADIUS,
    )
    features = triton.language.load(
        input_pointer + offsets, mask=offsets < total, other=0.0
    )
    triton.language.store(
        gradient_input_pointer + offsets,
        (features.to(triton.language.float32) * ratios).to(features.dtype),
        mask=offsets < total,
    )


def count_blocks(features: torch.Tensor) -> tuple[int]:
    """Return the launch grid: one program per block of elements."""
    return (triton.cdiv(features.numel(), TRITON_BLOCK_SIZE),)


def launch_forward(features: torch.Tensor, output: torch.Tensor) -> None:
    """Run the forward kernel over a contiguous NCHW tensor."""
    l2_pool_forward_kernel[count_blocks(features)](
        features,
        output,
        features.shape[-2],
        features.shape[-1],
        features.numel(),
        BLOCK_SIZE=TRITON_BLOCK_SIZE,
        RADIUS=TRITON_WINDOW_RADIUS,
        EPSILON=L2_POOL_EPSILON,
    )


def launch_backward(
    features: torch.Tensor,
    output: torch.Tensor,
    gradient_output: torch.Tensor,
    gradient_input: torch.Tensor,
) -> None:
    """Run the backward kernel over contiguous NCHW tensors."""
    l2_pool_backward_kernel[count_blocks(features)](
        features,
        output,
        gradient_output,
        gradient_input,
        features.shape[-2],
        features.shape[-1],
        features.numel(),
        BLOCK_SIZE=TRITON_BLOCK_SIZE,
        RADIUS=TRITON_WINDOW_RADIUS,
    )


class FusedL2Pool(torch.autograd.Function):
    """L2 pooling whose forward and backward passes are single Triton kernels."""

    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        context: FunctionCtx,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Pool the features and keep the input and the output for backward."""
        features = features.contiguous()
        output = torch.empty_like(features)
        launch_forward(features, output)
        context.save_for_backward(features, output)
        return output

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        context: FunctionCtx,
        gradient_output: torch.Tensor,
    ) -> torch.Tensor:
        """Return the gradient of the loss with respect to the input."""
        saved: object = getattr(context, "saved_tensors")
        features, output = cast(tuple[torch.Tensor, torch.Tensor], saved)
        gradient_input = torch.empty_like(features)
        launch_backward(features, output, gradient_output.contiguous(), gradient_input)
        return gradient_input


def check_window(kernel_size: int, stride: int, padding: int) -> None:
    """Refuse window settings that the fused kernels do not implement."""
    supported = (L2_POOL_KERNEL_SIZE, L2_POOL_STRIDE, L2_POOL_PADDING)
    if (kernel_size, stride, padding) != supported:
        raise ValueError(f"fused L2 pooling supports only {supported}")


def l2_pool_triton(
    features: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """L2-pool the features with the fused kernels."""
    check_window(kernel_size, stride, padding)
    pool = cast(Callable[[torch.Tensor], torch.Tensor], getattr(FusedL2Pool, "apply"))
    return pool(features)
