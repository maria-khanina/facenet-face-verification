"""Fused Triton kernels of L2 pooling with a 3x3 window, stride 1 and padding 1.

The eager version runs three kernels forward (square, window sum, square root)
and three backward, and keeps the squared input for the backward pass. Here
the forward kernel loads the window, sums the squares in float32 registers and
writes only the result; the backward kernel computes the input gradient in one
pass as x times the sum of grad_output / output over the 3x3 neighbours, so no
atomic additions are needed. Addresses come from the strides of the tensor, so
the same kernels run on NCHW and on channels-last (NHWC) tensors and keep the
layout of their input. Both passes are registered with torch.library as
operators, so torch.compile sees their inputs, outputs and layouts.
"""
from collections.abc import Callable
from typing import Any, cast

import torch
import triton

from face_verification.training_settings import (
    L2_POOL_EPSILON,
    L2_POOL_KERNEL_SIZE,
    L2_POOL_PADDING,
    L2_POOL_STRIDE,
    TRITON_BLOCK_SIZE,
    TRITON_WINDOW_RADIUS,
)

OPERATOR_NAMESPACE = "face_verification"
LIBRARY: Any = torch.library


@triton.jit
def block_offsets(BLOCK_SIZE: triton.language.constexpr) -> triton.language.tensor:
    """Return the memory offsets of the elements of this block."""
    block_start = triton.language.program_id(0) * BLOCK_SIZE
    return block_start + triton.language.arange(0, BLOCK_SIZE)


@triton.jit
def neighbour_valid(
    offsets: triton.language.tensor,
    total: int,
    height: int,
    width: int,
    row_stride: int,
    column_stride: int,
    row_shift: triton.language.constexpr,
    column_shift: triton.language.constexpr,
) -> triton.language.tensor:
    """Return whether the shifted position of every element lies in its map."""
    neighbour_row = (offsets // row_stride) % height + row_shift
    neighbour_column = (offsets // column_stride) % width + column_shift
    valid = (offsets < total) & (neighbour_row >= 0) & (neighbour_row < height)
    return valid & (neighbour_column >= 0) & (neighbour_column < width)


@triton.jit
def sum_window_squares(
    input_pointer: triton.language.tensor,
    offsets: triton.language.tensor,
    total: int,
    height: int,
    width: int,
    row_stride: int,
    column_stride: int,
    BLOCK_SIZE: triton.language.constexpr,
    RADIUS: triton.language.constexpr,
) -> triton.language.tensor:
    """Sum the squares of the window around every position in float32."""
    squares = triton.language.zeros([BLOCK_SIZE], dtype=triton.language.float32)
    for row_shift in triton.language.static_range(-RADIUS, RADIUS + 1):
        for column_shift in triton.language.static_range(-RADIUS, RADIUS + 1):
            valid = neighbour_valid(
                offsets,
                total,
                height,
                width,
                row_stride,
                column_stride,
                row_shift,
                column_shift,
            )
            shift = row_shift * row_stride + column_shift * column_stride
            value = triton.language.load(
                input_pointer + offsets + shift, mask=valid, other=0.0
            )
            widened = value.to(triton.language.float32)
            squares += widened * widened
    return squares


@triton.jit
def l2_pool_forward_kernel(
    input_pointer: triton.language.tensor,
    output_pointer: triton.language.tensor,
    height: int,
    width: int,
    row_stride: int,
    column_stride: int,
    total: int,
    BLOCK_SIZE: triton.language.constexpr,
    RADIUS: triton.language.constexpr,
    EPSILON: triton.language.constexpr,
) -> None:
    """Write sqrt(sum of squares over the window + epsilon) for one block."""
    offsets = block_offsets(BLOCK_SIZE)
    squares = sum_window_squares(
        input_pointer,
        offsets,
        total,
        height,
        width,
        row_stride,
        column_stride,
        BLOCK_SIZE,
        RADIUS,
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
    row_stride: int,
    column_stride: int,
    BLOCK_SIZE: triton.language.constexpr,
    RADIUS: triton.language.constexpr,
) -> triton.language.tensor:
    """Sum grad_output / output over the windows that contain every position."""
    ratios = triton.language.zeros([BLOCK_SIZE], dtype=triton.language.float32)
    for row_shift in triton.language.static_range(-RADIUS, RADIUS + 1):
        for column_shift in triton.language.static_range(-RADIUS, RADIUS + 1):
            valid = neighbour_valid(
                offsets,
                total,
                height,
                width,
                row_stride,
                column_stride,
                row_shift,
                column_shift,
            )
            shift = row_shift * row_stride + column_shift * column_stride
            ratios += load_gradient_ratio(
                gradient_pointer, output_pointer, offsets + shift, valid
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
    row_stride: int,
    column_stride: int,
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
        row_stride,
        column_stride,
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


def is_dense(features: torch.Tensor) -> bool:
    """Return whether the tensor fills its memory in NCHW or NHWC order."""
    channels_last = features.is_contiguous(memory_format=torch.channels_last)
    return features.is_contiguous() or channels_last


def densify(features: torch.Tensor) -> torch.Tensor:
    """Return the tensor itself when dense, otherwise an NCHW copy."""
    return features if is_dense(features) else features.contiguous()


def match_layout(gradient: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return the gradient with the same strides as the reference tensor."""
    if gradient.stride() == reference.stride():
        return gradient
    return torch.empty_like(reference).copy_(gradient)


def count_blocks(features: torch.Tensor) -> tuple[int]:
    """Return the launch grid: one program per block of elements."""
    return (triton.cdiv(features.numel(), TRITON_BLOCK_SIZE),)


def spatial_geometry(features: torch.Tensor) -> tuple[int, int, int, int]:
    """Return the height, the width and the strides of a row and of a column."""
    return (
        features.shape[-2],
        features.shape[-1],
        features.stride(-2),
        features.stride(-1),
    )


def launcher(kernel: object, features: torch.Tensor) -> Any:
    """Return the kernel wrapped for torch.compile and bound to its grid."""
    wrapped: Any = LIBRARY.wrap_triton(kernel)
    return wrapped[count_blocks(features)]


def pool_forward_launch(features: torch.Tensor) -> torch.Tensor:
    """Pool a dense NCHW or NHWC tensor and keep its layout."""
    output = torch.empty_like(features)
    launcher(l2_pool_forward_kernel, features)(
        features,
        output,
        *spatial_geometry(features),
        features.numel(),
        BLOCK_SIZE=TRITON_BLOCK_SIZE,
        RADIUS=TRITON_WINDOW_RADIUS,
        EPSILON=L2_POOL_EPSILON,
    )
    return output


def pool_backward_launch(
    features: torch.Tensor,
    output: torch.Tensor,
    gradient_output: torch.Tensor,
) -> torch.Tensor:
    """Return the input gradient, in the layout of the input."""
    gradient_output = match_layout(gradient_output, features)
    gradient_input = torch.empty_like(features)
    launcher(l2_pool_backward_kernel, features)(
        features,
        output,
        gradient_output,
        gradient_input,
        *spatial_geometry(features),
        features.numel(),
        BLOCK_SIZE=TRITON_BLOCK_SIZE,
        RADIUS=TRITON_WINDOW_RADIUS,
    )
    return gradient_input


def save_for_backward(**values: Any) -> None:
    """Keep the input and the output of the forward pass.

    torch.library passes the context, the inputs and the output by the names
    ctx, inputs and output.
    """
    inputs = cast(tuple[torch.Tensor], values["inputs"])
    values["ctx"].save_for_backward(inputs[0], values["output"])


def pool_backward(context: Any, gradient_output: torch.Tensor) -> torch.Tensor:
    """Return the gradient of the loss with respect to the input."""
    saved = cast(tuple[torch.Tensor, torch.Tensor], context.saved_tensors)
    return l2_pool_backward(saved[0], saved[1], gradient_output)


def register_operator(name: str, launch: Callable[..., torch.Tensor]) -> Any:
    """Register a launch function as a torch.library operator of Triton kernels."""
    return LIBRARY.triton_op(f"{OPERATOR_NAMESPACE}::{name}", mutates_args=())(launch)


FORWARD_OPERATOR = register_operator("l2_pool_forward", pool_forward_launch)
BACKWARD_OPERATOR = register_operator("l2_pool_backward", pool_backward_launch)
FORWARD_OPERATOR.register_autograd(pool_backward, setup_context=save_for_backward)
l2_pool_forward = cast(Callable[[torch.Tensor], torch.Tensor], FORWARD_OPERATOR)
l2_pool_backward = cast(
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    BACKWARD_OPERATOR,
)


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
    return l2_pool_forward(densify(features))
