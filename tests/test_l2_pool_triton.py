"""The fused Triton kernels agree with eager L2 pooling (needs CUDA and Triton)."""
import importlib.util

import pytest
import torch
import torch.nn.functional as functional

from face_verification.layers import l2_pool_eager, select_pool_function
from tests.fixture_settings import (
    FLOAT16_GRADIENT_TOLERANCE,
    FLOAT16_OUTPUT_TOLERANCE,
    FLOAT32_GRADIENT_TOLERANCE,
    KERNEL_SHAPES,
    WINDOW,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="the fused kernels need a CUDA GPU and Triton",
)


def run_pass(
    features: torch.Tensor,
    gradient: torch.Tensor,
    triton: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the output and the input gradient of one implementation."""
    leaf = features.detach().clone().requires_grad_()
    pool = select_pool_function("triton") if triton else l2_pool_eager
    output = pool(leaf, WINDOW, 1, 1)
    output.backward(gradient)  # pyright: ignore[reportUnknownMemberType]
    assert leaf.grad is not None
    return output.detach(), leaf.grad


@pytest.mark.parametrize("shape", KERNEL_SHAPES)
def test_float32_matches_eager(shape: tuple[int, int, int, int]) -> None:
    """Output and gradient match eager pooling in float32."""
    features = functional.relu(torch.randn(shape, device="cuda"))
    gradient = torch.randn(shape, device="cuda")
    fused = run_pass(features, gradient, True)
    eager = run_pass(features, gradient, False)
    tolerance = FLOAT32_GRADIENT_TOLERANCE
    torch.testing.assert_close(fused[0], eager[0])
    torch.testing.assert_close(fused[1], eager[1], rtol=tolerance, atol=tolerance)


def test_float16_keeps_dtype_and_stays_close() -> None:
    """In float16 the output keeps the input type and is close to float32."""
    features = functional.relu(torch.randn(KERNEL_SHAPES[-1], device="cuda"))
    gradient = torch.randn(KERNEL_SHAPES[-1], device="cuda")
    fused = run_pass(features.half(), gradient.half(), True)
    exact = run_pass(features, gradient, False)
    output_tolerance = FLOAT16_OUTPUT_TOLERANCE
    gradient_tolerance = FLOAT16_GRADIENT_TOLERANCE
    assert fused[0].dtype == torch.float16
    torch.testing.assert_close(
        fused[0].float(), exact[0], rtol=output_tolerance, atol=output_tolerance
    )
    torch.testing.assert_close(
        fused[1].float(), exact[1], rtol=output_tolerance, atol=gradient_tolerance
    )


def test_channels_last_keeps_layout_and_matches_eager() -> None:
    """On an NHWC tensor the kernels keep the layout and match eager pooling."""
    shape = KERNEL_SHAPES[-1]
    features = functional.relu(torch.randn(shape, device="cuda"))
    features = features.contiguous(memory_format=torch.channels_last)
    gradient = torch.randn(shape, device="cuda")
    fused = run_pass(features, gradient, True)
    eager = run_pass(features, gradient, False)
    tolerance = FLOAT32_GRADIENT_TOLERANCE
    assert fused[0].is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(fused[0], eager[0])
    torch.testing.assert_close(fused[1], eager[1], rtol=tolerance, atol=tolerance)


def test_compiled_function_matches_eager() -> None:
    """Inside torch.compile the registered operators give the eager result."""
    features = functional.relu(torch.randn(KERNEL_SHAPES[-1], device="cuda"))
    gradient = torch.randn(KERNEL_SHAPES[-1], device="cuda")
    pool = select_pool_function("triton")
    compiled = torch.compile(pool)  # pyright: ignore[reportUnknownMemberType]
    leaf = features.detach().clone().requires_grad_()
    output = compiled(leaf, WINDOW, 1, 1)
    output.backward(gradient)  # pyright: ignore[reportUnknownMemberType]
    eager = run_pass(features, gradient, False)
    torch.testing.assert_close(output.detach(), eager[0])
    torch.testing.assert_close(
        leaf.grad,
        eager[1],
        rtol=FLOAT32_GRADIENT_TOLERANCE,
        atol=FLOAT32_GRADIENT_TOLERANCE,
    )
