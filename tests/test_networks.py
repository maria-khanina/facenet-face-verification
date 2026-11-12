"""Eager L2 pooling, the NN2 network and its input pipeline on the CPU."""
from PIL import Image
import torch
from torch import nn

from face_verification.architecture_settings import EMBEDDING_SIZE, IMAGE_CHANNELS
from face_verification.layers import PointwiseBranches, l2_pool_eager
from face_verification.loaders import build_evaluation_transform, to_model_input
from face_verification.networks import FaceNetNN2
from face_verification.training_settings import INPUT_SIZE, L2_POOL_EPSILON
from tests.fixture_settings import (
    EMBEDDED_IMAGES,
    GRID_CENTRE,
    GRID_SIDE,
    OTHER_SEED,
    PIXEL_LEVELS,
    POINTWISE_INPUT,
    POINTWISE_OUTPUTS,
    WINDOW,
)


def test_eager_l2_pool_matches_the_definition() -> None:
    """The centre output is the root of the sum of squares of its 3x3 window."""
    grid = torch.arange(GRID_SIDE * GRID_SIDE, dtype=torch.float32)
    features = grid.reshape(1, 1, GRID_SIDE, GRID_SIDE)
    pooled = l2_pool_eager(features, WINDOW, 1, 1)
    around = slice(GRID_CENTRE - 1, GRID_CENTRE + 2)
    window = features[0, 0, around, around]
    expected = torch.sqrt((window * window).sum() + L2_POOL_EPSILON)
    assert pooled.shape == features.shape
    torch.testing.assert_close(pooled[0, 0, GRID_CENTRE, GRID_CENTRE], expected)


def test_nn2_returns_unit_embeddings() -> None:
    """Every image maps to an embedding of length one."""
    model = FaceNetNN2("eager", False).eval()
    with torch.no_grad():
        embeddings = model(torch.randn(EMBEDDED_IMAGES, IMAGE_CHANNELS, *INPUT_SIZE))
    assert embeddings.shape == (EMBEDDED_IMAGES, EMBEDDING_SIZE)
    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(EMBEDDED_IMAGES))


def copy_into_merged(separate: PointwiseBranches, merged: PointwiseBranches) -> None:
    """Stack the weights of the separate convolutions into the merged one."""
    target = merged.convolutions[0]
    sources: list[nn.Module] = list(separate.convolutions)
    for name, parameter in target.state_dict().items():
        if parameter.dim():
            parameter.copy_(torch.cat([part.state_dict()[name] for part in sources]))


def test_merged_pointwise_matches_separate() -> None:
    """One merged 1x1 convolution gives the outputs of the separate ones."""
    separate = PointwiseBranches(POINTWISE_INPUT, POINTWISE_OUTPUTS, False).eval()
    merged = PointwiseBranches(POINTWISE_INPUT, POINTWISE_OUTPUTS, True).eval()
    with torch.no_grad():
        copy_into_merged(separate, merged)
        features = torch.randn(EMBEDDED_IMAGES, POINTWISE_INPUT, GRID_SIDE, GRID_SIDE)
        expected = separate(features)
        produced = merged(features)
    for want, got in zip(expected, produced):
        torch.testing.assert_close(got, want)


def test_uint8_input_matches_float_pipeline() -> None:
    """Normalizing uint8 pixels on the device equals ToTensor and Normalize."""
    generator = torch.Generator().manual_seed(OTHER_SEED)
    size = (*INPUT_SIZE, IMAGE_CHANNELS)
    pixels = torch.randint(0, PIXEL_LEVELS, size, generator=generator)
    image = Image.fromarray(pixels.to(torch.uint8).numpy())
    expected = build_evaluation_transform(False)(image)
    raw = build_evaluation_transform(True)(image).unsqueeze(0)
    produced = to_model_input(raw, torch.device("cpu"))[0]
    torch.testing.assert_close(produced, expected)
