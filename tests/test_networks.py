"""Eager L2 pooling and the NN2 network on the CPU."""
import torch

from face_verification.architecture_settings import EMBEDDING_SIZE, IMAGE_CHANNELS
from face_verification.layers import l2_pool_eager
from face_verification.networks import FaceNetNN2
from face_verification.training_settings import INPUT_SIZE, L2_POOL_EPSILON
from tests.fixture_settings import EMBEDDED_IMAGES, GRID_CENTRE, GRID_SIDE, WINDOW


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
    model = FaceNetNN2("eager").eval()
    with torch.no_grad():
        embeddings = model(torch.randn(EMBEDDED_IMAGES, IMAGE_CHANNELS, *INPUT_SIZE))
    assert embeddings.shape == (EMBEDDED_IMAGES, EMBEDDING_SIZE)
    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(EMBEDDED_IMAGES))
