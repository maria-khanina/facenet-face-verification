"""Embedding networks: FaceNet NN2 from scratch and a pretrained ResNet-18."""
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as functional
from torchvision import models

from face_verification.architecture_settings import (
    EMBEDDING_SIZE,
    FINAL_CHANNELS,
    IMAGE_CHANNELS,
    INCEPTION_STAGE_FIVE,
    INCEPTION_STAGE_FOUR,
    INCEPTION_STAGE_THREE,
    REDUCTION_STAGE_FOUR,
    REDUCTION_STAGE_THREE,
    RESNET_EMBEDDING_INPUT,
    SECOND_STAGE_CHANNELS,
    SMALL_KERNEL,
    SMALL_PADDING,
    STEM_CONVOLUTION_CHANNELS,
    STEM_CONVOLUTION_KERNEL,
    STEM_CONVOLUTION_PADDING,
    STEM_CONVOLUTION_STRIDE,
    STEM_POOL_KERNEL,
    STEM_POOL_STRIDE,
    InceptionLayout,
)
from face_verification.layers import (
    BasicConv2d,
    InceptionReduction,
    NN2Inception,
    PoolBackend,
    pointwise_convolution,
)

ModelName = Literal["nn2", "resnet"]


def build_stem() -> nn.Sequential:
    """Return the convolutions and poolings before the first inception module."""
    return nn.Sequential(
        BasicConv2d(
            IMAGE_CHANNELS,
            STEM_CONVOLUTION_CHANNELS,
            STEM_CONVOLUTION_KERNEL,
            STEM_CONVOLUTION_STRIDE,
            STEM_CONVOLUTION_PADDING,
        ),
        nn.MaxPool2d(STEM_POOL_KERNEL, stride=STEM_POOL_STRIDE, ceil_mode=True),
        pointwise_convolution(STEM_CONVOLUTION_CHANNELS, STEM_CONVOLUTION_CHANNELS),
        BasicConv2d(
            STEM_CONVOLUTION_CHANNELS,
            SECOND_STAGE_CHANNELS,
            SMALL_KERNEL,
            1,
            SMALL_PADDING,
        ),
        nn.MaxPool2d(STEM_POOL_KERNEL, stride=STEM_POOL_STRIDE, ceil_mode=True),
    )


def build_inception_stage(
    layouts: tuple[InceptionLayout, ...],
    backend: PoolBackend,
) -> list[nn.Module]:
    """Return the inception modules of one stage."""
    return [NN2Inception(layout, backend) for layout in layouts]


def build_inception_body(backend: PoolBackend) -> nn.Sequential:
    """Return the three inception stages with their reduction modules."""
    modules: list[nn.Module] = [
        *build_inception_stage(INCEPTION_STAGE_THREE, backend),
        InceptionReduction(REDUCTION_STAGE_THREE),
        *build_inception_stage(INCEPTION_STAGE_FOUR, backend),
        InceptionReduction(REDUCTION_STAGE_FOUR),
        *build_inception_stage(INCEPTION_STAGE_FIVE, backend),
    ]
    return nn.Sequential(*modules)


class FaceNetNN2(nn.Module):
    """FaceNet NN2 that maps a face image to an L2-normalized embedding."""

    def __init__(self, backend: PoolBackend) -> None:
        """Create the stem, the inception body and the embedding layer."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.stem = build_stem()
        self.body = build_inception_body(backend)
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.full_layer = nn.Linear(FINAL_CHANNELS, EMBEDDING_SIZE)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unit-length embeddings of the images."""
        features = self.average_pool(self.body(self.stem(images)))
        embeddings = self.full_layer(torch.flatten(features, 1))
        return functional.normalize(embeddings, p=2.0, dim=1)


class PretrainedFaceResNet(nn.Module):
    """ImageNet-pretrained ResNet-18 with a new embedding layer."""

    def __init__(self) -> None:
        """Load the pretrained backbone and replace its classifier."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(RESNET_EMBEDDING_INPUT, EMBEDDING_SIZE)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unit-length embeddings of the images."""
        return functional.normalize(self.backbone(images), p=2.0, dim=1)


def build_model(name: ModelName, backend: PoolBackend) -> nn.Module:
    """Return a fresh embedding network of the given kind."""
    if name == "resnet":
        return PretrainedFaceResNet()
    return FaceNetNN2(backend)
