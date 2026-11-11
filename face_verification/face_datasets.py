"""Datasets of labelled face images and of image pairs for verification."""
from collections.abc import Callable
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset

ImageTransform = Callable[[Image.Image], torch.Tensor]
PathPair = tuple[Path, Path]


def load_image(path: Path, transform: ImageTransform) -> torch.Tensor:
    """Read an image as RGB and apply the transform."""
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


class IdentityImages(Dataset[tuple[torch.Tensor, int]]):
    """Every image of the given identities, labelled by identity index."""

    def __init__(
        self,
        paths_by_identity: dict[str, list[Path]],
        transform: ImageTransform,
    ) -> None:
        """List the images and their labels in identity order."""
        self.transform = transform
        self.samples: list[tuple[Path, int]] = [
            (path, label)
            for label, paths in enumerate(paths_by_identity.values())
            for path in paths
        ]
        self.targets: list[int] = [label for _, label in self.samples]

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.samples)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        """Return the transformed image and its label."""
        path, label = self.samples[position]
        return load_image(path, self.transform), label


def sample_pair(
    paths_by_identity: dict[str, list[Path]],
    generator: random.Random,
    same_person: bool,
) -> PathPair:
    """Draw two images of one person or of two different people."""
    if same_person:
        candidates: list[str] = [
            name for name, paths in paths_by_identity.items() if len(paths) > 1
        ]
        person = generator.choice(candidates)
        first, second = generator.sample(paths_by_identity[person], 2)
    else:
        one, other = generator.sample(sorted(paths_by_identity), 2)
        first = generator.choice(paths_by_identity[one])
        second = generator.choice(paths_by_identity[other])
    return (first, second) if first <= second else (second, first)


def draw_distinct_pairs(
    paths_by_identity: dict[str, list[Path]],
    generator: random.Random,
    same_person: bool,
    count: int,
) -> list[PathPair]:
    """Draw the given number of distinct pairs of one kind."""
    pairs: dict[PathPair, None] = {}
    while len(pairs) < count:
        pairs[sample_pair(paths_by_identity, generator, same_person)] = None
    return list(pairs)


class PairsDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Seeded balanced pairs: half of one person, half of two different people."""

    def __init__(
        self,
        paths_by_identity: dict[str, list[Path]],
        transform: ImageTransform,
        pair_count: int,
        seed: int,
    ) -> None:
        """Draw the positive pairs first, then the negative pairs."""
        generator = random.Random(seed)
        half = pair_count // 2
        positives = draw_distinct_pairs(paths_by_identity, generator, True, half)
        negatives = draw_distinct_pairs(paths_by_identity, generator, False, half)
        self.transform = transform
        self.pairs: list[PathPair] = positives + negatives
        self.labels: list[int] = [1] * len(positives) + [0] * len(negatives)

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self.pairs)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return both transformed images and whether they show one person."""
        first, second = self.pairs[position]
        return (
            load_image(first, self.transform),
            load_image(second, self.transform),
            self.labels[position],
        )
