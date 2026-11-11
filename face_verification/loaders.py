"""Transforms and data loaders of training, validation and the LFW test."""
from pathlib import Path

from pytorch_metric_learning import samplers
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from face_verification.face_datasets import IdentityImages, PairsDataset
from face_verification.identity_split import (
    IdentitySplit,
    collect_image_paths,
    list_identities,
)
from face_verification.training_settings import (
    BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    HORIZONTAL_FLIP_PROBABILITY,
    INPUT_SIZE,
    LFW_DIRECTORY,
    LFW_PAIRS,
    LFW_PAIRS_SEED,
    LOADER_WORKERS,
    NORMALIZATION_MEAN,
    NORMALIZATION_STANDARD_DEVIATION,
    PINS_DIRECTORY,
    PIN_LOADER_MEMORY,
    SAMPLES_PER_IDENTITY,
    VALIDATION_PAIRS,
    VALIDATION_PAIRS_SEED,
)

TrainingLoader = DataLoader[tuple[torch.Tensor, int]]
PairsLoader = DataLoader[tuple[torch.Tensor, torch.Tensor, int]]


def build_training_transform() -> transforms.Compose:
    """Resize, flip at random and normalize to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROBABILITY),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STANDARD_DEVIATION),
    ])


def build_evaluation_transform() -> transforms.Compose:
    """Resize and normalize to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STANDARD_DEVIATION),
    ])


def build_training_loader(split: IdentitySplit) -> TrainingLoader:
    """Batches with a fixed number of images of every sampled identity."""
    dataset = IdentityImages(
        collect_image_paths(PINS_DIRECTORY, split.training),
        build_training_transform(),
    )
    sampler = samplers.MPerClassSampler(
        dataset.targets,
        SAMPLES_PER_IDENTITY,
        BATCH_SIZE,
        len(dataset) // BATCH_SIZE,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=LOADER_WORKERS,
        pin_memory=PIN_LOADER_MEMORY,
    )


def build_pairs_loader(
    paths_by_identity: dict[str, list[Path]],
    pair_count: int,
    seed: int,
) -> PairsLoader:
    """Batches of seeded verification pairs in a fixed order."""
    dataset = PairsDataset(
        paths_by_identity,
        build_evaluation_transform(),
        pair_count,
        seed,
    )
    return DataLoader(
        dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=LOADER_WORKERS,
        pin_memory=PIN_LOADER_MEMORY,
    )


def build_validation_loader(split: IdentitySplit) -> PairsLoader:
    """Pairs of the held-out Pins identities."""
    paths = collect_image_paths(PINS_DIRECTORY, split.validation)
    return build_pairs_loader(paths, VALIDATION_PAIRS, VALIDATION_PAIRS_SEED)


def build_lfw_loader() -> PairsLoader:
    """Pairs of Labeled Faces in the Wild, the final test."""
    paths = collect_image_paths(LFW_DIRECTORY, list_identities(LFW_DIRECTORY))
    return build_pairs_loader(paths, LFW_PAIRS, LFW_PAIRS_SEED)
