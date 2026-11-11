"""Split of the Pins identities into disjoint training and validation people.

Verification is judged on people the network has never seen, as on LFW, so the
validation people are held out entirely rather than some images of everyone.
"""
import json
from pathlib import Path
import random

from pydantic import BaseModel


class IdentitySplit(BaseModel):
    """Names of the identity folders used for training and for validation."""

    seed: int
    training: list[str]
    validation: list[str]


def list_identities(directory: Path) -> list[str]:
    """Return the sorted names of the identity folders in a directory."""
    return sorted(entry.name for entry in directory.iterdir() if entry.is_dir())


def split_identities(
    identities: list[str],
    validation_count: int,
    seed: int,
) -> IdentitySplit:
    """Hold out a seeded random choice of identities for validation."""
    shuffled: list[str] = list(identities)
    random.Random(seed).shuffle(shuffled)
    validation = sorted(shuffled[:validation_count])
    training = sorted(shuffled[validation_count:])
    return IdentitySplit(seed=seed, training=training, validation=validation)


def save_split(split: IdentitySplit, path: Path) -> None:
    """Write the split as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.model_dump(), indent=2) + "\n", encoding="utf-8")


def load_split(path: Path) -> IdentitySplit:
    """Read a split written by save_split."""
    return IdentitySplit.model_validate_json(path.read_text(encoding="utf-8"))


def collect_image_paths(
    directory: Path,
    identities: list[str],
) -> dict[str, list[Path]]:
    """Map every identity to the sorted paths of its images."""
    return {
        identity: sorted((directory / identity).iterdir())
        for identity in identities
    }
