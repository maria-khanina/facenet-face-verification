"""Write the seeded split of the Pins identities into training and validation.

Usage (from the repository root):
  python -m scripts.split_identities
"""
from face_verification.identity_split import (
    list_identities,
    save_split,
    split_identities,
)
from face_verification.training_settings import (
    PINS_DIRECTORY,
    SPLIT_PATH,
    SPLIT_SEED,
    VALIDATION_IDENTITIES,
)


def main() -> None:
    """Split the identity folders and save the split."""
    identities = list_identities(PINS_DIRECTORY)
    split = split_identities(identities, VALIDATION_IDENTITIES, SPLIT_SEED)
    save_split(split, SPLIT_PATH)
    print(f"{len(split.training)} training and {len(split.validation)} validation")


if __name__ == "__main__":
    main()
