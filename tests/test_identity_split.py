"""The identity split holds out whole people and is reproducible."""
from face_verification.identity_split import split_identities
from tests.fixture_settings import HELD_OUT_COUNT, IDENTITY_COUNT, OTHER_SEED

IDENTITIES: list[str] = [f"person_{index:03d}" for index in range(IDENTITY_COUNT)]


def test_split_is_disjoint_and_complete() -> None:
    """Every identity is in exactly one part."""
    split = split_identities(IDENTITIES, HELD_OUT_COUNT, 0)
    assert len(split.validation) == HELD_OUT_COUNT
    assert set(split.training).isdisjoint(split.validation)
    assert sorted(split.training + split.validation) == IDENTITIES


def test_split_depends_only_on_the_seed() -> None:
    """The same seed gives the same split and another seed a different one."""
    first = split_identities(IDENTITIES, HELD_OUT_COUNT, 0)
    other = split_identities(IDENTITIES, HELD_OUT_COUNT, OTHER_SEED)
    assert first == split_identities(IDENTITIES, HELD_OUT_COUNT, 0)
    assert first.validation != other.validation
