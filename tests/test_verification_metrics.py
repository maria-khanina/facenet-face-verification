"""Verification scores on distances with a known answer."""
import numpy

from face_verification.verification_metrics import score_verification
from tests.fixture_settings import (
    IMPOSTOR_FIRST_DISTANCES,
    IMPOSTOR_FIRST_LABELS,
    LOOSE_FALSE_ACCEPT_RATE,
    SEPARATED_DISTANCES,
    SEPARATED_LABELS,
    STRICT_FALSE_ACCEPT_RATE,
)


def test_perfect_separation_scores_one() -> None:
    """Same-person pairs that are all closer than the others score 1."""
    distances = numpy.array(SEPARATED_DISTANCES, dtype=numpy.float32)
    labels = numpy.array(SEPARATED_LABELS, dtype=numpy.int64)
    scores = score_verification(distances, labels, (STRICT_FALSE_ACCEPT_RATE,))
    assert scores.roc_auc == 1.0
    assert scores.best_threshold_accuracy == 1.0
    assert list(scores.true_accept_rate_at_false_accept_rate.values()) == [1.0]


def test_true_accept_rate_respects_the_false_accept_limit() -> None:
    """With one impostor closest, no same-person pair is accepted at zero FAR."""
    distances = numpy.array(IMPOSTOR_FIRST_DISTANCES, dtype=numpy.float32)
    labels = numpy.array(IMPOSTOR_FIRST_LABELS, dtype=numpy.int64)
    limits = (0.0, LOOSE_FALSE_ACCEPT_RATE)
    scores = score_verification(distances, labels, limits)
    rates: list[float] = list(scores.true_accept_rate_at_false_accept_rate.values())
    assert rates == [0.0, 1.0]
