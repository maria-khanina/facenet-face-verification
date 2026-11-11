"""Verification scores of an embedding network on pairs of images.

A pair is accepted as one person when the distance of the embeddings is below
a threshold. ROC AUC summarizes all thresholds; the best-threshold accuracy is
the accuracy at the threshold that maximizes TPR - FPR, as in the notebook;
TAR@FAR is the share of same-person pairs accepted when at most the given
share of different-person pairs is accepted (the measure of the FaceNet paper).
"""
import numpy
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.metrics import auc, roc_curve
import torch
from torch import nn

from face_verification.loaders import PairsLoader

Distances = NDArray[numpy.float32]
Labels = NDArray[numpy.int64]


class VerificationScores(BaseModel):
    """Scores of one network on one set of pairs."""

    pairs: int
    roc_auc: float
    best_threshold_accuracy: float
    true_accept_rate_at_false_accept_rate: dict[str, float]


def compute_distances(
    model: nn.Module,
    loader: PairsLoader,
    device: torch.device,
) -> tuple[Distances, Labels]:
    """Return the embedding distance and the label of every pair."""
    model.eval()
    distances: list[Distances] = []
    labels: list[Labels] = []
    with torch.no_grad():
        for first, second, pair_labels in loader:
            gap = model(first.to(device)) - model(second.to(device))
            distances.append(gap.norm(p=2, dim=1).float().cpu().numpy())
            labels.append(pair_labels.numpy())
    return numpy.concatenate(distances), numpy.concatenate(labels)


def measure_true_accept_rate(
    false_accept_rates: NDArray[numpy.float64],
    true_accept_rates: NDArray[numpy.float64],
    limit: float,
) -> float:
    """Return the best true accept rate with false accept rate at most the limit."""
    allowed = true_accept_rates[false_accept_rates <= limit]
    return float(allowed.max()) if allowed.size else 0.0


def best_threshold_accuracy(
    distances: Distances,
    labels: Labels,
    rates: tuple[NDArray[numpy.float64], NDArray[numpy.float64]],
    thresholds: NDArray[numpy.float64],
) -> float:
    """Accuracy at the threshold that maximizes TPR - FPR."""
    false_accept_rates, true_accept_rates = rates
    best = int((true_accept_rates - false_accept_rates).argmax())
    predicted = (distances <= -thresholds[best]).astype(numpy.int64)
    return float((predicted == labels).mean())


def score_verification(
    distances: Distances,
    labels: Labels,
    false_accept_limits: tuple[float, ...],
) -> VerificationScores:
    """Score distances against the same-person labels."""
    false_accept_rates, true_accept_rates, thresholds = roc_curve(labels, -distances)
    rates = (false_accept_rates, true_accept_rates)
    accept_rates: dict[str, float] = {
        f"{limit:g}": measure_true_accept_rate(*rates, limit)
        for limit in false_accept_limits
    }
    return VerificationScores(
        pairs=int(labels.size),
        roc_auc=auc(false_accept_rates, true_accept_rates),
        best_threshold_accuracy=best_threshold_accuracy(
            distances, labels, rates, thresholds
        ),
        true_accept_rate_at_false_accept_rate=accept_rates,
    )


def evaluate_verification(
    model: nn.Module,
    loader: PairsLoader,
    device: torch.device,
    false_accept_limits: tuple[float, ...],
) -> VerificationScores:
    """Embed every pair and score the distances."""
    distances, labels = compute_distances(model, loader, device)
    return score_verification(distances, labels, false_accept_limits)
