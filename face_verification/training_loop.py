"""Training loop with fp16 autocast, triplet loss and semi-hard triplet mining.

The loop follows the notebook: AdamW, linear learning rate warmup per epoch, an
M-per-class sampler, semi-hard triplets mined in every batch, and the weights
of the best validation ROC AUC restored at the end. The loss is accumulated on
the GPU and read once per epoch, so steps do not wait for the host.
"""
import copy
from dataclasses import dataclass
import time

from pydantic import BaseModel
from pytorch_metric_learning import losses, miners
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LinearLR

from face_verification.loaders import PairsLoader, TrainingLoader
from face_verification.training_settings import (
    FALSE_ACCEPT_RATES,
    TRIPLET_MARGIN,
    WARMUP_START_FACTOR,
)
from face_verification.verification_metrics import (
    VerificationScores,
    evaluate_verification,
)

StateDictionary = dict[str, torch.Tensor]


class TrainingHyperparameters(BaseModel):
    """Optimizer and schedule settings of one model."""

    learning_rate: float
    weight_decay: float
    warmup_epochs: int
    epochs: int


class EpochRecord(BaseModel):
    """Loss, speed and validation scores after one epoch."""

    epoch: int
    training_loss: float
    seconds: float
    images_per_second: float
    validation: VerificationScores


@dataclass
class TrainingParts:
    """Optimizer, schedule, gradient scaler, miner and loss of one run."""

    optimizer: torch.optim.Optimizer
    scheduler: LinearLR
    scaler: GradScaler
    miner: miners.TripletMarginMiner
    loss: losses.TripletMarginLoss


@dataclass
class TrainingOutcome:
    """History of a run and the weights with the best validation ROC AUC."""

    history: list[EpochRecord]
    best_state: StateDictionary


def build_training_parts(
    model: nn.Module,
    hyperparameters: TrainingHyperparameters,
) -> TrainingParts:
    """Create the optimizer, the warmup schedule, the scaler and the loss."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )
    return TrainingParts(
        optimizer=optimizer,
        scheduler=LinearLR(
            optimizer,
            start_factor=WARMUP_START_FACTOR,
            total_iters=hyperparameters.warmup_epochs,
        ),
        scaler=GradScaler("cuda"),
        miner=miners.TripletMarginMiner(TRIPLET_MARGIN, "semihard"),
        loss=losses.TripletMarginLoss(TRIPLET_MARGIN),
    )


def train_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    parts: TrainingParts,
) -> torch.Tensor:
    """Run one optimizer step and return the loss without reading it."""
    parts.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        embeddings = model(images)
    embeddings = embeddings.float()
    loss = parts.loss(embeddings, labels, parts.miner(embeddings, labels))
    parts.scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]
    parts.scaler.step(parts.optimizer)
    parts.scaler.update()
    return loss.detach()


def train_one_epoch(
    model: nn.Module,
    loader: TrainingLoader,
    parts: TrainingParts,
    device: torch.device,
) -> tuple[float, int, float]:
    """Return the mean loss, the number of images and the seconds of an epoch."""
    model.train()
    started = time.perf_counter()
    total_loss = torch.zeros((), device=device)
    images_seen = 0
    for images, labels in loader:
        loss = train_step(
            model,
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
            parts,
        )
        total_loss += loss * images.shape[0]
        images_seen += images.shape[0]
    mean_loss = total_loss.item() / images_seen
    return mean_loss, images_seen, time.perf_counter() - started


def record_epoch(
    epoch: int,
    measured: tuple[float, int, float],
    validation: VerificationScores,
) -> EpochRecord:
    """Combine the training measurements and the validation scores."""
    mean_loss, images_seen, seconds = measured
    record = EpochRecord(
        epoch=epoch,
        training_loss=mean_loss,
        seconds=seconds,
        images_per_second=images_seen / seconds,
        validation=validation,
    )
    print(record.model_dump_json(), flush=True)
    return record


def run_epoch(
    model: nn.Module,
    loaders: tuple[TrainingLoader, PairsLoader],
    parts: TrainingParts,
    epoch: int,
) -> EpochRecord:
    """Train for one epoch, step the schedule and validate."""
    device = next(model.parameters()).device
    measured = train_one_epoch(model, loaders[0], parts, device)
    parts.scheduler.step()
    validation = evaluate_verification(model, loaders[1], device, FALSE_ACCEPT_RATES)
    return record_epoch(epoch, measured, validation)


def copy_state(model: nn.Module) -> StateDictionary:
    """Return a detached copy of the weights."""
    return copy.deepcopy(model.state_dict())


def run_training(
    model: nn.Module,
    loaders: tuple[TrainingLoader, PairsLoader],
    hyperparameters: TrainingHyperparameters,
) -> TrainingOutcome:
    """Train for all epochs and keep the weights of the best validation ROC AUC."""
    parts = build_training_parts(model, hyperparameters)
    outcome = TrainingOutcome(history=[], best_state=copy_state(model))
    best_roc_auc = -1.0
    for epoch in range(1, hyperparameters.epochs + 1):
        record = run_epoch(model, loaders, parts, epoch)
        outcome.history.append(record)
        if record.validation.roc_auc > best_roc_auc:
            best_roc_auc = record.validation.roc_auc
            outcome.best_state = copy_state(model)
    return outcome
