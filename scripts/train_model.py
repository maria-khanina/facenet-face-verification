"""Train an embedding network on Pins and test it on LFW pairs.

The run writes runs/<run name>/history.json (loss, speed and validation scores
of every epoch), summary.json (LFW scores, speed and peak GPU memory) and the
best weights (best.pt, not tracked by git).

Usage (from the repository root):
  python -m scripts.train_model --model nn2 --pool-backend eager --run-name nn2_eager
  python -m scripts.train_model --model nn2 --pool-backend triton --compile \
      --run-name nn2_triton_compiled
"""
import argparse
import json
import statistics
from typing import cast

from pydantic import BaseModel
import torch
from torch import nn

from face_verification.identity_split import load_split
from face_verification.layers import PoolBackend
from face_verification.loaders import (
    build_lfw_loader,
    build_training_loader,
    build_validation_loader,
)
from face_verification.networks import ModelName, build_model
from face_verification.training_loop import (
    EpochRecord,
    TrainingHyperparameters,
    TrainingOutcome,
    run_training,
)
from face_verification.training_settings import (
    BYTES_PER_GIBIBYTE,
    EPOCHS,
    FALSE_ACCEPT_RATES,
    NN2_LEARNING_RATE,
    NN2_WARMUP_EPOCHS,
    NN2_WEIGHT_DECAY,
    RESNET_LEARNING_RATE,
    RESNET_WARMUP_EPOCHS,
    RESNET_WEIGHT_DECAY,
    RUNS_DIRECTORY,
    SPLIT_PATH,
)
from face_verification.verification_metrics import (
    VerificationScores,
    evaluate_verification,
)


class RunSummary(BaseModel):
    """Configuration, speed, memory and test scores of one run."""

    model: str
    pool_backend: str
    compiled: bool
    epochs: int
    best_epoch: int
    best_validation_roc_auc: float
    first_epoch_seconds: float
    later_epoch_seconds: float
    later_images_per_second: float
    peak_gpu_gibibytes: float
    lfw: VerificationScores


def parse_arguments() -> argparse.Namespace:
    """Read the model, the pooling backend, compilation and the run name."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["nn2", "resnet"], required=True)
    parser.add_argument("--pool-backend", choices=["eager", "triton"], required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def choose_hyperparameters(model: ModelName, epochs: int) -> TrainingHyperparameters:
    """Return the notebook's optimizer settings of the model."""
    if model == "resnet":
        return TrainingHyperparameters(
            learning_rate=RESNET_LEARNING_RATE,
            weight_decay=RESNET_WEIGHT_DECAY,
            warmup_epochs=RESNET_WARMUP_EPOCHS,
            epochs=epochs,
        )
    return TrainingHyperparameters(
        learning_rate=NN2_LEARNING_RATE,
        weight_decay=NN2_WEIGHT_DECAY,
        warmup_epochs=NN2_WARMUP_EPOCHS,
        epochs=epochs,
    )


def prepare_model(arguments: argparse.Namespace, device: torch.device) -> nn.Module:
    """Build the network on the device and compile it when asked."""
    model_name = cast(ModelName, arguments.model)
    backend = cast(PoolBackend, arguments.pool_backend)
    model = build_model(model_name, backend).to(device)
    if arguments.compile:
        compiled = torch.compile(model)  # pyright: ignore[reportUnknownMemberType]
        return cast(nn.Module, compiled)
    return model


def find_best_epoch(history: list[EpochRecord]) -> EpochRecord:
    """Return the epoch with the highest validation ROC AUC."""
    return max(history, key=lambda record: record.validation.roc_auc)


def summarize(
    arguments: argparse.Namespace,
    outcome: TrainingOutcome,
    lfw: VerificationScores,
) -> RunSummary:
    """Collect the configuration, speed, memory and LFW scores of the run."""
    later: list[EpochRecord] = outcome.history[1:] or outcome.history
    best = find_best_epoch(outcome.history)
    return RunSummary(
        model=arguments.model,
        pool_backend=arguments.pool_backend,
        compiled=arguments.compile,
        epochs=len(outcome.history),
        best_epoch=best.epoch,
        best_validation_roc_auc=best.validation.roc_auc,
        first_epoch_seconds=outcome.history[0].seconds,
        later_epoch_seconds=statistics.mean(record.seconds for record in later),
        later_images_per_second=statistics.mean(
            record.images_per_second for record in later
        ),
        peak_gpu_gibibytes=torch.cuda.max_memory_allocated() / BYTES_PER_GIBIBYTE,
        lfw=lfw,
    )


def write_results(
    arguments: argparse.Namespace,
    model: nn.Module,
    outcome: TrainingOutcome,
    summary: RunSummary,
) -> None:
    """Write the history, the summary and the best weights of the run."""
    directory = RUNS_DIRECTORY / arguments.run_name
    directory.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = [
        record.model_dump() for record in outcome.history
    ]
    (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (directory / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    torch.save(model.state_dict(), directory / "best.pt")


def main() -> None:
    """Train, restore the best validation weights and test on LFW."""
    arguments = parse_arguments()
    device = torch.device("cuda")
    split = load_split(SPLIT_PATH)
    model = prepare_model(arguments, device)
    loaders = (build_training_loader(split), build_validation_loader(split))
    hyperparameters = choose_hyperparameters(arguments.model, arguments.epochs)
    outcome = run_training(model, loaders, hyperparameters)
    model.load_state_dict(outcome.best_state)
    lfw = evaluate_verification(model, build_lfw_loader(), device, FALSE_ACCEPT_RATES)
    write_results(arguments, model, outcome, summarize(arguments, outcome, lfw))


if __name__ == "__main__":
    main()
