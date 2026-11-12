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
from face_verification.loaders import (
    build_lfw_loader,
    build_training_loader,
    build_validation_loader,
)
from face_verification.networks import ModelName, build_model
from face_verification.speed_options import (
    SpeedOptions,
    add_speed_arguments,
    apply_global_options,
    compile_when_asked,
    read_speed_options,
)
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
    options: SpeedOptions
    epochs: int
    best_epoch: int
    best_validation_roc_auc: float
    first_epoch_seconds: float
    later_epoch_seconds: float
    later_images_per_second: float
    peak_gpu_gibibytes: float
    lfw: VerificationScores


def parse_arguments() -> argparse.Namespace:
    """Read the model, the optimizations, the epochs and the run name."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["nn2", "resnet"], required=True)
    add_speed_arguments(parser)
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


def prepare_model(
    arguments: argparse.Namespace,
    options: SpeedOptions,
    device: torch.device,
) -> nn.Module:
    """Build the network on the device and compile it when asked."""
    model_name = cast(ModelName, arguments.model)
    built = build_model(model_name, options.pool_backend, options.merge_pointwise)
    return compile_when_asked(built.to(device), options)


def find_best_epoch(history: list[EpochRecord]) -> EpochRecord:
    """Return the epoch with the highest validation ROC AUC."""
    return max(history, key=lambda record: record.validation.roc_auc)


def summarize(
    arguments: argparse.Namespace,
    options: SpeedOptions,
    outcome: TrainingOutcome,
    lfw: VerificationScores,
) -> RunSummary:
    """Collect the configuration, speed, memory and LFW scores of the run."""
    later: list[EpochRecord] = outcome.history[1:] or outcome.history
    best = find_best_epoch(outcome.history)
    return RunSummary(
        model=arguments.model,
        options=options,
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


def train(
    arguments: argparse.Namespace,
    options: SpeedOptions,
    model: nn.Module,
) -> TrainingOutcome:
    """Train on the Pins training identities and validate on the held-out ones."""
    split = load_split(SPLIT_PATH)
    uint8 = options.transfer_uint8
    loaders = (
        build_training_loader(split, uint8),
        build_validation_loader(split, uint8),
    )
    hyperparameters = choose_hyperparameters(arguments.model, arguments.epochs)
    return run_training(model, loaders, hyperparameters, options.compile_optimizer)


def test_on_lfw(
    model: nn.Module,
    options: SpeedOptions,
    device: torch.device,
) -> VerificationScores:
    """Score the restored best weights on the LFW pairs."""
    lfw_loader = build_lfw_loader(options.transfer_uint8)
    return evaluate_verification(model, lfw_loader, device, FALSE_ACCEPT_RATES)


def main() -> None:
    """Train, restore the best validation weights and test on LFW."""
    arguments = parse_arguments()
    options = read_speed_options(arguments)
    apply_global_options(options)
    device = torch.device("cuda")
    model = prepare_model(arguments, options, device)
    outcome = train(arguments, options, model)
    model.load_state_dict(outcome.best_state)
    lfw = test_on_lfw(model, options, device)
    summary = summarize(arguments, options, outcome, lfw)
    write_results(arguments, model, outcome, summary)


if __name__ == "__main__":
    main()
