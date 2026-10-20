#!/usr/bin/env python3
"""Train and evaluate admission-time PLOS prediction on one AI Runtime GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import mlflow
import torch
from serverless_gpu.data import DataLoader, UCVolumeDataset

from experiments.ai_runtime_exmedbert.shared.contracts import (
    canonical_sha256,
    validate_supervised_manifest,
)
from experiments.ai_runtime_exmedbert.training.checkpoint import (
    latest_complete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from experiments.ai_runtime_exmedbert.training.endpoint_metrics import (
    binary_metrics,
    select_youden_threshold,
)
from experiments.ai_runtime_exmedbert.training.supervised_data import (
    ExMedBertClassificationCollator,
    SupervisedExMedBertParquetDataset,
)
from experiments.ai_runtime_exmedbert.training.supervised_modeling import (
    ExMedBertForBinaryEndpoint,
    create_classifier,
    freeze_encoder,
    transfer_pretrained_encoder,
)
from experiments.ai_runtime_exmedbert.training.vocabulary import (
    VocabularyArtifacts,
    load_snapshot_metadata,
    sha256_file,
)


_CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d{6})_[0-9a-f]+$")
_VALIDATION_EVIDENCE_FILENAME = "validation_evidence.json"
_TRAINING_RESULT_FILENAME = "training_result.json"


@dataclass(frozen=True)
class ValidationCheckpoint:
    epoch: int
    checkpoint: str
    validation_average_precision: float


def complete_checkpoints(checkpoint_root: str) -> list[tuple[int, str]]:
    root = Path(checkpoint_root)
    if not root.exists():
        return []
    checkpoints = []
    for child in root.iterdir():
        match = _CHECKPOINT_PATTERN.match(child.name)
        if child.is_dir() and match and (child / ".metadata").is_file():
            checkpoints.append((int(match.group(1)), str(child)))
    return sorted(checkpoints, key=lambda item: (item[0], item[1]))


def write_validation_evidence(
    checkpoint: str,
    *,
    epoch: int,
    validation_average_precision: float,
    manifest_sha256: str,
    configuration_sha256: str,
) -> ValidationCheckpoint:
    checkpoint_path = Path(checkpoint)
    match = _CHECKPOINT_PATTERN.match(checkpoint_path.name)
    if not checkpoint_path.is_dir() or not (checkpoint_path / ".metadata").is_file():
        raise ValueError("validation evidence requires a complete checkpoint")
    if match is None or int(match.group(1)) != epoch:
        raise ValueError("validation evidence epoch does not match checkpoint name")
    if not math.isfinite(validation_average_precision) or not (
        0.0 <= validation_average_precision <= 1.0
    ):
        raise ValueError("validation average_precision must be finite and between zero and one")
    selection = ValidationCheckpoint(
        epoch=epoch,
        checkpoint=checkpoint,
        validation_average_precision=float(validation_average_precision),
    )
    payload = {
        "version": 1,
        "epoch": selection.epoch,
        "checkpoint": selection.checkpoint,
        "validation_average_precision": selection.validation_average_precision,
        "manifest_sha256": manifest_sha256,
        "configuration_sha256": configuration_sha256,
    }
    destination = checkpoint_path / _VALIDATION_EVIDENCE_FILENAME
    temporary = checkpoint_path / f".{_VALIDATION_EVIDENCE_FILENAME}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return selection


def write_training_result(
    checkpoint_root: str,
    *,
    manifest_sha256: str,
    configuration_sha256: str,
    configuration: dict[str, Any],
    result: dict[str, Any],
) -> Path:
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / _TRAINING_RESULT_FILENAME
    temporary = root / f".{_TRAINING_RESULT_FILENAME}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_sha256,
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
        "result": result,
    }
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def read_validation_evidence(
    checkpoint: str,
    *,
    expected_manifest_sha256: str,
    expected_configuration_sha256: str,
) -> ValidationCheckpoint | None:
    checkpoint_path = Path(checkpoint)
    evidence_path = checkpoint_path / _VALIDATION_EVIDENCE_FILENAME
    if not evidence_path.is_file():
        return None
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError(f"unsupported validation evidence version: {evidence_path}")
    if payload.get("manifest_sha256") != expected_manifest_sha256:
        return None
    if payload.get("configuration_sha256") != expected_configuration_sha256:
        return None
    match = _CHECKPOINT_PATTERN.match(checkpoint_path.name)
    epoch = payload.get("epoch")
    average_precision = payload.get("validation_average_precision")
    if (
        match is None
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch != int(match.group(1))
        or payload.get("checkpoint") != checkpoint
        or isinstance(average_precision, bool)
        or not isinstance(average_precision, (int, float))
        or not math.isfinite(float(average_precision))
        or not 0.0 <= float(average_precision) <= 1.0
    ):
        raise ValueError(f"invalid validation evidence: {evidence_path}")
    return ValidationCheckpoint(
        epoch=epoch,
        checkpoint=checkpoint,
        validation_average_precision=float(average_precision),
    )


def select_best_validation_checkpoint(
    checkpoint_root: str,
    *,
    expected_manifest_sha256: str,
    expected_configuration_sha256: str,
) -> ValidationCheckpoint | None:
    candidates = []
    for _, checkpoint in complete_checkpoints(checkpoint_root):
        evidence = read_validation_evidence(
            checkpoint,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_configuration_sha256=expected_configuration_sha256,
        )
        if evidence is not None:
            candidates.append(evidence)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (-item.validation_average_precision, item.epoch, item.checkpoint),
    )[0]


def restore_validation_checkpoint(
    selection: ValidationCheckpoint,
    new_model_and_optimizer: Callable[[], tuple[Any, Any]],
    *,
    expected_manifest_sha256: str,
    expected_configuration_sha256: str,
) -> tuple[Any, Any, int]:
    model, optimizer = new_model_and_optimizer()
    restored_epoch, restored_step = load_checkpoint(
        selection.checkpoint,
        model,
        optimizer,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_configuration_sha256=expected_configuration_sha256,
    )
    if restored_epoch != selection.epoch:
        raise AssertionError("selected checkpoint epoch does not match restored training counter")
    return model, optimizer, restored_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument(
        "--initialization",
        choices=("random", "frozen", "finetune"),
        default="finetune",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=288)
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument("--num-attention-heads", type=int, default=6)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=201214)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_metadata(snapshot_root: str):
    root = Path(snapshot_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("supervised snapshot requires manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_supervised_manifest(manifest)
    pretrain_root = manifest["pretraining"]["snapshot_root"]
    pretrain_manifest, vocabulary, pretrain_manifest_hash = load_snapshot_metadata(pretrain_root)
    if pretrain_manifest_hash != manifest["pretraining"]["manifest_sha256"]:
        raise ValueError("supervised snapshot references a different pretraining manifest")
    vocabulary_path = Path(pretrain_root) / "vocabulary.json"
    if sha256_file(vocabulary_path) != manifest["pretraining"]["vocabulary_sha256"]:
        raise ValueError("supervised snapshot references a different vocabulary")
    return manifest, pretrain_manifest, vocabulary, sha256_file(manifest_path), pretrain_manifest_hash


def expected_files(manifest: dict[str, Any], split: str) -> dict[str, int]:
    return {item["path"]: int(item["bytes"]) for item in manifest["artifacts"][split]}


def make_loader(
    snapshot_root: str,
    split: str,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[SupervisedExMedBertParquetDataset, DataLoader]:
    paths = UCVolumeDataset(f"{snapshot_root}/{split}")
    dataset = SupervisedExMedBertParquetDataset(paths, expected_files(manifest, split))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
        pin_memory=True,
        drop_last=False,
        collate_fn=ExMedBertClassificationCollator(
            max_sequence_length=args.max_sequence_length
        ),
    )
    return dataset, loader


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_inputs(batch: dict[str, Any], *, include_labels: bool = True):
    keys = [
        "input_ids",
        "entity_ids",
        "age_ids",
        "sex_ids",
        "state_ids",
        "position_ids",
        "attention_mask",
    ]
    if include_labels:
        keys.append("endpoint_labels")
    return {key: batch[key] for key in keys}


def assert_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient in {name}")


def run_split(
    model: ExMedBertForBinaryEndpoint,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    encoder_frozen: bool,
) -> tuple[dict[str, Any], dict[str, list[Any]], dict[str, torch.Tensor] | None]:
    training = optimizer is not None
    model.train(training)
    if encoder_frozen:
        model.bert.eval()
    total_loss = 0.0
    examples = 0
    optimizer_steps = 0
    predictions: dict[str, list[Any]] = {
        "example_ids": [],
        "patient_ids": [],
        "labels": [],
        "probabilities": [],
    }
    probe_batch = None
    started = time.monotonic()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            predictions["example_ids"].extend(raw_batch["example_ids"])
            predictions["patient_ids"].extend(raw_batch["patient_ids"])
            batch = move_batch(raw_batch, device)
            if probe_batch is None:
                probe_batch = {
                    key: value.detach().clone()
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(**model_inputs(batch))
            loss = output["loss"]
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("non-finite ExMed-BERT PLOS loss")
            if training:
                loss.backward()
                assert_finite_gradients(model)
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    1.0,
                )
                optimizer.step()
                optimizer_steps += 1
            labels = batch["endpoint_labels"].detach().cpu().reshape(-1)
            probabilities = torch.sigmoid(output["logits"]).detach().cpu().reshape(-1)
            count = int(labels.numel())
            total_loss += float(loss.detach().item()) * count
            examples += count
            predictions["labels"].extend(labels.to(torch.int64).tolist())
            predictions["probabilities"].extend(probabilities.tolist())
    if examples == 0:
        raise RuntimeError("split produced no supervised examples")
    if len(predictions["example_ids"]) != len(set(predictions["example_ids"])):
        raise RuntimeError("a supervised example was consumed more than once")
    metrics = binary_metrics(predictions["labels"], predictions["probabilities"])
    metrics.update(
        {
            "loss": total_loss / examples,
            "optimizer_steps": optimizer_steps,
            "seconds": time.monotonic() - started,
        }
    )
    return metrics, predictions, probe_batch


def probe_logits(model: ExMedBertForBinaryEndpoint, batch: dict[str, torch.Tensor]):
    model.eval()
    with torch.no_grad():
        return model(**model_inputs(batch, include_labels=False))["logits"].detach().cpu()


def ensure_validation_evidence(
    checkpoint_root: str,
    new_model_and_optimizer: Callable[[], tuple[Any, Any]],
    validation_loader: DataLoader,
    device: torch.device,
    *,
    encoder_frozen: bool,
    expected_examples: int,
    manifest_sha256: str,
    configuration_sha256: str,
) -> ValidationCheckpoint:
    for epoch, checkpoint in complete_checkpoints(checkpoint_root):
        existing = read_validation_evidence(
            checkpoint,
            expected_manifest_sha256=manifest_sha256,
            expected_configuration_sha256=configuration_sha256,
        )
        if existing is not None:
            continue
        candidate = ValidationCheckpoint(
            epoch=epoch,
            checkpoint=checkpoint,
            validation_average_precision=0.0,
        )
        model, optimizer, _ = restore_validation_checkpoint(
            candidate,
            new_model_and_optimizer,
            expected_manifest_sha256=manifest_sha256,
            expected_configuration_sha256=configuration_sha256,
        )
        validation_metrics, _, _ = run_split(
            model,
            validation_loader,
            device,
            optimizer=None,
            encoder_frozen=encoder_frozen,
        )
        if validation_metrics["examples"] != expected_examples:
            raise RuntimeError("validation example coverage does not match manifest")
        evidence = write_validation_evidence(
            checkpoint,
            epoch=epoch,
            validation_average_precision=float(validation_metrics["average_precision"]),
            manifest_sha256=manifest_sha256,
            configuration_sha256=configuration_sha256,
        )
        print(
            json.dumps(
                {
                    "backfilled_validation_evidence": {
                        "epoch": evidence.epoch,
                        "checkpoint": evidence.checkpoint,
                        "validation_average_precision": evidence.validation_average_precision,
                    }
                },
                indent=2,
            ),
            flush=True,
        )
        del model, optimizer
        torch.cuda.empty_cache()

    selection = select_best_validation_checkpoint(
        checkpoint_root,
        expected_manifest_sha256=manifest_sha256,
        expected_configuration_sha256=configuration_sha256,
    )
    if selection is None:
        raise RuntimeError("no complete downstream checkpoint has validation evidence")
    return selection


def start_air_mlflow_run() -> bool:
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id or mlflow.active_run() is not None:
        return False
    mlflow.start_run(run_id=run_id)
    return True


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to fall back to CPU")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise RuntimeError("this milestone is single-GPU; distributed training is deferred")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs, batch size, and worker count are invalid")
    if args.initialization != "random" and not args.pretrained_checkpoint:
        raise ValueError("frozen and finetune modes require --pretrained-checkpoint")
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-3 if args.initialization == "frozen" else 3e-5

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    (
        manifest,
        pretrain_manifest,
        vocabulary,
        manifest_hash,
        pretrain_manifest_hash,
    ) = load_metadata(args.snapshot_root)
    if int(manifest["configuration"]["max_sequence_length"]) != args.max_sequence_length:
        raise ValueError("training max_sequence_length must match the supervised snapshot")
    train_statistics = manifest["quality"]["split_statistics"]["train"]
    positive_weight = float(train_statistics["negative"]) / float(train_statistics["positive"])
    encoder_frozen = args.initialization == "frozen"
    configuration = {
        "task": "binary_plos",
        "initialization": args.initialization,
        "pretrained_checkpoint": (
            args.pretrained_checkpoint if args.initialization != "random" else "none"
        ),
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "max_sequence_length": args.max_sequence_length,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "intermediate_size": args.intermediate_size,
        "positive_weight": positive_weight,
        "pooling": "attention_masked_mean",
        "seed": args.seed,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "world_size": 1,
    }
    configuration_hash = canonical_sha256(configuration)
    opened_mlflow_run = start_air_mlflow_run()

    def new_model_and_optimizer(*, transfer: bool):
        model = create_classifier(
            vocabulary,
            pretrain_manifest,
            positive_weight=positive_weight,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_sequence_length=args.max_sequence_length,
        )
        if transfer:
            transfer_pretrained_encoder(
                model,
                vocabulary,
                pretrain_manifest,
                args.pretrained_checkpoint,
                pretrain_manifest_sha256=pretrain_manifest_hash,
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                num_attention_heads=args.num_attention_heads,
                intermediate_size=args.intermediate_size,
                max_sequence_length=args.max_sequence_length,
            )
        if encoder_frozen:
            freeze_encoder(model)
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=0.01,
        )
        return model, optimizer

    try:
        resume_checkpoint = latest_complete_checkpoint(args.checkpoint_root) if args.resume else None
        model, optimizer = new_model_and_optimizer(
            transfer=resume_checkpoint is None and args.initialization != "random"
        )
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("model was not placed on CUDA")
        runtime_info = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda or "unknown",
            "gpu": torch.cuda.get_device_name(0),
            "serverless_gpu": getattr(__import__("serverless_gpu"), "__version__", "unknown"),
            "transformers": __import__("transformers").__version__,
        }
        print(json.dumps({"runtime": runtime_info, "configuration": configuration}, indent=2), flush=True)
        if mlflow.active_run() is not None:
            mlflow.log_params(
                {**configuration, **{f"runtime_{key}": value for key, value in runtime_info.items()}}
            )
            mlflow.log_params(
                {
                    "dataset_version": manifest["version"],
                    "manifest_sha256": manifest_hash,
                    "pretrain_manifest_sha256": pretrain_manifest_hash,
                }
            )
            mlflow.log_metric("model_parameters", sum(p.numel() for p in model.parameters()))
            mlflow.log_metric(
                "trainable_model_parameters",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )

        start_epoch = 0
        global_step = 0
        if resume_checkpoint:
            start_epoch, global_step = load_checkpoint(
                resume_checkpoint,
                model,
                optimizer,
                expected_manifest_sha256=manifest_hash,
                expected_configuration_sha256=configuration_hash,
            )
            print(
                f"Restored {resume_checkpoint}; next_epoch={start_epoch}, global_step={global_step}",
                flush=True,
            )
            if mlflow.active_run() is not None:
                mlflow.log_param("resumed_from", resume_checkpoint)

        incomplete = Path(args.checkpoint_root) / "epoch_999999_incomplete"
        incomplete.mkdir(parents=True, exist_ok=True)
        (incomplete / "README").write_text("controlled incomplete checkpoint probe\n", encoding="utf-8")
        if latest_complete_checkpoint(args.checkpoint_root) == str(incomplete):
            raise AssertionError("incomplete checkpoint was selected")

        train_dataset, train_loader = make_loader(args.snapshot_root, "train", manifest, args)
        validation_dataset, validation_loader = make_loader(
            args.snapshot_root, "validation", manifest, args
        )
        test_dataset, test_loader = make_loader(args.snapshot_root, "test", manifest, args)
        expected_counts = {
            split: int(manifest["quality"]["split_statistics"][split]["examples"])
            for split in ("train", "validation", "test")
        }

        for epoch in range(start_epoch, args.epochs):
            train_metrics, _, _ = run_split(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                encoder_frozen=encoder_frozen,
            )
            validation_metrics, _, probe_batch = run_split(
                model,
                validation_loader,
                device,
                optimizer=None,
                encoder_frozen=encoder_frozen,
            )
            if train_metrics["examples"] != expected_counts["train"]:
                raise RuntimeError("training example coverage does not match manifest")
            if validation_metrics["examples"] != expected_counts["validation"]:
                raise RuntimeError("validation example coverage does not match manifest")
            if args.num_workers == 0:
                if train_dataset.validated_files != set(expected_files(manifest, "train")):
                    raise RuntimeError("not every training Parquet shard was validated")
                if validation_dataset.validated_files != set(expected_files(manifest, "validation")):
                    raise RuntimeError("not every validation Parquet shard was validated")

            global_step += int(train_metrics["optimizer_steps"])
            before = probe_logits(model, probe_batch)
            checkpoint_started = time.monotonic()
            checkpoint = save_checkpoint(
                args.checkpoint_root,
                model,
                optimizer,
                next_epoch=epoch + 1,
                global_step=global_step,
                manifest_sha256=manifest_hash,
                configuration_sha256=configuration_hash,
            )
            checkpoint_seconds = time.monotonic() - checkpoint_started
            fresh_model, fresh_optimizer = new_model_and_optimizer(transfer=False)
            restored_epoch, restored_step = load_checkpoint(
                checkpoint,
                fresh_model,
                fresh_optimizer,
                expected_manifest_sha256=manifest_hash,
                expected_configuration_sha256=configuration_hash,
            )
            after = probe_logits(fresh_model, probe_batch)
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
            if restored_epoch != epoch + 1 or restored_step != global_step:
                raise AssertionError("restored downstream training counters do not match")
            model, optimizer = fresh_model, fresh_optimizer
            validation_evidence = write_validation_evidence(
                checkpoint,
                epoch=epoch + 1,
                validation_average_precision=float(validation_metrics["average_precision"]),
                manifest_sha256=manifest_hash,
                configuration_sha256=configuration_hash,
            )

            evidence = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
                "checkpoint": checkpoint,
                "validation_evidence": str(
                    Path(validation_evidence.checkpoint) / _VALIDATION_EVIDENCE_FILENAME
                ),
                "checkpoint_seconds": checkpoint_seconds,
                "fresh_restore_probe": "passed",
            }
            print(json.dumps(evidence, indent=2), flush=True)
            if mlflow.active_run() is not None:
                mlflow.log_metrics(
                    {
                        "train_loss": float(train_metrics["loss"]),
                        "train_auroc": float(train_metrics["auroc"]),
                        "train_average_precision": float(train_metrics["average_precision"]),
                        "validation_loss": float(validation_metrics["loss"]),
                        "validation_auroc": float(validation_metrics["auroc"]),
                        "validation_average_precision": float(
                            validation_metrics["average_precision"]
                        ),
                        "checkpoint_seconds": checkpoint_seconds,
                    },
                    step=epoch + 1,
                )
                mlflow.set_tag("latest_checkpoint", checkpoint)

        del model, optimizer
        torch.cuda.empty_cache()
        best_checkpoint = ensure_validation_evidence(
            args.checkpoint_root,
            lambda: new_model_and_optimizer(transfer=False),
            validation_loader,
            device,
            encoder_frozen=encoder_frozen,
            expected_examples=expected_counts["validation"],
            manifest_sha256=manifest_hash,
            configuration_sha256=configuration_hash,
        )
        model, optimizer, best_global_step = restore_validation_checkpoint(
            best_checkpoint,
            lambda: new_model_and_optimizer(transfer=False),
            expected_manifest_sha256=manifest_hash,
            expected_configuration_sha256=configuration_hash,
        )
        del optimizer
        torch.cuda.empty_cache()
        model_selection = {
            "criterion": "maximum validation average_precision",
            "tie_breaker": "earliest epoch",
            "best_epoch": best_checkpoint.epoch,
            "best_checkpoint": best_checkpoint.checkpoint,
            "best_validation_average_precision": best_checkpoint.validation_average_precision,
            "restored_global_step": best_global_step,
        }
        print(json.dumps({"model_selection": model_selection}, indent=2), flush=True)
        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "best_epoch": float(best_checkpoint.epoch),
                    "best_validation_average_precision": (
                        best_checkpoint.validation_average_precision
                    ),
                }
            )
            mlflow.set_tags(
                {
                    "checkpoint_selection_metric": "validation_average_precision",
                    "checkpoint_selection_tie_breaker": "earliest_epoch",
                    "best_checkpoint": best_checkpoint.checkpoint,
                    "best_epoch": str(best_checkpoint.epoch),
                    "best_validation_average_precision": str(
                        best_checkpoint.validation_average_precision
                    ),
                }
            )
            mlflow.log_dict(model_selection, "best_checkpoint.json")

        validation_metrics, validation_predictions, _ = run_split(
            model,
            validation_loader,
            device,
            optimizer=None,
            encoder_frozen=encoder_frozen,
        )
        if not math.isclose(
            float(validation_metrics["average_precision"]),
            best_checkpoint.validation_average_precision,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                "restored best checkpoint did not reproduce its validation average_precision"
            )
        decision_threshold = select_youden_threshold(
            validation_predictions["labels"], validation_predictions["probabilities"]
        )
        validation_at_threshold = binary_metrics(
            validation_predictions["labels"],
            validation_predictions["probabilities"],
            threshold=decision_threshold,
        )
        validation_at_threshold.update(
            {key: validation_metrics[key] for key in ("loss", "optimizer_steps", "seconds")}
        )
        test_metrics, test_predictions, _ = run_split(
            model,
            test_loader,
            device,
            optimizer=None,
            encoder_frozen=encoder_frozen,
        )
        test_at_threshold = binary_metrics(
            test_predictions["labels"],
            test_predictions["probabilities"],
            threshold=decision_threshold,
        )
        test_at_threshold.update(
            {key: test_metrics[key] for key in ("loss", "optimizer_steps", "seconds")}
        )
        if test_at_threshold["examples"] != expected_counts["test"]:
            raise RuntimeError("test example coverage does not match manifest")
        if args.num_workers == 0 and test_dataset.validated_files != set(
            expected_files(manifest, "test")
        ):
            raise RuntimeError("not every test Parquet shard was validated")

        final_evidence = {
            "initialization": args.initialization,
            "epochs": args.epochs,
            "model_selection": model_selection,
            "validation_threshold_source": "maximum Youden J on validation",
            "validation": validation_at_threshold,
            "test": test_at_threshold,
        }
        print(json.dumps({"final": final_evidence}, indent=2), flush=True)
        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    f"final_validation_{key}": float(value)
                    for key, value in validation_at_threshold.items()
                    if isinstance(value, (int, float))
                }
            )
            mlflow.log_metrics(
                {
                    f"final_test_{key}": float(value)
                    for key, value in test_at_threshold.items()
                    if isinstance(value, (int, float))
                }
            )
            mlflow.log_dict(final_evidence, "plos_metrics.json")
            mlflow.set_tag("endpoint", "admission_time_plos_gt_7d")
        result_path = write_training_result(
            args.checkpoint_root,
            manifest_sha256=manifest_hash,
            configuration_sha256=configuration_hash,
            configuration=configuration,
            result=final_evidence,
        )
        print(f"Published completed training result: {result_path}", flush=True)
        print("AIR ExMed-BERT PLOS training completed successfully", flush=True)
    finally:
        if opened_mlflow_run and mlflow.active_run() is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
