#!/usr/bin/env python3
"""Train the repository ExMed-BERT model on one AI Runtime GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
from serverless_gpu.data import DataLoader, UCVolumeDataset

from experiments.ai_runtime_exmedbert.shared.contracts import canonical_sha256
from experiments.ai_runtime_exmedbert.training.checkpoint import (
    latest_complete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from experiments.ai_runtime_exmedbert.training.data import (
    DeterministicExMedBertCollator,
    ExMedBertParquetDataset,
)
from experiments.ai_runtime_exmedbert.training.modeling import create_model
from experiments.ai_runtime_exmedbert.training.vocabulary import load_snapshot_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=288)
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument("--num-attention-heads", type=int, default=6)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=201214)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--max-masked-tokens", type=int, default=20)
    parser.add_argument("--mask-medications", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def expected_files(manifest: dict[str, Any], split: str) -> dict[str, int]:
    return {item["path"]: int(item["bytes"]) for item in manifest["artifacts"][split]}


def make_loader(
    snapshot_root: str,
    split: str,
    manifest: dict[str, Any],
    collator: DeterministicExMedBertCollator,
    args: argparse.Namespace,
) -> tuple[ExMedBertParquetDataset, DataLoader]:
    paths = UCVolumeDataset(f"{snapshot_root}/{split}")
    dataset = ExMedBertParquetDataset(paths, expected_files(manifest, split))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )
    return dataset, loader


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_inputs(batch: dict[str, Any], *, include_labels: bool = True) -> dict[str, torch.Tensor]:
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
        keys.append("code_labels")
    return {key: batch[key] for key in keys}


def assert_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient in {name}")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[dict[str, float | int], dict[str, Any] | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    scored_tokens = 0
    optimizer_steps = 0
    patient_ids: list[str] = []
    probe_batch = None
    started = time.monotonic()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            patient_ids.extend(raw_batch["patient_ids"])
            batch = move_batch(raw_batch, device)
            token_count = int((batch["code_labels"] != -100).sum().item())
            if token_count == 0:
                continue
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
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite ExMed-BERT MLM loss")
            if training:
                loss.backward()
                assert_finite_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer_steps += 1
            total_loss += float(loss.detach().item()) * token_count
            scored_tokens += token_count
    if scored_tokens == 0:
        raise RuntimeError("epoch produced no scorable MLM tokens")
    if len(patient_ids) != len(set(patient_ids)):
        raise RuntimeError("a patient was consumed more than once in one pass")
    return (
        {
            "loss": total_loss / scored_tokens,
            "patients": len(patient_ids),
            "scored_tokens": scored_tokens,
            "optimizer_steps": optimizer_steps,
            "seconds": time.monotonic() - started,
        },
        probe_batch,
    )


def probe_logits(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(**model_inputs(batch, include_labels=False))["logits"].detach().cpu()


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
        raise RuntimeError("this milestone is single-GPU; distributed data ownership is intentionally deferred")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    manifest, vocabulary, manifest_hash = load_snapshot_metadata(args.snapshot_root)
    if int(manifest["configuration"]["max_sequence_length"]) != args.max_sequence_length:
        raise ValueError("training max_sequence_length must match the snapshot")
    configuration = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_sequence_length": args.max_sequence_length,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "intermediate_size": args.intermediate_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "mask_probability": args.mask_probability,
        "max_masked_tokens": args.max_masked_tokens,
        "mask_medications": args.mask_medications,
        "world_size": 1,
    }
    configuration_hash = canonical_sha256(configuration)
    opened_mlflow_run = start_air_mlflow_run()

    def new_model_and_optimizer():
        model = create_model(
            vocabulary,
            manifest,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_sequence_length=args.max_sequence_length,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        return model, optimizer

    try:
        model, optimizer = new_model_and_optimizer()
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
            mlflow.log_params({**configuration, **{f"runtime_{key}": value for key, value in runtime_info.items()}})
            mlflow.log_params({"dataset_version": manifest["version"], "manifest_sha256": manifest_hash})
            mlflow.log_metric("model_parameters", sum(parameter.numel() for parameter in model.parameters()))
            mlflow.log_metric(
                "trainable_model_parameters", sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            )

        start_epoch = 0
        global_step = 0
        checkpoint = latest_complete_checkpoint(args.checkpoint_root) if args.resume else None
        if checkpoint:
            start_epoch, global_step = load_checkpoint(
                checkpoint,
                model,
                optimizer,
                expected_manifest_sha256=manifest_hash,
                expected_configuration_sha256=configuration_hash,
            )
            print(f"Restored {checkpoint}; next_epoch={start_epoch}, global_step={global_step}", flush=True)
            if mlflow.active_run() is not None:
                mlflow.log_param("resumed_from", checkpoint)

        incomplete = Path(args.checkpoint_root) / "epoch_999999_incomplete"
        incomplete.mkdir(parents=True, exist_ok=True)
        (incomplete / "README").write_text("controlled incomplete checkpoint probe\n", encoding="utf-8")
        if latest_complete_checkpoint(args.checkpoint_root) == str(incomplete):
            raise AssertionError("incomplete checkpoint was selected")

        train_collator = DeterministicExMedBertCollator(
            code_entity_ids=vocabulary.code_entity_ids,
            maskable_code_ids=vocabulary.maskable_code_ids,
            max_sequence_length=args.max_sequence_length,
            seed=args.seed,
            mask_probability=args.mask_probability,
            max_masked_tokens=args.max_masked_tokens,
            mask_medications=args.mask_medications,
        )
        validation_collator = DeterministicExMedBertCollator(
            code_entity_ids=vocabulary.code_entity_ids,
            maskable_code_ids=vocabulary.maskable_code_ids,
            max_sequence_length=args.max_sequence_length,
            seed=args.seed,
            mask_probability=args.mask_probability,
            max_masked_tokens=args.max_masked_tokens,
            mask_medications=args.mask_medications,
        )
        validation_collator.set_epoch(-1)
        train_dataset, train_loader = make_loader(args.snapshot_root, "train", manifest, train_collator, args)
        validation_dataset, validation_loader = make_loader(
            args.snapshot_root, "validation", manifest, validation_collator, args
        )

        if start_epoch >= args.epochs:
            print(f"Checkpoint already reached requested epochs ({args.epochs}); nothing to train", flush=True)

        for epoch in range(start_epoch, args.epochs):
            train_collator.set_epoch(epoch)
            train_metrics, _ = run_epoch(model, train_loader, device, optimizer=optimizer)
            validation_metrics, probe_batch = run_epoch(model, validation_loader, device, optimizer=None)
            expected_train = int(manifest["quality"]["split_statistics"]["train"]["patients"])
            expected_validation = int(manifest["quality"]["split_statistics"]["validation"]["patients"])
            if train_metrics["patients"] != expected_train:
                raise RuntimeError("training patient coverage does not match manifest")
            if validation_metrics["patients"] != expected_validation:
                raise RuntimeError("validation patient coverage does not match manifest")
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
            fresh_model, fresh_optimizer = new_model_and_optimizer()
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
                raise AssertionError("restored training counters do not match")
            model, optimizer = fresh_model, fresh_optimizer

            evidence = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
                "checkpoint": checkpoint,
                "checkpoint_seconds": checkpoint_seconds,
                "fresh_restore_probe": "passed",
            }
            print(json.dumps(evidence, indent=2), flush=True)
            if mlflow.active_run() is not None:
                mlflow.log_metrics(
                    {
                        "train_loss": float(train_metrics["loss"]),
                        "validation_loss": float(validation_metrics["loss"]),
                        "train_patients": int(train_metrics["patients"]),
                        "validation_patients": int(validation_metrics["patients"]),
                        "train_scored_tokens": int(train_metrics["scored_tokens"]),
                        "validation_scored_tokens": int(validation_metrics["scored_tokens"]),
                        "checkpoint_seconds": checkpoint_seconds,
                    },
                    step=epoch + 1,
                )
                mlflow.set_tag("latest_checkpoint", checkpoint)
        print("AIR ExMed-BERT training completed successfully", flush=True)
    finally:
        if opened_mlflow_run and mlflow.active_run() is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
