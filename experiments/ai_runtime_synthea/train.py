#!/usr/bin/env python3
"""Single-GPU AI Runtime smoke test for native UC loading and DCP recovery."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
from serverless_gpu.data import DataLoader, UCVolumeDataset

from runtime import (
    DeterministicMLMCollator,
    ParquetPatientDataset,
    TinyMedicalBert,
    assert_finite_gradients,
    canonical_sha256,
    latest_complete_checkpoint,
    load_checkpoint,
    load_snapshot_metadata,
    loss_sum_and_tokens,
    parameter_count,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=201214)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_model(vocab_size: int, args: argparse.Namespace) -> TinyMedicalBert:
    return TinyMedicalBert(
        vocab_size=vocab_size,
        max_sequence_length=args.max_sequence_length,
        use_time=args.use_time,
    )


def expected_hash(manifest: dict[str, Any], split: str) -> dict[str, str]:
    artifact = manifest["files"][f"{split}_parquet"]
    return {Path(artifact["path"]).name: artifact["sha256"]}


def make_loader(
    snapshot_root: str,
    split: str,
    manifest: dict[str, Any],
    collator: DeterministicMLMCollator,
    args: argparse.Namespace,
) -> tuple[ParquetPatientDataset, DataLoader]:
    paths = UCVolumeDataset(f"{snapshot_root}/{split}")
    dataset = ParquetPatientDataset(paths, expected_hash(manifest, split))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        # serverless_gpu.data.DataLoader defaults to prefetch_factor=4, but
        # PyTorch rejects prefetching when the deterministic baseline uses no
        # multiprocessing workers.
        prefetch_factor=None,
        collate_fn=collator,
    )
    return dataset, loader


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: TinyMedicalBert,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[dict[str, float | int], dict[str, Any] | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    scored_tokens = 0
    patient_ids: list[str] = []
    probe_batch: dict[str, Any] | None = None
    started = time.monotonic()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            if probe_batch is None:
                probe_batch = {key: value.detach().clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
            patient_ids.extend(raw_batch["patient_ids"])
            logits = model(batch["input_ids"], batch["time_gaps"], batch["attention_mask"])
            loss_sum, token_count = loss_sum_and_tokens(logits, batch["labels"])
            if token_count == 0:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
                (loss_sum / token_count).backward()
                assert_finite_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss_sum.detach().item())
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
            "seconds": time.monotonic() - started,
        },
        probe_batch,
    )


def probe_logits(model: TinyMedicalBert, batch: dict[str, Any]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(batch["input_ids"], batch["time_gaps"], batch["attention_mask"]).detach().cpu()


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
    if args.num_workers != 0:
        raise ValueError("version 1 pins num_workers=0")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    manifest, vocabulary, manifest_hash = load_snapshot_metadata(args.snapshot_root)
    configuration = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_sequence_length": args.max_sequence_length,
        "seed": args.seed,
        "use_time": args.use_time,
        "vocab_size": len(vocabulary),
        "world_size": 1,
        "num_workers": args.num_workers,
    }
    configuration_hash = canonical_sha256(configuration)
    opened_mlflow_run = start_air_mlflow_run()

    try:
        model = make_model(len(vocabulary), args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("model was not placed on CUDA")

        runtime_info = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda or "unknown",
            "gpu": torch.cuda.get_device_name(0),
            "serverless_gpu": __import__("serverless_gpu").__version__ if hasattr(__import__("serverless_gpu"), "__version__") else "unknown",
        }
        print(json.dumps({"runtime": runtime_info, "configuration": configuration}, indent=2), flush=True)
        if mlflow.active_run() is not None:
            mlflow.log_params({**configuration, **{f"runtime_{key}": value for key, value in runtime_info.items()}})
            mlflow.log_params({"dataset_version": manifest["version"], "manifest_sha256": manifest_hash})
            mlflow.log_metric("model_parameters", parameter_count(model))

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

        # An incomplete directory must never be selected by discovery.
        incomplete = Path(args.checkpoint_root) / "epoch_999999_incomplete"
        incomplete.mkdir(parents=True, exist_ok=True)
        (incomplete / "README").write_text("controlled incomplete checkpoint probe\n", encoding="utf-8")
        if latest_complete_checkpoint(args.checkpoint_root) == str(incomplete):
            raise AssertionError("incomplete checkpoint was selected")

        train_collator = DeterministicMLMCollator(
            vocab_size=len(vocabulary), max_sequence_length=args.max_sequence_length, seed=args.seed
        )
        validation_collator = DeterministicMLMCollator(
            vocab_size=len(vocabulary), max_sequence_length=args.max_sequence_length, seed=args.seed
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
            if train_metrics["patients"] != manifest["counts"]["train_patients"]:
                raise RuntimeError("training patient coverage does not match manifest")
            if validation_metrics["patients"] != manifest["counts"]["validation_patients"]:
                raise RuntimeError("validation patient coverage does not match manifest")
            if train_dataset.validated_files != {"patients.parquet"}:
                raise RuntimeError("training Parquet checksum was not validated through the local cache")
            if validation_dataset.validated_files != {"patients.parquet"}:
                raise RuntimeError("validation Parquet checksum was not validated through the local cache")

            global_step += (int(train_metrics["patients"]) + args.batch_size - 1) // args.batch_size
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

            fresh_model = make_model(len(vocabulary), args).to(device)
            fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=args.learning_rate)
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

            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "checkpoint": checkpoint,
                        "checkpoint_seconds": checkpoint_seconds,
                        "fresh_restore_probe": "passed",
                    },
                    indent=2,
                ),
                flush=True,
            )
            if mlflow.active_run() is not None:
                step = epoch + 1
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
                    step=step,
                )
                mlflow.set_tag("latest_checkpoint", checkpoint)

        print("AIR Synthea smoke test completed successfully", flush=True)
    finally:
        if opened_mlflow_run and mlflow.active_run() is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
