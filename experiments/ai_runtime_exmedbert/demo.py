#!/usr/bin/env python3
"""Operate the reproducible Synthea -> Spark -> AIR ExMed-BERT demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "demo_config.json"
TERMINAL_AIR_STATES = {"SUCCESS", "FAILED", "CANCELED", "TIMEDOUT", "INTERNAL_ERROR"}
CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d{6})_[0-9a-f]+$")
DOWNSTREAM_METRICS = (
    "average_precision",
    "auroc",
    "loss_brier",
    "calibration_error",
)


@dataclass(frozen=True)
class DemoPaths:
    source_root: str
    pretrain_snapshot_root: str
    pretrain_checkpoint_root: str
    audit_root: str
    audit_path: str
    plos_snapshot_root: str
    plos_checkpoint_root: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "validate", "run", "status"))
    parser.add_argument("--profile", required=True, help="Explicit Databricks CLI profile")
    parser.add_argument("--target", default="dev", help="Bundle target")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--include-comparators",
        action="store_true",
        help="Also run random and frozen PLOS controls after fine-tuning",
    )
    parser.add_argument(
        "--downstream-mode",
        action="append",
        choices=("finetune", "random", "frozen"),
        help="Run only the selected downstream initialization; repeat for multiple modes",
    )
    parser.add_argument(
        "--resume-proof",
        action="store_true",
        help="Launch a second MLM process for one additional epoch",
    )
    parser.add_argument(
        "--force-air",
        action="store_true",
        help="Submit AIR even when the requested complete checkpoint already exists",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    required_strings = (
        "volume_root",
        "source_release",
        "pretrain_snapshot_version",
        "audit_version",
        "plos_snapshot_version",
    )
    required_positive = (
        "pretrain_epochs",
        "pretrain_batch_size",
        "downstream_epochs",
        "downstream_batch_size",
        "num_workers",
        "prefetch_factor",
        "num_shards",
        "downstream_subset_seed",
    )
    for key in required_strings:
        if not isinstance(document.get(key), str) or not document[key].strip():
            raise ValueError(f"configuration requires a non-empty {key}")
    if not document["volume_root"].startswith("/Volumes/"):
        raise ValueError("volume_root must be a Unity Catalog /Volumes path")
    for key in required_positive:
        if isinstance(document.get(key), bool) or not isinstance(document.get(key), int) or document[key] < 1:
            raise ValueError(f"configuration requires a positive integer {key}")
    seeds = document.get("downstream_training_seeds")
    legacy_seed = document.get("downstream_training_seed")
    if seeds is None:
        seeds = [legacy_seed]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("configuration requires a non-empty downstream_training_seeds list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 1 for seed in seeds):
        raise ValueError("downstream_training_seeds must contain only positive integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("downstream_training_seeds must not contain duplicates")
    if legacy_seed is not None:
        if isinstance(legacy_seed, bool) or not isinstance(legacy_seed, int) or legacy_seed < 1:
            raise ValueError("downstream_training_seed must be a positive integer")
        if legacy_seed != seeds[0]:
            raise ValueError(
                "downstream_training_seed must match the first downstream_training_seeds value"
            )
    document["downstream_training_seeds"] = seeds
    document["downstream_training_seed"] = seeds[0]
    maximum_oov = document.get("max_oov_fraction")
    if not isinstance(maximum_oov, (int, float)) or not 0 <= maximum_oov <= 1:
        raise ValueError("max_oov_fraction must be between zero and one")
    train_fraction = document.get("downstream_train_fraction")
    if isinstance(train_fraction, bool) or not isinstance(train_fraction, (int, float)):
        raise ValueError("downstream_train_fraction must be numeric")
    if not 0 < train_fraction <= 1:
        raise ValueError("downstream_train_fraction must be greater than zero and at most one")
    return document


def derive_paths(config: dict[str, Any]) -> DemoPaths:
    root = config["volume_root"].rstrip("/")
    pretrain_version = config["pretrain_snapshot_version"]
    plos_version = config["plos_snapshot_version"]
    audit_root = f"{root}/exmedbert/audits/{config['audit_version']}"
    return DemoPaths(
        source_root=f"{root}/synthea/raw/{config['source_release']}",
        pretrain_snapshot_root=f"{root}/exmedbert/snapshots/{pretrain_version}",
        pretrain_checkpoint_root=f"{root}/exmedbert/checkpoints/{pretrain_version}",
        audit_root=audit_root,
        audit_path=f"{audit_root}/audit.json",
        plos_snapshot_root=f"{root}/exmedbert/downstream/plos/snapshots/{plos_version}",
        plos_checkpoint_root=f"{root}/exmedbert/downstream/plos/checkpoints/{plos_version}",
    )


def command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["COPYFILE_DISABLE"] = "1"
    if platform.system() == "Darwin":
        gnubin = Path("/opt/homebrew/opt/gnu-tar/libexec/gnubin")
        if gnubin.is_dir():
            environment["PATH"] = f"{gnubin}:{environment.get('PATH', '')}"
    return environment


def display_command(command: Sequence[str]) -> str:
    return " ".join(_quote(argument) for argument in command)


def _quote(value: str) -> str:
    if value and all(character.isalnum() or character in "-._/:=@" for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def execute(
    command: Sequence[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {display_command(command)}", flush=True)
    return subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=command_environment(),
        check=check,
        text=True,
        capture_output=capture,
    )


def parse_json_output(output: str) -> Any:
    """Parse regular JSON or the final AIR JSON envelope from mixed CLI output."""
    stripped = output.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for line in reversed(stripped.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"command did not return JSON: {stripped[-500:]}")


def volume_uri(path: str) -> str:
    if not path.startswith("/Volumes/"):
        raise ValueError(f"not a UC volume path: {path}")
    return f"dbfs:{path}"


def remote_exists(path: str, profile: str) -> bool:
    operation = "cat" if path.endswith(".json") else "ls"
    command = ["databricks", "fs", operation, volume_uri(path), "--profile", profile]
    if operation == "ls":
        command.extend(("--output", "json"))
    result = execute(
        command,
        capture=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    message = f"{result.stdout}\n{result.stderr}".strip()
    missing_markers = (
        "resource_does_not_exist",
        "no such file",
        "does not exist",
        "not found",
    )
    if any(marker in message.lower() for marker in missing_markers):
        return False
    raise RuntimeError(f"cannot inspect {path}: {message[-1000:]}")


def read_remote_json(path: str, profile: str) -> dict[str, Any]:
    result = execute(
        ["databricks", "fs", "cat", volume_uri(path), "--profile", profile],
        capture=True,
    )
    document = json.loads(result.stdout)
    if not isinstance(document, dict):
        raise ValueError(f"remote JSON document must be an object: {path}")
    return document


def validate_plos_manifest_reuse(
    manifest: dict[str, Any], config: dict[str, Any], paths: DemoPaths
) -> None:
    configuration = manifest.get("configuration", {})
    source = manifest.get("source", {})
    pretraining = manifest.get("pretraining", {})
    actual_fraction = configuration.get("train_label_fraction", 1.0)
    actual_subset_seed = configuration.get(
        "train_label_subset_seed", configuration.get("seed")
    )
    expected = {
        "version": config["plos_snapshot_version"],
        "source.root": paths.source_root,
        "pretraining.snapshot_root": paths.pretrain_snapshot_root,
        "configuration.train_label_fraction": float(config["downstream_train_fraction"]),
        "configuration.train_label_subset_seed": config["downstream_subset_seed"],
        "configuration.num_shards_per_split": config["num_shards"],
    }
    actual = {
        "version": manifest.get("version"),
        "source.root": source.get("root"),
        "pretraining.snapshot_root": pretraining.get("snapshot_root"),
        "configuration.train_label_fraction": actual_fraction,
        "configuration.train_label_subset_seed": actual_subset_seed,
        "configuration.num_shards_per_split": configuration.get("num_shards_per_split"),
    }
    mismatches = [
        f"{key}: expected {expected[key]!r}, found {actual[key]!r}"
        for key in expected
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise ValueError(
            "existing PLOS snapshot does not match the requested experiment; "
            "choose a new plos_snapshot_version (" + "; ".join(mismatches) + ")"
        )


def downstream_lineage_checkpoint_root(
    paths: DemoPaths, pretrained_checkpoint: str, training_seed: int
) -> str:
    root = f"{paths.plos_checkpoint_root}/from_{Path(pretrained_checkpoint).name}"
    if training_seed != 201214:
        root = f"{root}/seed_{training_seed}"
    return root


def validate_training_result(
    training_result: dict[str, Any], *, mode: str, seed: int, source: str
) -> dict[str, float]:
    configuration = training_result.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"training result has no configuration object: {source}")
    if configuration.get("seed") != seed:
        raise ValueError(
            f"training result seed mismatch at {source}: expected {seed}, "
            f"found {configuration.get('seed')!r}"
        )
    if configuration.get("initialization") != mode:
        raise ValueError(
            f"training result mode mismatch at {source}: expected {mode!r}, "
            f"found {configuration.get('initialization')!r}"
        )
    test_metrics = training_result.get("result", {}).get("test")
    if not isinstance(test_metrics, dict):
        raise ValueError(f"training result has no test metrics object: {source}")
    metrics: dict[str, float] = {}
    for metric in DOWNSTREAM_METRICS:
        value = test_metrics.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"training result is missing numeric test metric {metric}: {source}")
        metrics[metric] = float(value)
    return metrics


def aggregate_downstream_results(
    results: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot aggregate downstream results without seeds")
    paired: list[dict[str, Any]] = []
    for seed, modes in results.items():
        missing_modes = [mode for mode in ("finetune", "random") if mode not in modes]
        if missing_modes:
            raise ValueError(
                f"seed {seed} is missing paired downstream mode(s): {', '.join(missing_modes)}"
            )
        finetune = validate_training_result(
            modes["finetune"], mode="finetune", seed=seed, source=f"seed {seed} finetune"
        )
        random = validate_training_result(
            modes["random"], mode="random", seed=seed, source=f"seed {seed} random"
        )
        paired.append(
            {
                "seed": seed,
                "finetune": finetune,
                "random": random,
                "difference_finetune_minus_random": {
                    metric: finetune[metric] - random[metric]
                    for metric in DOWNSTREAM_METRICS
                },
            }
        )
    summary: dict[str, dict[str, float | None]] = {}
    for metric in DOWNSTREAM_METRICS:
        differences = [row["difference_finetune_minus_random"][metric] for row in paired]
        summary[metric] = {
            "mean": statistics.mean(differences),
            "sample_standard_deviation": (
                statistics.stdev(differences) if len(differences) > 1 else None
            ),
        }
    return {
        "comparison": "finetune_minus_random",
        "metrics": list(DOWNSTREAM_METRICS),
        "interpretation": {
            "average_precision": "positive favors finetune",
            "auroc": "positive favors finetune",
            "loss_brier": "negative favors finetune",
            "calibration_error": "negative favors finetune",
        },
        "per_seed": paired,
        "summary": summary,
    }


def latest_complete_checkpoint(root: str, profile: str) -> tuple[int, str] | None:
    result = execute(
        ["databricks", "fs", "ls", volume_uri(root), "--profile", profile, "--output", "json"],
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    candidates: list[tuple[int, str]] = []
    for entry in parse_json_output(result.stdout):
        match = CHECKPOINT_PATTERN.match(entry.get("name", ""))
        if entry.get("is_directory") and match:
            candidates.append((int(match.group(1)), f"{root}/{entry['name']}"))
    for epoch, path in sorted(candidates, reverse=True):
        metadata = execute(
            [
                "databricks",
                "fs",
                "ls",
                volume_uri(path),
                "--profile",
                profile,
                "--output",
                "json",
            ],
            capture=True,
            check=False,
        )
        if metadata.returncode == 0 and any(
            item.get("name") == ".metadata" for item in parse_json_output(metadata.stdout)
        ):
            return epoch, path
    return None


def downstream_lineages(root: str, profile: str) -> dict[str, dict[str, tuple[int, str] | None]]:
    result = execute(
        ["databricks", "fs", "ls", volume_uri(root), "--profile", profile, "--output", "json"],
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    lineages: dict[str, dict[str, tuple[int, str] | None]] = {}
    for entry in parse_json_output(result.stdout):
        name = entry.get("name", "")
        if entry.get("is_directory") and name.startswith("from_epoch_"):
            lineages[name] = {
                mode: latest_complete_checkpoint(f"{root}/{name}/{mode}", profile)
                for mode in ("finetune", "random", "frozen")
            }
    return lineages


def bundle_command(profile: str, target: str, action: str, *arguments: str) -> list[str]:
    values = list(arguments)
    flags = ["-t", target, "--profile", profile]
    if "--" in values:
        separator = values.index("--")
        return ["databricks", "bundle", action, *values[:separator], *flags, *values[separator:]]
    return ["databricks", "bundle", action, *values, *flags]


def air_overrides(spec: str, config: dict[str, Any], paths: DemoPaths, **extra: str) -> list[str]:
    if spec == "train.yaml":
        values = {
            "SNAPSHOT_ROOT": paths.pretrain_snapshot_root,
            "CHECKPOINT_ROOT": paths.pretrain_checkpoint_root,
            "EPOCHS": str(config["pretrain_epochs"]),
            "BATCH_SIZE": str(config["pretrain_batch_size"]),
            "NUM_WORKERS": str(config["num_workers"]),
            "PREFETCH_FACTOR": str(config["prefetch_factor"]),
        }
    else:
        values = {
            "SNAPSHOT_ROOT": paths.plos_snapshot_root,
            "CHECKPOINT_ROOT": paths.plos_checkpoint_root,
            "EPOCHS": str(config["downstream_epochs"]),
            "BATCH_SIZE": str(config["downstream_batch_size"]),
            "NUM_WORKERS": str(config["num_workers"]),
            "PREFETCH_FACTOR": str(config["prefetch_factor"]),
            "SEED": str(config["downstream_training_seed"]),
        }
    values.update(extra)
    return [f"env_variables.{key}={value}" for key, value in values.items()]


def air_command(
    profile: str,
    spec: str,
    overrides: Sequence[str],
    *,
    dry_run: bool = False,
    idempotency_key: str | None = None,
) -> list[str]:
    command = ["air", "run", "--profile", profile, "--file", spec, "--json"]
    if dry_run:
        command.append("--dry-run")
    if idempotency_key:
        command.extend(("--idempotency-key", idempotency_key))
    command.extend(("--override", *overrides))
    return command


def wait_for_air(run_id: str, profile: str, poll_seconds: int = 15) -> dict[str, Any]:
    while True:
        result = execute(
            ["air", "get", "run", run_id, "--profile", profile, "--json"],
            capture=True,
        )
        document = parse_json_output(result.stdout)["data"]
        state = document["status"]
        print(f"AIR run {run_id}: {state}", flush=True)
        if state in TERMINAL_AIR_STATES:
            if state != "SUCCESS":
                raise RuntimeError(
                    f"AIR run {run_id} ended as {state}: {document.get('termination_reason')}"
                )
            return document
        time.sleep(poll_seconds)


def idempotency_key(stage: str, values: Sequence[str]) -> str:
    source_files = (
        PROJECT_ROOT / "demo.py",
        PROJECT_ROOT / "train.yaml",
        PROJECT_ROOT / "train_plos.yaml",
        PROJECT_ROOT / "training" / "train.py",
        PROJECT_ROOT / "training" / "train_plos.py",
    )
    source_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files]
    digest = hashlib.sha256(
        "\n".join((*values, *source_hashes)).encode("utf-8")
    ).hexdigest()[:20]
    return f"exmedbert-{stage}-{digest}"


def submit_air(
    profile: str,
    spec: str,
    overrides: Sequence[str],
    stage: str,
) -> dict[str, Any]:
    result = execute(
        air_command(
            profile,
            spec,
            overrides,
            idempotency_key=idempotency_key(stage, overrides),
        ),
        capture=True,
    )
    document = parse_json_output(result.stdout)
    run_id = str(document["data"]["run_id"])
    return wait_for_air(run_id, profile)


def spark_commands(profile: str, target: str, config: dict[str, Any], paths: DemoPaths):
    pretrain_output_root = str(Path(paths.pretrain_snapshot_root).parent)
    audit_output_root = str(Path(paths.audit_root).parent)
    plos_output_root = str(Path(paths.plos_snapshot_root).parent)
    return {
        "pretrain_snapshot": bundle_command(
            profile,
            target,
            "run",
            "prepare_exmedbert_snapshot",
            "--",
            "--source-root",
            paths.source_root,
            "--output-root",
            pretrain_output_root,
            "--snapshot-version",
            config["pretrain_snapshot_version"],
            "--num-shards",
            str(config["num_shards"]),
            "--max-oov-fraction",
            str(config["max_oov_fraction"]),
        ),
        "endpoint_audit": bundle_command(
            profile,
            target,
            "run",
            "profile_exmedbert_endpoints",
            "--",
            "--source-root",
            paths.source_root,
            "--output-root",
            audit_output_root,
            "--audit-version",
            config["audit_version"],
        ),
        "plos_snapshot": bundle_command(
            profile,
            target,
            "run",
            "prepare_exmedbert_plos_snapshot",
            "--",
            "--source-root",
            paths.source_root,
            "--pretrain-snapshot-root",
            paths.pretrain_snapshot_root,
            "--output-root",
            plos_output_root,
            "--snapshot-version",
            config["plos_snapshot_version"],
            "--num-shards",
            str(config["num_shards"]),
            "--train-label-fraction",
            str(config["downstream_train_fraction"]),
            "--subset-seed",
            str(config["downstream_subset_seed"]),
        ),
    }


def print_plan(profile: str, target: str, config: dict[str, Any], paths: DemoPaths) -> None:
    commands = spark_commands(profile, target, config, paths)
    print("ExMed-BERT AIR demo plan\n")
    print(json.dumps(asdict(paths), indent=2))
    print("\nStages:")
    print(f"1. Validate and deploy bundle: {display_command(bundle_command(profile, target, 'deploy'))}")
    print(f"2. Spark pretraining snapshot: {display_command(commands['pretrain_snapshot'])}")
    print("3. AIR MLM pretraining: train.yaml")
    print(f"4. Spark endpoint audit: {display_command(commands['endpoint_audit'])}")
    print(f"5. Spark PLOS snapshot: {display_command(commands['plos_snapshot'])}")
    print("6. AIR PLOS fine-tuning and MLflow evidence: train_plos.yaml")
    print("Complete immutable Spark artifacts and requested AIR checkpoints are reused.")


def validate(profile: str, target: str, config: dict[str, Any], paths: DemoPaths) -> None:
    for executable in ("databricks", "air"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is not installed: {executable}")
    execute(["databricks", "--version"])
    execute(["air", "--version"])
    execute(["databricks", "current-user", "me", "--profile", profile])
    execute(bundle_command(profile, target, "validate", "--strict"))
    execute(
        air_command(
            profile,
            "train.yaml",
            air_overrides("train.yaml", config, paths),
            dry_run=True,
        )
    )
    checkpoint = latest_complete_checkpoint(paths.pretrain_checkpoint_root, profile)
    pretrained_checkpoint = (
        checkpoint[1] if checkpoint else f"{paths.pretrain_checkpoint_root}/generated_checkpoint"
    )
    validation_seed = config["downstream_training_seeds"][0]
    validation_paths = replace(
        paths,
        plos_checkpoint_root=downstream_lineage_checkpoint_root(
            paths, pretrained_checkpoint, validation_seed
        ),
    )
    plos_extra = {
        "PRETRAINED_CHECKPOINT": pretrained_checkpoint,
        "INITIALIZATION": "finetune",
    }
    execute(
        air_command(
            profile,
            "train_plos.yaml",
            air_overrides(
                "train_plos.yaml",
                {**config, "downstream_training_seed": validation_seed},
                validation_paths,
                **plos_extra,
            ),
            dry_run=True,
        )
    )
    print("Validation passed.")


def run_demo(args: argparse.Namespace, config: dict[str, Any], paths: DemoPaths) -> dict[str, Any]:
    profile, target = args.profile, args.target
    validate(profile, target, config, paths)
    execute(bundle_command(profile, target, "deploy"))
    if not remote_exists(paths.source_root, profile):
        raise FileNotFoundError(
            f"raw Synthea release is missing: {paths.source_root}; see synthea/README.md"
        )

    evidence: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "target": target,
        "configuration": config,
        "paths": asdict(paths),
        "spark": {},
        "air": {},
    }
    commands = spark_commands(profile, target, config, paths)
    for stage, marker in (
        ("pretrain_snapshot", f"{paths.pretrain_snapshot_root}/manifest.json"),
    ):
        if remote_exists(marker, profile):
            evidence["spark"][stage] = {"status": "REUSED", "marker": marker}
            print(f"Reusing complete {stage}: {marker}")
        else:
            execute(commands[stage])
            if not remote_exists(marker, profile):
                raise RuntimeError(f"{stage} completed without its marker: {marker}")
            evidence["spark"][stage] = {"status": "CREATED", "marker": marker}

    desired_pretrain_epoch = int(config["pretrain_epochs"])
    checkpoint = latest_complete_checkpoint(paths.pretrain_checkpoint_root, profile)
    if args.force_air or checkpoint is None or checkpoint[0] < desired_pretrain_epoch:
        overrides = air_overrides("train.yaml", config, paths)
        evidence["air"]["pretrain"] = submit_air(profile, "train.yaml", overrides, "pretrain")
        checkpoint = latest_complete_checkpoint(paths.pretrain_checkpoint_root, profile)
    else:
        evidence["air"]["pretrain"] = {
            "status": "REUSED",
            "checkpoint": checkpoint[1],
        }
    if checkpoint is None or checkpoint[0] < desired_pretrain_epoch:
        raise RuntimeError("AIR pretraining did not publish the requested complete checkpoint")
    evidence["air"]["pretrain"]["checkpoint"] = checkpoint[1]

    if args.resume_proof:
        resume_config = {**config, "pretrain_epochs": checkpoint[0] + 1}
        overrides = air_overrides("train.yaml", resume_config, paths)
        evidence["air"]["pretrain_resume"] = submit_air(
            profile, "train.yaml", overrides, "pretrain-resume"
        )
        checkpoint = latest_complete_checkpoint(paths.pretrain_checkpoint_root, profile)
        if checkpoint is None or checkpoint[0] < resume_config["pretrain_epochs"]:
            raise RuntimeError("separate-process resume did not publish its checkpoint")
        evidence["air"]["pretrain_resume"]["checkpoint"] = checkpoint[1]

    if remote_exists(paths.audit_path, profile):
        evidence["spark"]["endpoint_audit"] = {
            "status": "REUSED",
            "marker": paths.audit_path,
        }
        print(f"Reusing complete endpoint_audit: {paths.audit_path}")
    else:
        execute(commands["endpoint_audit"])
        if not remote_exists(paths.audit_path, profile):
            raise RuntimeError(f"endpoint audit completed without its marker: {paths.audit_path}")
        evidence["spark"]["endpoint_audit"] = {
            "status": "CREATED",
            "marker": paths.audit_path,
        }

    marker = f"{paths.plos_snapshot_root}/manifest.json"
    if remote_exists(marker, profile):
        plos_manifest = read_remote_json(marker, profile)
        validate_plos_manifest_reuse(plos_manifest, config, paths)
        evidence["spark"]["plos_snapshot"] = {"status": "REUSED", "marker": marker}
        print(f"Reusing complete plos_snapshot: {marker}")
    else:
        execute(commands["plos_snapshot"])
        if not remote_exists(marker, profile):
            raise RuntimeError(f"PLOS snapshot completed without its marker: {marker}")
        plos_manifest = read_remote_json(marker, profile)
        validate_plos_manifest_reuse(plos_manifest, config, paths)
        evidence["spark"]["plos_snapshot"] = {"status": "CREATED", "marker": marker}

    requires_training_result = "train_label_fraction" in plos_manifest.get("configuration", {})
    modes = list(dict.fromkeys(args.downstream_mode or ["finetune"]))
    if args.include_comparators:
        modes = list(dict.fromkeys((*modes, "random", "frozen")))
    seeds = config["downstream_training_seeds"]
    collected_results: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in seeds}
    for seed in seeds:
        seed_config = {**config, "downstream_training_seed": seed}
        for mode in modes:
            evidence_key = f"plos_{mode}" if len(seeds) == 1 else f"plos_{mode}_seed_{seed}"
            lineage_checkpoint_root = downstream_lineage_checkpoint_root(
                paths, checkpoint[1], seed
            )
            mode_checkpoint_root = f"{lineage_checkpoint_root}/{mode}"
            complete = latest_complete_checkpoint(mode_checkpoint_root, profile)
            result_marker = f"{mode_checkpoint_root}/training_result.json"
            result_complete = remote_exists(result_marker, profile) if complete else False
            desired_epoch = int(config["downstream_epochs"])
            if (
                not args.force_air
                and complete
                and complete[0] >= desired_epoch
                and (result_complete or not requires_training_result)
            ):
                evidence["air"][evidence_key] = {
                    "status": "REUSED",
                    "seed": seed,
                    "checkpoint": complete[1],
                    "result": result_marker if result_complete else "legacy checkpoint-only evidence",
                }
                if result_complete:
                    training_result = read_remote_json(result_marker, profile)
                    validate_training_result(
                        training_result, mode=mode, seed=seed, source=result_marker
                    )
                    evidence["air"][evidence_key]["training_result"] = training_result
                    collected_results[seed][mode] = training_result
                continue
            overrides = air_overrides(
                "train_plos.yaml",
                seed_config,
                replace(paths, plos_checkpoint_root=lineage_checkpoint_root),
                PRETRAINED_CHECKPOINT=checkpoint[1],
                INITIALIZATION=mode,
            )
            evidence["air"][evidence_key] = submit_air(
                profile, "train_plos.yaml", overrides, f"plos-{mode}-seed-{seed}"
            )
            evidence["air"][evidence_key]["seed"] = seed
            complete = latest_complete_checkpoint(mode_checkpoint_root, profile)
            if complete is None or complete[0] < desired_epoch:
                raise RuntimeError(
                    f"PLOS {mode} seed {seed} did not publish the requested complete checkpoint"
                )
            if not remote_exists(result_marker, profile):
                raise RuntimeError(
                    f"PLOS {mode} seed {seed} completed without its training result: {result_marker}"
                )
            training_result = read_remote_json(result_marker, profile)
            validate_training_result(training_result, mode=mode, seed=seed, source=result_marker)
            evidence["air"][evidence_key]["checkpoint"] = complete[1]
            evidence["air"][evidence_key]["result"] = result_marker
            evidence["air"][evidence_key]["training_result"] = training_result
            collected_results[seed][mode] = training_result

    if "finetune" in modes and "random" in modes:
        evidence["downstream_aggregation"] = aggregate_downstream_results(collected_results)

    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    reports = PROJECT_ROOT / "runs"
    reports.mkdir(exist_ok=True)
    report = reports / f"demo-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Demo completed. Evidence report: {report}")
    return evidence


def status(profile: str, config: dict[str, Any], paths: DemoPaths) -> None:
    execute(["databricks", "current-user", "me", "--profile", profile])
    rows = {
        "raw_source": remote_exists(paths.source_root, profile),
        "pretrain_snapshot": remote_exists(f"{paths.pretrain_snapshot_root}/manifest.json", profile),
        "endpoint_audit": remote_exists(paths.audit_path, profile),
        "plos_snapshot": remote_exists(f"{paths.plos_snapshot_root}/manifest.json", profile),
    }
    checkpoints = {
        "pretrain": latest_complete_checkpoint(paths.pretrain_checkpoint_root, profile),
        "legacy_plos_finetune": latest_complete_checkpoint(
            f"{paths.plos_checkpoint_root}/finetune", profile
        ),
        "legacy_plos_random": latest_complete_checkpoint(
            f"{paths.plos_checkpoint_root}/random", profile
        ),
        "legacy_plos_frozen": latest_complete_checkpoint(
            f"{paths.plos_checkpoint_root}/frozen", profile
        ),
    }
    print(
        json.dumps(
            {
                "artifacts": rows,
                "checkpoints": checkpoints,
                "downstream_lineages": downstream_lineages(
                    paths.plos_checkpoint_root, profile
                ),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = derive_paths(config)
    if args.command == "plan":
        print_plan(args.profile, args.target, config, paths)
    elif args.command == "validate":
        validate(args.profile, args.target, config, paths)
    elif args.command == "run":
        run_demo(args, config, paths)
    else:
        status(args.profile, config, paths)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; remote jobs continue unless canceled explicitly.", file=sys.stderr)
        raise SystemExit(130)
