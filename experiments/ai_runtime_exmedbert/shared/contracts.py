"""Dependency-light contracts shared by Spark preparation and AIR training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

SCHEMA_VERSION = 1
SPECIAL_TOKENS = ("PAD", "UNK", "CLS", "SEP", "MASK", "NA")
SPECIAL_TOKEN_IDS = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
ENTITY_ORDER = ("medication", "diagnosis", "procedure")
ENTITY_IDS = {"default": 0, **{name: index + 1 for index, name in enumerate(ENTITY_ORDER)}}
TRAIN_LABEL_SELECTION_METHOD = "sha256_rank_prefix_by_patient_label_profile"
TRAIN_LABEL_HASH_INPUT = "{seed}:train-label:{patient_id}"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_bucket(patient_id: str, seed: int) -> int:
    """Match the Spark SHA-256 split expression without relying on xxhash."""
    digest = hashlib.sha256(f"{seed}:split:{patient_id}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) % 10_000


def split_name(bucket: int, train_ratio: float, validation_ratio: float) -> str:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")
    if validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("validation_ratio must leave a positive test split")
    train_end = round(train_ratio * 10_000)
    validation_end = round((train_ratio + validation_ratio) * 10_000)
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


def validate_train_label_fraction(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("train_label_fraction must be a number")
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise ValueError("train_label_fraction must be greater than zero and at most one")
    return fraction


def train_label_patient_key(patient_id: str, seed: int) -> str:
    """Match the Spark SHA-256 rank used for deterministic low-label selection."""
    return hashlib.sha256(f"{seed}:train-label:{patient_id}".encode("utf-8")).hexdigest()


def train_label_patient_count(total_patients: int, fraction: float) -> int:
    """Select a nested prefix while retaining one patient from each non-empty stratum."""
    fraction = validate_train_label_fraction(fraction)
    if (
        isinstance(total_patients, bool)
        or not isinstance(total_patients, int)
        or total_patients < 0
    ):
        raise ValueError("total_patients must be a non-negative integer")
    if not total_patients:
        return 0
    return min(total_patients, max(1, math.ceil(total_patients * fraction)))


def train_label_selection_configuration() -> dict[str, str]:
    return {
        "unit": "patient",
        "method": TRAIN_LABEL_SELECTION_METHOD,
        "hash_input": TRAIN_LABEL_HASH_INPUT,
        "stratification": "patient train-label profile: negative_only, positive_only, or mixed",
        "rounding": "ceil within each non-empty stratum",
        "validation_and_test": "unchanged",
    }


def semantic_tag(description: str | None) -> str:
    match = re.search(r"\(([^()]*)\)\s*$", description or "")
    return match.group(1).strip().lower() if match else "unknown"


def validate_vocabulary(document: dict[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported vocabulary schema version")
    tokens = document.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("vocabulary tokens must be a non-empty list")
    ids = [entry.get("id") for entry in tokens]
    labels = [entry.get("token") for entry in tokens]
    if ids != list(range(len(tokens))):
        raise ValueError("vocabulary IDs must be contiguous and ordered")
    if labels[: len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS):
        raise ValueError("vocabulary special-token order is incompatible with ExMed-BERT")
    if len(labels) != len(set(labels)):
        raise ValueError("vocabulary contains duplicate token labels")
    for entry in tokens:
        entity = entry.get("entity")
        if entity not in ENTITY_IDS:
            raise ValueError(f"unsupported token entity: {entity}")
        if entry.get("entity_id") != ENTITY_IDS[entity]:
            raise ValueError(f"incorrect entity ID for {entity}")


def validate_supervised_manifest(document: dict[str, Any]) -> None:
    """Validate the immutable AIR binary-endpoint snapshot contract."""
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported supervised manifest schema version")
    configuration = document.get("configuration", {})
    if configuration.get("task") != "binary_plos":
        raise ValueError("supervised snapshot must describe the binary PLOS task")
    if configuration.get("history_rule") != (
        "event_time strictly before index_time; target encounter excluded"
    ):
        raise ValueError("supervised snapshot does not declare the leakage-safe history rule")
    pretraining = document.get("pretraining", {})
    for key in ("snapshot_root", "manifest_sha256", "vocabulary_sha256"):
        if not pretraining.get(key):
            raise ValueError(f"supervised snapshot is missing pretraining.{key}")
    quality = document.get("quality", {})
    if int(quality.get("temporal_leakage_rows", -1)) != 0:
        raise ValueError("supervised snapshot contains temporal leakage")
    statistics = quality.get("split_statistics", {})
    artifacts = document.get("artifacts", {})
    for split in ("train", "validation", "test"):
        values = statistics.get(split, {})
        examples = int(values.get("examples", 0))
        positive = int(values.get("positive", 0))
        negative = int(values.get("negative", 0))
        if examples <= 0 or positive <= 0 or negative <= 0:
            raise ValueError(f"{split} must contain both PLOS classes")
        if positive + negative != examples:
            raise ValueError(f"{split} class counts do not match its example count")
        files = artifacts.get(split)
        if not isinstance(files, list) or not files:
            raise ValueError(f"{split} has no Parquet artifacts")
        names = [item.get("path") for item in files]
        if len(names) != len(set(names)) or any(
            not isinstance(name, str) or not name.endswith(".parquet") for name in names
        ):
            raise ValueError(f"{split} artifact names are invalid")
        if any(int(item.get("bytes", 0)) <= 0 for item in files):
            raise ValueError(f"{split} contains an empty artifact")

    if "train_label_fraction" not in configuration:
        return

    fraction = validate_train_label_fraction(configuration["train_label_fraction"])
    subset_seed = configuration.get("train_label_subset_seed")
    if isinstance(subset_seed, bool) or not isinstance(subset_seed, int):
        raise ValueError("supervised snapshot has an invalid train-label subset seed")
    selection = configuration.get("train_label_selection")
    expected_selection = train_label_selection_configuration()
    if selection != expected_selection:
        raise ValueError("supervised snapshot has an unsupported train-label selection method")

    train_selection = quality.get("train_selection")
    if not isinstance(train_selection, dict):
        raise ValueError("supervised snapshot is missing quality.train_selection")
    if train_selection.get("requested_fraction") != fraction:
        raise ValueError("train-label quality fraction does not match configuration")
    if train_selection.get("subset_seed") != subset_seed:
        raise ValueError("train-label quality subset seed does not match configuration")

    summaries: dict[str, dict[str, Any]] = {}
    for name in ("full", "selected"):
        summary = train_selection.get(name)
        if not isinstance(summary, dict):
            raise ValueError(f"train-label quality is missing {name} summary")
        for key in ("examples", "patients", "positive", "negative"):
            value = summary.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"train-label {name}.{key} must be a non-negative integer")
        if summary["positive"] + summary["negative"] != summary["examples"]:
            raise ValueError(f"train-label {name} class counts do not match its example count")
        patient_strata = summary.get("patient_strata")
        if not isinstance(patient_strata, dict) or set(patient_strata) != {
            "mixed",
            "negative_only",
            "positive_only",
        }:
            raise ValueError(f"train-label {name} patient strata are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in patient_strata.values()
        ):
            raise ValueError(f"train-label {name} patient strata must be non-negative integers")
        if sum(patient_strata.values()) != summary["patients"]:
            raise ValueError(f"train-label {name} patient strata do not match its patient count")
        summaries[name] = summary

    selected = summaries["selected"]
    train_statistics = statistics["train"]
    for key in ("examples", "patients", "positive", "negative"):
        if selected[key] != int(train_statistics[key]):
            raise ValueError(f"selected train-label {key} does not match split statistics")

    full = summaries["full"]
    for key in ("examples", "patients", "positive", "negative"):
        if selected[key] > full[key]:
            raise ValueError(f"selected train-label {key} exceeds the full training set")
    for stratum, full_count in full["patient_strata"].items():
        expected_count = train_label_patient_count(full_count, fraction)
        if selected["patient_strata"][stratum] != expected_count:
            raise ValueError(f"selected train-label {stratum} count does not match the contract")

    excluded = train_selection.get("excluded")
    if not isinstance(excluded, dict):
        raise ValueError("train-label quality is missing excluded summary")
    for key in ("examples", "patients", "positive", "negative"):
        expected_count = full[key] - selected[key]
        if excluded.get(key) != expected_count:
            raise ValueError(f"excluded train-label {key} does not match full minus selected")
