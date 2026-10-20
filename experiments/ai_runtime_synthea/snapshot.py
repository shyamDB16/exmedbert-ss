#!/usr/bin/env python3
"""Build a small, reproducible ExMed-BERT snapshot from Synthea CSV exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SPECIAL_TOKENS = {"PAD": 0, "MASK": 1, "UNK": 2}
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def split_for_patient(patient_id: str, seed: int, train_ratio: float) -> str:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    threshold = int(train_ratio * 10_000)
    return "train" if stable_hash(f"split:{patient_id}", seed) % 10_000 < threshold else "validation"


def gap_bucket(days: float | None) -> int:
    """Map a past-to-current encounter gap to one of seven stable buckets."""
    if days is None:
        return 0
    if days < 1:
        return 1
    if days <= 7:
        return 2
    if days <= 30:
        return 3
    if days <= 180:
        return 4
    if days <= 365:
        return 5
    return 6


def _parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("empty timestamp")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalized_row(row: dict[str, str | None]) -> dict[str, str]:
    return {(key or "").strip().upper(): (value or "").strip() for key, value in row.items()}


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        for row in reader:
            yield _normalized_row(row)


def _code_namespace(system: str) -> str:
    lowered = system.lower()
    if "snomed" in lowered:
        return "SNOMED"
    if "loinc" in lowered:
        return "LOINC"
    return system.rstrip("/").rsplit("/", 1)[-1].upper() or "UNKNOWN"


def load_patient_events(
    conditions_csv: Path,
    encounters_csv: Path,
) -> tuple[dict[str, list[tuple[datetime, str, list[str]]]], dict[str, int]]:
    encounters: dict[str, tuple[str, datetime]] = {}
    duplicate_encounters = 0
    for row in _read_csv(encounters_csv):
        encounter_id = row.get("ID", "")
        patient_id = row.get("PATIENT", "")
        if not encounter_id or not patient_id:
            raise ValueError("encounters.csv requires ID and PATIENT")
        if encounter_id in encounters:
            duplicate_encounters += 1
            continue
        encounters[encounter_id] = (patient_id, _parse_timestamp(row.get("START", "")))

    codes_by_encounter: dict[str, set[str]] = defaultdict(set)
    excluded_missing_encounter = 0
    excluded_patient_mismatch = 0
    empty_codes = 0
    input_conditions = 0
    for row in _read_csv(conditions_csv):
        input_conditions += 1
        encounter_id = row.get("ENCOUNTER", "")
        patient_id = row.get("PATIENT", "")
        code = row.get("CODE", "")
        if not code:
            empty_codes += 1
            continue
        encounter = encounters.get(encounter_id)
        if encounter is None:
            excluded_missing_encounter += 1
            continue
        if patient_id != encounter[0]:
            excluded_patient_mismatch += 1
            continue
        namespace = _code_namespace(row.get("SYSTEM", ""))
        codes_by_encounter[encounter_id].add(f"{namespace}:{code}")

    patients: dict[str, list[tuple[datetime, str, list[str]]]] = defaultdict(list)
    for encounter_id, codes in codes_by_encounter.items():
        patient_id, start = encounters[encounter_id]
        patients[patient_id].append((start, encounter_id, sorted(codes)))
    for visits in patients.values():
        visits.sort(key=lambda item: (item[0], item[1]))

    counts = {
        "input_encounters": len(encounters),
        "duplicate_encounters": duplicate_encounters,
        "input_conditions": input_conditions,
        "excluded_missing_encounter": excluded_missing_encounter,
        "excluded_patient_mismatch": excluded_patient_mismatch,
        "excluded_empty_code": empty_codes,
        "patients_with_conditions": len(patients),
    }
    return dict(patients), counts


def patient_sequence(
    visits: list[tuple[datetime, str, list[str]]],
    max_sequence_length: int,
) -> tuple[list[str], list[int]]:
    flattened: list[tuple[datetime, str, str]] = []
    for start, encounter_id, codes in visits:
        flattened.extend((start, encounter_id, code) for code in codes)
    retained = flattened[-max_sequence_length:]
    if not retained:
        return [], []

    codes: list[str] = []
    gaps: list[int] = []
    previous_encounter: str | None = None
    previous_start: datetime | None = None
    for start, encounter_id, code in retained:
        if encounter_id != previous_encounter:
            days = None if previous_start is None else max(0.0, (start - previous_start).total_seconds() / 86_400)
            current_bucket = gap_bucket(days)
            previous_encounter = encounter_id
            previous_start = start
        codes.append(code)
        gaps.append(current_bucket)
    return codes, gaps


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("patient_id", pa.string(), nullable=False),
            pa.field("codes", pa.list_(pa.int32()), nullable=False),
            pa.field("time_gaps", pa.list_(pa.int8()), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def build_snapshot(
    conditions_csv: Path,
    encounters_csv: Path,
    output_dir: Path,
    *,
    version: str,
    max_patients: int = 256,
    max_sequence_length: int = 128,
    train_ratio: float = 0.8,
    seed: int = 201214,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"snapshot already exists: {output_dir}")
    if max_patients < 2:
        raise ValueError("max_patients must be at least 2")
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")

    patient_events, exclusions = load_patient_events(conditions_csv, encounters_csv)
    selected_ids = sorted(patient_events, key=lambda patient_id: (stable_hash(f"cohort:{patient_id}", seed), patient_id))[
        :max_patients
    ]
    sequences: dict[str, dict[str, Any]] = {}
    for patient_id in selected_ids:
        codes, time_gaps = patient_sequence(patient_events[patient_id], max_sequence_length)
        if codes:
            sequences[patient_id] = {"codes_text": codes, "time_gaps": time_gaps}

    splits = {"train": [], "validation": []}
    for patient_id in sorted(sequences):
        split = split_for_patient(patient_id, seed, train_ratio)
        splits[split].append(patient_id)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("stable split produced an empty train or validation set")
    if set(splits["train"]) & set(splits["validation"]):
        raise AssertionError("patient leakage between train and validation")

    train_codes = sorted({code for patient_id in splits["train"] for code in sequences[patient_id]["codes_text"]})
    vocabulary = dict(SPECIAL_TOKENS)
    vocabulary.update({code: index for index, code in enumerate(train_codes, start=len(SPECIAL_TOKENS))})

    encoded_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    unknown_validation_tokens = 0
    for split, patient_ids in splits.items():
        for patient_id in patient_ids:
            text_codes = sequences[patient_id]["codes_text"]
            encoded = [vocabulary.get(code, SPECIAL_TOKENS["UNK"]) for code in text_codes]
            if split == "validation":
                unknown_validation_tokens += sum(code_id == SPECIAL_TOKENS["UNK"] for code_id in encoded)
            encoded_rows[split].append(
                {
                    "patient_id": patient_id,
                    "codes": encoded,
                    "time_gaps": sequences[patient_id]["time_gaps"],
                }
            )

    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        artifact_paths: dict[str, Path] = {}
        for split in ("train", "validation"):
            split_dir = staging / split
            split_dir.mkdir()
            artifact_paths[f"{split}_parquet"] = split_dir / "patients.parquet"
            _write_parquet(encoded_rows[split], artifact_paths[f"{split}_parquet"])

        vocabulary_path = staging / "vocabulary.json"
        vocabulary_path.write_text(json.dumps(vocabulary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_paths["vocabulary"] = vocabulary_path

        files = {
            name: {
                "path": str(path.relative_to(staging)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset": "synthea-diagnoses",
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "conditions_csv": conditions_csv.name,
                "conditions_sha256": sha256_file(conditions_csv),
                "encounters_csv": encounters_csv.name,
                "encounters_sha256": sha256_file(encounters_csv),
            },
            "configuration": {
                "seed": seed,
                "max_patients": max_patients,
                "max_sequence_length": max_sequence_length,
                "train_ratio": train_ratio,
                "split": "sha256(seed, patient_id) modulo 10000",
                "code_system_policy": "preserve source namespace",
                "dynamic_masking": "training time",
                "time_gap_buckets": ["start", "<1d", "1-7d", ">7-30d", ">30-180d", ">180-365d", ">365d"],
            },
            "counts": {
                **exclusions,
                "selected_patients": len(sequences),
                "train_patients": len(encoded_rows["train"]),
                "validation_patients": len(encoded_rows["validation"]),
                "train_tokens": sum(len(row["codes"]) for row in encoded_rows["train"]),
                "validation_tokens": sum(len(row["codes"]) for row in encoded_rows["validation"]),
                "validation_unknown_tokens": unknown_validation_tokens,
                "vocabulary_size": len(vocabulary),
            },
            "files": files,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--encounters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--max-patients", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=201214)
    args = parser.parse_args()
    manifest = build_snapshot(
        args.conditions,
        args.encounters,
        args.output,
        version=args.version,
        max_patients=args.max_patients,
        max_sequence_length=args.max_sequence_length,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
