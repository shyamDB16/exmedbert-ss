#!/usr/bin/env python3
"""Profile leakage-safe supervised endpoint candidates from Synthea encounters."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--audit-version", required=True)
    parser.add_argument("--observation-end-date", default="2025-01-01")
    parser.add_argument("--plos-threshold-days", type=int, default=7)
    parser.add_argument("--readmission-days", type=int, nargs="+", default=[30, 180])
    parser.add_argument("--seed", type=int, default=201214)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args()


def plos_label(duration_hours: float, threshold_days: int) -> int:
    if duration_hours < 0 or threshold_days < 1:
        raise ValueError("duration must be non-negative and threshold must be positive")
    return int(duration_hours > threshold_days * 24)


def readmission_label(gap_days: float | None, horizon_days: int) -> int:
    if horizon_days < 1:
        raise ValueError("horizon must be positive")
    return int(gap_days is not None and 0 < gap_days <= horizon_days)


def read_csv(spark: Any, path: str) -> Any:
    from pyspark.sql import functions as F

    frame = spark.read.option("header", True).option("mode", "FAILFAST").csv(path)
    return frame.select(*(F.col(f"`{name}`").alias(name.strip().lower()) for name in frame.columns))


def require_columns(frame: Any, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def split_expression(seed: int, train_ratio: float, validation_ratio: float) -> Any:
    from pyspark.sql import functions as F

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")
    if validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("validation_ratio must leave a positive test split")
    digest = F.sha2(F.concat(F.lit(f"{seed}:split:"), F.col("patient_id")), 256)
    bucket = F.pmod(F.conv(F.substring(digest, 1, 15), 16, 10).cast("long"), F.lit(10_000))
    train_end = round(train_ratio * 10_000)
    validation_end = round((train_ratio + validation_ratio) * 10_000)
    return (
        F.when(bucket < train_end, F.lit("train"))
        .when(bucket < validation_end, F.lit("validation"))
        .otherwise(F.lit("test"))
    )


def candidate_metrics(frame: Any) -> dict[str, Any]:
    from pyspark.sql import functions as F

    rows = frame.groupBy("split").agg(
        F.count("*").alias("eligible"),
        F.sum(F.col("label").cast("long")).alias("positive"),
        F.countDistinct("patient_id").alias("patients"),
    ).collect()
    by_split = {}
    total_eligible = 0
    total_positive = 0
    total_patients = 0
    for row in rows:
        eligible = int(row.eligible)
        positive = int(row.positive or 0)
        by_split[row.split] = {
            "eligible": eligible,
            "positive": positive,
            "negative": eligible - positive,
            "prevalence": positive / eligible if eligible else 0.0,
            "patients": int(row.patients),
        }
        total_eligible += eligible
        total_positive += positive
        total_patients += int(row.patients)
    return {
        "eligible": total_eligible,
        "positive": total_positive,
        "negative": total_eligible - total_positive,
        "prevalence": total_positive / total_eligible if total_eligible else 0.0,
        "patients_across_splits": total_patients,
        "splits": by_split,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    from pyspark.sql import SparkSession, functions as F

    args = parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.observation_end_date):
        raise ValueError("observation-end-date must use YYYY-MM-DD")
    if args.plos_threshold_days < 1 or any(value < 1 for value in args.readmission_days):
        raise ValueError("all endpoint horizons must be positive")
    if len(set(args.readmission_days)) != len(args.readmission_days):
        raise ValueError("readmission horizons must be unique")

    spark = SparkSession.builder.getOrCreate()
    source_root = args.source_root.rstrip("/")
    final_root = Path(args.output_root.rstrip("/")) / args.audit_version
    if final_root.exists():
        raise FileExistsError(f"immutable audit path already exists: {final_root}")

    encounters_path = f"{source_root}/encounters.csv"
    raw = read_csv(spark, encounters_path)
    require_columns(raw, {"id", "patient", "start", "stop", "encounterclass"}, encounters_path)
    source_rows = raw.count()
    duplicate_ids = source_rows - raw.select("id").distinct().count()
    if duplicate_ids:
        raise ValueError(f"encounters.csv contains {duplicate_ids} duplicate IDs")

    encounters = raw.select(
        F.trim("id").alias("encounter_id"),
        F.trim("patient").alias("patient_id"),
        F.to_timestamp("start").alias("start_time"),
        F.to_timestamp("stop").alias("stop_time"),
        F.lower(F.trim("encounterclass")).alias("encounter_class"),
    )
    missing_patient = encounters.where(F.col("patient_id").isNull() | (F.col("patient_id") == "")).count()
    missing_start = encounters.where(F.col("start_time").isNull()).count()
    missing_stop = encounters.where(F.col("stop_time").isNull()).count()
    encounters = encounters.where(
        F.col("patient_id").isNotNull()
        & (F.col("patient_id") != "")
        & F.col("start_time").isNotNull()
        & F.col("stop_time").isNotNull()
    ).withColumn(
        "duration_hours",
        (F.col("stop_time").cast("long") - F.col("start_time").cast("long")) / F.lit(3600.0),
    )
    invalid_duration = encounters.where(F.col("duration_hours") < 0).count()
    valid = encounters.where(F.col("duration_hours") >= 0).withColumn(
        "split", split_expression(args.seed, args.train_ratio, args.validation_ratio)
    )
    class_counts = {
        row.encounter_class or "unknown": int(row["count"])
        for row in valid.groupBy("encounter_class").count().collect()
    }
    observation_end = F.to_timestamp(F.lit(f"{args.observation_end_date} 00:00:00"))
    inpatient = valid.where(
        (F.col("encounter_class") == "inpatient") & (F.col("stop_time") <= observation_end)
    )
    inpatient_count = inpatient.count()
    if not inpatient_count:
        raise ValueError("source contains no valid inpatient encounters")

    plos = inpatient.select(
        "encounter_id",
        "patient_id",
        "split",
        (F.col("duration_hours") > F.lit(args.plos_threshold_days * 24)).cast("int").alias("label"),
    )
    duration_quantiles = inpatient.agg(
        F.expr("percentile_approx(duration_hours, array(0.5, 0.9, 0.95, 0.99), 10000)").alias("values")
    ).first().values
    candidates = {
        f"plos_gt_{args.plos_threshold_days}d": {
            "prediction_time": "inpatient admission",
            "label": f"completed inpatient duration > {args.plos_threshold_days} days",
            **candidate_metrics(plos),
            "duration_hours_quantiles": {
                name: float(value)
                for name, value in zip(("p50", "p90", "p95", "p99"), duration_quantiles)
            },
        }
    }

    future = inpatient.select(
        F.col("patient_id").alias("next_patient_id"),
        F.col("start_time").alias("next_start_time"),
    )
    for horizon in sorted(args.readmission_days):
        current = inpatient.where(
            F.col("stop_time") <= observation_end - F.expr(f"INTERVAL {horizon} DAYS")
        )
        next_times = (
            current.alias("current")
            .join(
                future.alias("future"),
                (F.col("current.patient_id") == F.col("future.next_patient_id"))
                & (F.col("future.next_start_time") > F.col("current.stop_time")),
                "left",
            )
            .groupBy(F.col("current.encounter_id").alias("encounter_id"))
            .agg(F.min(F.col("future.next_start_time")).alias("next_inpatient_start"))
        )
        readmission = current.join(next_times, "encounter_id", "left").select(
            "encounter_id",
            "patient_id",
            "split",
            (
                F.col("next_inpatient_start").isNotNull()
                & (
                    F.col("next_inpatient_start").cast("long") - F.col("stop_time").cast("long")
                    <= F.lit(horizon * 86_400)
                )
            ).cast("int").alias("label"),
        )
        candidates[f"inpatient_readmission_{horizon}d"] = {
            "prediction_time": "inpatient discharge",
            "label": f"next inpatient admission within {horizon} days after discharge",
            "censoring": f"discharge on or before {args.observation_end_date} minus {horizon} days",
            **candidate_metrics(readmission),
        }

    provenance_path = Path(source_root) / "provenance.json"
    audit = {
        "schema_version": 1,
        "dataset": "synthea-endpoint-audit",
        "version": args.audit_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "root": source_root,
            "encounter_rows": source_rows,
            "provenance_sha256": sha256_file(provenance_path) if provenance_path.exists() else None,
        },
        "configuration": {
            "observation_end_date": args.observation_end_date,
            "plos_threshold_days": args.plos_threshold_days,
            "readmission_days": sorted(args.readmission_days),
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1 - args.train_ratio - args.validation_ratio,
            "split": "sha256(seed:split:patient_id) first 15 hex modulo 10000",
        },
        "quality": {
            "duplicate_encounter_ids": duplicate_ids,
            "missing_patient": missing_patient,
            "missing_start": missing_start,
            "missing_stop": missing_stop,
            "invalid_duration": invalid_duration,
            "valid_inpatient_encounters": inpatient_count,
            "encounter_class_counts": class_counts,
        },
        "candidates": candidates,
    }
    final_root.mkdir(parents=True, exist_ok=False)
    write_json(final_root / "audit.json", audit)
    print(json.dumps({"audit": str(final_root / "audit.json"), **audit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
