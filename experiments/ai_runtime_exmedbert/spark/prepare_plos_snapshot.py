#!/usr/bin/env python3
"""Build an immutable admission-time PLOS snapshot using only prior events."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_FILE = Path(globals().get("__file__") or globals().get("filename") or ".").resolve()
PROJECT_ROOT = SOURCE_FILE.parents[1] if SOURCE_FILE.is_file() else SOURCE_FILE
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.contracts import (  # noqa: E402
    canonical_sha256,
    train_label_selection_configuration,
    validate_train_label_fraction,
    validate_supervised_manifest,
    validate_vocabulary,
)
from spark.prepare_snapshot import (  # noqa: E402
    prepare_domain,
    read_csv,
    require_columns,
    split_expression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--pretrain-snapshot-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--snapshot-version", required=True)
    parser.add_argument("--observation-end-date", default="2025-01-01")
    parser.add_argument("--plos-threshold-days", type=int, default=7)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--min-history-events", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--seed", type=int, default=201214)
    parser.add_argument("--subset-seed", type=int)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--train-label-fraction", type=float, default=1.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_statistics(frame: Any, split: str) -> dict[str, Any]:
    from pyspark.sql import functions as F

    row = frame.where(F.col("split") == split).agg(
        F.count("*").alias("examples"),
        F.countDistinct("patient_id").alias("patients"),
        F.sum(F.col("label").cast("long")).alias("positive"),
        F.sum(F.size("codes")).alias("tokens"),
        F.min(F.size("codes")).alias("min_length"),
        F.max(F.size("codes")).alias("max_length"),
        F.sum(F.expr("size(filter(codes, x -> x = 1))")).alias("unknown_tokens"),
        F.sum(
            F.when(
                (F.size("codes") != F.size("entity_ids"))
                | (F.size("codes") != F.size("age_ids"))
                | (F.size("codes") != F.size("position_ids")),
                1,
            ).otherwise(0)
        ).alias("unaligned_rows"),
    ).first()
    examples = int(row.examples or 0)
    positive = int(row.positive or 0)
    tokens = int(row.tokens or 0)
    unknown = int(row.unknown_tokens or 0)
    return {
        "examples": examples,
        "patients": int(row.patients or 0),
        "positive": positive,
        "negative": examples - positive,
        "prevalence": positive / examples if examples else 0.0,
        "tokens": tokens,
        "unknown_tokens": unknown,
        "unknown_fraction": unknown / tokens if tokens else 0.0,
        "min_length": int(row.min_length or 0),
        "max_length": int(row.max_length or 0),
        "unaligned_rows": int(row.unaligned_rows or 0),
    }


def train_patient_strata(frame: Any) -> dict[str, int]:
    from pyspark.sql import functions as F

    rows = (
        frame.where(F.col("split") == "train")
        .groupBy("patient_id")
        .agg(F.min("label").alias("min_label"), F.max("label").alias("max_label"))
        .withColumn(
            "label_profile",
            F.when(F.col("max_label") == 0, F.lit("negative_only"))
            .when(F.col("min_label") == 1, F.lit("positive_only"))
            .otherwise(F.lit("mixed")),
        )
        .groupBy("label_profile")
        .count()
        .collect()
    )
    counts = {row.label_profile: int(row["count"]) for row in rows}
    return {
        "mixed": counts.get("mixed", 0),
        "negative_only": counts.get("negative_only", 0),
        "positive_only": counts.get("positive_only", 0),
    }


def select_training_fraction(frame: Any, fraction: float, seed: int) -> Any:
    from pyspark.sql import Window, functions as F

    fraction = validate_train_label_fraction(fraction)
    if fraction == 1.0:
        return frame

    train = frame.where(F.col("split") == "train")
    patients = (
        train.groupBy("patient_id")
        .agg(F.min("label").alias("min_label"), F.max("label").alias("max_label"))
        .withColumn(
            "label_profile",
            F.when(F.col("max_label") == 0, F.lit("negative_only"))
            .when(F.col("min_label") == 1, F.lit("positive_only"))
            .otherwise(F.lit("mixed")),
        )
        .withColumn(
            "selection_key",
            F.sha2(F.concat(F.lit(f"{seed}:train-label:"), F.col("patient_id")), 256),
        )
    )
    stratum = Window.partitionBy("label_profile")
    rank = stratum.orderBy("selection_key", "patient_id")
    selected_patients = (
        patients.withColumn("selection_rank", F.row_number().over(rank))
        .withColumn("stratum_patients", F.count(F.lit(1)).over(stratum))
        .withColumn(
            "selected_patients",
            F.greatest(
                F.lit(1),
                F.ceil(F.col("stratum_patients") * F.lit(fraction)).cast("long"),
            ),
        )
        .where(F.col("selection_rank") <= F.col("selected_patients"))
        .select("patient_id")
    )
    selected_train = train.join(selected_patients, "patient_id", "left_semi")
    return frame.where(F.col("split") != "train").unionByName(selected_train)


def training_summary(statistics: dict[str, Any], patient_strata: dict[str, int]) -> dict[str, Any]:
    return {
        "examples": statistics["examples"],
        "patients": statistics["patients"],
        "positive": statistics["positive"],
        "negative": statistics["negative"],
        "patient_strata": patient_strata,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    from pyspark.sql import SparkSession, Window, functions as F

    args = parse_args()
    if args.plos_threshold_days < 1 or args.max_sequence_length < 2:
        raise ValueError("PLOS threshold must be positive and max sequence length must be at least two")
    if args.min_history_events < 1 or args.min_history_events > args.max_sequence_length:
        raise ValueError("min history events must be between one and max sequence length")
    if args.num_shards < 1:
        raise ValueError("num shards must be positive")
    train_label_fraction = validate_train_label_fraction(args.train_label_fraction)
    subset_seed = args.seed if args.subset_seed is None else args.subset_seed

    spark = SparkSession.builder.getOrCreate()
    source_root = args.source_root.rstrip("/")
    pretrain_root = Path(args.pretrain_snapshot_root.rstrip("/"))
    output_root = Path(args.output_root.rstrip("/"))
    final_root = output_root / args.snapshot_version
    if final_root.exists():
        raise FileExistsError(f"immutable PLOS snapshot already exists: {final_root}")

    pretrain_manifest_path = pretrain_root / "manifest.json"
    vocabulary_path = pretrain_root / "vocabulary.json"
    if not pretrain_manifest_path.is_file() or not vocabulary_path.is_file():
        raise FileNotFoundError("pretraining snapshot requires manifest.json and vocabulary.json")
    pretrain_manifest = json.loads(pretrain_manifest_path.read_text(encoding="utf-8"))
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    validate_vocabulary(vocabulary)
    if sha256_file(vocabulary_path) != pretrain_manifest["vocabulary"]["sha256"]:
        raise ValueError("pretraining vocabulary checksum does not match its manifest")
    if int(pretrain_manifest["configuration"]["max_sequence_length"]) != args.max_sequence_length:
        raise ValueError("PLOS max sequence length must match pretraining")

    token_rows = [
        (entry["entity"], entry["token"], int(entry["id"]), int(entry["entity_id"]))
        for entry in vocabulary["tokens"]
        if entry["entity"] != "default"
    ]
    token_vocabulary = spark.createDataFrame(
        token_rows, "entity string, code string, code_id int, entity_id int"
    )
    state_rows = [
        (state, int(index))
        for state, index in vocabulary["state_tokens"].items()
        if state not in {"PAD", "UNK"}
    ]
    state_vocabulary = spark.createDataFrame(state_rows, "state string, state_id int")

    patients_raw = read_csv(spark, f"{source_root}/patients.csv")
    require_columns(patients_raw, {"id", "birthdate", "gender", "state"}, "patients.csv")
    patient_rows = patients_raw.count()
    duplicate_patients = patient_rows - patients_raw.select("id").distinct().count()
    if duplicate_patients:
        raise ValueError(f"patients.csv contains {duplicate_patients} duplicate IDs")
    patients = patients_raw.select(
        F.trim("id").alias("patient_id"),
        F.to_date("birthdate").alias("birth_date"),
        F.upper(F.trim("gender")).alias("gender"),
        F.upper(F.trim("state")).alias("state"),
    )

    encounters_raw = read_csv(spark, f"{source_root}/encounters.csv")
    require_columns(
        encounters_raw,
        {"id", "patient", "start", "stop", "encounterclass"},
        "encounters.csv",
    )
    encounter_rows = encounters_raw.count()
    duplicate_encounters = encounter_rows - encounters_raw.select("id").distinct().count()
    if duplicate_encounters:
        raise ValueError(f"encounters.csv contains {duplicate_encounters} duplicate IDs")
    encounters = encounters_raw.select(
        F.trim("id").alias("encounter_id"),
        F.trim("patient").alias("patient_id"),
        F.to_timestamp("start").alias("encounter_time"),
        F.to_timestamp("stop").alias("stop_time"),
        F.lower(F.trim("encounterclass")).alias("encounter_class"),
    ).where(F.col("encounter_time").isNotNull())

    observation_end = F.to_timestamp(F.lit(f"{args.observation_end_date} 00:00:00"))
    duration_hours = (
        F.col("stop_time").cast("long") - F.col("encounter_time").cast("long")
    ) / F.lit(3600.0)
    indexes = (
        encounters.where(
            (F.col("encounter_class") == "inpatient")
            & F.col("stop_time").isNotNull()
            & (F.col("stop_time") <= observation_end)
            & (duration_hours >= 0)
        )
        .withColumn("duration_hours", duration_hours)
        .withColumn(
            "label", (F.col("duration_hours") > F.lit(args.plos_threshold_days * 24)).cast("int")
        )
        .withColumn("split", split_expression(args.seed, args.train_ratio, args.validation_ratio))
        .select(
            F.col("encounter_id").alias("example_id"),
            "patient_id",
            F.col("encounter_time").alias("index_time"),
            "duration_hours",
            "label",
            "split",
        )
    )
    eligible_examples = indexes.count()
    if not eligible_examples:
        raise ValueError("no eligible completed inpatient encounters")

    source_metrics: dict[str, dict[str, int]] = {}
    domain_frames = []
    for filename, entity, fallback in (
        ("conditions.csv", "diagnosis", "SNOMED"),
        ("medications.csv", "medication", "RXNORM"),
        ("procedures.csv", "procedure", "SNOMED"),
    ):
        frame, metrics = prepare_domain(
            spark,
            encounters.select("patient_id", "encounter_id", "encounter_time"),
            f"{source_root}/{filename}",
            entity=entity,
            fallback_namespace=fallback,
        )
        domain_frames.append(frame)
        source_metrics[entity] = metrics
    events = domain_frames[0]
    for frame in domain_frames[1:]:
        events = events.unionByName(frame)
    events_before_deduplication = events.count()
    events = events.dropDuplicates(["patient_id", "encounter_id", "entity", "code"])
    duplicate_events_removed = events_before_deduplication - events.count()

    encoded = (
        events.join(token_vocabulary, ["entity", "code"], "left")
        .withColumn("code_id", F.coalesce("code_id", F.lit(1)).cast("int"))
        .withColumn("entity_id", F.coalesce("entity_id", F.lit(0)).cast("int"))
        .join(patients, "patient_id", "inner")
        .join(state_vocabulary, "state", "left")
        .withColumn("state_id", F.coalesce("state_id", F.lit(1)).cast("int"))
        .withColumn(
            "sex_id",
            F.when(F.col("gender").isin("M", "MALE"), 2)
            .when(F.col("gender").isin("F", "FEMALE"), 3)
            .otherwise(1)
            .cast("int"),
        )
    )
    age_years = F.months_between(F.to_date("encounter_time"), F.col("birth_date")) / F.lit(12.0)
    encoded = encoded.withColumn(
        "age_id",
        F.when((age_years >= 0) & (age_years <= 110), F.round(age_years * 10).cast("int") + 2)
        .otherwise(1)
        .cast("int"),
    )
    visit_window = Window.partitionBy("patient_id").orderBy("encounter_time", "encounter_id")
    encoded = encoded.withColumn(
        "position_id",
        F.least(F.dense_rank().over(visit_window), F.lit(args.max_sequence_length - 1)).cast("int"),
    )

    history = indexes.alias("index").join(
        encoded.alias("event"),
        (F.col("index.patient_id") == F.col("event.patient_id"))
        & (F.col("event.event_time") < F.col("index.index_time"))
        & (F.col("event.encounter_id") != F.col("index.example_id")),
        "inner",
    )
    ordered_event = F.struct(
        F.col("event.event_time").alias("event_time"),
        F.col("event.encounter_id").alias("encounter_id"),
        F.col("event.entity_id").alias("entity_id"),
        F.col("event.code").alias("code"),
        F.col("event.code_id").alias("code_id"),
        F.col("event.age_id").alias("age_id"),
        F.col("event.position_id").alias("position_id"),
    )
    grouped = history.groupBy(
        F.col("index.example_id").alias("example_id"),
        F.col("index.patient_id").alias("patient_id"),
        F.col("index.index_time").alias("index_time"),
        F.col("index.duration_hours").alias("duration_hours"),
        F.col("index.label").alias("label"),
        F.col("index.split").alias("split"),
    ).agg(
        F.sort_array(F.collect_list(ordered_event)).alias("events"),
        F.first(F.col("event.sex_id"), ignorenulls=True).alias("sex_id"),
        F.first(F.col("event.state_id"), ignorenulls=True).alias("state_id"),
    )
    examples = (
        grouped.where(F.size("events") >= args.min_history_events)
        .withColumn(
            "events",
            F.slice(
                "events",
                F.greatest(F.size("events") - F.lit(args.max_sequence_length) + F.lit(1), F.lit(1)),
                args.max_sequence_length,
            ),
        )
        .select(
            "example_id",
            "patient_id",
            "index_time",
            "duration_hours",
            "label",
            "split",
            F.expr("transform(events, x -> x.code_id)").alias("codes"),
            F.expr("transform(events, x -> x.entity_id)").alias("entity_ids"),
            F.expr("transform(events, x -> x.age_id)").alias("age_ids"),
            F.expr("transform(events, x -> x.position_id)").alias("position_ids"),
            "sex_id",
            "state_id",
            F.expr("aggregate(events, cast(null as timestamp), (acc, x) -> greatest(acc, x.event_time))").alias(
                "last_event_time"
            ),
        )
    )
    full_retained_examples = examples.count()
    excluded_insufficient_history = eligible_examples - full_retained_examples
    leakage_rows = examples.where(F.col("last_event_time") >= F.col("index_time")).count()
    if leakage_rows:
        raise ValueError(f"PLOS snapshot contains {leakage_rows} temporal leakage rows")

    full_train_statistics = split_statistics(examples, "train")
    full_train_patient_strata = train_patient_strata(examples)
    examples = select_training_fraction(examples, train_label_fraction, subset_seed)
    statistics = {
        split: split_statistics(examples, split)
        for split in ("train", "validation", "test")
    }
    selected_train_patient_strata = train_patient_strata(examples)
    for split, values in statistics.items():
        if not values["examples"] or not values["positive"] or not values["negative"]:
            raise ValueError(f"{split} must contain positive and negative examples")
        if values["unaligned_rows"]:
            raise ValueError(f"{split} contains unaligned arrays")
        if values["max_length"] > args.max_sequence_length:
            raise ValueError(f"{split} exceeds max sequence length")

    final_root.mkdir(parents=True, exist_ok=False)
    artifact_columns = [
        "example_id",
        "patient_id",
        "index_time",
        "duration_hours",
        "label",
        "codes",
        "entity_ids",
        "age_ids",
        "position_ids",
        "sex_id",
        "state_id",
    ]
    for split in ("train", "validation", "test"):
        destination = final_root / split
        (
            examples.where(F.col("split") == split)
            .select(*artifact_columns)
            .repartition(args.num_shards, "example_id")
            .write.mode("overwrite")
            .parquet(str(destination))
        )
        for child in destination.iterdir():
            if child.is_file() and child.suffix != ".parquet":
                child.unlink()

    artifacts = {}
    for split in ("train", "validation", "test"):
        files = sorted((final_root / split).glob("*.parquet"))
        if not files:
            raise RuntimeError(f"no Parquet files written for {split}")
        artifacts[split] = [{"path": path.name, "bytes": path.stat().st_size} for path in files]

    manifest = {
        "schema_version": 1,
        "dataset": "synthea-exmedbert-plos",
        "version": args.snapshot_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "root": source_root,
            "patient_rows": patient_rows,
            "encounter_rows": encounter_rows,
            **source_metrics,
        },
        "pretraining": {
            "snapshot_root": str(pretrain_root),
            "manifest_sha256": sha256_file(pretrain_manifest_path),
            "vocabulary_path": str(vocabulary_path),
            "vocabulary_sha256": sha256_file(vocabulary_path),
            "vocabulary_size": len(vocabulary["tokens"]),
        },
        "configuration": {
            "task": "binary_plos",
            "prediction_time": "inpatient admission",
            "label": f"completed inpatient duration > {args.plos_threshold_days} days",
            "plos_threshold_days": args.plos_threshold_days,
            "observation_end_date": args.observation_end_date,
            "history_rule": "event_time strictly before index_time; target encounter excluded",
            "max_sequence_length": args.max_sequence_length,
            "min_history_events": args.min_history_events,
            "num_shards_per_split": args.num_shards,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1 - args.train_ratio - args.validation_ratio,
            "train_label_fraction": train_label_fraction,
            "train_label_subset_seed": subset_seed,
            "train_label_selection": train_label_selection_configuration(),
        },
        "quality": {
            "eligible_inpatient_encounters": eligible_examples,
            "full_retained_examples": full_retained_examples,
            "retained_examples": sum(values["examples"] for values in statistics.values()),
            "excluded_insufficient_history": excluded_insufficient_history,
            "excluded_train_label_fraction": full_train_statistics["examples"]
            - statistics["train"]["examples"],
            "duplicate_events_removed": duplicate_events_removed,
            "temporal_leakage_rows": leakage_rows,
            "split_statistics": statistics,
            "train_selection": {
                "requested_fraction": train_label_fraction,
                "subset_seed": subset_seed,
                "full": training_summary(full_train_statistics, full_train_patient_strata),
                "selected": training_summary(
                    statistics["train"], selected_train_patient_strata
                ),
                "excluded": {
                    key: full_train_statistics[key] - statistics["train"][key]
                    for key in ("examples", "patients", "positive", "negative")
                },
            },
        },
        "artifacts": artifacts,
    }
    manifest["contract_sha256"] = canonical_sha256(
        {"configuration": manifest["configuration"], "pretraining": manifest["pretraining"]}
    )
    validate_supervised_manifest(manifest)
    write_json(final_root / "manifest.json", manifest)
    print(json.dumps({"snapshot": str(final_root), "quality": manifest["quality"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
