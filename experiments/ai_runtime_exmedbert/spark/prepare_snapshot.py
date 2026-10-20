#!/usr/bin/env python3
"""Prepare an immutable ExMed-BERT snapshot from Synthea CSV files with Spark."""

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
    ENTITY_IDS,
    ENTITY_ORDER,
    SCHEMA_VERSION,
    SPECIAL_TOKENS,
    canonical_sha256,
    validate_vocabulary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="UC volume directory containing Synthea CSV files")
    parser.add_argument("--output-root", required=True, help="UC volume parent directory for versioned snapshots")
    parser.add_argument("--snapshot-version", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=201214)
    parser.add_argument("--num-shards", type=int, default=4, help="Parquet shards per split")
    parser.add_argument("--max-oov-fraction", type=float, default=0.05)
    parser.add_argument("--max-train-token-fraction", type=float, default=0.25)
    return parser.parse_args()


def require_columns(frame: Any, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def read_csv(spark: Any, path: str) -> Any:
    from pyspark.sql import functions as F

    frame = spark.read.option("header", True).option("mode", "FAILFAST").csv(path)
    return frame.select(*(F.col(f"`{name}`").alias(name.strip().lower()) for name in frame.columns))


def path_exists(spark: Any, path: str) -> bool:
    del spark
    return Path(path).exists()


def code_namespace(system_column: Any, fallback: str) -> Any:
    from pyspark.sql import functions as F

    system = F.lower(F.trim(system_column))
    tail = F.upper(F.regexp_extract(F.trim(system_column), r"([^/]+)$", 1))
    return (
        F.when(system.contains("snomed"), F.lit("SNOMED"))
        .when(system.contains("rxnorm"), F.lit("RXNORM"))
        .when(system.contains("loinc"), F.lit("LOINC"))
        .when(system.contains("icd-10") | system.contains("icd10"), F.lit("ICD10"))
        .when(system.contains("cpt"), F.lit("CPT"))
        .when((tail != "") & tail.isNotNull(), tail)
        .otherwise(F.lit(fallback))
    )


def prepare_domain(
    spark: Any,
    encounters: Any,
    path: str,
    *,
    entity: str,
    fallback_namespace: str,
) -> tuple[Any, dict[str, int]]:
    from pyspark.sql import functions as F

    source = read_csv(spark, path)
    require_columns(source, {"patient", "encounter", "code", "start"}, path)
    raw_count = source.count()
    valid = source.where(
        (F.trim("patient") != "")
        & (F.trim("encounter") != "")
        & (F.trim("code") != "")
        & F.col("start").isNotNull()
    )
    invalid_count = raw_count - valid.count()
    keys = encounters.select("patient_id", "encounter_id")
    normalized = valid.select(
        F.trim("patient").alias("patient_id"),
        F.trim("encounter").alias("encounter_id"),
        F.to_timestamp("start").alias("source_time"),
        F.trim("code").alias("source_code"),
        F.trim("description").alias("description") if "description" in valid.columns else F.lit("").alias("description"),
        F.col("system").alias("source_system") if "system" in valid.columns else F.lit(fallback_namespace).alias("source_system"),
    )
    orphan_count = normalized.join(keys, ["patient_id", "encounter_id"], "left_anti").count()
    joined = normalized.join(encounters, ["patient_id", "encounter_id"], "inner")
    namespace = code_namespace(F.col("source_system"), fallback_namespace)
    prepared = joined.select(
        "patient_id",
        "encounter_id",
        "encounter_time",
        F.coalesce("source_time", "encounter_time").alias("event_time"),
        F.lit(entity).alias("entity"),
        F.lit(ENTITY_IDS[entity]).cast("int").alias("source_entity_id"),
        F.concat(
            F.lit(f"{entity.upper()}:"), namespace, F.lit(":"), F.col("source_code")
        ).alias("code"),
        "description",
        F.lower(F.regexp_extract("description", r"\(([^()]*)\)\s*$", 1)).alias("semantic_tag"),
    )
    return prepared, {"input_rows": raw_count, "invalid_rows": invalid_count, "orphan_rows": orphan_count}


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


def build_vocabulary(events: Any) -> tuple[Any, dict[str, Any]]:
    from pyspark.sql import Window, functions as F

    entity_rank = F.create_map(
        *[item for entity in ENTITY_ORDER for item in (F.lit(entity), F.lit(ENTITY_IDS[entity]))]
    )
    train_codes = (
        events.where(F.col("split") == "train")
        .select("entity", "code")
        .distinct()
        .withColumn("entity_id", entity_rank[F.col("entity")].cast("int"))
    )
    order = Window.orderBy("entity_id", "code")
    vocabulary = train_codes.withColumn(
        "code_id", (F.row_number().over(order) + len(SPECIAL_TOKENS) - 1).cast("int")
    )
    entries = [
        {"id": index, "token": token, "entity": "default", "entity_id": 0}
        for index, token in enumerate(SPECIAL_TOKENS)
    ]
    entries.extend(
        {
            "id": int(row.code_id),
            "token": row.code,
            "entity": row.entity,
            "entity_id": int(row.entity_id),
        }
        for row in vocabulary.orderBy("code_id").collect()
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "special_tokens": {token: index for index, token in enumerate(SPECIAL_TOKENS)},
        "entity_ids": ENTITY_IDS,
        "tokens": entries,
    }
    validate_vocabulary(document)
    return vocabulary, document


def split_statistics(frame: Any, split: str) -> dict[str, Any]:
    from pyspark.sql import functions as F

    selected = frame.where(F.col("split") == split)
    row = selected.agg(
        F.count("*").alias("patients"),
        F.coalesce(F.sum(F.size("codes")), F.lit(0)).alias("tokens"),
        F.coalesce(F.min(F.size("codes")), F.lit(0)).alias("min_length"),
        F.coalesce(F.max(F.size("codes")), F.lit(0)).alias("max_length"),
        F.coalesce(F.sum(F.expr("size(filter(codes, x -> x = 1))")), F.lit(0)).alias("unknown_tokens"),
        F.sum(
            F.when(
                (F.size("codes") != F.size("entity_ids"))
                | (F.size("codes") != F.size("age_ids"))
                | (F.size("codes") != F.size("position_ids")),
                1,
            ).otherwise(0)
        ).alias("unaligned_rows"),
    ).first()
    tokens = int(row.tokens)
    unknown = int(row.unknown_tokens)
    return {
        "patients": int(row.patients),
        "tokens": tokens,
        "min_length": int(row.min_length),
        "max_length": int(row.max_length),
        "unknown_tokens": unknown,
        "unknown_fraction": unknown / tokens if tokens else 0.0,
        "unaligned_rows": int(row.unaligned_rows or 0),
    }


def write_json(path: str, value: dict[str, Any]) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    from pyspark.sql import SparkSession, Window, functions as F

    args = parse_args()
    if args.max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least two")
    if args.num_shards < 1:
        raise ValueError("num_shards must be positive")

    spark = SparkSession.builder.getOrCreate()

    source_root = args.source_root.rstrip("/")
    final_root = f"{args.output_root.rstrip('/')}/{args.snapshot_version}"
    if path_exists(spark, final_root):
        raise FileExistsError(f"immutable snapshot path already exists: {final_root}")

    patients = read_csv(spark, f"{source_root}/patients.csv")
    require_columns(patients, {"id", "birthdate", "gender", "state"}, "patients.csv")
    patient_count = patients.count()
    duplicate_patients = patient_count - patients.select("id").distinct().count()
    if duplicate_patients:
        raise ValueError(f"patients.csv contains {duplicate_patients} duplicate IDs")
    patients = patients.select(
        F.trim("id").alias("patient_id"),
        F.to_date("birthdate").alias("birth_date"),
        F.upper(F.trim("gender")).alias("gender"),
        F.upper(F.trim("state")).alias("state"),
    )

    encounters_raw = read_csv(spark, f"{source_root}/encounters.csv")
    require_columns(encounters_raw, {"id", "patient", "start"}, "encounters.csv")
    encounter_count = encounters_raw.count()
    duplicate_encounters = encounter_count - encounters_raw.select("id").distinct().count()
    if duplicate_encounters:
        raise ValueError(f"encounters.csv contains {duplicate_encounters} duplicate IDs")
    encounters = encounters_raw.select(
        F.trim("id").alias("encounter_id"),
        F.trim("patient").alias("patient_id"),
        F.to_timestamp("start").alias("encounter_time"),
    ).where(F.col("encounter_time").isNotNull())

    sources: dict[str, dict[str, int]] = {}
    domain_frames = []
    domain_specs = [
        ("conditions.csv", "diagnosis", "SNOMED"),
        ("medications.csv", "medication", "RXNORM"),
        ("procedures.csv", "procedure", "SNOMED"),
    ]
    for filename, entity, fallback in domain_specs:
        path = f"{source_root}/{filename}"
        if not path_exists(spark, path):
            if entity == "diagnosis":
                raise FileNotFoundError(path)
            sources[entity] = {"input_rows": 0, "invalid_rows": 0, "orphan_rows": 0}
            continue
        frame, metrics = prepare_domain(
            spark, encounters, path, entity=entity, fallback_namespace=fallback
        )
        domain_frames.append(frame)
        sources[entity] = metrics
    if not domain_frames:
        raise ValueError("no clinical event sources were loaded")

    events = domain_frames[0]
    for frame in domain_frames[1:]:
        events = events.unionByName(frame)
    event_rows_before_deduplication = events.count()
    events = events.dropDuplicates(["patient_id", "encounter_id", "entity", "code"])
    duplicate_events_removed = event_rows_before_deduplication - events.count()
    events = events.join(patients, "patient_id", "inner")

    patient_splits = events.select("patient_id").distinct().withColumn(
        "split", split_expression(args.seed, args.train_ratio, args.validation_ratio)
    )
    events = events.join(patient_splits, "patient_id", "inner")
    split_counts = {row.split: int(row["count"]) for row in patient_splits.groupBy("split").count().collect()}
    Path(final_root).mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation", "test"):
        if split_counts.get(split, 0) == 0:
            raise ValueError(f"stable split produced no {split} patients")

    vocabulary, vocabulary_document = build_vocabulary(events)
    state_labels = [
        row.state
        for row in events.where((F.col("split") == "train") & F.col("state").isNotNull() & (F.col("state") != ""))
        .select("state")
        .distinct()
        .orderBy("state")
        .collect()
    ]
    state_rows = [(state, index + 2) for index, state in enumerate(state_labels)]
    state_vocabulary = spark.createDataFrame(state_rows, "state string, state_id int")
    vocabulary_document["state_tokens"] = {"PAD": 0, "UNK": 1, **{state: index for state, index in state_rows}}
    vocabulary_document["sex_tokens"] = {"PAD": 0, "UNK": 1, "MALE": 2, "FEMALE": 3}
    vocabulary_document["age"] = {"pad_id": 0, "unknown_id": 1, "minimum": 0.0, "maximum": 110.0, "step": 0.1, "size": 1103}
    vocabulary_document["sha256"] = canonical_sha256(vocabulary_document)

    encoded = (
        events.join(vocabulary.select("entity", "code", "code_id", "entity_id"), ["entity", "code"], "left")
        .withColumn("code_id", F.coalesce("code_id", F.lit(1)).cast("int"))
        .withColumn("entity_id", F.coalesce("entity_id", F.lit(0)).cast("int"))
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

    ordered_event = F.struct(
        F.col("encounter_time").alias("encounter_time"),
        F.col("encounter_id").alias("encounter_id"),
        F.col("entity_id").alias("entity_id"),
        F.col("code").alias("code"),
        F.col("code_id").alias("code_id"),
        F.col("age_id").alias("age_id"),
        F.col("position_id").alias("position_id"),
    )
    sequences = encoded.groupBy("patient_id", "split").agg(
        F.sort_array(F.collect_list(ordered_event)).alias("events"),
        F.first("sex_id", ignorenulls=True).alias("sex_id"),
        F.first("state_id", ignorenulls=True).alias("state_id"),
    )
    sequences = sequences.withColumn(
        "events",
        F.slice(
            "events",
            F.greatest(F.size("events") - F.lit(args.max_sequence_length) + F.lit(1), F.lit(1)),
            args.max_sequence_length,
        ),
    ).select(
        "patient_id",
        "split",
        F.expr("transform(events, x -> x.code_id)").alias("codes"),
        F.expr("transform(events, x -> x.entity_id)").alias("entity_ids"),
        F.expr("transform(events, x -> x.age_id)").alias("age_ids"),
        F.expr("transform(events, x -> x.position_id)").alias("position_ids"),
        "sex_id",
        "state_id",
    )

    statistics = {split: split_statistics(sequences, split) for split in ("train", "validation", "test")}
    for split, values in statistics.items():
        if values["unaligned_rows"]:
            raise ValueError(f"{split} contains unaligned arrays")
        if values["max_length"] > args.max_sequence_length:
            raise ValueError(f"{split} exceeds max_sequence_length")
        if split != "train" and values["unknown_fraction"] > args.max_oov_fraction:
            raise ValueError(
                f"{split} unknown-token fraction {values['unknown_fraction']:.4f} exceeds {args.max_oov_fraction:.4f}"
            )

    train_token_counts = encoded.where(F.col("split") == "train").groupBy("code").count()
    top_token = train_token_counts.orderBy(F.desc("count"), "code").first()
    train_event_count = encoded.where(F.col("split") == "train").count()
    top_token_fraction = int(top_token["count"]) / train_event_count
    if top_token_fraction > args.max_train_token_fraction:
        raise ValueError(
            f"top training token fraction {top_token_fraction:.4f} exceeds {args.max_train_token_fraction:.4f}"
        )
    rare_vocabulary_tokens = train_token_counts.where(F.col("count") == 1).count()
    semantic_tags = {
        row.semantic_tag or "unknown": int(row["count"])
        for row in events.where(F.col("entity") == "diagnosis").groupBy("semantic_tag").count().collect()
    }

    for split in ("train", "validation", "test"):
        destination = f"{final_root}/{split}"
        (
            sequences.where(F.col("split") == split)
            .drop("split")
            .repartition(args.num_shards, "patient_id")
            .write.mode("overwrite")
            .parquet(destination)
        )
        for child in Path(destination).iterdir():
            if child.is_file() and child.suffix != ".parquet":
                child.unlink()

    write_json(f"{final_root}/vocabulary.json", vocabulary_document)
    artifacts = {}
    for split in ("train", "validation", "test"):
        files = sorted(Path(f"{final_root}/{split}").glob("*.parquet"))
        artifacts[split] = [{"path": file.name, "bytes": file.stat().st_size} for file in files]
        if not files:
            raise RuntimeError(f"no Parquet files were written for {split}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "synthea-exmedbert",
        "version": args.snapshot_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"root": source_root, "patient_rows": patient_count, "encounter_rows": encounter_count, **sources},
        "configuration": {
            "seed": args.seed,
            "split": "sha256(seed:split:patient_id) first 15 hex modulo 10000",
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": round(1 - args.train_ratio - args.validation_ratio, 10),
            "max_sequence_length": args.max_sequence_length,
            "num_shards_per_split": args.num_shards,
            "max_oov_fraction": args.max_oov_fraction,
            "max_train_token_fraction": args.max_train_token_fraction,
            "dynamic_masking": "AIR training time",
            "code_policy": "domain and source-system namespaced; no lossy crosswalk",
        },
        "quality": {
            "duplicate_events_removed": duplicate_events_removed,
            "split_patient_counts": split_counts,
            "split_statistics": statistics,
            "top_train_token": top_token.code,
            "top_train_token_count": int(top_token["count"]),
            "top_train_token_fraction": top_token_fraction,
            "rare_train_vocabulary_tokens": int(rare_vocabulary_tokens),
            "diagnosis_semantic_tags": semantic_tags,
        },
        "vocabulary": {
            "path": "vocabulary.json",
            "sha256": hashlib.sha256(
                (json.dumps(vocabulary_document, indent=2, sort_keys=True) + "\n").encode("utf-8")
            ).hexdigest(),
            "code_size": len(vocabulary_document["tokens"]),
            "age_size": vocabulary_document["age"]["size"],
            "sex_size": len(vocabulary_document["sex_tokens"]),
            "state_size": len(vocabulary_document["state_tokens"]),
            "entity_size": len(ENTITY_IDS),
        },
        "artifacts": artifacts,
    }
    write_json(f"{final_root}/manifest.json", manifest)
    print(json.dumps({"snapshot": final_root, "quality": manifest["quality"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
