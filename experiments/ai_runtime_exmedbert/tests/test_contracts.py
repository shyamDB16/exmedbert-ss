from experiments.ai_runtime_exmedbert.shared.contracts import (
    ENTITY_IDS,
    SPECIAL_TOKENS,
    semantic_tag,
    split_bucket,
    split_name,
    train_label_patient_count,
    train_label_patient_key,
    train_label_selection_configuration,
    validate_supervised_manifest,
    validate_train_label_fraction,
    validate_vocabulary,
)


def test_patient_split_is_deterministic_and_exclusive():
    bucket = split_bucket("patient-1", 201214)
    assert bucket == split_bucket("patient-1", 201214)
    assert split_name(bucket, 0.8, 0.1) in {"train", "validation", "test"}


def test_train_label_patient_prefixes_are_deterministic_and_nested():
    patient_ids = [f"patient-{index}" for index in range(20)]
    ranked = sorted(patient_ids, key=lambda patient_id: train_label_patient_key(patient_id, 17))
    small = set(ranked[: train_label_patient_count(len(ranked), 0.1)])
    medium = set(ranked[: train_label_patient_count(len(ranked), 0.25)])
    assert small < medium
    assert ranked == sorted(
        patient_ids, key=lambda patient_id: train_label_patient_key(patient_id, 17)
    )
    assert ranked != sorted(
        patient_ids, key=lambda patient_id: train_label_patient_key(patient_id, 18)
    )


def test_train_label_fraction_validation_and_rounding():
    assert validate_train_label_fraction(1) == 1.0
    assert train_label_patient_count(0, 0.1) == 0
    assert train_label_patient_count(3, 0.1) == 1
    assert train_label_patient_count(3, 0.5) == 2
    for value in (True, 0, -0.1, 1.1, "0.5"):
        try:
            validate_train_label_fraction(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid train-label fraction accepted: {value!r}")


def test_semantic_tag_is_explicit_when_missing():
    assert semantic_tag("Gingivitis (disorder)") == "disorder"
    assert semantic_tag("code without a tag") == "unknown"


def test_exmedbert_vocabulary_contract():
    tokens = [
        {"id": index, "token": token, "entity": "default", "entity_id": 0}
        for index, token in enumerate(SPECIAL_TOKENS)
    ]
    tokens.append(
        {
            "id": len(tokens),
            "token": "DIAGNOSIS:SNOMED:123",
            "entity": "diagnosis",
            "entity_id": ENTITY_IDS["diagnosis"],
        }
    )
    validate_vocabulary({"schema_version": 1, "tokens": tokens})


def test_supervised_manifest_requires_leakage_safe_balanced_splits():
    manifest = {
        "schema_version": 1,
        "configuration": {
            "task": "binary_plos",
            "history_rule": "event_time strictly before index_time; target encounter excluded",
        },
        "pretraining": {
            "snapshot_root": "/Volumes/catalog/schema/volume/pretrain",
            "manifest_sha256": "a" * 64,
            "vocabulary_sha256": "b" * 64,
        },
        "quality": {
            "temporal_leakage_rows": 0,
            "split_statistics": {
                split: {"examples": 3, "positive": 1, "negative": 2}
                for split in ("train", "validation", "test")
            },
        },
        "artifacts": {
            split: [{"path": f"{split}.parquet", "bytes": 10}]
            for split in ("train", "validation", "test")
        },
    }
    validate_supervised_manifest(manifest)
    manifest["quality"]["temporal_leakage_rows"] = 1
    try:
        validate_supervised_manifest(manifest)
    except ValueError as error:
        assert "leakage" in str(error)
    else:
        raise AssertionError("temporal leakage must be rejected")


def test_supervised_manifest_validates_low_label_selection():
    manifest = {
        "schema_version": 1,
        "configuration": {
            "task": "binary_plos",
            "history_rule": "event_time strictly before index_time; target encounter excluded",
            "train_label_fraction": 0.5,
            "train_label_subset_seed": 17,
            "train_label_selection": train_label_selection_configuration(),
        },
        "pretraining": {
            "snapshot_root": "/Volumes/catalog/schema/volume/pretrain",
            "manifest_sha256": "a" * 64,
            "vocabulary_sha256": "b" * 64,
        },
        "quality": {
            "temporal_leakage_rows": 0,
            "split_statistics": {
                "train": {"examples": 5, "patients": 3, "positive": 2, "negative": 3},
                "validation": {"examples": 3, "patients": 3, "positive": 1, "negative": 2},
                "test": {"examples": 3, "patients": 3, "positive": 1, "negative": 2},
            },
            "train_selection": {
                "requested_fraction": 0.5,
                "subset_seed": 17,
                "full": {
                    "examples": 9,
                    "patients": 5,
                    "positive": 4,
                    "negative": 5,
                    "patient_strata": {"mixed": 1, "negative_only": 2, "positive_only": 2},
                },
                "selected": {
                    "examples": 5,
                    "patients": 3,
                    "positive": 2,
                    "negative": 3,
                    "patient_strata": {"mixed": 1, "negative_only": 1, "positive_only": 1},
                },
                "excluded": {"examples": 4, "patients": 2, "positive": 2, "negative": 2},
            },
        },
        "artifacts": {
            split: [{"path": f"{split}.parquet", "bytes": 10}]
            for split in ("train", "validation", "test")
        },
    }
    validate_supervised_manifest(manifest)
    manifest["quality"]["train_selection"]["selected"]["patients"] = 2
    try:
        validate_supervised_manifest(manifest)
    except ValueError as error:
        assert "patient strata" in str(error)
    else:
        raise AssertionError("inconsistent low-label patient counts must be rejected")
