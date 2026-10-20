import json
import math
from pathlib import Path

from experiments.ai_runtime_exmedbert.demo import (
    air_overrides,
    aggregate_downstream_results,
    bundle_command,
    derive_paths,
    downstream_lineage_checkpoint_root,
    load_config,
    parse_json_output,
    spark_commands,
    validate_training_result,
    validate_plos_manifest_reuse,
)


def fixture_config():
    return {
        "volume_root": "/Volumes/catalog/schema/volume",
        "source_release": "synthea-v1",
        "pretrain_snapshot_version": "pretrain-v1",
        "audit_version": "audit-v1",
        "plos_snapshot_version": "plos-v1",
        "pretrain_epochs": 10,
        "pretrain_batch_size": 32,
        "downstream_epochs": 5,
        "downstream_batch_size": 64,
        "downstream_train_fraction": 1.0,
        "downstream_subset_seed": 201214,
        "downstream_training_seed": 201214,
        "num_workers": 6,
        "prefetch_factor": 4,
        "num_shards": 16,
        "max_oov_fraction": 0.01,
    }


def training_result(seed, mode, **metrics):
    values = {
        "average_precision": 0.4,
        "auroc": 0.8,
        "loss_brier": 0.2,
        "calibration_error": 0.1,
    }
    values.update(metrics)
    return {
        "configuration": {"seed": seed, "initialization": mode},
        "result": {"test": values},
    }


def test_paths_are_derived_from_one_volume_root():
    paths = derive_paths(fixture_config())
    assert paths.source_root == "/Volumes/catalog/schema/volume/synthea/raw/synthea-v1"
    assert paths.pretrain_snapshot_root.endswith("/exmedbert/snapshots/pretrain-v1")
    assert paths.audit_path.endswith("/exmedbert/audits/audit-v1/audit.json")
    assert paths.plos_snapshot_root.endswith("/exmedbert/downstream/plos/snapshots/plos-v1")


def test_bundle_flags_precede_forwarded_task_arguments():
    command = bundle_command(
        "profile",
        "dev",
        "run",
        "prepare",
        "--",
        "--source-root",
        "/Volumes/a/b/c",
    )
    separator = command.index("--")
    assert command.index("--profile") < separator
    assert command.index("-t") < separator


def test_air_overrides_use_derived_paths():
    config = fixture_config()
    paths = derive_paths(config)
    overrides = air_overrides("train.yaml", config, paths)
    assert "env_variables.EPOCHS=10" in overrides
    assert f"env_variables.SNAPSHOT_ROOT={paths.pretrain_snapshot_root}" in overrides


def test_downstream_overrides_include_training_seed():
    config = fixture_config()
    paths = derive_paths(config)
    overrides = air_overrides("train_plos.yaml", config, paths)
    assert "env_variables.SEED=201214" in overrides


def test_plos_snapshot_command_includes_low_label_contract():
    config = fixture_config()
    command = spark_commands("profile", "dev", config, derive_paths(config))["plos_snapshot"]
    assert command[command.index("--train-label-fraction") + 1] == "1.0"
    assert command[command.index("--subset-seed") + 1] == "201214"


def test_config_rejects_invalid_training_fraction(tmp_path: Path):
    config = fixture_config()
    config["downstream_train_fraction"] = 0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        load_config(path)
    except ValueError as error:
        assert "downstream_train_fraction" in str(error)
    else:
        raise AssertionError("zero training fraction must be rejected")


def test_config_normalizes_legacy_training_seed(tmp_path: Path):
    config = fixture_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["downstream_training_seeds"] == [201214]
    assert loaded["downstream_training_seed"] == 201214


def test_config_accepts_ordered_training_seeds(tmp_path: Path):
    config = fixture_config()
    config.pop("downstream_training_seed")
    config["downstream_training_seeds"] = [201214, 7, 99]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["downstream_training_seeds"] == [201214, 7, 99]
    assert loaded["downstream_training_seed"] == 201214


def test_config_rejects_duplicate_or_ambiguous_training_seeds(tmp_path: Path):
    for seeds, legacy in (([7, 7], None), ([7, 8], 8)):
        config = fixture_config()
        config["downstream_training_seeds"] = seeds
        if legacy is None:
            config.pop("downstream_training_seed")
        else:
            config["downstream_training_seed"] = legacy
        path = tmp_path / f"config-{seeds}-{legacy}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        try:
            load_config(path)
        except ValueError as error:
            assert "downstream_training_seed" in str(error)
        else:
            raise AssertionError("duplicate or ambiguous training seeds must be rejected")


def test_downstream_mode_argument_accepts_primary_comparison(monkeypatch):
    from experiments.ai_runtime_exmedbert import demo

    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "demo.py",
            "plan",
            "--profile",
            "profile",
            "--downstream-mode",
            "finetune",
            "--downstream-mode",
            "random",
        ],
    )
    args = demo.parse_args()
    assert args.downstream_mode == ["finetune", "random"]


def test_plos_manifest_reuse_requires_matching_fraction_and_seed():
    config = fixture_config()
    paths = derive_paths(config)
    manifest = {
        "version": config["plos_snapshot_version"],
        "source": {"root": paths.source_root},
        "pretraining": {"snapshot_root": paths.pretrain_snapshot_root},
        "configuration": {
            "seed": 201214,
            "num_shards_per_split": 16,
        },
    }
    validate_plos_manifest_reuse(manifest, config, paths)
    config["downstream_train_fraction"] = 0.1
    try:
        validate_plos_manifest_reuse(manifest, config, paths)
    except ValueError as error:
        assert "plos_snapshot_version" in str(error)
    else:
        raise AssertionError("a full snapshot must not satisfy a low-label request")


def test_nondefault_training_seed_gets_isolated_checkpoint_root():
    paths = derive_paths(fixture_config())
    default = downstream_lineage_checkpoint_root(paths, "/checkpoints/epoch_000011_abcd", 201214)
    alternate = downstream_lineage_checkpoint_root(paths, "/checkpoints/epoch_000011_abcd", 99)
    assert default.endswith("/from_epoch_000011_abcd")
    assert alternate.endswith("/from_epoch_000011_abcd/seed_99")


def test_mixed_air_output_uses_final_json_envelope():
    envelope = {"v": 1, "data": {"run_id": "123"}}
    assert parse_json_output("[INFO] profile selected\n" + json.dumps(envelope)) == envelope


def test_training_result_validation_rejects_seed_mismatch():
    try:
        validate_training_result(
            training_result(7, "finetune"),
            mode="finetune",
            seed=8,
            source="fixture",
        )
    except ValueError as error:
        assert "seed mismatch" in str(error)
    else:
        raise AssertionError("a result from another seed must never be reused")


def test_aggregation_produces_paired_differences_mean_and_sample_standard_deviation():
    results = {
        1: {
            "finetune": training_result(
                1,
                "finetune",
                average_precision=0.5,
                auroc=0.8,
                loss_brier=0.15,
                calibration_error=0.10,
            ),
            "random": training_result(
                1,
                "random",
                average_precision=0.3,
                auroc=0.7,
                loss_brier=0.20,
                calibration_error=0.15,
            ),
        },
        2: {
            "finetune": training_result(
                2,
                "finetune",
                average_precision=0.6,
                auroc=0.9,
                loss_brier=0.12,
                calibration_error=0.08,
            ),
            "random": training_result(
                2,
                "random",
                average_precision=0.2,
                auroc=0.6,
                loss_brier=0.22,
                calibration_error=0.18,
            ),
        },
    }
    aggregation = aggregate_downstream_results(results)
    assert [row["seed"] for row in aggregation["per_seed"]] == [1, 2]
    assert math.isclose(aggregation["summary"]["average_precision"]["mean"], 0.3)
    assert math.isclose(
        aggregation["summary"]["average_precision"]["sample_standard_deviation"],
        math.sqrt(0.02),
    )
    assert math.isclose(aggregation["summary"]["loss_brier"]["mean"], -0.075)


def test_single_seed_aggregation_has_null_sample_standard_deviation():
    aggregation = aggregate_downstream_results(
        {7: {"finetune": training_result(7, "finetune"), "random": training_result(7, "random")}}
    )
    assert aggregation["summary"]["auroc"]["sample_standard_deviation"] is None


def test_aggregation_fails_for_missing_pair_or_metric():
    try:
        aggregate_downstream_results({7: {"finetune": training_result(7, "finetune")}})
    except ValueError as error:
        assert "missing paired" in str(error)
    else:
        raise AssertionError("unpaired results must be rejected")

    incomplete = training_result(7, "random")
    del incomplete["result"]["test"]["calibration_error"]
    try:
        aggregate_downstream_results(
            {7: {"finetune": training_result(7, "finetune"), "random": incomplete}}
        )
    except ValueError as error:
        assert "calibration_error" in str(error)
    else:
        raise AssertionError("missing aggregation metrics must be rejected")
