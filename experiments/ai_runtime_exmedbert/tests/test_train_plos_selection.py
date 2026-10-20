import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def stub_module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def load_train_plos_module():
    module_name = "_test_train_plos_selection"
    module_path = Path(__file__).parents[1] / "training" / "train_plos.py"
    checkpoint = stub_module(
        "experiments.ai_runtime_exmedbert.training.checkpoint",
        latest_complete_checkpoint=lambda *_args, **_kwargs: None,
        load_checkpoint=lambda *_args, **_kwargs: (0, 0),
        save_checkpoint=lambda *_args, **_kwargs: "",
    )
    stubs = {
        "mlflow": stub_module("mlflow"),
        "torch": stub_module("torch"),
        "serverless_gpu": stub_module("serverless_gpu"),
        "serverless_gpu.data": stub_module(
            "serverless_gpu.data", DataLoader=object, UCVolumeDataset=object
        ),
        "experiments.ai_runtime_exmedbert.shared.contracts": stub_module(
            "experiments.ai_runtime_exmedbert.shared.contracts",
            canonical_sha256=lambda value: value,
            validate_supervised_manifest=lambda manifest: manifest,
        ),
        "experiments.ai_runtime_exmedbert.training.checkpoint": checkpoint,
        "experiments.ai_runtime_exmedbert.training.endpoint_metrics": stub_module(
            "experiments.ai_runtime_exmedbert.training.endpoint_metrics",
            binary_metrics=lambda *_args, **_kwargs: {},
            select_youden_threshold=lambda *_args, **_kwargs: 0.5,
        ),
        "experiments.ai_runtime_exmedbert.training.supervised_data": stub_module(
            "experiments.ai_runtime_exmedbert.training.supervised_data",
            ExMedBertClassificationCollator=object,
            SupervisedExMedBertParquetDataset=object,
        ),
        "experiments.ai_runtime_exmedbert.training.supervised_modeling": stub_module(
            "experiments.ai_runtime_exmedbert.training.supervised_modeling",
            ExMedBertForBinaryEndpoint=object,
            create_classifier=lambda *_args, **_kwargs: None,
            freeze_encoder=lambda *_args, **_kwargs: None,
            transfer_pretrained_encoder=lambda *_args, **_kwargs: None,
        ),
        "experiments.ai_runtime_exmedbert.training.vocabulary": stub_module(
            "experiments.ai_runtime_exmedbert.training.vocabulary",
            VocabularyArtifacts=object,
            load_snapshot_metadata=lambda *_args, **_kwargs: None,
            sha256_file=lambda *_args, **_kwargs: "",
        ),
    }
    previous = {name: sys.modules.get(name) for name in (*stubs, module_name)}
    sys.modules.update(stubs)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, existing in previous.items():
            if existing is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = existing
    return module


train_plos = load_train_plos_module()


def make_checkpoint(root: Path, epoch: int, suffix: str) -> Path:
    checkpoint = root / f"epoch_{epoch:06d}_{suffix}"
    checkpoint.mkdir()
    (checkpoint / ".metadata").write_text("complete\n", encoding="utf-8")
    return checkpoint


def write_evidence(checkpoint: Path, epoch: int, average_precision: float):
    return train_plos.write_validation_evidence(
        str(checkpoint),
        epoch=epoch,
        validation_average_precision=average_precision,
        manifest_sha256="manifest",
        configuration_sha256="configuration",
    )


def test_selects_highest_validation_ap_and_prefers_earliest_tie(tmp_path):
    first = make_checkpoint(tmp_path, 1, "a1")
    second = make_checkpoint(tmp_path, 2, "b2")
    third = make_checkpoint(tmp_path, 3, "c3")
    write_evidence(first, 1, 0.40)
    write_evidence(second, 2, 0.75)
    write_evidence(third, 3, 0.75)

    selected = train_plos.select_best_validation_checkpoint(
        str(tmp_path),
        expected_manifest_sha256="manifest",
        expected_configuration_sha256="configuration",
    )

    assert selected.epoch == 2
    assert selected.checkpoint == str(second)
    assert selected.validation_average_precision == 0.75
    payload = json.loads((second / "validation_evidence.json").read_text(encoding="utf-8"))
    assert payload["manifest_sha256"] == "manifest"
    assert payload["configuration_sha256"] == "configuration"


def test_ignores_incomplete_and_mismatched_evidence(tmp_path):
    complete = make_checkpoint(tmp_path, 1, "a1")
    write_evidence(complete, 1, 0.80)
    incomplete = tmp_path / "epoch_000002_b2"
    incomplete.mkdir()

    selected = train_plos.select_best_validation_checkpoint(
        str(tmp_path),
        expected_manifest_sha256="different",
        expected_configuration_sha256="configuration",
    )

    assert selected is None
    assert train_plos.complete_checkpoints(str(tmp_path)) == [(1, str(complete))]


def test_restore_uses_fresh_factory_and_validates_epoch(monkeypatch):
    created_model = object()
    created_optimizer = object()
    calls = []

    def fake_load(checkpoint, model, optimizer, **kwargs):
        calls.append((checkpoint, model, optimizer, kwargs))
        return 4, 99

    monkeypatch.setattr(train_plos, "load_checkpoint", fake_load)
    selection = train_plos.ValidationCheckpoint(4, "/checkpoint", 0.61)
    model, optimizer, step = train_plos.restore_validation_checkpoint(
        selection,
        lambda: (created_model, created_optimizer),
        expected_manifest_sha256="manifest",
        expected_configuration_sha256="configuration",
    )

    assert (model, optimizer, step) == (created_model, created_optimizer, 99)
    assert calls == [
        (
            "/checkpoint",
            created_model,
            created_optimizer,
            {
                "expected_manifest_sha256": "manifest",
                "expected_configuration_sha256": "configuration",
            },
        )
    ]

    monkeypatch.setattr(train_plos, "load_checkpoint", lambda *_args, **_kwargs: (5, 99))
    with pytest.raises(AssertionError, match="epoch does not match"):
        train_plos.restore_validation_checkpoint(
            selection,
            lambda: (created_model, created_optimizer),
            expected_manifest_sha256="manifest",
            expected_configuration_sha256="configuration",
        )


def test_rejects_invalid_validation_evidence(tmp_path):
    checkpoint = make_checkpoint(tmp_path, 1, "a1")
    with pytest.raises(ValueError, match="between zero and one"):
        write_evidence(checkpoint, 1, float("nan"))
    with pytest.raises(ValueError, match="does not match checkpoint name"):
        write_evidence(checkpoint, 2, 0.5)


def test_backfills_legacy_checkpoint_evidence(tmp_path, monkeypatch):
    first = make_checkpoint(tmp_path, 1, "a1")
    second = make_checkpoint(tmp_path, 2, "b2")
    average_precision = {str(first): 0.72, str(second): 0.68}

    def fake_restore(selection, *_args, **_kwargs):
        return SimpleNamespace(checkpoint=selection.checkpoint), object(), 10

    def fake_run_split(model, *_args, **_kwargs):
        return {
            "examples": 12,
            "average_precision": average_precision[model.checkpoint],
        }, {}, None

    monkeypatch.setattr(train_plos, "restore_validation_checkpoint", fake_restore)
    monkeypatch.setattr(train_plos, "run_split", fake_run_split)
    monkeypatch.setattr(
        train_plos.torch,
        "cuda",
        SimpleNamespace(empty_cache=lambda: None),
        raising=False,
    )

    selected = train_plos.ensure_validation_evidence(
        str(tmp_path),
        lambda: (object(), object()),
        object(),
        object(),
        encoder_frozen=False,
        expected_examples=12,
        manifest_sha256="manifest",
        configuration_sha256="configuration",
    )

    assert selected.epoch == 1
    assert (first / "validation_evidence.json").is_file()
    assert (second / "validation_evidence.json").is_file()


def test_training_result_is_published_only_as_final_stage_marker(tmp_path):
    path = train_plos.write_training_result(
        str(tmp_path),
        manifest_sha256="manifest",
        configuration_sha256="configuration",
        configuration={"seed": 17},
        result={"test": {"average_precision": 0.7}},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "training_result.json"
    assert payload["configuration"] == {"seed": 17}
    assert payload["result"]["test"]["average_precision"] == 0.7
