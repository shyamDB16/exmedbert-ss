import csv
from pathlib import Path

import pytest

from experiments.ai_runtime_exmedbert.synthea.generate import (
    SYNTHEA_JAR_SHA256,
    build_command,
    validate_raw_release,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fixture_release(root: Path) -> None:
    write_csv(
        root / "patients.csv",
        ["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "STATE"],
        [
            {
                "Id": "patient-1",
                "BIRTHDATE": "1980-01-01",
                "DEATHDATE": "",
                "GENDER": "F",
                "STATE": "MA",
            }
        ],
    )
    write_csv(
        root / "encounters.csv",
        ["Id", "PATIENT", "START"],
        [{"Id": "encounter-1", "PATIENT": "patient-1", "START": "2020-01-01T00:00:00Z"}],
    )
    for filename in ("conditions.csv", "medications.csv", "procedures.csv"):
        write_csv(
            root / filename,
            ["PATIENT", "ENCOUNTER", "START", "CODE"],
            [
                {
                    "PATIENT": "patient-1",
                    "ENCOUNTER": "encounter-1",
                    "START": "2020-01-01T00:00:00Z",
                    "CODE": "123",
                }
            ],
        )


def test_raw_release_contract_and_checksums(tmp_path: Path):
    fixture_release(tmp_path)
    result = validate_raw_release(tmp_path, expected_living_patients=1)
    assert result["patients"] == 1
    assert result["living_patients"] == 1
    assert result["clinical_events"] == 3
    assert result["orphan_events"] == 0
    assert set(result["files"]) == {
        "patients.csv",
        "encounters.csv",
        "conditions.csv",
        "medications.csv",
        "procedures.csv",
    }


def test_raw_release_rejects_patient_encounter_mismatch(tmp_path: Path):
    fixture_release(tmp_path)
    path = tmp_path / "conditions.csv"
    write_csv(
        path,
        ["PATIENT", "ENCOUNTER", "START", "CODE"],
        [{"PATIENT": "wrong", "ENCOUNTER": "encounter-1", "START": "2020-01-01", "CODE": "123"}],
    )
    with pytest.raises(ValueError, match="does not own encounter"):
        validate_raw_release(tmp_path)


def test_command_pins_both_seeds_and_reference_date():
    command = build_command(
        "java",
        Path("synthea.jar"),
        Path("synthea.properties"),
        population=10_000,
        seed=201_214,
        clinician_seed=201_214,
        reference_date="20250101",
        state=None,
        city=None,
    )
    assert command[-10:] == [
        "-c",
        "synthea.properties",
        "-s",
        "201214",
        "-cs",
        "201214",
        "-p",
        "10000",
        "-r",
        "20250101",
    ]
    assert len(SYNTHEA_JAR_SHA256) == 64
