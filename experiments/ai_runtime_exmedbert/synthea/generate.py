#!/usr/bin/env python3
"""Generate an immutable, provenance-complete Synthea CSV release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SYNTHEA_VERSION = "v4.0.0"
SYNTHEA_RELEASE_URL = f"https://github.com/synthetichealth/synthea/releases/tag/{SYNTHEA_VERSION}"
SYNTHEA_JAR_URL = (
    f"https://github.com/synthetichealth/synthea/releases/download/{SYNTHEA_VERSION}/"
    "synthea-with-dependencies.jar"
)
SYNTHEA_JAR_SHA256 = "ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1"
REQUIRED_COLUMNS = {
    "patients.csv": {"ID", "BIRTHDATE", "DEATHDATE", "GENDER", "STATE"},
    "encounters.csv": {"ID", "PATIENT", "START"},
    "conditions.csv": {"PATIENT", "ENCOUNTER", "START", "CODE"},
    "medications.csv": {"PATIENT", "ENCOUNTER", "START", "CODE"},
    "procedures.csv": {"PATIENT", "ENCOUNTER", "START", "CODE"},
}
PROVENANCE_FILE = "provenance.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rows(path: Path) -> tuple[list[str], Iterable[tuple[int, dict[str, str]]]]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(handle)
    fieldnames = [(name or "").strip().upper() for name in (reader.fieldnames or [])]

    def iterator() -> Iterable[tuple[int, dict[str, str]]]:
        try:
            for row_number, row in enumerate(reader, start=2):
                yield row_number, {
                    (key or "").strip().upper(): (value or "").strip()
                    for key, value in row.items()
                }
        finally:
            handle.close()

    return fieldnames, iterator()


def _require_headers(path: Path, headers: list[str]) -> None:
    missing = REQUIRED_COLUMNS[path.name] - set(headers)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")


def validate_raw_release(csv_root: Path, expected_living_patients: int | None = None) -> dict[str, Any]:
    """Validate source keys and return row/checksum metadata without Spark."""
    missing_files = sorted(name for name in REQUIRED_COLUMNS if not (csv_root / name).is_file())
    if missing_files:
        raise ValueError(f"missing required Synthea CSV files: {missing_files}")

    patient_path = csv_root / "patients.csv"
    headers, rows = _rows(patient_path)
    _require_headers(patient_path, headers)
    patient_ids: set[str] = set()
    patient_rows = 0
    living_patients = 0
    for row_number, row in rows:
        patient_id = row.get("ID", "")
        if not patient_id:
            raise ValueError(f"patients.csv:{row_number} has an empty ID")
        if patient_id in patient_ids:
            raise ValueError(f"patients.csv:{row_number} has duplicate ID {patient_id}")
        patient_ids.add(patient_id)
        patient_rows += 1
        living_patients += not bool(row.get("DEATHDATE", ""))
    if expected_living_patients is not None and living_patients != expected_living_patients:
        raise ValueError(f"expected {expected_living_patients} living patients, found {living_patients}")

    encounter_path = csv_root / "encounters.csv"
    headers, rows = _rows(encounter_path)
    _require_headers(encounter_path, headers)
    encounter_to_patient: dict[str, str] = {}
    encounter_rows = 0
    for row_number, row in rows:
        encounter_id = row.get("ID", "")
        patient_id = row.get("PATIENT", "")
        if not encounter_id or not patient_id or not row.get("START", ""):
            raise ValueError(f"encounters.csv:{row_number} has an empty required value")
        if encounter_id in encounter_to_patient:
            raise ValueError(f"encounters.csv:{row_number} has duplicate ID {encounter_id}")
        if patient_id not in patient_ids:
            raise ValueError(f"encounters.csv:{row_number} references unknown patient {patient_id}")
        encounter_to_patient[encounter_id] = patient_id
        encounter_rows += 1

    row_counts = {"patients.csv": patient_rows, "encounters.csv": encounter_rows}
    for filename in ("conditions.csv", "medications.csv", "procedures.csv"):
        path = csv_root / filename
        headers, rows = _rows(path)
        _require_headers(path, headers)
        count = 0
        for row_number, row in rows:
            patient_id = row.get("PATIENT", "")
            encounter_id = row.get("ENCOUNTER", "")
            if not patient_id or not encounter_id or not row.get("START", "") or not row.get("CODE", ""):
                raise ValueError(f"{filename}:{row_number} has an empty required value")
            expected_patient = encounter_to_patient.get(encounter_id)
            if expected_patient is None:
                raise ValueError(f"{filename}:{row_number} references unknown encounter {encounter_id}")
            if expected_patient != patient_id:
                raise ValueError(
                    f"{filename}:{row_number} patient {patient_id} does not own encounter {encounter_id}"
                )
            count += 1
        row_counts[filename] = count

    files = {}
    for filename in REQUIRED_COLUMNS:
        path = csv_root / filename
        files[filename] = {
            "bytes": path.stat().st_size,
            "rows": row_counts[filename],
            "sha256": sha256_file(path),
        }
    return {
        "patients": patient_rows,
        "living_patients": living_patients,
        "deceased_patients": patient_rows - living_patients,
        "encounters": encounter_rows,
        "clinical_events": sum(row_counts[name] for name in row_counts if name not in {"patients.csv", "encounters.csv"}),
        "orphan_events": 0,
        "files": files,
    }


def download_jar(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if actual != SYNTHEA_JAR_SHA256:
            raise ValueError(f"cached Synthea JAR checksum mismatch: {actual}")
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--proto",
                "=https",
                "--retry",
                "3",
                "--output",
                str(partial),
                SYNTHEA_JAR_URL,
            ],
            check=True,
        )
        actual = sha256_file(partial)
        if actual != SYNTHEA_JAR_SHA256:
            raise ValueError(f"downloaded Synthea JAR checksum mismatch: {actual}")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def java_version(java: str) -> str:
    result = subprocess.run([java, "-version"], text=True, capture_output=True, check=True)
    output = (result.stderr or result.stdout).strip()
    match = re.search(r'version "(\d+)', output)
    if not match or int(match.group(1)) < 17:
        raise RuntimeError(f"Synthea {SYNTHEA_VERSION} requires Java 17 or newer; found: {output}")
    return output.splitlines()[0]


def build_command(
    java: str,
    jar: Path,
    config: Path,
    *,
    population: int,
    seed: int,
    clinician_seed: int,
    reference_date: str,
    state: str | None,
    city: str | None,
) -> list[str]:
    command = [
        java,
        "-jar",
        str(jar),
        "-c",
        str(config),
        "-s",
        str(seed),
        "-cs",
        str(clinician_seed),
        "-p",
        str(population),
        "-r",
        reference_date,
    ]
    if state:
        command.append(state)
    if city:
        command.append(city)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=201_214)
    parser.add_argument("--clinician-seed", type=int, default=201_214)
    parser.add_argument("--reference-date", default="20250101", help="Fixed simulation date in YYYYMMDD")
    parser.add_argument("--state", default="Massachusetts", help="US state; defaults to Massachusetts")
    parser.add_argument("--city", help="Optional city; requires --state")
    parser.add_argument("--java", default="java")
    parser.add_argument(
        "--jar-cache",
        type=Path,
        default=Path.home() / ".cache" / "exmedbert" / SYNTHEA_VERSION / "synthea-with-dependencies.jar",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.population < 1:
        raise ValueError("population must be positive")
    if not re.fullmatch(r"\d{8}", args.reference_date):
        raise ValueError("reference-date must use YYYYMMDD")
    if args.city and not args.state:
        raise ValueError("--city requires --state")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable raw release already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = Path(__file__).with_name("synthea.properties").resolve()
    jar = download_jar(args.jar_cache.resolve())
    java = java_version(args.java)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    command = build_command(
        args.java,
        jar,
        config,
        population=args.population,
        seed=args.seed,
        clinician_seed=args.clinician_seed,
        reference_date=args.reference_date,
        state=args.state,
        city=args.city,
    )
    try:
        result = subprocess.run(command, cwd=staging, text=True, capture_output=True)
        (staging / "synthea.stdout.log").write_text(result.stdout, encoding="utf-8")
        (staging / "synthea.stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(
                f"Synthea failed with exit code {result.returncode}; logs are in {staging}"
            )
        csv_root = staging / "output" / "csv"
        quality = validate_raw_release(csv_root, expected_living_patients=args.population)
        reproducibility = {
            "synthea_version": SYNTHEA_VERSION,
            "jar_sha256": SYNTHEA_JAR_SHA256,
            "properties_sha256": sha256_file(config),
            "population": args.population,
            "population_seed": args.seed,
            "clinician_seed": args.clinician_seed,
            "reference_date": args.reference_date,
            "state": args.state,
            "city": args.city,
            "thread_pool_size": 1,
            "years_of_history": 0,
        }
        provenance = {
            "schema_version": 1,
            "dataset": "synthea-raw",
            "complete": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "release": SYNTHEA_RELEASE_URL,
                "jar": SYNTHEA_JAR_URL,
                "jar_sha256": SYNTHEA_JAR_SHA256,
            },
            "reproducibility": reproducibility,
            "reproducibility_sha256": canonical_sha256(reproducibility),
            "environment": {"java": java},
            "quality": {key: value for key, value in quality.items() if key != "files"},
            "files": quality["files"],
        }
        (csv_root / PROVENANCE_FILE).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(csv_root, output)
        shutil.rmtree(staging)
        print(json.dumps({"output": str(output), **provenance["quality"]}, indent=2, sort_keys=True))
    except Exception:
        print(f"generation staging directory retained for diagnosis: {staging}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
