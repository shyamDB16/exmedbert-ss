#!/usr/bin/env python3
"""Publish a complete Synthea raw release to an immutable UC Volume path."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from .generate import PROVENANCE_FILE, REQUIRED_COLUMNS, sha256_file, validate_raw_release
except ImportError:  # Direct execution: python synthea/publish.py
    from generate import PROVENANCE_FILE, REQUIRED_COLUMNS, sha256_file, validate_raw_release


def run_databricks(arguments: list[str], *, profile: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["databricks", *arguments, "--profile", profile],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Databricks CLI failed ({' '.join(arguments)}): {detail}")
    return result


def load_provenance(source: Path) -> dict[str, Any]:
    path = source / PROVENANCE_FILE
    if not path.is_file():
        raise ValueError(f"missing completion marker: {path}")
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance.get("dataset") != "synthea-raw" or provenance.get("complete") is not True:
        raise ValueError("invalid Synthea provenance document")
    population = int(provenance["reproducibility"]["population"])
    quality = validate_raw_release(source, expected_living_patients=population)
    if quality["files"] != provenance.get("files"):
        raise ValueError("raw CSV content no longer matches provenance.json")
    return provenance


def remote_path(path: str) -> str:
    if not path.startswith("/Volumes/") or path.endswith("/"):
        raise ValueError("destination must be a versioned /Volumes/... path without a trailing slash")
    return f"dbfs:{path}"


def assert_destination_absent(destination: str, profile: str) -> None:
    result = run_databricks(["fs", "ls", destination, "--output", "json"], profile=profile, check=False)
    if result.returncode == 0:
        raise FileExistsError(f"immutable destination already exists: {destination}")
    detail = f"{result.stdout}\n{result.stderr}".lower()
    absent_messages = ("does not exist", "resource_does_not_exist", "not found", "no such directory")
    if not any(message in detail for message in absent_messages):
        raise RuntimeError(f"could not inspect destination: {result.stderr.strip()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--profile", default="fevm-default")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing publication against the local release without writing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    provenance = load_provenance(source)
    destination = remote_path(args.destination)
    filenames = [*REQUIRED_COLUMNS, PROVENANCE_FILE]
    if not args.validate_only:
        assert_destination_absent(destination, args.profile)
        run_databricks(["fs", "mkdir", destination], profile=args.profile)

        # Upload the completion marker last. Consumers must reject directories
        # without provenance.json, which makes interrupted publications visible.
        for filename in filenames:
            local = source / filename
            target = f"{destination}/{filename}"
            result = run_databricks(["fs", "cp", str(local), target], profile=args.profile)
            if result.stdout.strip():
                print(result.stdout.strip())

    listing = run_databricks(["fs", "ls", destination, "--output", "json"], profile=args.profile)
    remote_entries = {
        item.get("name") or Path(item["path"]).name: item
        for item in json.loads(listing.stdout)
    }
    expected_sizes = {
        filename: (source / filename).stat().st_size
        for filename in filenames
    }
    actual_sizes = {
        filename: int(
            remote_entries.get(filename, {}).get(
                "size", remote_entries.get(filename, {}).get("file_size", -1)
            )
        )
        for filename in filenames
    }
    if expected_sizes != actual_sizes:
        raise RuntimeError(f"remote size validation failed: expected={expected_sizes}, actual={actual_sizes}")
    remote_provenance = run_databricks(
        ["fs", "cat", f"{destination}/{PROVENANCE_FILE}"], profile=args.profile
    )
    if json.loads(remote_provenance.stdout) != provenance:
        raise RuntimeError("remote provenance.json does not match the validated local release")
    print(
        json.dumps(
            {
                "destination": args.destination,
                "patients": provenance["quality"]["patients"],
                "files": len(filenames),
                "bytes": sum(actual_sizes.values()),
                "provenance_sha256": sha256_file(source / PROVENANCE_FILE),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
