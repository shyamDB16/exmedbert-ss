"""Reconstruct legacy ExMed-BERT vocabulary objects from JSON artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.ai_runtime_exmedbert.shared.contracts import ENTITY_IDS, validate_vocabulary


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VocabularyArtifacts:
    document: dict[str, Any]

    @property
    def code_size(self) -> int:
        return len(self.document["tokens"])

    @property
    def code_entity_ids(self) -> list[int]:
        return [int(entry["entity_id"]) for entry in self.document["tokens"]]

    @property
    def maskable_code_ids(self) -> list[int]:
        return [int(entry["id"]) for entry in self.document["tokens"] if entry["entity"] != "default"]

    def build_code_dict(self):
        from exmed_bert.data.encoding import CodeDict

        grouped = {entity: [] for entity in ("medication", "diagnosis", "procedure")}
        for entry in self.document["tokens"]:
            if entry["entity"] in grouped:
                grouped[entry["entity"]].append(entry["token"])
        code_dict = CodeDict(
            atc_codes=grouped["medication"],
            icd_codes=grouped["diagnosis"],
            cpt_codes=grouped["procedure"],
            rx_to_atc_map=None,
        )
        for entry in self.document["tokens"]:
            if code_dict.labels_to_id.get(entry["token"]) != entry["id"]:
                raise ValueError(f"CodeDict ID mismatch for {entry['token']}")
        expected_entities = {
            "default": ENTITY_IDS["default"],
            "atc": ENTITY_IDS["medication"],
            "icd": ENTITY_IDS["diagnosis"],
            "cpt": ENTITY_IDS["procedure"],
        }
        if code_dict.entity_to_id != expected_entities:
            raise ValueError(f"CodeDict entity mapping mismatch: {code_dict.entity_to_id}")
        return code_dict


def load_snapshot_metadata(snapshot_root: str) -> tuple[dict[str, Any], VocabularyArtifacts, str]:
    root = Path(snapshot_root)
    manifest_path = root / "manifest.json"
    vocabulary_path = root / "vocabulary.json"
    if not manifest_path.is_file() or not vocabulary_path.is_file():
        raise FileNotFoundError("snapshot requires manifest.json and vocabulary.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    document = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    validate_vocabulary(document)
    if sha256_file(vocabulary_path) != manifest["vocabulary"]["sha256"]:
        raise ValueError("vocabulary checksum does not match manifest")
    artifacts = VocabularyArtifacts(document)
    if artifacts.code_size != int(manifest["vocabulary"]["code_size"]):
        raise ValueError("vocabulary size does not match manifest")
    return manifest, artifacts, sha256_file(manifest_path)
