"""UC-volume Parquet decoding and deterministic ExMed-BERT collation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


class ExMedBertParquetDataset(IterableDataset):
    def __init__(self, path_dataset: Iterable[str], expected_files: dict[str, int]):
        super().__init__()
        self.path_dataset = path_dataset
        self.expected_files = expected_files
        self.validated_files: set[str] = set()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for cached_path in self.path_dataset:
            path = Path(str(cached_path))
            expected_size = self.expected_files.get(path.name)
            if expected_size is None:
                raise ValueError(f"unexpected file in training split: {path.name}")
            if path.stat().st_size != expected_size:
                raise ValueError(f"cached file size mismatch: {path.name}")
            self.validated_files.add(path.name)
            parquet = pq.ParquetFile(path)
            columns = [
                "patient_id",
                "codes",
                "entity_ids",
                "age_ids",
                "position_ids",
                "sex_id",
                "state_id",
            ]
            for batch in parquet.iter_batches(columns=columns, batch_size=256):
                yield from batch.to_pylist()


class DeterministicExMedBertCollator:
    def __init__(
        self,
        *,
        code_entity_ids: list[int],
        maskable_code_ids: list[int],
        max_sequence_length: int,
        seed: int,
        mask_probability: float = 0.15,
        max_masked_tokens: int = 20,
        mask_medications: bool = True,
    ):
        self.code_entity_ids = torch.tensor(code_entity_ids, dtype=torch.long)
        self.maskable_code_ids = torch.tensor(maskable_code_ids, dtype=torch.long)
        self.max_sequence_length = max_sequence_length
        self.seed = seed
        self.mask_probability = mask_probability
        self.max_masked_tokens = max_masked_tokens
        self.mask_medications = mask_medications
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _generator(self, patient_id: str) -> torch.Generator:
        payload = f"{self.seed}:{self.epoch}:{patient_id}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
        return torch.Generator().manual_seed(value)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        length = min(self.max_sequence_length, max(len(sample["codes"]) for sample in samples))
        batch_size = len(samples)
        input_ids = torch.zeros((batch_size, length), dtype=torch.long)
        entity_ids = torch.zeros((batch_size, length), dtype=torch.long)
        age_ids = torch.zeros((batch_size, length), dtype=torch.long)
        position_ids = torch.zeros((batch_size, length), dtype=torch.long)
        sex_ids = torch.zeros((batch_size, length), dtype=torch.long)
        state_ids = torch.zeros((batch_size, length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, length), dtype=torch.long)
        code_labels = torch.full((batch_size, length), -100, dtype=torch.long)
        patient_ids: list[str] = []

        for row_index, sample in enumerate(samples):
            patient_id = str(sample["patient_id"])
            patient_ids.append(patient_id)
            codes = torch.as_tensor(sample["codes"][-length:], dtype=torch.long).clone()
            entities = torch.as_tensor(sample["entity_ids"][-length:], dtype=torch.long).clone()
            ages = torch.as_tensor(sample["age_ids"][-length:], dtype=torch.long)
            positions = torch.as_tensor(sample["position_ids"][-length:], dtype=torch.long)
            if not (len(codes) == len(entities) == len(ages) == len(positions)):
                raise ValueError(f"unaligned model arrays for patient {patient_id}")
            size = len(codes)
            input_ids[row_index, :size] = codes
            entity_ids[row_index, :size] = entities
            age_ids[row_index, :size] = ages
            position_ids[row_index, :size] = positions
            sex_ids[row_index, :size] = int(sample["sex_id"])
            state_ids[row_index, :size] = int(sample["state_id"])
            attention_mask[row_index, :size] = 1

            eligible = torch.nonzero(codes >= 6, as_tuple=False).flatten()
            if not self.mask_medications:
                eligible = eligible[entities[eligible] != 1]
            maximum = max(0, len(eligible) - 1)
            selected_count = min(
                self.max_masked_tokens,
                maximum,
                max(1, int(round(len(eligible) * self.mask_probability))) if len(eligible) else 0,
            )
            if selected_count == 0:
                continue
            generator = self._generator(patient_id)
            selected = eligible[torch.randperm(len(eligible), generator=generator)[:selected_count]]
            code_labels[row_index, selected] = codes[selected]
            decisions = torch.rand(len(selected), generator=generator)
            mask_positions = selected[decisions < 0.8]
            random_positions = selected[(decisions >= 0.8) & (decisions < 0.9)]
            input_ids[row_index, mask_positions] = 4
            entity_ids[row_index, mask_positions] = 0
            if len(random_positions):
                random_indexes = torch.randint(
                    0, len(self.maskable_code_ids), (len(random_positions),), generator=generator
                )
                replacements = self.maskable_code_ids[random_indexes]
                input_ids[row_index, random_positions] = replacements
                entity_ids[row_index, random_positions] = self.code_entity_ids[replacements]

        return {
            "patient_ids": patient_ids,
            "input_ids": input_ids,
            "entity_ids": entity_ids,
            "age_ids": age_ids,
            "sex_ids": sex_ids,
            "state_ids": state_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "code_labels": code_labels,
        }
