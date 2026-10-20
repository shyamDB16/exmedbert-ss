"""AIR file-streaming adapter and collation for supervised ExMed-BERT data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


class SupervisedExMedBertParquetDataset(IterableDataset):
    """Decode cached Parquet files yielded by ``UCVolumeDataset``."""

    def __init__(self, path_dataset: Iterable[str], expected_files: dict[str, int]):
        super().__init__()
        self.path_dataset = path_dataset
        self.expected_files = expected_files
        self.validated_files: set[str] = set()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        columns = [
            "example_id",
            "patient_id",
            "codes",
            "entity_ids",
            "age_ids",
            "position_ids",
            "sex_id",
            "state_id",
            "label",
        ]
        for cached_path in self.path_dataset:
            path = Path(str(cached_path))
            expected_size = self.expected_files.get(path.name)
            if expected_size is None:
                raise ValueError(f"unexpected file in supervised split: {path.name}")
            if path.stat().st_size != expected_size:
                raise ValueError(f"cached file size mismatch: {path.name}")
            self.validated_files.add(path.name)
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(columns=columns, batch_size=256):
                yield from batch.to_pylist()


class ExMedBertClassificationCollator:
    def __init__(self, *, max_sequence_length: int):
        self.max_sequence_length = max_sequence_length

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        length = min(self.max_sequence_length, max(len(sample["codes"]) for sample in samples))
        batch_size = len(samples)
        tensors = {
            name: torch.zeros((batch_size, length), dtype=torch.long)
            for name in (
                "input_ids",
                "entity_ids",
                "age_ids",
                "position_ids",
                "sex_ids",
                "state_ids",
                "attention_mask",
            )
        }
        endpoint_labels = torch.zeros((batch_size, 1), dtype=torch.float32)
        example_ids: list[str] = []
        patient_ids: list[str] = []

        source_names = {
            "input_ids": "codes",
            "entity_ids": "entity_ids",
            "age_ids": "age_ids",
            "position_ids": "position_ids",
        }
        for row_index, sample in enumerate(samples):
            example_id = str(sample["example_id"])
            patient_id = str(sample["patient_id"])
            example_ids.append(example_id)
            patient_ids.append(patient_id)
            arrays = {
                target: torch.as_tensor(sample[source][-length:], dtype=torch.long)
                for target, source in source_names.items()
            }
            sizes = {len(value) for value in arrays.values()}
            if len(sizes) != 1:
                raise ValueError(f"unaligned model arrays for example {example_id}")
            size = sizes.pop()
            if size == 0:
                raise ValueError(f"empty model history for example {example_id}")
            for name, values in arrays.items():
                tensors[name][row_index, :size] = values
            tensors["sex_ids"][row_index, :size] = int(sample["sex_id"])
            tensors["state_ids"][row_index, :size] = int(sample["state_id"])
            tensors["attention_mask"][row_index, :size] = 1
            label = int(sample["label"])
            if label not in (0, 1):
                raise ValueError(f"invalid binary label for example {example_id}: {label}")
            endpoint_labels[row_index, 0] = label

        return {
            "example_ids": example_ids,
            "patient_ids": patient_ids,
            **tensors,
            "endpoint_labels": endpoint_labels,
        }
