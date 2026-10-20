"""AI Runtime data, model, and distributed-checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import IterableDataset


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_snapshot_metadata(snapshot_root: str) -> tuple[dict[str, Any], dict[str, int], str]:
    root = Path(snapshot_root)
    manifest_path = root / "manifest.json"
    vocabulary_path = root / "vocabulary.json"
    if not manifest_path.is_file() or not vocabulary_path.is_file():
        raise FileNotFoundError("snapshot requires manifest.json and vocabulary.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    expected_vocab_hash = manifest["files"]["vocabulary"]["sha256"]
    if sha256_file(vocabulary_path) != expected_vocab_hash:
        raise ValueError("vocabulary checksum does not match manifest")
    if vocabulary.get("PAD") != 0 or vocabulary.get("MASK") != 1 or vocabulary.get("UNK") != 2:
        raise ValueError("unexpected special-token IDs")
    return manifest, {str(key): int(value) for key, value in vocabulary.items()}, sha256_file(manifest_path)


class ParquetPatientDataset(IterableDataset):
    """Decode locally cached Parquet paths yielded by UCVolumeDataset."""

    def __init__(self, path_dataset: IterableDataset, expected_hashes: dict[str, str]):
        super().__init__()
        self.path_dataset = path_dataset
        self.expected_hashes = expected_hashes
        self.validated_files: set[str] = set()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for local_path in self.path_dataset:
            local_path = str(local_path)
            basename = Path(local_path).name
            expected = self.expected_hashes.get(basename)
            if expected and basename not in self.validated_files:
                actual = sha256_file(local_path)
                if actual != expected:
                    raise ValueError(f"checksum mismatch for {basename}")
                self.validated_files.add(basename)
            parquet_file = pq.ParquetFile(local_path)
            for batch in parquet_file.iter_batches(columns=["patient_id", "codes", "time_gaps"], batch_size=256):
                yield from batch.to_pylist()


class DeterministicMLMCollator:
    def __init__(
        self,
        *,
        vocab_size: int,
        max_sequence_length: int,
        seed: int,
        mask_probability: float = 0.15,
    ):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.seed = seed
        self.mask_probability = mask_probability
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
        time_gaps = torch.zeros((batch_size, length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, length), dtype=torch.bool)
        labels = torch.full((batch_size, length), -100, dtype=torch.long)
        patient_ids: list[str] = []

        for row_index, sample in enumerate(samples):
            patient_id = str(sample["patient_id"])
            patient_ids.append(patient_id)
            codes = torch.as_tensor(sample["codes"][-length:], dtype=torch.long).clone()
            gaps = torch.as_tensor(sample["time_gaps"][-length:], dtype=torch.long)
            if len(codes) != len(gaps):
                raise ValueError(f"unaligned codes/time_gaps for patient {patient_id}")
            size = len(codes)
            input_ids[row_index, :size] = codes
            time_gaps[row_index, :size] = gaps
            attention_mask[row_index, :size] = True

            eligible = torch.nonzero(codes >= 3, as_tuple=False).flatten()
            if len(eligible) == 0:
                continue
            generator = self._generator(patient_id)
            selected_count = max(1, int(round(len(eligible) * self.mask_probability)))
            selected = eligible[torch.randperm(len(eligible), generator=generator)[:selected_count]]
            labels[row_index, selected] = codes[selected]
            decisions = torch.rand(len(selected), generator=generator)
            mask_positions = selected[decisions < 0.8]
            random_positions = selected[(decisions >= 0.8) & (decisions < 0.9)]
            input_ids[row_index, mask_positions] = 1
            if len(random_positions):
                input_ids[row_index, random_positions] = torch.randint(
                    3, self.vocab_size, (len(random_positions),), generator=generator
                )

        return {
            "patient_ids": patient_ids,
            "input_ids": input_ids,
            "time_gaps": time_gaps,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class TinyMedicalBert(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_sequence_length: int = 128,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        intermediate_size: int = 512,
        dropout: float = 0.1,
        use_time: bool = True,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.use_time = use_time
        self.code_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_sequence_length, hidden_size)
        self.time_embeddings = nn.Embedding(7, hidden_size) if use_time else None
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, time_gaps: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.code_embeddings(input_ids) + self.position_embeddings(positions)
        if self.time_embeddings is not None:
            hidden = hidden + self.time_embeddings(time_gaps)
        hidden = self.dropout(self.layer_norm(hidden))
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        return self.lm_head(hidden)


def loss_sum_and_tokens(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    token_count = int((labels != -100).sum().item())
    if token_count == 0:
        return logits.sum() * 0.0, 0
    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum"
    )
    return loss_sum, token_count


_CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d{6})_[0-9a-f]+$")


def latest_complete_checkpoint(checkpoint_root: str) -> str | None:
    root = Path(checkpoint_root)
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = _CHECKPOINT_PATTERN.match(child.name)
        if child.is_dir() and match and (child / ".metadata").is_file():
            candidates.append((int(match.group(1)), child))
    return str(max(candidates, key=lambda item: item[0])[1]) if candidates else None


def _text_tensor(value: str) -> torch.Tensor:
    return torch.tensor(list(value.encode("ascii")), dtype=torch.uint8)


def _tensor_text(value: torch.Tensor) -> str:
    return bytes(value.cpu().tolist()).decode("ascii")


def save_checkpoint(
    checkpoint_root: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    next_epoch: int,
    global_step: int,
    manifest_sha256: str,
    configuration_sha256: str,
) -> str:
    import torch.distributed.checkpoint as dcp
    from serverless_gpu.data import UCVolumeWriter
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "next_epoch": torch.tensor(next_epoch, dtype=torch.int64),
        "global_step": torch.tensor(global_step, dtype=torch.int64),
        "manifest_sha256": _text_tensor(manifest_sha256),
        "configuration_sha256": _text_tensor(configuration_sha256),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    destination = str(Path(checkpoint_root) / f"epoch_{next_epoch:06d}_{uuid.uuid4().hex}")
    dcp.save(state, storage_writer=UCVolumeWriter(destination))
    if not (Path(destination) / ".metadata").is_file():
        raise RuntimeError(f"DCP checkpoint did not publish .metadata: {destination}")
    return destination


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_manifest_sha256: str,
    expected_configuration_sha256: str,
) -> tuple[int, int]:
    import torch.distributed.checkpoint as dcp
    from serverless_gpu.data import UCVolumeReader
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "next_epoch": torch.tensor(0, dtype=torch.int64),
        "global_step": torch.tensor(0, dtype=torch.int64),
        "manifest_sha256": _text_tensor(expected_manifest_sha256),
        "configuration_sha256": _text_tensor(expected_configuration_sha256),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    dcp.load(state, storage_reader=UCVolumeReader(checkpoint_path))
    if _tensor_text(state["manifest_sha256"]) != expected_manifest_sha256:
        raise ValueError("checkpoint belongs to a different dataset manifest")
    if _tensor_text(state["configuration_sha256"]) != expected_configuration_sha256:
        raise ValueError("checkpoint belongs to a different training configuration")
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    return int(state["next_epoch"].item()), int(state["global_step"].item())


def assert_finite_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient in {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(math.prod(parameter.shape) for parameter in model.parameters())
