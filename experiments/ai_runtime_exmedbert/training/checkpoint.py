"""PyTorch DCP helpers backed by Unity Catalog volumes."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import torch
from torch import nn

_CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d{6})_[0-9a-f]+$")


def latest_complete_checkpoint(checkpoint_root: str) -> str | None:
    root = Path(checkpoint_root)
    if not root.exists():
        return None
    candidates = []
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
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    return int(state["next_epoch"].item()), int(state["global_step"].item())


def load_model_for_transfer(
    checkpoint_path: str,
    model: nn.Module,
    *,
    expected_manifest_sha256: str,
) -> None:
    """Load model weights and dataset identity from a DCP checkpoint.

    Optimizer state and training counters are intentionally omitted so an MLM
    checkpoint can initialize a downstream encoder without coupling the two
    training configurations.
    """
    import torch.distributed.checkpoint as dcp
    from serverless_gpu.data import UCVolumeReader
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

    model_state = get_model_state_dict(model)
    state = {
        "model": model_state,
        "manifest_sha256": _text_tensor(expected_manifest_sha256),
    }
    dcp.load(state, storage_reader=UCVolumeReader(checkpoint_path))
    if _tensor_text(state["manifest_sha256"]) != expected_manifest_sha256:
        raise ValueError("pretraining checkpoint belongs to a different dataset manifest")
    set_model_state_dict(model, model_state_dict=state["model"])
