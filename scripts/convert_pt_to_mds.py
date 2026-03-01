#!/usr/bin/env python
"""
Convert a PatientDataset .pt file to Mosaic Data Shard (MDS) format for streaming pre-training.

Use this to create minimal MDS data from dummy or existing .pt datasets without Spark,
so you can run the streaming smoke test (e.g. with --max-steps 5).

Usage:
    python scripts/convert_pt_to_mds.py \\
      --train-pt pretrain_data/dummy_train.pt \\
      --val-pt pretrain_data/dummy_val.pt \\
      --output-dir pretrain_data_mds

Then run streaming pretrain with:
    python scripts/pretrain-exmed-bert-streaming.py \\
      pretrain_data_mds/train pretrain_data_mds/val \\
      pretrain_data/embeddings.pt output/minimal_streaming output/minimal_streaming_data \\
      --local-cache-dir /tmp/mds_cache --max-steps 5 --no-register-model
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MDS column schema (must match convert_spark_to_mds and StreamingPatientDataset)
MDS_COLUMNS = {
    "patient_id": "str",
    "codes": "ndarray:int32",
    "ages": "ndarray:int32",
    "visits": "ndarray:int32",
    "segments": "ndarray:int32",
    "sex": "int32",
    "birth_year": "int32",
    "state": "int32",
    "code_labels": "ndarray:int32",
    "code_mask": "ndarray:uint8",
    "plos_label": "float32",
    "endpoint_labels": "ndarray:float32",
    "unpadded_length": "int32",
}


def sample_to_mds_row(idx: int, sample: dict, num_endpoints: int) -> dict:
    """Map PatientDataset __getitem__ output to MDS row dict."""
    def to_np(t, dtype=None):
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        if dtype is not None:
            t = np.asarray(t, dtype=dtype)
        return t

    codes = to_np(sample["input_ids"], np.int32)
    ages = to_np(sample["age_ids"], np.int32)
    visits = to_np(sample["position_ids"], np.int32)
    segments = to_np(sample["entity_ids"], np.int32)
    sex = int(sample["sex_ids"].flatten()[0].item()) if sample["sex_ids"].numel() else 0
    state = int(sample["state_ids"].flatten()[0].item()) if sample["state_ids"].numel() else 0
    code_labels = to_np(sample["code_labels"], np.int32)
    code_mask = (code_labels != -100).astype(np.uint8)
    plos_label = float(sample["plos_label"].item()) if sample["plos_label"].numel() else 0.0
    attention_mask = sample.get("attention_mask")
    if attention_mask is None:
        unpadded_length = int((codes != 0).sum())
    else:
        unpadded_length = int(to_np(attention_mask).sum())

    return {
        "patient_id": str(idx),
        "codes": codes,
        "ages": ages,
        "visits": visits,
        "segments": segments,
        "sex": np.int32(sex),
        "birth_year": np.int32(0),
        "state": np.int32(state),
        "code_labels": code_labels,
        "code_mask": code_mask,
        "plos_label": np.float32(plos_label),
        "endpoint_labels": np.zeros(num_endpoints, dtype=np.float32),
        "unpadded_length": np.int32(unpadded_length),
    }


def write_pt_to_mds(dataset_path: str, output_path: str, compression: str = "zstd") -> None:
    try:
        from streaming.base import MDSWriter
    except ImportError:
        raise ImportError("mosaicml-streaming is required. Install with: pip install mosaicml-streaming")

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    dataset = torch.load(dataset_path, weights_only=False)
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError("Expected a Dataset with __len__ and __getitem__")

    ep_dict = getattr(dataset, "endpoint_dict", None)
    num_endpoints = max(1, len(ep_dict.endpoints)) if ep_dict is not None else 1

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Writing MDS shards to %s (%d samples)", output_path, len(dataset))

    with MDSWriter(
        out=str(output_path),
        columns=MDS_COLUMNS,
        compression=compression,
        size_limit="64MB",
    ) as writer:
        for i in range(len(dataset)):
            sample = dataset[i]
            row = sample_to_mds_row(i, sample, num_endpoints)
            writer.write(row)

    logger.info("Wrote %s", output_path)


def main():
    p = argparse.ArgumentParser(description="Convert .pt PatientDataset to MDS for streaming pretrain")
    p.add_argument("--train-pt", required=True, help="Path to training .pt file")
    p.add_argument("--val-pt", required=True, help="Path to validation .pt file")
    p.add_argument("--output-dir", default="pretrain_data_mds", help="Output directory (train/ and val/ created under it)")
    p.add_argument("--compression", default="zstd", help="MDS compression (e.g. zstd)")
    args = p.parse_args()

    out = Path(args.output_dir)
    write_pt_to_mds(args.train_pt, str(out / "train"), compression=args.compression)
    write_pt_to_mds(args.val_pt, str(out / "val"), compression=args.compression)
    logger.info("Done. Train: %s/train  Val: %s/val", out, out)


if __name__ == "__main__":
    main()
