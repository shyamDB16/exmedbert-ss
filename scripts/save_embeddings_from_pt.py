#!/usr/bin/env python
"""
Save embedding dictionaries from an existing PatientDataset .pt file.

Use this to create embeddings.pt for streaming pre-training when you already
have a .pt dataset (e.g. from synthetic data or a previous export).

Usage:
    python scripts/save_embeddings_from_pt.py pretrain_data/train.pt pretrain_data/embeddings.pt
"""

import sys
from pathlib import Path

import torch


def main(dataset_path: str, output_path: str) -> None:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load dataset (weights_only=False for custom classes)
    dataset = torch.load(dataset_path, weights_only=False)

    # Expect PatientDataset or similar with embed attributes
    for key in ("code_embed", "age_embed", "sex_embed", "state_embed", "endpoint_dict"):
        if not hasattr(dataset, key):
            raise AttributeError(f"Dataset has no attribute '{key}'. Is this a PatientDataset .pt file?")

    embeddings = {
        "code_embed": dataset.code_embed,
        "age_embed": dataset.age_embed,
        "sex_embed": dataset.sex_embed,
        "state_embed": dataset.state_embed,
        "endpoint_dict": dataset.endpoint_dict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: save_embeddings_from_pt.py <dataset.pt> <embeddings.pt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
