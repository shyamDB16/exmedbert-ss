# Mosaic Streaming Quick Start

> **Scope: classic/Mosaic Streaming path.** This is retained for comparison and
> rollback. For the recommended serverless GPU workflow, use the
> [AI Runtime project](../experiments/ai_runtime_exmedbert/README.md).

Quick reference for running ExMed-BERT training with Mosaic Streaming.

## Installation

```bash
pip install mosaicml-streaming
```

## Step 1: Convert Data to MDS Format

```bash
python scripts/convert_spark_to_mds.py \
    --input-table "catalog.schema.patients" \
    --output-mds-path "/Volumes/main/exmedbert/data/mds" \
    --embeddings-path "/Volumes/main/exmedbert/data/embeddings" \
    --train-test-split 0.8
```

## Step 2: Train with Streaming

### Local MDS Files

```bash
python scripts/train_entrypoint_streaming.py \
    --local-data-dir "pretrain_data_mds" \
    --embeddings-path "pretrain_data/embeddings.pt"
```

### Unity Catalog Volumes (Databricks)

```bash
# Set authentication
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"

# Run training
python scripts/train_entrypoint_streaming.py \
    --use-uc-volumes \
    --uc-catalog "main" \
    --uc-schema "exmedbert" \
    --uc-volume "training_data"
```

## Python API

```python
from exmed_bert.data.streaming_dataset import create_streaming_datasets
from exmed_bert.models.streaming_trainer import MedBertStreamingTrainer

# Load datasets
train_data, val_data = create_streaming_datasets(
    train_remote="path/to/mds/train",
    val_remote="path/to/mds/val",
    local_base="/tmp/mds_cache",
    embeddings_path="path/to/embeddings.pt",
    batch_size=256,
)

# Train
trainer = MedBertStreamingTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    use_streaming_dataloader=True,
)
trainer.train()
```

## File Structure

After conversion, you should have:

```
/Volumes/main/exmedbert/data/
├── mds_train/              # Training MDS shards
│   ├── index.json
│   ├── shard.00000.mds
│   ├── shard.00001.mds
│   └── ...
├── mds_val/                # Validation MDS shards
│   ├── index.json
│   ├── shard.00000.mds
│   └── ...
└── embeddings.pt           # Vocabulary embeddings
```

## New Files Created

| File | Purpose |
|------|---------|
| `scripts/convert_spark_to_mds.py` | Convert Spark DataFrame to MDS format |
| `scripts/train_entrypoint_streaming.py` | Training entry point for streaming |
| `scripts/pretrain-exmed-bert-streaming.py` | Updated training script |
| `exmed_bert/data/streaming_dataset.py` | StreamingDataset wrapper |
| `exmed_bert/models/streaming_trainer.py` | Enhanced trainer with streaming support |

## Key Parameters

### StreamingPatientDataset

```python
dataset = StreamingPatientDataset(
    remote="path/to/mds",        # Remote MDS location
    local="/tmp/cache",          # Local cache directory
    batch_size=256,              # Batch size (for optimization)
    predownload=256,             # Samples to predownload
    shuffle=True,                # Shuffle data
    shuffle_seed=42,             # Shuffle seed
    dynamic_masking=True,        # Apply dynamic masking
    max_masked=20,               # Max tokens to mask
)
```

### MedBertStreamingTrainer

```python
trainer = MedBertStreamingTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    use_streaming_dataloader=True,  # Enable streaming features
)
```

## Troubleshooting

### Authentication Error (Unity Catalog)

```bash
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"
```

### Slow Startup

Increase predownload:
```python
predownload=512  # Default: 256
```

### Out of Disk Space

Clean cache:
```bash
rm -rf /tmp/mds_cache/*
```

## Migration Checklist

- [ ] Install `mosaicml-streaming`
- [ ] Convert data to MDS format
- [ ] Save embedding dictionaries separately
- [ ] Update import statements in training code
- [ ] Test with small dataset first
- [ ] Configure Databricks auth (if using UC volumes)
- [ ] Run full training
- [ ] Benchmark performance vs. old method

For detailed information, see [MOSAIC_STREAMING_MIGRATION.md](./MOSAIC_STREAMING_MIGRATION.md).

**Minimal GPU test (Databricks or SSH):** See [MINIMAL_GPU_TEST.md](./MINIMAL_GPU_TEST.md) to smoke-test pre-training on a GPU cluster before sharing with customers.

