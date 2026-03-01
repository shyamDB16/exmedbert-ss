# Mosaic Streaming Migration Guide

This guide explains how to migrate ExMed-BERT training from the current `.pt` file-based data loading to Mosaic Streaming for efficient cloud-based data loading.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Migration Steps](#migration-steps)
4. [Architecture Changes](#architecture-changes)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Performance Considerations](#performance-considerations)

---

## Overview

### What is Mosaic Streaming?

Mosaic Streaming is an open-source data loading library that enables efficient streaming from cloud storage (S3, Azure, DBFS, Unity Catalog volumes) to PyTorch training. It provides:

- **Cloud-native data loading**: Stream directly from S3, Azure, Unity Catalog volumes
- **Memory efficiency**: Load data on-demand instead of keeping everything in RAM
- **Fast mid-epoch resumption**: Resume training exactly where you left off
- **Distributed training support**: Optimized for multi-GPU and multi-node training
- **Data integrity**: Built-in hashing and validation

### Why Migrate?

**Benefits:**
- ✅ Eliminate the Spark → `.pt` conversion bottleneck
- ✅ Work with datasets larger than available RAM
- ✅ Faster iteration when data is already in Spark
- ✅ Better integration with Databricks workflows
- ✅ Improved distributed training performance

**When to use:**
- You have a Spark pipeline for data preprocessing
- Your datasets are large (>100GB)
- You're training on Databricks or cloud infrastructure
- You need efficient distributed training

**When NOT to use:**
- Small datasets that fit in memory (<10GB)
- Local training on a single machine
- No existing Spark pipeline

---

## Prerequisites

### Software Requirements

```bash
# Mosaic Streaming (included in Databricks Runtime 15.2 ML+)
pip install mosaicml-streaming

# Existing ExMed-BERT dependencies
pip install -r requirements.txt
```

### Databricks Setup (for Unity Catalog Volumes)

If you plan to use Unity Catalog volumes for data storage:

```python
# Set these environment variables
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-access-token"
```

**How to get a token:**
1. Go to your Databricks workspace
2. Click User Settings → Developer
3. Click "Manage" next to "Access tokens"
4. Generate a new token

---

## Migration Steps

### Step 1: Convert Spark DataFrame to MDS Format

The first step is to convert your Spark-based patient data into MDS (Mosaic Data Shard) format.

**Option A: Using the conversion script**

```python
# Run the conversion script
python scripts/convert_spark_to_mds.py \
    --input-table "catalog.schema.patients" \
    --output-mds-path "/Volumes/main/exmedbert/training_data/mds_train" \
    --embeddings-path "/Volumes/main/exmedbert/training_data/embeddings" \
    --train-test-split 0.8
```

**Option B: Custom Spark job**

```python
from pyspark.sql import SparkSession
from streaming.base.converters import dataframe_to_mds

# Load your patient data
spark = SparkSession.builder.appName("ExMedBert-MDS-Conversion").getOrCreate()
df = spark.table("catalog.schema.patients")

# Define schema for MDS
columns = {
    "patient_id": "str",
    "codes": "ndarray:int32",
    "ages": "ndarray:int32",
    "visits": "ndarray:int32",
    "segments": "ndarray:int32",
    "sex": "int32",
    "birth_year": "int32",
    "state": "int32",
    "code_labels": "ndarray:int32",
    "code_mask": "ndarray:bool",
    "plos_label": "float32",
    "endpoint_labels": "ndarray:float32",
    "unpadded_length": "int32",
}

# Convert to MDS
dataframe_to_mds(
    df=df,
    merge_index=True,
    mds_kwargs={
        "out": "/Volumes/main/exmedbert/training_data/mds_train",
        "columns": columns,
        "compression": "zstd",
        "hashes": ["sha256"],
        "size_limit": "100mb",
    }
)
```

**Important:** Save your embedding dictionaries separately:

```python
import torch

embeddings = {
    "code_embed": code_embed,
    "age_embed": age_embed,
    "sex_embed": sex_embed,
    "state_embed": state_embed,
    "endpoint_dict": endpoint_dict,
}

torch.save(embeddings, "/Volumes/main/exmedbert/training_data/embeddings.pt")
```

### Step 2: Update Training Code

Replace the old data loading code with the new streaming version.

**Old code (pretrain-exmed-bert-clinvec.py):**

```python
from exmed_bert.data.dataset import PatientDataset

train_data = PatientDataset.load_dataset("pretrain_data/train.pt")
val_data = PatientDataset.load_dataset("pretrain_data/val.pt")
```

**New code (pretrain-exmed-bert-streaming.py):**

```python
from exmed_bert.data.streaming_dataset import create_streaming_datasets

train_data, val_data = create_streaming_datasets(
    train_remote="/Volumes/main/exmedbert/training_data/mds_train",
    val_remote="/Volumes/main/exmedbert/training_data/mds_val",
    local_base="/tmp/mds_cache",
    embeddings_path="/Volumes/main/exmedbert/training_data/embeddings.pt",
    batch_size=256,
    dynamic_masking=True,
)
```

### Step 3: Update Trainer

Use the streaming-aware trainer for better checkpoint support.

**Old code:**

```python
from exmed_bert.models.trainer import MedBertTrainer

trainer = MedBertTrainer(
    model=model,
    args=training_config,
    train_dataset=train_data,
    eval_dataset=val_data,
)
```

**New code:**

```python
from exmed_bert.models.streaming_trainer import MedBertStreamingTrainer

trainer = MedBertStreamingTrainer(
    model=model,
    args=training_config,
    train_dataset=train_data,
    eval_dataset=val_data,
    use_streaming_dataloader=True,  # Enable streaming features
)
```

### Step 4: Run Training

Use the new streaming entry point:

```bash
# For local MDS data
python scripts/train_entrypoint_streaming.py \
    --local-data-dir "pretrain_data_mds" \
    --embeddings-path "pretrain_data/embeddings.pt"

# For Unity Catalog volumes
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"

python scripts/train_entrypoint_streaming.py \
    --use-uc-volumes \
    --uc-catalog "main" \
    --uc-schema "exmedbert" \
    --uc-volume "training_data"
```

---

## Architecture Changes

### Before: File-Based Loading

```
┌─────────────────┐
│  Spark Pipeline │
└────────┬────────┘
         │
         ├─ Generate synthetic patients
         ├─ Apply preprocessing
         ├─ Create train/val splits
         │
         ▼
┌──────────────────────┐
│  Save to .pt files   │
│  (torch.save)        │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  PatientDataset      │
│  .load_dataset()     │
│  (loads entire file) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  PyTorch DataLoader  │
└──────────────────────┘
```

### After: Streaming-Based Loading

```
┌─────────────────┐
│  Spark Pipeline │
└────────┬────────┘
         │
         ├─ Generate synthetic patients
         ├─ Apply preprocessing
         ├─ Create train/val splits
         │
         ▼
┌─────────────────────────────┐
│  Convert to MDS format      │
│  (dataframe_to_mds)         │
│  - Sharded files            │
│  - Compressed (zstd)        │
│  - Indexed for fast access  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Save to Cloud Storage      │
│  - S3                       │
│  - Unity Catalog Volumes    │
│  - DBFS                     │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  StreamingPatientDataset    │
│  - Streams from cloud       │
│  - Local cache              │
│  - On-demand loading        │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  StreamingDataLoader        │
│  - Checkpoint support       │
│  - Deterministic shuffling  │
└─────────────────────────────┘
```

### Key Differences

| Aspect | Old (File-Based) | New (Streaming) |
|--------|------------------|-----------------|
| **Data Storage** | Single `.pt` files | Sharded MDS files |
| **Loading** | Load entire dataset | Stream on-demand |
| **Memory** | Must fit in RAM | Minimal memory footprint |
| **Startup** | Slow (load all data) | Fast (index only) |
| **Distributed** | Manual sharding | Automatic sharding |
| **Resumption** | Approximate | Exact mid-epoch |
| **Cloud Integration** | Copy files locally | Direct streaming |

---

## Usage Examples

### Example 1: Local Training with MDS Files

```python
from exmed_bert.data.streaming_dataset import StreamingPatientDataset
from exmed_bert.models.streaming_trainer import MedBertStreamingTrainer

# Load embeddings
embeddings_path = "pretrain_data/embeddings.pt"

# Create dataset
train_dataset = StreamingPatientDataset.from_embeddings_file(
    remote="pretrain_data_mds/train",
    local="/tmp/mds_cache/train",
    embeddings_path=embeddings_path,
    batch_size=256,
    shuffle=True,
    dynamic_masking=True,
)

# Train as usual
trainer = MedBertStreamingTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    use_streaming_dataloader=True,
)
trainer.train()
```

### Example 2: Databricks with Unity Catalog Volumes

```python
# Set authentication
import os
os.environ["DATABRICKS_HOST"] = "https://your-workspace.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "your-token"

# Load from UC volume
train_dataset = StreamingPatientDataset.from_embeddings_file(
    remote="/Volumes/main/exmedbert/training_data/mds_train",
    local="/local_disk0/mds_cache/train",
    embeddings_path="/Volumes/main/exmedbert/training_data/embeddings.pt",
    batch_size=256,
)
```

### Example 3: Multi-GPU Training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="output/streaming_pretrain",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    # Streaming-specific optimizations
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,  # Important!
)

trainer = MedBertStreamingTrainer(
    model=model,
    args=training_args,
    train_dataset=streaming_train_dataset,
    eval_dataset=streaming_val_dataset,
    use_streaming_dataloader=True,
)

# Launch with torchrun or accelerate
trainer.train()
```

---

## Troubleshooting

### Issue 1: Authentication Error with Unity Catalog

**Error:**
```
ValueError: default auth: cannot configure default credentials
```

**Solution:**
```bash
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"

# For distributed training, set on all worker nodes
# Or pass as arguments:
python pretrain-exmed-bert-streaming.py \
    --databricks-host "https://..." \
    --databricks-token "your-token" \
    ...
```

### Issue 2: Python 3.11 Shared Memory Issues

**Error:** Transient shared memory errors on Databricks Runtime 15.4 LTS ML

**Solution:** Upgrade to Databricks Runtime 16.4 LTS ML (Python 3.12)

### Issue 3: Slow Startup

**Problem:** Training takes a long time to start

**Solutions:**
- Increase `predownload` parameter (default: 256)
- Use local storage for cache (e.g., `/local_disk0/` on Databricks)
- Enable compression (`compression="zstd"`) during MDS conversion

```python
train_dataset = StreamingPatientDataset(
    remote=remote_path,
    local=local_cache,
    predownload=512,  # Increase predownload
    ...
)
```

### Issue 4: Out of Disk Space

**Problem:** Local cache fills up disk

**Solutions:**
- Use a larger disk volume
- Reduce the local cache size (streams more frequently)
- Clean up old cache periodically

```python
import shutil
shutil.rmtree("/tmp/mds_cache", ignore_errors=True)
```

### Issue 5: Data Mismatch Errors

**Problem:** Data format doesn't match expected schema

**Solution:** Ensure your Spark DataFrame schema matches the expected format:

```python
# Check your DataFrame schema
df.printSchema()

# Expected columns:
# - codes: array<int>
# - ages: array<int>
# - visits: array<int>
# - segments: array<int>
# - sex: int
# - state: int
# - code_labels: array<int>
# - code_mask: array<bool>
# - plos_label: float
# - endpoint_labels: array<float>
```

---

## Performance Considerations

### Memory Usage

| Configuration | Memory Usage (Train Dataset) |
|---------------|------------------------------|
| Old (.pt file, 100k patients) | ~20GB RAM |
| New (Streaming, 100k patients) | ~2GB RAM |

### Training Speed

Streaming performance is comparable to file-based loading after the initial cache warmup:

- **First epoch:** May be 5-10% slower (downloading shards)
- **Subsequent epochs:** Within 1-2% of file-based performance
- **Multi-GPU:** Streaming often faster due to better load balancing

### Optimization Tips

1. **Shard Size:** Use 50-100MB shards for optimal performance
   ```python
   mds_kwargs={"size_limit": "100mb"}
   ```

2. **Compression:** zstd provides best compression/speed tradeoff
   ```python
   mds_kwargs={"compression": "zstd"}
   ```

3. **Predownload:** Tune based on batch size and network speed
   ```python
   predownload=batch_size * 4  # Download 4 batches ahead
   ```

4. **Local Cache:** Use fast local storage (NVMe SSD > HDD)
   ```python
   local="/local_disk0/mds_cache"  # Fast local disk on Databricks
   ```

5. **Workers:** Use dataloader workers for better throughput
   ```python
   TrainingArguments(
       dataloader_num_workers=4,  # Adjust based on CPU count
       dataloader_pin_memory=True,
   )
   ```

---

## Backwards Compatibility

The streaming implementation maintains API compatibility with the original `PatientDataset`:

```python
# Old code still works
from exmed_bert.data.dataset import PatientDataset
dataset = PatientDataset.load_dataset("train.pt")

# New code has similar interface
from exmed_bert.data.streaming_dataset import StreamingPatientDataset
dataset = StreamingPatientDataset.from_embeddings_file(
    remote="mds_train",
    local="/tmp/cache",
    embeddings_path="embeddings.pt"
)

# Both expose the same methods:
# - __len__(), __getitem__()
# - .endpoints(selected_endpoints)
# - .code_embed, .age_embed, etc.
```

---

## Next Steps

1. **Test with small dataset:** Start with a small subset to verify the pipeline
2. **Benchmark performance:** Compare training speed with your existing setup
3. **Update CI/CD:** Integrate MDS conversion into your data pipeline
4. **Monitor costs:** Track storage and compute costs in production

For questions or issues, refer to:
- [Mosaic Streaming Documentation](https://docs.mosaicml.com/projects/streaming/)
- [Databricks ML Documentation](https://docs.databricks.com/machine-learning/)
- [ExMed-BERT Documentation](./README.md)

