# AI Runtime Data Loading and Checkpointing Architecture

## Purpose

This document defines an AI Runtime-native alternative to Mosaic Streaming for
ExMed-BERT. It covers the boundary between Spark preprocessing and distributed
PyTorch training, file-based ingestion from Unity Catalog volumes, and resilient
checkpointing. The single-GPU transport and epoch-recovery path has now been
implemented and verified; see
[the Synthea experiment](../experiments/ai_runtime_synthea/README.md) and its
[workspace results](../experiments/ai_runtime_synthea/RESULTS.md).
The subsequent Spark preparation and full `ExMedBertForMaskedLM` port are also
verified in [the ExMed-BERT AIR project](../experiments/ai_runtime_exmedbert/README.md).

The central design decision is:

> Do not pass a Spark DataFrame directly into distributed training. Use Spark to
> create an immutable, versioned training snapshot, then let AI Runtime stream
> that snapshot from a Unity Catalog volume.

This separates two workloads with different execution characteristics:

- Spark performs joins, filtering, code normalization, sequence construction,
  train/validation splitting, and large-scale feature engineering.
- AI Runtime performs GPU-local decoding, batching, distributed training, and
  checkpointing.

## Scope and current state

The current streaming path converts a Spark DataFrame to Mosaic Data Shard
(MDS) files with `scripts/convert_spark_to_mds.py`. Training then uses
`StreamingPatientDataset`, `StreamingDataLoader`, and
`MedBertStreamingTrainer`.

Mosaic Streaming currently provides several capabilities together:

- Remote file streaming and local caching.
- Rank-aware dataset partitioning.
- Shuffling and prefetching.
- Dataloader state used for mid-epoch resume.
- MDS serialization and shard indexing.

An AI Runtime-native design replaces these capabilities with more than one
component:

| Current responsibility | AI Runtime-native component |
|---|---|
| Spark-to-training serialization | Versioned Parquet shards in a UC volume |
| File caching and rank/worker partitioning | `serverless_gpu.data.UCVolumeDataset` |
| Concurrent loading and prefetching | `serverless_gpu.data.DataLoader` |
| Model and optimizer checkpoints | PyTorch Distributed Checkpoint (DCP) with `UCVolumeWriter` and `UCVolumeReader` |
| Dataloader position and random state | Explicit training-state checkpoint managed by ExMed-BERT |

`UCVolumeWriter` and `UCVolumeReader` replace checkpoint storage. They do not,
on their own, capture the current position of the training data pipeline.

### Current platform constraints

As of September 2026, the relevant AI Runtime capabilities are in Preview or
Beta and should be revalidated before a production migration:

- AI Runtime data access goes through Unity Catalog. Source tables and exported
  files must be registered in or governed by Unity Catalog and accessible to the
  run identity.
- `UCVolumeDataset`, `serverless_gpu.data.DataLoader`, `UCVolumeWriter`, and
  `UCVolumeReader` require GPU environment 5 or above. These APIs were verified
  with `serverless_gpu 0.5.23+patch1` on numeric environment version `"5"`.
- Distributed notebook training with `@distributed` currently uses a single
  8xH100 node. It is not a general replacement for classic multi-node training.
- The native distributed API supports DDP, FSDP, and DeepSpeed. DDP is the
  appropriate starting point for the current ExMed-BERT model because it fits
  on one GPU; FSDP or DeepSpeed would add complexity without a memory need.

## Recommended architecture

```text
Source Delta tables
        |
        v
Spark preprocessing job
  - normalize clinical codes
  - build patient sequences
  - create stable train/validation/test splits
  - validate the training schema
        |
        v
Immutable dataset snapshot in a UC volume
  - manifest.json
  - train/*.parquet
  - validation/*.parquet
  - test/*.parquet
  - vocabulary/config artifacts
        |
        v
AI Runtime training process
  - UCVolumeDataset
  - Parquet-decoding IterableDataset
  - serverless_gpu.data.DataLoader
  - single-GPU PyTorch first; DDP or Ray Train after sharding tests
        |
        v
UC volume checkpoints
  - DCP model and optimizer state
  - dataset position, epoch, step, and RNG state
```

### Why materialize a snapshot?

A Spark DataFrame is a lazy query plan rather than a portable training dataset.
Reading it independently from every GPU rank can repeat Spark work, complicate
partition ownership, and introduce a CPU/network bottleneck during training.
Large DataFrames also cannot be serialized into an `@distributed` function.

An immutable export provides:

- Reproducibility: each run references one dataset version.
- A stable boundary between CPU preprocessing and GPU training.
- Efficient rank-aware file distribution and local caching.
- Independent scaling and retry behavior for preprocessing and training.
- Auditable lineage from source Delta versions to model runs.

The additional Parquet copy is intentional. It is a training-optimized snapshot,
not the system of record.

## Dataset snapshot contract

Each Parquet row should represent one model-ready patient sequence. The initial
schema should preserve the current ExMed-BERT batch contract:

| Column | Type | Description |
|---|---|---|
| `patient_id` | string | Stable, de-identified patient identifier |
| `codes` | array&lt;int&gt; | Tokenized medical-code sequence |
| `ages` | array&lt;int&gt; | Age token at each sequence position |
| `visits` | array&lt;int&gt; | Visit index at each sequence position |
| `segments` | array&lt;int&gt; | Segment/type token at each position |
| `sex` | int | Tokenized sex value |
| `state` | int | Tokenized region/state value |
| `code_labels` | array&lt;int&gt; | Masked-language-model labels; `-100` means ignored |
| `code_mask` | array&lt;boolean&gt; | Positions selected for code prediction |
| `plos_label` | float | Prolonged-length-of-stay target |
| `endpoint_labels` | array&lt;float&gt; | Optional downstream endpoint labels |
| `unpadded_length` | int | Sequence length before padding |

Where practical, dynamic masking should happen during training rather than in
the Spark export. This permits a patient to receive different masks across
epochs. The snapshot can then omit `code_labels` and `code_mask`, or retain them
only for deterministic evaluation.

### Manifest

Every export should include a manifest containing at least:

- Dataset name and immutable version.
- Creation timestamp and producing job/run ID.
- Source table names and Delta versions or timestamps.
- Schema version.
- Split algorithm and seed.
- Vocabulary version and artifact paths.
- Number of patients, samples, and files per split.
- Sequence-length statistics.
- Optional file checksums.

MLflow should log the manifest and dataset version with each training run.

## Spark export

The existing `prepare_patient_dataframe` transformation remains useful. Replace
the MDS-specific conversion with a versioned Parquet export to a UC volume.

Illustrative Spark code:

```python
from pyspark.sql import functions as F

dataset_version = "2026-09-10-v1"
output_root = (
    "/Volumes/<catalog>/<schema>/<volume>/"
    f"exmedbert/datasets/{dataset_version}"
)

prepared = prepare_patient_dataframe(spark.table("<catalog>.<schema>.<table>"))

# Prefer a stable patient-level split over randomSplit so reruns assign a
# patient to the same split.
prepared = prepared.withColumn(
    "split_bucket",
    F.pmod(F.xxhash64("patient_id"), F.lit(1000)),
)

train = prepared.where(F.col("split_bucket") < 800).drop("split_bucket")
validation = prepared.where(
    (F.col("split_bucket") >= 800) & (F.col("split_bucket") < 900)
).drop("split_bucket")
test = prepared.where(F.col("split_bucket") >= 900).drop("split_bucket")

for name, frame in {
    "train": train,
    "validation": validation,
    "test": test,
}.items():
    (
        frame
        # Tune this using measured row sizes and desired file size.
        .repartition(<number_of_output_files>)
        .write
        .mode("errorifexists")
        .parquet(f"{output_root}/{name}")
    )
```

Avoid very small files. A reasonable initial target is approximately 256-512 MB
per Parquet file, followed by measurement on representative patient sequences.
The export job should fail rather than overwrite an existing immutable version.

## AI Runtime data loading

`UCVolumeDataset` yields locally cached file paths. It does not decode Parquet
rows into model inputs. ExMed-BERT therefore needs a small `IterableDataset`
adapter that:

1. Consumes each local path immediately.
2. Opens the Parquet file or row groups with PyArrow.
3. Converts rows to the tensor dictionary expected by the model.
4. Applies dynamic masking and other per-epoch transformations.
5. Yields samples to `serverless_gpu.data.DataLoader`.

Illustrative structure:

```python
from serverless_gpu.data import DataLoader, UCVolumeDataset
from torch.utils.data import IterableDataset


class ExMedBertParquetDataset(IterableDataset):
    def __init__(self, path_dataset, vocabulary, dynamic_masking=True):
        self.path_dataset = path_dataset
        self.vocabulary = vocabulary
        self.dynamic_masking = dynamic_masking

    def __iter__(self):
        for local_path in self.path_dataset:
            # Open and consume the cached path in this iteration. Paths are
            # ephemeral and may be evicted after iteration advances.
            yield from decode_parquet_file(
                local_path,
                vocabulary=self.vocabulary,
                dynamic_masking=self.dynamic_masking,
            )


paths = UCVolumeDataset(
    "/Volumes/<catalog>/<schema>/<volume>/exmedbert/datasets/<version>/train"
)
dataset = ExMedBertParquetDataset(paths, vocabulary)
loader = DataLoader(dataset, batch_size=..., num_workers=...)
```

Construct the datasets and dataloaders inside the `@distributed` function. Once
`torch.distributed` is initialized, `UCVolumeDataset` partitions files across
ranks and workers. Do not add a second `DistributedSampler` around it.

All ranks must use the same `num_workers`; otherwise files can be duplicated or
skipped. Start with the AI Runtime loader defaults and tune workers, prefetching,
and batch size using measured GPU utilization and batch-fetch timing. Worker
counts tuned for classic clusters should not be copied without remeasurement.

### Why not point `UCVolumeDataset` at MDS files?

An MDS dataset contains coordinated index, shard, and optional compression files.
`UCVolumeDataset` partitions individual files, while the MDS reader expects to
control shard discovery and indexing itself. Combining the two would preserve
Mosaic dependencies without preserving a clear ownership model.

For the native path, export self-contained Parquet or Arrow shards. Keep the
existing Mosaic path available during migration for comparison and rollback.

## When direct Spark Connect is appropriate

AI Runtime can read Delta tables with Spark Connect. This is appropriate for:

- Vocabulary and dataset metadata queries.
- Small smoke-test datasets.
- Validation samples that fit safely in pandas or NumPy memory.
- Interactive exploration before creating a snapshot.

It is not the preferred main path for large distributed pre-training. Converting
a large Spark DataFrame to pandas centralizes the data, while having every GPU
rank issue its own Spark query makes partitioning and retry semantics difficult.

For data that is too large to materialize as one in-memory object, create the
snapshot in a preceding Spark job task and make training depend on successful
completion of that task.

## Distributed checkpointing

Use PyTorch Distributed Checkpoint rather than `torch.save` for distributed
model and optimizer state. On AI Runtime, use:

- `serverless_gpu.data.UCVolumeWriter` when saving.
- `serverless_gpu.data.UCVolumeReader` when restoring.
- `torch.distributed.checkpoint.async_save` where appropriate to overlap upload
  with training.

Checkpoints should be written to versioned step directories:

```text
/Volumes/<catalog>/<schema>/<volume>/exmedbert/checkpoints/<run-id>/
  step_000001000/
  step_000002000/
  ...
```

A checkpoint is valid only after DCP metadata has been published. Resume logic
should select the newest complete checkpoint and ignore partially written ones.

### Training state required for exact resume

Model and optimizer state are insufficient for a correct mid-epoch resume. Save
the following state alongside them:

- Dataset version and manifest path.
- Global step and epoch.
- Number of globally consumed samples.
- Completed shard IDs and, if needed, the row offset in the current shard.
- Python, NumPy, CPU Torch, and CUDA RNG state.
- Dynamic-masking RNG state.
- Scheduler and gradient-scaler state, when used.

There are three possible resume guarantees:

1. **Epoch-boundary resume:** simplest, but can lose up to one epoch of work.
2. **Shard-boundary resume:** record completed Parquet shards; a practical first
   resilient implementation.
3. **Exact sample resume:** record shard and row offsets plus all relevant RNG
   states; closest to the current Mosaic dataloader checkpoint behavior.

Shard-boundary resume is the recommended first implementation. Exact sample
resume can follow if failure-recovery tests show that the extra complexity is
needed.

## MLflow and rank ownership

Calling an `@distributed` function creates an MLflow run automatically. Avoid
opening an independent top-level run from every GPU process.

The implementation should:

- Configure the parent experiment before calling `.distributed()`.
- Log high-frequency metrics only from the intended process or through a
  distributed-aware integration.
- Restrict final model logging and Unity Catalog model registration to global
  rank 0.
- Log dataset version, manifest, effective global batch size, world size, and
  checkpoint location.

## Migration plan

### Phase 1: Establish a reproducible Spark boundary

- Define and validate the patient-sequence schema.
- Replace `randomSplit` with a stable patient-level split.
- Add versioned Parquet export and manifest generation.
- Retain the MDS export path for comparison.

### Phase 2: Implement native AI Runtime loading

- Add the Parquet-decoding `IterableDataset`.
- Integrate `UCVolumeDataset` and `serverless_gpu.data.DataLoader`.
- Run single-GPU correctness tests against the existing Mosaic dataset.
- Validate that every sample is consumed exactly once across eight ranks.

### Phase 3: Add distributed execution and checkpoints

- Add the `@distributed(gpus=8)` DDP entrypoint.
- Add DCP save and restore with UC volume storage.
- Implement shard-boundary data-position checkpoints.
- Make MLflow logging, model saving, and registration rank-safe.

### Phase 4: Benchmark and decide

Compare Mosaic and the native path using the same dataset snapshot:

- Samples per second and time to first batch.
- GPU utilization and GPU memory.
- Data-fetch latency and local-cache behavior.
- Checkpoint duration and recovery time.
- Correctness after interruption and resume.
- Operational complexity and dependency footprint.

Do not remove Mosaic Streaming until the native path meets correctness and
recovery requirements.

## Open design questions

- Should the Parquet snapshot contain pre-tokenized arrays or normalized raw
  codes tokenized by the training adapter?
- Is shard-boundary resume sufficient for expected job duration and failure
  rate?
- What target shard size provides the best trade-off between parallelism,
  metadata overhead, and recovery granularity?
- Should dynamic masking use one deterministic seed per `(epoch, patient_id)` to
  simplify exact resume?
- How should vocabulary artifacts be represented long-term instead of pickled
  Python objects in `embeddings.pt`?

## References

- [Distributed training in notebooks](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training)
- [Load data on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading)
- [Experiment tracking and observability](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/tracking-observability)
- [Improve training performance and resiliency on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/guides/performance-and-resiliency)
