# AI Runtime Synthea GPU smoke test

This experiment is an implemented and workspace-verified proof of concept for
moving ExMed-BERT training from classic-cluster/Mosaic Streaming assumptions to
Databricks AI Runtime.

It proves the narrow infrastructure path first:

1. Convert Synthea diagnosis events into a deterministic Parquet snapshot.
2. Store the immutable files and manifest in a Unity Catalog volume.
3. Stream and locally cache files with `UCVolumeDataset`.
4. Train a small masked-code BERT model on an AI Runtime GPU.
5. Save and restore complete PyTorch DCP checkpoints through
   `UCVolumeWriter` and `UCVolumeReader`.
6. Resume from the latest complete epoch in a separate AIR process.

The successful runs and metrics are recorded in [RESULTS.md](RESULTS.md). This
is an engineering validation, not a clinical-quality result or a port of the
full ExMed-BERT architecture.

For the broader Mosaic-to-AI-Runtime design, including the Spark boundary, see
[AI Runtime data loading and checkpointing](../../docs/AI_RUNTIME_DATA_AND_CHECKPOINTING.md).

## Validated architecture

```text
Synthea CSVs
    |
    | snapshot.py: join, validate, split, train-only vocabulary
    v
Immutable UC volume snapshot
    |-- manifest.json
    |-- vocabulary.json
    |-- train/patients.parquet
    `-- validation/patients.parquet
             |
             | UCVolumeDataset + local cache
             | PyArrow row decoding + native DataLoader
             v
      Tiny masked-code BERT on one A10 GPU
             |
             | PyTorch Distributed Checkpoint
             v
      UCVolumeWriter / UCVolumeReader
             |
             `-- fresh-object and fresh-process restore tests
```

Spark remains the appropriate place for large source-table joins, feature
engineering, sequence construction, and snapshot publication. A Spark
DataFrame is a lazy execution plan, so it should not be passed into or
independently queried by every distributed GPU worker. Materialize a versioned
Parquet/Arrow snapshot in a UC volume, then make that snapshot the reproducible
input to training.

## Files

| File | Purpose |
|---|---|
| `snapshot.py` | Deterministic Synthea CSV-to-Parquet snapshot builder |
| `runtime.py` | Parquet adapter, deterministic MLM collator, tiny model, and DCP helpers |
| `train.py` | CUDA-only training, coverage checks, MLflow logging, checkpoint and recovery probes |
| `train.yaml` | AIR CLI workload definition for one A10 GPU |
| `tests/test_snapshot.py` | Local deterministic transformation tests |
| `RESULTS.md` | Workspace evidence, metrics, paths, and limitations |

## Current workspace configuration

The checked-in YAML points to the validated development resources:

```text
Profile:          fevm-default
Snapshot:         /Volumes/shyam_internal_ssa_catalog/deep_learning/exmedbert_air/synthea/snapshots/v1
Checkpoint root:  /Volumes/shyam_internal_ssa_catalog/deep_learning/exmedbert_air/synthea/checkpoints/v1
```

Change these values before using another workspace. Do not overwrite an
existing snapshot version or reuse a checkpoint root with a different dataset
manifest or training configuration.

## Snapshot contract

Each Parquet row has this minimal schema:

| Column | Type | Meaning |
|---|---|---|
| `patient_id` | string | Stable synthetic patient identifier |
| `codes` | array<int32> | Diagnosis token IDs from an uncorrupted sequence |
| `time_gaps` | array<int8> | Past-to-current encounter gap bucket for each code |

The builder:

- joins conditions to encounters with patient/encounter validation;
- preserves source code namespaces such as `SNOMED:<code>`;
- deduplicates codes within an encounter and uses stable chronological ordering;
- retains at most 128 recent diagnosis events per patient;
- uses a deterministic patient-level train/validation split;
- builds the vocabulary from training patients only, reserving PAD/MASK/UNK;
- records source and output SHA-256 hashes and all counts in `manifest.json`;
- writes to a staging directory and publishes the snapshot atomically.

Build a local snapshot with Python and PyArrow available:

```bash
python snapshot.py \
  --conditions /path/to/synthea/conditions.csv \
  --encounters /path/to/synthea/encounters.csv \
  --output /path/to/new/snapshot/v1 \
  --version v1
```

Publish that complete directory into a new UC volume version. For a production
Spark pipeline, implement the same contract with stable patient splits and
multiple appropriately sized, self-contained Parquet shards. Keep Spark commit
and staging files outside the directories passed to `UCVolumeDataset`.

## Run with the AIR CLI

Prerequisites:

- Databricks CLI and AIR CLI installed and authenticated.
- Access to the configured UC volume.
- AI Runtime GPU environment 5 or later.
- The snapshot already published at the YAML's `--snapshot-root`.

The workspace used for this test does not expose the managed
`databricks_ai_v5` base environment. The validated YAML therefore uses numeric
environment version `"5"` and adds only `torch`, `mlflow>=3.7.0`, and
`pyarrow>=14.0.0`.

Run from this experiment directory because the YAML packages `root_path: .`:

```bash
cd experiments/ai_runtime_synthea

COPYFILE_DISABLE=1 air run \
  --profile fevm-default \
  --file train.yaml \
  --dry-run

COPYFILE_DISABLE=1 air run \
  --profile fevm-default \
  --file train.yaml \
  --watch
```

The default execution trains through epoch 2. To demonstrate a later process
resuming from that checkpoint and training only epoch 3:

```bash
COPYFILE_DISABLE=1 air run \
  --profile fevm-default \
  --file train.yaml \
  --override env_variables.EPOCHS=3 \
  --watch
```

`train.py` enables resume by default. It selects only checkpoint directories
matching `epoch_<number>_<attempt-id>` that contain DCP `.metadata`, validates
the dataset/configuration hashes, restores model/optimizer/RNG/counters, and
continues at `next_epoch`.

Useful inspection commands:

```bash
air get run --profile fevm-default <run-id>
air logs --profile fevm-default <run-id>
```

### macOS AIR CLI 1.2 packaging notes

- Set `COPYFILE_DISABLE=1` for `air run` so macOS does not add AppleDouble
  `._*` entries to the source snapshot.
- AIR CLI 1.2 can invoke GNU-tar-only exclusion flags while the host provides
  BSD tar. Remove generated `.pytest_cache` and `__pycache__` directories from
  the packaged source tree before launch if packaging fails on ignored files.
- `LIBARCHIVE.xattr.com.apple.provenance` warnings may still appear during
  extraction; they were harmless in the successful runs.

With `num_workers=0`, keep `prefetch_factor=None`; PyTorch rejects a non-`None`
prefetch factor when multiprocessing workers are disabled.

## Correctness checks performed by training

- Refuses to run without CUDA and verifies model parameters are on the GPU.
- Validates vocabulary and cached Parquet checksums against the manifest.
- Uses deterministic masking keyed by `(seed, epoch, patient_id)` and a fixed
  validation corruption.
- Ensures each patient is consumed once and split coverage matches the manifest.
- Computes finite token-weighted MLM loss and rejects non-finite gradients.
- Writes model, optimizer, next epoch, global step, dataset/configuration hashes,
  and Torch CPU/CUDA RNG state in one DCP transaction.
- Reloads every saved checkpoint into fresh model and optimizer objects and
  compares fixed-probe logits.
- Creates an incomplete checkpoint directory and verifies discovery ignores it.

The recovery guarantee is deliberately epoch-boundary only. A failure during an
epoch replays that epoch from the prior complete checkpoint. Mid-epoch recovery,
worker prefetch state, and distributed resharding are not claimed.

## Scaling beyond the smoke test

Do not scale the current one-file-per-split snapshot by only increasing GPU
count. Native file partitioning needs enough balanced shards for all ranks and
workers, plus explicit global coverage and equal-step validation.

The next useful sequence is:

1. Export representative Spark data into multiple immutable Parquet shards.
2. Port the existing ExMed-BERT batch/model contract behind the proven adapter.
3. Compare native loading against Mosaic on throughput, time to first batch,
   recovery time, and operational complexity.
4. Add DDP or Ray Train only after rank-level data ownership and checkpoint
   participation tests exist.
5. Keep the Mosaic path for rollback until those comparisons pass.

Ray Train is relevant when multi-node orchestration is an immediate need. Use
either native file partitioning or Ray Data as the owner of data sharding, not
both. The official [Ray Train distributed example](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/examples/ray-train-distributed)
is a scaling reference; Ray is intentionally absent from this one-GPU baseline.

## Model and data direction

For this structured BERT encoder, suitable next pretraining sources are governed
longitudinal EHR/claims data already normalized in Unity Catalog, MIMIC-IV or
eICU for credentialed research use, and OMOP-formatted cohorts. Synthea is ideal
for software validation but not for clinical-quality pretraining. PubMed/PMC or
clinical-note corpora imply a text-model objective and are a separate data/model
track. Verify each source's license, data-use agreement, de-identification, and
code-system mapping before ingestion.

RLHF, RLAIF, PPO, and GRPO are not natural post-training methods for this
masked-code encoder: it does not generate actions or responses. Its relevant
post-training is supervised adaptation to downstream prediction tasks. If the
goal is also to showcase modern LLM post-training, make that a separate AIR
experiment based on a small open causal/instruction model: establish evaluation,
run SFT, then prefer a simple preference-optimization stage before attempting a
reward-model-plus-PPO pipeline. GRPO is worthwhile only when the task has a
reliable automatic reward. Do not mix that demonstration into the ExMed-BERT
migration or treat synthetic preference gains as clinical evidence.

## References

- [AIR CLI quickstart](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/quickstart)
- [Load data on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading)
- [Performance and resiliency](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/guides/performance-and-resiliency)
- [Distributed training](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training)
- [Synthea](https://synthea.mitre.org/)
