# ExMed-BERT on Databricks AI Runtime

This project is the next step after the verified
[`ai_runtime_synthea`](../ai_runtime_synthea/README.md) transport test. It keeps
Spark data preparation and GPU training as separate workloads connected by one
immutable Unity Catalog volume snapshot.

**Status: implemented and workspace-verified.** See [RESULTS.md](RESULTS.md) for
the Spark, AIR, MLflow, and recovery evidence.

For the packaged end-to-end operator workflow, start with [DEMO.md](DEMO.md):

```bash
python3 demo.py validate --profile fevm-default
python3 demo.py run --profile fevm-default --resume-proof --include-comparators
```

Ray is deliberately out of scope. The first acceptance target is the repository's
`ExMedBertForMaskedLM` on one A10 GPU with complete epoch-boundary recovery.

## Project boundary

```text
Synthea CSV or governed Delta sources
              |
              | Declarative Automation Bundle / serverless Spark
              v
Canonical diagnosis + medication + procedure events
              |
              | deterministic split, train-only vocabulary, quality gates
              v
Immutable Parquet snapshot in a UC volume
              |
              | UCVolumeDataset + cached-path Parquet decoder
              v
ExMedBertForMaskedLM on one AIR GPU
              |
              | DCP + UCVolumeWriter / UCVolumeReader
              v
Durable epoch checkpoints
```

Spark never enters the GPU training loop. The AIR process never receives a
Spark DataFrame.

## Snapshot schema

Each patient row contains unpadded, unmasked arrays:

| Column | Meaning |
|---|---|
| `patient_id` | Stable synthetic or de-identified identifier |
| `codes` | Train-vocabulary token IDs |
| `entity_ids` | Medication, diagnosis, or procedure token domain |
| `age_ids` | Age-at-encounter in 0.1-year buckets |
| `position_ids` | Visit index, starting at 1 and capped to the model limit |
| `sex_id` | Fixed ExMed-BERT-compatible sex vocabulary |
| `state_id` | Train-only region vocabulary |

Codes preserve both domain and source namespace, for example
`DIAGNOSIS:SNOMED:66383009` or `MEDICATION:RXNORM:12345`. The pipeline does not
invent an ICD, ATC, or CPT crosswalk. The compatibility layer represents these
domains through the existing `CodeDict` entity slots while preserving the full
source token.

The vocabulary keeps the legacy special-token IDs exactly:

```text
PAD=0, UNK=1, CLS=2, SEP=3, MASK=4, NA=5
```

Masking and padding happen in AIR, not in Spark.

## Spark preparation

The bundle defines a serverless Spark Python task. Its default development paths
are parameterized in `databricks.yml`.

Before the first run, publish the unmodified Synthea CSV directory to:

```text
/Volumes/shyam_internal_ssa_catalog/deep_learning/exmedbert_air/synthea/raw/sample-20260910
```

That official 108-patient sample is retained as the original acceptance
fixture. For a pinned, regenerable 10,000-patient source release, use the
[`synthea`](synthea/README.md) generation and publication stage.

Validate, deploy, and run from this directory:

```bash
databricks bundle validate --strict -t dev
databricks bundle deploy -t dev
databricks bundle run prepare_exmedbert_snapshot -t dev
```

Override task arguments without changing the deployed job or editing source:

```bash
databricks bundle run prepare_exmedbert_snapshot -t dev \
  -- \
  --source-root /Volumes/<catalog>/<schema>/<volume>/synthea/raw/<version> \
  --output-root /Volumes/<catalog>/<schema>/<volume>/exmedbert/snapshots \
  --snapshot-version synthea-v3 \
  --num-shards 8 \
  --max-oov-fraction 0.05
```

The producer:

- validates patient and encounter key uniqueness;
- rejects malformed rows and records orphan event counts;
- deduplicates identical domain/code events within an encounter;
- uses stable chronological and within-visit ordering;
- assigns train/validation/test at patient level with SHA-256;
- constructs code and state vocabularies from training patients only;
- records OOV, sequence alignment, semantic-tag, rare-code, and token-dominance metrics;
- writes multiple self-contained Parquet files per split;
- writes `manifest.json` last as the snapshot completion marker;
- refuses to overwrite an existing snapshot version.

The code default rejects validation or test OOV above 5%. The 108-patient
Synthea integration fixture uses a documented 10% ceiling because its observed
test OOV is 5.71%; a representative pretraining cohort should return to the
stricter threshold.

For production sources, replace the CSV readers with canonical UC Delta tables
while preserving the output and quality contract.

## AIR training

Run from this project directory so `root_path: ../..` resolves to the repository:

AIR CLI 1.2 selective snapshots require GNU tar on macOS (`brew install
gnu-tar`). The commands below put its compatible `tar` executable on `PATH`.

```bash
PATH="/opt/homebrew/opt/gnu-tar/libexec/gnubin:$PATH" \
COPYFILE_DISABLE=1 air run \
  --profile fevm-default \
  --file train.yaml \
  --dry-run

PATH="/opt/homebrew/opt/gnu-tar/libexec/gnubin:$PATH" \
COPYFILE_DISABLE=1 air run \
  --profile fevm-default \
  --file train.yaml \
  --watch
```

The initial configuration uses one A10, environment version 5, full
`ExMedBertForMaskedLM` inputs, trainable randomly initialized code embeddings,
dynamic MLM corruption, and no PLOS head. It pins `num_workers=0` for the first
recovery proof; after correctness passes, benchmark the documented native
DataLoader defaults (`num_workers=6`, `prefetch_factor=4`).

The 10,000-patient benchmark has now verified the latter profile: six workers
reduced training epoch time from 15.88 to 10.40 seconds (1.53x throughput) with
complete patient coverage. Iterable worker partitions produced four additional
final partial-batch optimizer steps; see
[`synthea/RESULTS.md`](synthea/RESULTS.md#dataloader-worker-benchmark) for the
comparison and DDP caveat.

The snapshot path, checkpoint root, epochs, batch size, worker count, and
prefetch factor are environment variables in `train.yaml`. This preserves the
verified small-fixture defaults while allowing a larger immutable snapshot to
be selected with AIR `--override env_variables.<NAME>=<VALUE>` arguments.

Training validates every cached Parquet filename and size, patient coverage,
finite losses/gradients, CUDA placement, and snapshot/configuration identity. It
saves model, optimizer, counters, and Torch RNG state with DCP, reloads each save
into fresh objects, and ignores incomplete checkpoint directories.

## Acceptance sequence

1. Spark job publishes `synthea-v2` and all quality gates pass.
2. AIR YAML dry-run resolves the selective repository snapshot.
3. ExMed-BERT completes two finite-loss epochs on CUDA.
4. Fresh model/optimizer checkpoint probes pass.
5. A second AIR process restores epoch 2 and trains only epoch 3.
6. Only after this sequence: tune workers, batch size, and shard sizes.

Distributed DDP, Ray Data, Ray Train, ClinVec initialization, PLOS labels, and
model registration are later milestones.

## Downstream endpoint audit

The bundle also defines `profile_exmedbert_endpoints`, a serverless Spark task
that profiles completed inpatient encounters for PLOS over seven days and
30-/180-day inpatient readmission. Readmission cohorts are censored against the
fixed Synthea observation end date, and all candidates reuse the pretraining
patient split hash. The task writes an immutable `audit.json`; model training
does not begin until label prevalence and split balance have been reviewed.

The 10,000-patient audit selected admission-time PLOS as the first endpoint:
22,742 eligible inpatient encounters, 3,452 positive labels, and 15.18%
prevalence. The 30- and 180-day readmission candidates remain useful follow-on
tasks but are not needed to prove the downstream AIR path.

## Admission-time PLOS snapshot

`prepare_exmedbert_plos_snapshot` publishes one row per eligible completed
inpatient admission. The label is completed inpatient duration strictly greater
than seven days. Model inputs contain only events with `event_time < index_time`;
the target encounter is explicitly excluded. The job reuses the pretraining
patient split and vocabulary, keeps the latest 128 prior events, requires at
least five history events, and rejects any temporal leakage before writing the
manifest completion marker.

```bash
databricks bundle run prepare_exmedbert_plos_snapshot -t dev \
  --profile fevm-default
```

The resulting split directories contain only Parquet shards. Spark does not run
inside the GPU process, and the AIR trainer does not read a Spark DataFrame.

## AIR PLOS transfer-learning matrix

`train_plos.yaml` is a separate AIR project entry point for downstream
classification. Its `UCVolumeDataset` adapter streams and caches supervised
Parquet shards, while the ExMed-BERT classifier uses attention-mask-aware mean
pooling so padded positions cannot affect a prediction.

Run each initialization as a separate AIR/MLflow run:

```bash
PATH="/opt/homebrew/opt/gnu-tar/libexec/gnubin:$PATH" \
COPYFILE_DISABLE=1 air run --profile fevm-default --file train_plos.yaml \
  --override env_variables.INITIALIZATION=random --watch

PATH="/opt/homebrew/opt/gnu-tar/libexec/gnubin:$PATH" \
COPYFILE_DISABLE=1 air run --profile fevm-default --file train_plos.yaml \
  --override env_variables.INITIALIZATION=frozen --watch

PATH="/opt/homebrew/opt/gnu-tar/libexec/gnubin:$PATH" \
COPYFILE_DISABLE=1 air run --profile fevm-default --file train_plos.yaml \
  --override env_variables.INITIALIZATION=finetune --watch
```

- `random`: random encoder and classifier, all parameters trainable.
- `frozen`: verified MLM checkpoint transfer, frozen encoder, trainable head.
- `finetune`: verified MLM checkpoint transfer, all parameters trainable.

The trainer derives `pos_weight` from the training split only. It reports AUROC,
average precision, Brier score, expected calibration error, sensitivity, and
specificity. The decision threshold maximizes Youden's J on validation and is
then applied unchanged to test. Each mode has its own DCP directory below the
configured checkpoint root and performs a fresh-object restore/logit probe after
every epoch using `UCVolumeWriter` and `UCVolumeReader`.

The latest epoch checkpoint remains the recovery checkpoint. Final validation
threshold selection and test evaluation restore the checkpoint with the highest
validation average precision, preferring the earliest epoch on a tie. Every
successful downstream stage publishes `training_result.json` only after final
evaluation and MLflow logging complete.

The Spark snapshot builder can retain a deterministic, nested patient-level
fraction of the training split while leaving validation and test unchanged. Use
a new immutable snapshot version for every fraction/subset-seed contract. This
supports controlled pretrained-versus-random comparisons in lower-label regimes
without changing the AIR `DataLoader` path.

The complete workspace comparison and separate-process resume proof have passed;
see [RESULTS.md](RESULTS.md#air-downstream-implementation). The result is
appropriately mixed: full fine-tuning beats a frozen encoder and improves
calibration over random initialization, but does not beat the random baseline on
test AUROC after this short two-epoch synthetic pretraining run.

PPO, GRPO, RLHF, and RLAIF are not a natural fit for this encoder-only clinical
event prediction task: it has no generated policy or preference signal. Those
methods belong in a later generative clinical-language-model track; supervised
transfer learning is the appropriate proof for ExMed-BERT.
