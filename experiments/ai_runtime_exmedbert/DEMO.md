# Reproducible ExMed-BERT AI Runtime demo

This is the operator runbook for the complete synthetic healthcare demo. One
launcher connects the already-pinned Synthea source, serverless Spark data
preparation, Unity Catalog volume artifacts, AI Runtime GPU training, DCP
recovery, and MLflow evidence.

The demo establishes platform compatibility and reproducibility. Synthea is
synthetic, so its downstream metrics are not evidence of clinical utility.

## What the launcher operates

```text
Pinned Synthea release in a UC volume
  -> serverless Spark pretraining snapshot
  -> AIR ExMed-BERT MLM training
  -> DCP checkpoint in a UC volume
  -> serverless Spark endpoint audit and leakage-safe PLOS snapshot
  -> AIR PLOS fine-tuning
  -> MLflow metrics, logs, and local evidence report
```

Spark owns joins, temporal filtering, labels, patient splits, vocabulary, and
immutable Parquet publication. AIR receives files through `UCVolumeDataset` and
uses `UCVolumeWriter`/`UCVolumeReader` for DCP. Spark DataFrames and Mosaic
Streaming do not enter the AIR process.

## Prerequisites

- Databricks CLI 1.0 or newer and the AIR CLI
- An explicitly authenticated OAuth profile with access to the configured UC volume
- GNU tar on macOS (`brew install gnu-tar`); the launcher configures its PATH automatically
- The pinned raw Synthea release published according to [synthea/README.md](synthea/README.md)

Authentication is deliberately outside automation. If validation reports a
credential-store or refresh-token error, run:

```bash
databricks auth login --profile fevm-default
```

Use your own explicit profile name when working in another workspace.

## Configuration

Edit [demo_config.json](demo_config.json) or pass another file with `--config`.
`volume_root` is the only storage base; all snapshot, audit, and checkpoint
paths are derived from the three immutable version names.

For a genuinely new build, choose new snapshot/audit version names. The Spark
producers never overwrite an existing version. Complete existing artifacts are
reused by their final `manifest.json` or `audit.json` marker.

`downstream_train_fraction` retains a deterministic patient-level fraction of
the training split only. `downstream_subset_seed` defines the nested subset,
while the ordered `downstream_training_seeds` list controls model initialization
and training repetitions. Validation and test remain unchanged. Changing the
fraction or subset seed requires a new `plos_snapshot_version`; never reuse a
snapshot version for a different experiment contract.

## Commands

Run these from `experiments/ai_runtime_exmedbert`:

```bash
# Inspect paths and stage commands without contacting the workspace.
python3 demo.py plan --profile fevm-default

# Check authentication, strict bundle validation, and both AIR payloads.
python3 demo.py validate --profile fevm-default

# Inspect immutable artifacts and latest complete DCP checkpoints.
python3 demo.py status --profile fevm-default

# Run/reuse the primary end-to-end path.
python3 demo.py run --profile fevm-default
```

For the full showcase, include a separate-process MLM recovery proof and the two
downstream controls:

```bash
python3 demo.py run --profile fevm-default \
  --resume-proof \
  --include-comparators
```

`--force-air` resubmits GPU stages even when the requested epoch already has a
complete checkpoint. It does not overwrite Spark artifacts.

To validate the low-label mechanics, copy the configuration, set a new
`plos_snapshot_version`, set `downstream_train_fraction` to `0.1`, and run only
the primary comparison:

```bash
python3 demo.py run --profile fevm-default --config /path/to/plos-10pct.json \
  --downstream-mode finetune \
  --downstream-mode random
```

The launcher runs every selected downstream mode for every configured seed,
isolates their checkpoint roots, validates seed/mode identity before reuse, and
writes paired means and sample standard deviations into the evidence report.
Keep the default configuration single-seed; use an explicit experiment config
when launching a multi-seed matrix.

## Safety and restart behavior

- Every Spark artifact is immutable and considered complete only after its JSON marker exists.
- AIR resumes from the highest DCP directory containing `.metadata`.
- Training checkpoints are currently written at epoch boundaries, so an
  interrupted epoch is repeated rather than resumed at an exact sample index.
- Incomplete checkpoint directories are ignored.
- Downstream checkpoint roots include the exact MLM checkpoint directory name,
  preventing a fine-tune from being silently reused after pretraining changes.
- Non-default downstream training seeds receive separate checkpoint roots.
- New supervised snapshots require `training_result.json` before a downstream
  stage is considered reusable; this marker is written after final evaluation.
- The profile is required; the launcher never selects a workspace implicitly.
- Ctrl-C stops local monitoring but does not cancel a remote run.

## Evidence

AIR automatically creates MLflow runs. The launcher waits for terminal status
and writes run IDs, dashboard URLs, MLflow URLs, configuration, paths, and reuse
decisions to ignored files under `runs/`.

The workspace-verified reference execution and metrics are documented in
[RESULTS.md](RESULTS.md#air-downstream-implementation).
