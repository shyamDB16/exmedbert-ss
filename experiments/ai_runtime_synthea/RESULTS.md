# AI Runtime Synthea smoke-test results

## Outcome

**Passed on 10 September 2026.** A one-GPU AIR job trained for two epochs, saved
and reloaded a DCP checkpoint after each epoch, and a separate AIR process
restored the epoch-2 checkpoint before training epoch 3.

## Workspace and snapshot

- Profile: `fevm-default`
- Identity: `shyam.sankararaman@databricks.com`
- Volume: `shyam_internal_ssa_catalog.deep_learning.exmedbert_air`
- Snapshot: `/Volumes/shyam_internal_ssa_catalog/deep_learning/exmedbert_air/synthea/snapshots/v1`
- Source: official Synthea latest CSV sample ZIP downloaded on 10 September 2026
- Source ZIP SHA-256: `d61417b551e5b0997c33851b339c157421751f0ea68c18ea686ceb1850907c35`
- Manifest SHA-256: `1d11208d9d92c71262341d0b996e15250e17bdc1a996c9503fe776e569ed632c`
- Patients: 108 total; 78 train and 30 validation
- Tokens: 2,210 train and 1,128 validation
- Vocabulary: 172 entries, built from training only
- Validation unknowns: 20 tokens
- Exclusions: no missing encounters, patient mismatches, empty codes, or duplicate encounters

The manifest records source and artifact hashes. During AIR execution, the
locally cached train and validation Parquet files were hashed and matched before
their coverage checks passed.

## Runtime

- Databricks CLI: 1.4.0
- AIR CLI: 1.2.0
- AIR environment: numeric version `5` with inline dependencies
- Accelerator: one NVIDIA A10G (`GPU_1xA10`)
- Python: 3.12.3
- PyTorch: `2.14.0+cu130`
- CUDA runtime: 13.0
- `serverless_gpu`: `0.5.23+patch1`
- MLflow resolved by the run: 3.16.0
- Batch size: 16
- DataLoader workers: 0 (`prefetch_factor=None`)
- Model objective: deterministic dynamic masked-code prediction with time-gap embeddings

## AIR runs

### Initial two-epoch run

- Run ID: `520513910385452`
- Status: `SUCCESS`
- AIR duration: 92 seconds; user-code execution: 35 seconds
- [Job dashboard](https://fevm-shyam-internal-ssa.cloud.databricks.com/jobs/runs/520513910385452?o=7474645041746813)
- [MLflow artifacts/logs](https://fevm-shyam-internal-ssa.cloud.databricks.com/ml/experiments/964216389379452/runs/5a5f5931bd064ff79d49249c0a181ff0/artifacts/logs/node_0?o=7474645041746813)

| Epoch | Train loss | Validation loss | Train/validation patients | Scored train/validation tokens | Checkpoint save |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.0425 | 4.4360 | 78 / 30 | 334 / 166 | 5.42 s |
| 2 | 4.0774 | 4.0092 | 78 / 30 | 334 / 166 | 4.96 s |

Checkpoints:

- `.../checkpoints/v1/epoch_000001_81fe76afc6314d8fb74debe0ef24f27f`
- `.../checkpoints/v1/epoch_000002_b43b8099443a4b2a8f0e6f63c35d5ca1`

Both fresh model/optimizer restore probes passed.

### Separate-process resume run

- Run ID: `911800130597532`
- Status: `SUCCESS`
- AIR duration: 72 seconds; user-code execution: 21 seconds
- [Job dashboard](https://fevm-shyam-internal-ssa.cloud.databricks.com/jobs/runs/911800130597532?o=7474645041746813)
- [MLflow artifacts/logs](https://fevm-shyam-internal-ssa.cloud.databricks.com/ml/experiments/964216389379452/runs/fefbdbe16d05496ab13fd478a165164c/artifacts/logs/node_0?o=7474645041746813)

The new process discovered and restored the epoch-2 checkpoint with
`next_epoch=2` and `global_step=10`, then trained only epoch 3:

- Train loss: 4.0717 across 78 patients and 334 scored tokens
- Validation loss: 3.8966 across 30 patients and 166 scored tokens
- Checkpoint save: 1.11 seconds
- Final checkpoint: `.../checkpoints/v1/epoch_000003_f4e6d6cbaa8e4e7d82595b31c1dd5900`
- Fresh model/optimizer restore probe: passed

The separate-process restore also demonstrates that the prior DCP checkpoint
and its `.metadata` were durably visible through `UCVolumeReader`.

A direct UC volume listing confirmed that epochs 1, 2, and 3 each contain
`.metadata` and `__0_0.distcp`. The controlled `epoch_999999_incomplete`
directory contains only its probe `README` and was correctly ignored.

## Acceptance evidence

- [x] AIR YAML dry-run passed.
- [x] Snapshot checksums and disjoint patient splits passed.
- [x] Data loaded through `UCVolumeDataset`; cached-file checksums passed.
- [x] Model and batches ran on CUDA with no CPU fallback.
- [x] Expected patient coverage passed for train and validation in every epoch.
- [x] Two initial epochs completed with finite token-weighted MLM loss.
- [x] DCP saves through `UCVolumeWriter` completed and published `.metadata`.
- [x] Fresh model and optimizer restores through `UCVolumeReader` passed.
- [x] Controlled incomplete checkpoint discovery test passed.
- [x] A separate AIR process resumed the latest complete checkpoint and completed epoch 3.
- [x] Local deterministic tests passed (`3 passed`).

## Constraints encountered

- The workspace does not provide `jobs_serverless_managed_base_environments`, so
  `databricks_ai_v5` was unavailable. Numeric environment version `"5"` with
  inline dependencies succeeded.
- With `num_workers=0`, the native loader must receive
  `prefetch_factor=None`.
- On macOS, AIR CLI 1.2 packaging requires `COPYFILE_DISABLE=1` to avoid
  AppleDouble files. Generated ignored cache directories can trigger unsupported
  GNU-tar flags under BSD tar; keep them outside the packaged experiment tree.
- The successful jobs emitted harmless Apple provenance xattr warnings while
  extracting the source snapshot.

## Limitations

- Synthetic diagnoses only; no clinical-performance claim.
- One GPU, fixed patient order, and epoch-boundary recovery only.
- Tiny transport model, not the full ExMed-BERT model or batch schema.
- One Parquet file per split; this layout is intentionally not distributed-ready.
- DCP ran in single-process mode; distributed resharding and rank failure were not tested.
- Falling validation loss on this tiny synthetic sample is an engineering signal only.
