# Reproducible Synthea generation results

## Raw release

**Generated and published on 11 September 2026.** The release was created with
Synthea `v4.0.0`, one generation thread, population and clinician seeds
`201214`, a fixed `2025-01-01` reference date, and complete history.

- Synthea JAR SHA-256: `ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1`
- Generator properties SHA-256: `99c5d27f67721e33b9092dbe55fa7681eee62eb8bb7a5088cb87fb5f8c77bcad`
- Reproducibility specification SHA-256: `c2b29a08e7c29e8be829ba42f53b57a3a3ab87d313c3503a11e1c74f7912598e`
- Provenance document SHA-256: `3f24a28769d9aea2555fc26dfa09fb3eb629c5ee16bf06fc0ca5050d1a83605c`
- UC Volume path: `/Volumes/shyam_internal_ssa_catalog/deep_learning/exmedbert_air/synthea/raw/synthea-v4.0.0-p10000-seed201214-ref20250101`
- Published bytes: 2,097,402,812 across five CSVs plus `provenance.json`

| Entity | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| Patients | 11,761 | 3,532,193 | `23801c645a22f3a59e3735a0bfe5562d9f3e4bf70d6942c0bb6df32f1022f401` |
| Encounters | 2,020,907 | 666,466,156 | `9a83981f3f4969a58766fdfd36cd2cf91eeeb69f28fdd3925924f554a9e28da3` |
| Conditions | 1,144,672 | 166,320,472 | `30c0674196bd799d3e7ddcd98ad073e7a01d31307467d1cac9ba1e9568992d93` |
| Medications | 1,132,197 | 297,733,498 | `be0cef7098dea9b399cd5aae3c80d5daa359db50b73ceb8286581cb2cc6657e0` |
| Procedures | 4,790,459 | 963,348,421 | `48d83716f1f2500b65552fb8bb166118d812e5ce6777899364c847ce4f7ddd60` |

The patient rows comprise 10,000 living and 1,761 deceased histories. Local
validation found 7,067,328 clinical events and zero unknown-patient,
unknown-encounter, or patient/encounter ownership violations. Remote validation
matched all file sizes and the complete provenance document.

A separate 25-living-patient smoke cohort was generated twice. All five CSV
SHA-256 values matched byte-for-byte across the independent runs.

## Spark handoff

**Passed on 11 September 2026.** The 16-shard-per-split serverless Spark
conversion completed under the normal 5% validation/test OOV ceiling:

- Job ID: `60940400724286`
- Run ID: `396168146279894`
- [Spark run](https://fevm-shyam-internal-ssa.cloud.databricks.com/?o=7474645041746813#job/60940400724286/run/396168146279894)
- Status: `TERMINATED / SUCCESS`
- Duration: 308 seconds
- Target snapshot: `/Volumes/shyam_internal_ssa_catalog/deep_learning/exmedbert_air/exmedbert/snapshots/synthea-v4-p10000-v1`
- Vocabulary: 2,073 codes, built from training patients only
- Duplicate within-encounter events removed: 629,832
- Top training token fraction: 5.04%, below the 25% gate

| Split | Patients | Tokens | OOV tokens | OOV fraction | Length range |
|---|---:|---:|---:|---:|---:|
| Train | 9,406 | 1,081,112 | 0 | 0.0000% | 3–128 |
| Validation | 1,167 | 134,470 | 9 | 0.0067% | 6–128 |
| Test | 1,188 | 137,090 | 10 | 0.0073% | 4–128 |

All sequence arrays aligned, and direct UC Volume inspection confirmed exactly
16 Parquet files in each split plus `vocabulary.json` and the final
`manifest.json` completion marker.

## AIR ExMed-BERT training

**Passed on 11 September 2026.** The repository's `ExMedBertForMaskedLM`
trained on one NVIDIA A10G using `UCVolumeDataset`, batch size 32, six loader
workers, and prefetch factor 4. Both runs used environment version 5, PyTorch
`2.14.0+cu130`, CUDA 13.0, Transformers 4.46.3, and `serverless_gpu`
`0.5.23+patch1`.

### Initial epoch

- Run ID: `561178542401040`
- Status: `SUCCESS`; duration 103 seconds; user code 37 seconds
- [AIR dashboard](https://fevm-shyam-internal-ssa.cloud.databricks.com/jobs/runs/561178542401040?o=7474645041746813)
- [MLflow logs](https://fevm-shyam-internal-ssa.cloud.databricks.com/ml/experiments/964216389379452/runs/4680a74ed6d84061ae73851e80ca17e4/artifacts/logs/node_0?o=7474645041746813)
- Train: loss 5.8915, 9,406 patients, 160,643 scored tokens, 298 optimizer steps, 10.40 seconds
- Validation: loss 5.0125, 1,167 patients, 19,979 scored tokens, 0.64 seconds
- Checkpoint save: 1.66 seconds
- Checkpoint: `epoch_000001_3ba5cbe9fb3d4167a993b0c8fdc66114`
- Fresh model/optimizer restore: passed

### Separate-process resume

- Run ID: `697227262334177`
- Status: `SUCCESS`; duration 112 seconds; user code 38 seconds
- [AIR dashboard](https://fevm-shyam-internal-ssa.cloud.databricks.com/jobs/runs/697227262334177?o=7474645041746813)
- [MLflow logs](https://fevm-shyam-internal-ssa.cloud.databricks.com/ml/experiments/964216389379452/runs/4b3096c4761547eca6965713cb292157/artifacts/logs/node_0?o=7474645041746813)
- Restored epoch 1 with `next_epoch=1` and `global_step=298`
- Epoch 2 train: loss 4.7596, 9,406 patients, 160,643 scored tokens, 298 optimizer steps, 10.16 seconds
- Epoch 2 validation: loss 4.5218, 1,167 patients, 19,979 scored tokens, 0.71 seconds
- Checkpoint save: 2.05 seconds
- Checkpoint: `epoch_000002_a20e86968ea643619df515cff875c52a`
- Fresh model/optimizer restore: passed

Both complete checkpoint directories contain DCP `.metadata` and
`__0_0.distcp`. The controlled `epoch_999999_incomplete` directory remains and
was ignored during recovery.

## DataLoader worker benchmark

A separate one-epoch control changed only the loader worker count and used an
isolated checkpoint root.

- Run ID: `1013938137136491`
- Status: `SUCCESS`; duration 103 seconds; user code 44 seconds
- [AIR dashboard](https://fevm-shyam-internal-ssa.cloud.databricks.com/jobs/runs/1013938137136491?o=7474645041746813)
- [MLflow logs](https://fevm-shyam-internal-ssa.cloud.databricks.com/ml/experiments/964216389379452/runs/51fe5acd03cc4bd0a8a3658ad52c325b/artifacts/logs/node_0?o=7474645041746813)
- Checkpoint: `epoch_000001_f93bcf4c990b4a7cb12e891851d54b49`
- Fresh model/optimizer restore: passed

| Metric | 0 workers | 6 workers | Result |
|---|---:|---:|---:|
| Train epoch | 15.88 s | 10.40 s | 1.53x faster |
| Train patients/s | 592 | 904 | 52.7% higher |
| Validation epoch | 1.81 s | 0.64 s | 2.84x faster |
| User-code execution | 44 s | 37 s | 15.9% lower |
| End-to-end AIR duration | 103 s | 103 s | unchanged |
| Patients / scored train tokens | 9,406 / 160,643 | 9,406 / 160,643 | identical coverage |
| Optimizer steps | 294 | 298 | worker-partition partial batches |

Six workers are the recommended setting for this snapshot: training epoch time
fell by 34.5% despite four additional partial-batch optimizer steps. End-to-end
time did not improve for a single short epoch because environment and workload
startup dominate. For longer pretraining runs, epoch throughput is the useful
measure.

With an iterable dataset, each worker partition can contribute a final partial
batch when `drop_last=false`. This preserves every patient but changes the
number and sizes of optimizer steps slightly. Any future DDP/global-batch design
must define this behavior explicitly rather than assuming
`ceil(total_patients / batch_size)` steps.
