# ExMed-BERT AIR port results

## Outcome

**Passed on 10 September 2026.** Serverless Spark produced a quality-gated,
multi-file ExMed-BERT snapshot. The repository's `ExMedBertForMaskedLM` then
trained for two epochs on one AIR A10 GPU, and a separate AIR process restored
epoch 2 before training only epoch 3.

## Spark snapshot

- Bundle: `exmedbert-ai-runtime`, target `dev`
- Job ID: `60940400724286`
- Successful run ID: `684053475617634`
- [Spark run](https://fevm-shyam-internal-ssa.cloud.databricks.com/?o=7474645041746813#job/60940400724286/run/684053475617634)
- Result: `TERMINATED / SUCCESS`
- Run duration: 102 seconds
- Raw source: `/Volumes/shyam_internal_ssa_catalog/deep_learning/exmedbert_air/synthea/raw/sample-20260910`
- Source ZIP SHA-256: `d61417b551e5b0997c33851b339c157421751f0ea68c18ea686ceb1850907c35`
- Snapshot: `/Volumes/shyam_internal_ssa_catalog/deep_learning/exmedbert_air/exmedbert/snapshots/synthea-v2`
- Four Parquet shards per train, validation, and test split; no extra files in the loader directories
- Code vocabulary: 558 entries, constructed from training patients only
- Age/sex/state/entity vocabulary sizes: 1,103 / 4 / 3 / 4

| Split | Patients | Tokens | OOV tokens | OOV fraction | Length range |
|---|---:|---:|---:|---:|---:|
| Train | 79 | 8,389 | 0 | 0.00% | 15–128 |
| Validation | 14 | 1,489 | 33 | 2.22% | 1–128 |
| Test | 15 | 1,560 | 89 | 5.71% | 57–128 |

Quality evidence:

- No malformed source rows or orphan encounter joins.
- 1,284 duplicate domain/code events within encounters removed.
- Patient-level split sets are exclusive and all model arrays are aligned.
- Top training token is `PROCEDURE:SNOMED:710824005` at 4.32%, below the 25% dominance ceiling.
- The tiny fixture's test OOV exceeded the general 5% default; its bundle configuration explicitly allows 10% and records the observed value.
- `manifest.json` was written after all split and vocabulary artifacts.

## AIR runtime

- AIR CLI: 1.2.0
- Environment: numeric GPU version `5`
- GPU: NVIDIA A10G
- Python: 3.12.3
- PyTorch: `2.14.0+cu130`
- CUDA runtime: 13.0
- Transformers: 4.46.3
- `serverless_gpu`: `0.5.23+patch1`
- Model: repository `ExMedBertForMaskedLM`, 288 hidden units, 6 layers, 6 heads
- Objective: dynamic masked code prediction; PLOS disabled
- Batch size: 16; DataLoader workers: 0

### Initial two-epoch run

- Run ID: `118675160909861`
- Status: `SUCCESS`
- Duration: 92 seconds; user code: 29 seconds
- [AIR dashboard](https://fevm-shyam-internal-ssa.cloud.databricks.com/jobs/runs/118675160909861?o=7474645041746813)
- [MLflow logs](https://fevm-shyam-internal-ssa.cloud.databricks.com/ml/experiments/964216389379452/runs/b5ba6d25c81f441ea9de771a87176ce7/artifacts/logs/node_0?o=7474645041746813)

| Epoch | Train loss | Validation loss | Patients train/validation | Scored tokens train/validation | Checkpoint |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.2815 | 6.2772 | 79 / 14 | 1,248 / 217 | 1.55 s |
| 2 | 6.1744 | 6.1594 | 79 / 14 | 1,248 / 217 | 1.78 s |

Both fresh model/optimizer restores and fixed-logit probes passed.

### Separate-process resume

- Run ID: `1087555817010656`
- Status: `SUCCESS`
- Duration: 112 seconds; user code: 37 seconds
- [AIR dashboard](https://fevm-shyam-internal-ssa.cloud.databricks.com/jobs/runs/1087555817010656?o=7474645041746813)
- [MLflow logs](https://fevm-shyam-internal-ssa.cloud.databricks.com/ml/experiments/964216389379452/runs/a6a93a7f65d147a0832a89a259e8d280/artifacts/logs/node_0?o=7474645041746813)

The new process restored
`epoch_000002_f44d88dcfc844e0faea34bfd882899cb` with
`next_epoch=2, global_step=10`, then trained only epoch 3:

- Train loss: 6.0728 over 79 patients and 1,248 scored tokens
- Validation loss: 6.0570 over 14 patients and 217 scored tokens
- Final checkpoint: `epoch_000003_4355420079e74004a9c702f95b71549a`
- Checkpoint save: 5.04 seconds
- Fresh model/optimizer restore: passed

UC volume checkpoint directories:

- `epoch_000001_93992dfe85314f07a6afebc31e748462`
- `epoch_000002_f44d88dcfc844e0faea34bfd882899cb`
- `epoch_000003_4355420079e74004a9c702f95b71549a`
- `epoch_999999_incomplete`, the controlled incomplete-checkpoint probe

Each completed epoch contains `.metadata` and `__0_0.distcp`; the incomplete
probe contains only its `README` marker.

## Acceptance evidence

- [x] Bundle strict validation passed.
- [x] Serverless Spark snapshot job completed.
- [x] Train/validation/test quality gates passed under the documented fixture threshold.
- [x] AIR YAML dry-run passed.
- [x] `ExMedBertForMaskedLM` ran on CUDA.
- [x] Two epochs completed with finite MLM loss and complete patient coverage.
- [x] Fresh model/optimizer DCP restore passed for every epoch.
- [x] Separate-process epoch-3 resume passed.
- [x] New contract tests and original snapshot tests passed (`3 + 3`).

## Platform findings

- Serverless Spark Python tasks execute source through `exec`, so `__file__` is not guaranteed.
- Serverless Spark Connect forbids direct `sparkContext`, `_jvm`, and `_jsc` access.
- Spark Connect does not support `mode("errorifexists")`; the job reserves a new version directory with `exist_ok=False` and writes the manifest last.
- AIR CLI 1.2 selective snapshots on macOS require GNU tar on `PATH`; `COPYFILE_DISABLE=1` prevents AppleDouble files.
- Transformers 4.46.3 emits a harmless deprecation warning for the legacy `get_extended_attention_mask` call; Transformers 5 is intentionally not enabled.

## Limitations

- Synthea is an integration fixture, not clinical pretraining evidence.
- The official sample is reproducible by checksum but was not generated with a locally recorded Synthea seed.
- One GPU and epoch-boundary recovery only; Ray and DDP are intentionally deferred.
- Semantic-tag counts are reported, but SNOMED concept filtering still requires an approved terminology policy.
- No ClinVec initialization, PLOS objective, model registration, or downstream clinical evaluation.

## Reproducible 10,000-patient extension

The follow-on Synthea v4.0.0 release, strict Spark snapshot, six-worker AIR
training run, and separate-process recovery also passed on 11 September 2026.
The larger snapshot contains 9,406 training patients and 1,081,112 training
tokens; its validation/test OOV fractions are below 0.008%. See
[`synthea/RESULTS.md`](synthea/RESULTS.md) for source checksums, Spark run
evidence, AIR metrics, and checkpoint IDs.

## Downstream endpoint selection

The serverless Spark endpoint audit completed successfully in job
`719775019936724`, run `765945925953213`:

| Candidate | Eligible examples | Positive prevalence |
|---|---:|---:|
| PLOS > 7 days | 22,742 | 15.18% |
| 30-day inpatient readmission | — | 10.98% |
| 180-day inpatient readmission | — | 25.64% |

PLOS was selected because it has sufficient support, stable train/validation/test
prevalence, and direct alignment with the repository's intended auxiliary task.
The production definition is admission-time prediction from events strictly
before admission; the target encounter is excluded.

## AIR downstream implementation

The repository now includes an AIR-native PLOS training path with three explicit
comparators: random initialization, frozen pretrained encoder, and full
fine-tuning. It uses no Mosaic Streaming and no Spark DataFrame in AIR.

### Immutable PLOS snapshot

- Spark job: `807244604730586`
- Successful run: `1100125327633137`
- Duration: 176 seconds
- Snapshot: `/Volumes/shyam_internal_ssa_catalog/deep_learning/exmedbert_air/exmedbert/downstream/plos/snapshots/synthea-v4-p10000-plos7d-v1`
- 16 Parquet shards in each train, validation, and test directory
- 22,742 eligible admissions; 22,734 retained; eight excluded for insufficient history
- Zero temporal leakage and zero unaligned rows
- Pretraining vocabulary and patient split identity reused

| Split | Examples | Patients | Positive | Prevalence | Tokens | OOV fraction |
|---|---:|---:|---:|---:|---:|---:|
| Train | 18,328 | 5,617 | 2,805 | 15.30% | 2,133,291 | 0.0000% |
| Validation | 2,119 | 703 | 286 | 13.50% | 244,157 | 0.0045% |
| Test | 2,287 | 711 | 356 | 15.57% | 263,122 | 0.0068% |

### Five-epoch AIR comparison

All runs used one A10, batch size 64, six loader workers, prefetch factor four,
masked mean pooling, and the train-only positive weight 5.534. The threshold was
selected by maximum Youden J on validation and applied unchanged to test.

| Initialization | AIR run | Test AUROC | Test AP | Brier | Calibration error | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | `1076941076461433` | 0.8120 | 0.3750 | 0.1632 | 0.1595 | 92.13% | 61.83% |
| Frozen pretrained encoder | `364960634219771` | 0.7441 | 0.3098 | 0.1712 | 0.2087 | 78.37% | 60.28% |
| Full fine-tuning | `37526256118625` | 0.8088 | 0.3756 | 0.1497 | 0.1375 | 89.89% | 65.56% |

All three runs completed every epoch with complete split coverage, finite losses
and gradients, and fresh-object checkpoint/logit restore probes. Full fine-tuning
is clearly better than freezing. Against random initialization, ranking quality
is effectively tied on this synthetic endpoint: random is 0.0032 higher in
AUROC, while fine-tuning is 0.0006 higher in average precision. Fine-tuning is
better calibrated (Brier lower by 0.0135 and calibration error lower by 0.0220).
This does not yet establish a transfer-learning benefit from only two MLM epochs;
the next honest tests are a longer/more diverse pretraining corpus and a
low-label downstream regime.

Final epoch-5 checkpoints:

- Fine-tune: `epoch_000005_58153f07f1154a1683799369032db5ad`
- Frozen: `epoch_000005_3927b4b6683e4f7f9c0fc836b0e4d37d`
- Random: `epoch_000005_c5975ef0664f4e509b9172241df4c9d2`

Each contains DCP `.metadata` and `__0_0.distcp`; each mode also has the
controlled `epoch_999999_incomplete` directory, which was ignored.

### Separate-process downstream resume

AIR run `606438502319567` restored the fine-tuning epoch-5 checkpoint with
`next_epoch=5, global_step=1440`, trained only epoch 6, and saved
`epoch_000006_a3890f369279480882d487492ab23230`. Its fresh restore probe and
complete train/validation/test coverage passed. The run finished successfully
with test AUROC 0.8057 and average precision 0.3742.

## Packaged end-to-end reference run

The consolidated [demo launcher](DEMO.md) passed on 15 September 2026. It
strictly validated the bundle, dry-ran both AIR specifications, deployed the
bundle, verified and reused all immutable Spark artifacts, continued MLM
pretraining, proved separate-process recovery, discovered the resulting DCP
checkpoint, and passed it to a lineage-isolated PLOS fine-tune.

| Stage | AIR run | Result | Key evidence |
|---|---:|---|---|
| MLM epochs 3–10 | `413476394194209` | SUCCESS | Epoch-10 validation loss 3.7058; all fresh restore probes passed |
| Separate-process epoch 11 | `183289068887637` | SUCCESS | Restored epoch 10 at global step 2,980; validation loss 3.6654 |
| PLOS from epoch 11 | `86394788467641` | SUCCESS | Test AUROC 0.8160; AP 0.3890; Brier 0.1415 |

The epoch-11 checkpoint is
`epoch_000011_5d7cc938d7664475ab99d0d09edbd5aa`. The downstream DCP root embeds
that exact name and ended at
`finetune/epoch_000005_2be355b6816c44ddb36e453e91cee512`.

At the validation-selected threshold, the new test result had 91.85%
sensitivity and 63.70% specificity. Compared with the earlier five-epoch random
baseline, longer-pretrained fine-tuning is 0.0040 higher in AUROC and 0.0140
higher in average precision. This is useful synthetic demo evidence, not a
clinical or statistically robust transfer-learning claim.

## Ten-percent labeled-training pilot

On September 15, 2026, the updated launcher created the immutable snapshot
`synthea-v4-p10000-plos7d-train10-v1`. Patient-level selection was deterministic,
stratified by each training patient's label profile, and nested by SHA-256 rank.
Validation and test were copied unchanged from the full snapshot.

| Split | Examples | Patients | Positive | Prevalence |
|---|---:|---:|---:|---:|
| Full training pool | 18,328 | 5,617 | 2,805 | 15.30% |
| Selected training subset | 1,777 | 563 | 269 | 15.14% |
| Validation, unchanged | 2,119 | 703 | 286 | 13.50% |
| Test, unchanged | 2,287 | 711 | 356 | 15.57% |

Both AIR runs used seed `201214`, five epochs, one A10, batch size 64, and the
same untouched validation/test examples. Final evaluation restored the checkpoint
with maximum validation average precision; epoch 5 was selected for both runs.

| Initialization | AIR run | Validation AP | Test AUROC | Test AP | Brier | Calibration error |
|---|---:|---:|---:|---:|---:|---:|
| Epoch-11 pretrained, full fine-tuning | `628722691781735` | 0.2902 | 0.7713 | 0.3229 | 0.1774 | 0.1929 |
| Random initialization | `222250388147513` | 0.2701 | 0.7261 | 0.2758 | 0.2177 | 0.2883 |

### Three-seed confirmation

On September 21, 2026, the 10% experiment was repeated with training seeds
`201214`, `202609`, and `314159`. Each seed used the same immutable training
subset and untouched validation/test sets, so these are paired comparisons of
optimization randomness rather than different data samples.

| Seed | Initialization | AIR run | Test AUROC | Test AP | Brier | Calibration error |
|---:|---|---:|---:|---:|---:|---:|
| `201214` | Epoch-11 pretrained, full fine-tuning | `628722691781735` | 0.7713 | 0.3229 | 0.1774 | 0.1929 |
| `201214` | Random initialization | `222250388147513` | 0.7261 | 0.2758 | 0.2177 | 0.2883 |
| `202609` | Epoch-11 pretrained, full fine-tuning | `598111794043977` | 0.7686 | 0.3203 | 0.1776 | 0.1892 |
| `202609` | Random initialization | `919590939221367` | 0.7372 | 0.2935 | 0.2140 | 0.2809 |
| `314159` | Epoch-11 pretrained, full fine-tuning | `848423398292672` | 0.7690 | 0.3277 | 0.1834 | 0.2063 |
| `314159` | Random initialization | `2292685170953` | 0.7255 | 0.2898 | 0.2200 | 0.2915 |

Paired differences are fine-tuning minus random initialization. Positive values
favor fine-tuning for ranking metrics; negative values favor fine-tuning for
error metrics.

| Metric | Seed `201214` | Seed `202609` | Seed `314159` | Mean paired difference ± sample SD |
|---|---:|---:|---:|---:|
| Test AUROC | +0.0452 | +0.0313 | +0.0435 | +0.0400 ± 0.0076 |
| Test AP | +0.0471 | +0.0269 | +0.0379 | +0.0373 ± 0.0101 |
| Brier | -0.0403 | -0.0365 | -0.0365 | -0.0378 ± 0.0022 |
| Calibration error | -0.0954 | -0.0917 | -0.0852 | -0.0908 ± 0.0052 |

The pretraining signal persisted in the favorable direction for every metric in
all three paired seeds. This is stronger engineering evidence than the original
single-seed pilot: under this low-label synthetic setup, pretrained fine-tuning
consistently improved ranking and calibration over training the same architecture
from scratch. It remains a small-sample Synthea experiment, not a clinical
efficacy claim or a substitute for evaluation on representative real-world data.
The next useful expansion is the registered 100%, 25%, and 10% label-fraction
matrix, retaining the same paired seeds.

The original execution report with AIR run IDs is stored in
`runs/demo-20260915T210144Z.json`; the successful no-training replay with embedded
metrics is `runs/demo-20260915T210343Z.json`. Both downstream checkpoint roots
contain `training_result.json`, and the snapshot reported zero temporal leakage
rows. The complete three-seed aggregation is stored in
`runs/demo-20260921T113043Z.json`.
