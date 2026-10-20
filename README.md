# ExMed-BERT on Databricks

This repository contains ExMed-BERT pretraining and downstream evaluation
workflows for longitudinal healthcare event data. The recommended demonstration
uses synthetic Synthea data, serverless Spark preparation, and Databricks AI
Runtime GPU training.

Synthea is synthetic. The demo establishes platform compatibility,
reproducibility, checkpoint recovery, and experiment mechanics; its metrics are
not evidence of clinical utility.

## Recommended path: Databricks AI Runtime

The AI Runtime implementation separates data engineering from model training:

```text
Synthea or governed Delta sources
  -> serverless Spark quality gates and immutable Parquet snapshots
  -> Unity Catalog volume
  -> AI Runtime ExMed-BERT training with UCVolumeDataset
  -> distributed checkpoints with UCVolumeWriter and UCVolumeReader
  -> MLflow evidence
```

Start with the [AI Runtime project](experiments/ai_runtime_exmedbert/README.md).
The packaged operator workflow is documented in its
[demo runbook](experiments/ai_runtime_exmedbert/DEMO.md).

From `experiments/ai_runtime_exmedbert`:

```bash
python3 demo.py plan --profile fevm-default
python3 demo.py validate --profile fevm-default
python3 demo.py run --profile fevm-default --resume-proof --include-comparators
```

Replace `fevm-default` with an authenticated Databricks CLI profile for your
workspace. The source dataset generation and publication procedure is in the
[Synthea guide](experiments/ai_runtime_exmedbert/synthea/README.md).

## Documentation

| Document | Purpose |
|---|---|
| [AI Runtime project](experiments/ai_runtime_exmedbert/README.md) | Architecture, data contract, training, and acceptance sequence |
| [Demo runbook](experiments/ai_runtime_exmedbert/DEMO.md) | Reproducible end-to-end operator workflow |
| [AI Runtime data and checkpointing](docs/AI_RUNTIME_DATA_AND_CHECKPOINTING.md) | Design rationale for UC volume loading and distributed checkpoints |
| [Workspace results](experiments/ai_runtime_exmedbert/RESULTS.md) | Verified runs, recovery evidence, and limitations |
| [Synthea generation](experiments/ai_runtime_exmedbert/synthea/README.md) | Pinned synthetic dataset generation and publication |

## Classic cluster path

The original implementation based on Mosaic Streaming remains available for
comparison, rollback, and classic-cluster users. It is not used by the AI
Runtime workflow.

- [Mosaic Streaming implementation](MOSAIC_STREAMING_IMPLEMENTATION.md)
- [Mosaic Streaming migration guide](docs/MOSAIC_STREAMING_MIGRATION.md)
- [MLflow integration for the Mosaic path](MLFLOW_INTEGRATION_SUMMARY.md)
- [Streaming quick start](docs/STREAMING_QUICKSTART.md)

## Repository layout

```text
exmed_bert/                         Core ExMed-BERT model and data APIs
experiments/ai_runtime_exmedbert/  Recommended AI Runtime demonstration
experiments/ai_runtime_synthea/    Minimal AI Runtime data-transport proof
scripts/                            Classic/Mosaic utilities and entry points
docs/                               Architecture and operational guides
```
