# Getting started with ExMed-BERT

Choose the workflow that matches your goal. The recommended demonstration uses
Databricks AI Runtime; the original local and Mosaic Streaming paths remain in
the repository for model development and migration comparisons.

## Databricks AI Runtime demo

Use this path to demonstrate governed Spark preparation, Unity Catalog volume
data loading, GPU training, distributed checkpoints, recovery, and MLflow
evidence with synthetic Synthea data.

Prerequisites:

- Databricks CLI 1.0 or newer and the AIR CLI.
- An authenticated Databricks CLI profile with access to the configured Unity
  Catalog volume.
- GNU tar on macOS.
- The pinned Synthea release described in the
  [Synthea generation guide](../experiments/ai_runtime_exmedbert/synthea/README.md).

From `experiments/ai_runtime_exmedbert`:

```bash
python3 demo.py plan --profile fevm-default
python3 demo.py validate --profile fevm-default
python3 demo.py run --profile fevm-default --resume-proof --include-comparators
```

Replace `fevm-default` with your explicitly authenticated profile. Read the
[AI Runtime project guide](../experiments/ai_runtime_exmedbert/README.md) for
storage configuration, Spark publication, acceptance criteria, and individual
AIR commands. Use the [demo runbook](../experiments/ai_runtime_exmedbert/DEMO.md)
for restart behavior and operator details.

## Local model development

The original Python scripts support focused model, ClinVec, and synthetic-data
experiments outside AI Runtime. Begin with:

- [Local training guide](TRAINING_GUIDE.md)
- [ClinVec integration](ClinVec_Integration.md)
- [Local synthetic test generator](SYNTHETIC_DATA_GUIDE.md)
- [Local troubleshooting](TROUBLESHOOTING.md)

Some historical Windows launcher references remain in the local training guide,
but those launchers are not present in this repository. Invoke the documented
Python scripts directly.

## Classic cluster and Mosaic Streaming

Use this path only when maintaining or comparing the original classic-cluster
implementation:

- [Mosaic Streaming quick start](STREAMING_QUICKSTART.md)
- [Mosaic Streaming migration guide](MOSAIC_STREAMING_MIGRATION.md)
- [Classic-cluster GPU smoke test](MINIMAL_GPU_TEST.md)
- [MLflow integration for streaming training](MLFLOW_DATABRICKS_GUIDE.md)

The Mosaic path is not used by the AI Runtime demo.

## Documentation index

See the [documentation map](README.md) for the complete routing table.
