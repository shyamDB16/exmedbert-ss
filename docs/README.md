# Documentation map

The recommended repository path is the Databricks AI Runtime demonstration.
Older local, classic-cluster, and Mosaic Streaming documents remain available
for model background, comparison, and rollback, but they are not prerequisites
for the AI Runtime workflow.

## AI Runtime

| Document | Purpose |
|---|---|
| [AI Runtime architecture](AI_RUNTIME_DATA_AND_CHECKPOINTING.md) | Spark boundary, immutable Parquet snapshots, UC volume loading, and distributed checkpointing |
| [AI Runtime project](../experiments/ai_runtime_exmedbert/README.md) | Implemented data contract, training workflow, and acceptance sequence |
| [Demo runbook](../experiments/ai_runtime_exmedbert/DEMO.md) | End-to-end operator commands |
| [Synthea generation](../experiments/ai_runtime_exmedbert/synthea/README.md) | Pinned synthetic source generation and publication |
| [Verified results](../experiments/ai_runtime_exmedbert/RESULTS.md) | Workspace evidence, recovery results, and limitations |

## Classic and Mosaic Streaming path

These documents describe the retained classic-cluster implementation. They do
not describe the AI Runtime data loader or checkpoint lifecycle.

- [Mosaic Streaming migration](MOSAIC_STREAMING_MIGRATION.md)
- [Mosaic Streaming quick start](STREAMING_QUICKSTART.md)
- [Classic-cluster GPU smoke test](MINIMAL_GPU_TEST.md)
- [MLflow integration for streaming training](MLFLOW_DATABRICKS_GUIDE.md)

## Local model-development references

These guides are runtime-neutral or describe the original local workflow:

- [ClinVec integration](ClinVec_Integration.md)
- [Local training guide](TRAINING_GUIDE.md)
- [Local synthetic test generator](SYNTHETIC_DATA_GUIDE.md)
- [Local environment troubleshooting](TROUBLESHOOTING.md)

Start at [Getting started](GETTING_STARTED.md) if you are deciding which path
to use.
