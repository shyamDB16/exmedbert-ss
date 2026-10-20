# MLflow Integration with Databricks & Unity Catalog

> **Scope: classic/Mosaic Streaming path.** This guide documents explicit
> MLflow integration in `scripts/train_entrypoint_streaming.py`. AI Runtime
> creates and manages training-run observability differently; start with the
> [AI Runtime project](../experiments/ai_runtime_exmedbert/README.md) and its
> [demo runbook](../experiments/ai_runtime_exmedbert/DEMO.md).

This guide explains how to use MLflow tracking and Unity Catalog Model Registry with ExMed-BERT streaming training on Databricks.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Training with MLflow](#training-with-mlflow)
4. [Viewing Experiments](#viewing-experiments)
5. [Model Registry](#model-registry)
6. [What Gets Tracked](#what-gets-tracked)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The ExMed-BERT streaming implementation includes comprehensive MLflow integration that automatically:

✅ **Tracks all training metrics** (loss, accuracy, learning rate)  
✅ **Logs all hyperparameters** (model architecture, training config, streaming settings)  
✅ **Logs model artifacts** to MLflow  
✅ **Registers models** to Unity Catalog Model Registry  
✅ **Tracks data lineage** (UC Volume paths, dataset sizes)  
✅ **Auto-detects Databricks environment**  

##Setup

### Prerequisites

1. **Databricks Workspace** with Unity Catalog enabled
2. **MLflow** installed (pre-installed on Databricks Runtime 15.2 ML+)
3. **Unity Catalog** configured with catalog and schema
4. **Access Token** for authentication (if running externally)

### Authentication (if needed)

If running from outside Databricks or in distributed training:

```bash
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-personal-access-token"
```

**To create a token:**
1. Go to Databricks UI → User Settings
2. Click "Developer" → "Access tokens"
3. Click "Generate new token"
4. Copy and save the token securely

---

## Training with MLflow

### Basic Usage

```bash
python scripts/train_entrypoint_streaming.py \
    --use-uc-volumes \
    --uc-catalog "main" \
    --uc-schema "exmedbert" \
    --uc-volume "training_data"
```

**Result:**
- ✅ Metrics logged to MLflow automatically
- ✅ Experiment created: `/Users/{your_username}/exmedbert-streaming`
- ✅ Model registered as: `main.exmedbert.exmedbert_streaming`

### Custom Experiment and Model Names

```bash
python scripts/train_entrypoint_streaming.py \
    --use-uc-volumes \
    --uc-catalog "main" \
    --uc-schema "exmedbert" \
    --uc-volume "training_data" \
    --mlflow-experiment-name "/Users/me/my-exmedbert-experiments" \
    --mlflow-run-name "exp1-pretrain-v1" \
    --uc-model-name "exmedbert_pretrained_v1" \
    --model-alias "Champion"
```

### Disable Model Registration

If you only want tracking without registration:

```bash
python scripts/train_entrypoint_streaming.py \
    ... \
    --no-register-model
```

### Python API

```python
from scripts.pretrain_exmed_bert_streaming import pretrain

pretrain(
    train_remote="/Volumes/main/exmedbert/data/mds/train",
    val_remote="/Volumes/main/exmedbert/data/mds/val",
    embeddings_path="/Volumes/main/exmedbert/data/embeddings.pt",
    output_dir="output/streaming_pretrain",
    output_data_dir="output/streaming_pretrain_data",
    
    # MLflow settings
    mlflow_experiment_name="/Users/me/exmedbert-experiments",
    mlflow_run_name="streaming-pretrain-20260129",
    log_model=True,
    
    # UC Model Registry settings
    register_model=True,
    uc_model_catalog="main",
    uc_model_schema="exmedbert",
    uc_model_name="exmedbert_streaming_v1",
    model_alias="Champion",
    
    # ... other training params
)
```

---

## Viewing Experiments

### In Databricks UI

1. **Navigate to Experiments:**
   - Click "Machine Learning" in left sidebar
   - Click "Experiments"
   - Find your experiment (e.g., `/Users/you/exmedbert-streaming`)

2. **View Runs:**
   - Click on experiment name
   - See all runs with metrics, parameters, and artifacts
   - Compare runs side-by-side

3. **Explore Run Details:**
   - Click on a run name
   - **Metrics tab**: View loss curves, accuracy graphs
   - **Parameters tab**: See all hyperparameters
   - **Artifacts tab**: Download model, configs
   - **System Metrics tab**: GPU utilization, memory usage

### Programmatically

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("databricks")

# Get experiment
experiment = mlflow.get_experiment_by_name("/Users/me/exmedbert-streaming")

# List runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
print(runs[["run_id", "metrics.eval_loss", "params.learning_rate"]])

# Get specific run
run = mlflow.get_run(run_id="abc123...")
print(run.data.metrics)
print(run.data.params)
```

---

## Model Registry

### Registered Models in Unity Catalog

After training, your model is automatically registered to Unity Catalog:

**Model Name Format:** `{catalog}.{schema}.{model_name}`  
**Example:** `main.exmedbert.exmedbert_streaming_v1`

### View Models in UC

#### Databricks UI

1. Click "Catalog" in left sidebar
2. Navigate to: Catalog → Schema → Models
3. Click on model name
4. See versions, aliases, tags

#### Python API

```python
from mlflow import MlflowClient

client = MlflowClient()

# Get model info
model_name = "main.exmedbert.exmedbert_streaming"
model = client.get_registered_model(model_name)
print(f"Model: {model.name}")
print(f"Description: {model.description}")

# List versions
versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")
```

### Model Aliases

Set aliases to mark specific versions:

```python
client.set_registered_model_alias(
    name="main.exmedbert.exmedbert_streaming",
    alias="Champion",  # or "Production", "Staging", "Challenger"
    version=3
)
```

**Common Aliases:**
- **Champion**: Best performing model
- **Production**: Currently deployed model
- **Staging**: Model under testing
- **Challenger**: New model being evaluated

### Load Model for Inference

```python
import mlflow

# Load by version
model = mlflow.transformers.load_model(
    "models:/main.exmedbert.exmedbert_streaming/3"
)

# Load by alias
model = mlflow.transformers.load_model(
    "models:/main.exmedbert.exmedbert_streaming@Champion"
)

# Use for inference
predictions = model.predict(input_data)
```

---

## What Gets Tracked

### Training Metrics (Auto-logged every `logging_steps`)

```
train_loss: 2.45, 2.32, 2.18, ...
train_learning_rate: 3e-5, 2.8e-5, 2.5e-5, ...
train_epoch: 0.1, 0.2, 0.3, ...
train_runtime: 150.2
```

### Evaluation Metrics (Auto-logged every `eval_steps`)

```
eval_loss: 2.40, 2.28, 2.15, ...
eval_code_accuracy: 0.65, 0.72, 0.78, ...
eval_plos_accuracy: 0.58, 0.63, 0.69, ...
eval_runtime: 15.3
eval_samples_per_second: 650.2
```

### Model Architecture Parameters

```
model_hidden_size: 288
model_num_layers: 6
model_num_heads: 6
model_intermediate_size: 512
model_max_seq_length: 512
model_code_vocab_size: 15420
model_age_vocab_size: 100
model_sex_vocab_size: 3
model_state_vocab_size: 52
```

### Training Configuration Parameters

```
train_learning_rate: 3e-5
train_batch_size: 124
train_epochs: 1
train_warmup_steps: 1000
train_grad_accum_steps: 1
train_weight_decay: 0.01
train_max_steps: -1
train_seed: 201214
```

### Streaming-Specific Parameters

```
data_train_remote: /Volumes/main/exmedbert/data/mds/train
data_val_remote: /Volumes/main/exmedbert/data/mds/val
data_embeddings_path: /Volumes/main/exmedbert/data/embeddings.pt
streaming_local_cache: /tmp/mds_cache
streaming_predownload: 256
streaming_shuffle_seed: 201214
dataset_train_size: 50000
dataset_val_size: 10000
uc_catalog: main
uc_schema: exmedbert
uc_volume: training_data
```

### System Information

```
system_platform: Linux-5.4.0-...-x86_64
system_python_version: 3.11.5
system_pytorch_version: 2.1.0
system_cuda_available: True
system_cuda_version: 12.1
system_gpu_count: 4
system_gpu_name: Tesla V100-SXM2-32GB
system_databricks_runtime: 16.4.x-gpu-ml-scala2.12
```

### Model Artifacts

```
model/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
└── training_args.bin
```

---

## Best Practices

### 1. Experiment Organization

```bash
# Use hierarchical experiment names
/Users/{username}/exmedbert-experiments/pretraining
/Users/{username}/exmedbert-experiments/finetuning
/Shared/team/exmedbert/production-experiments
```

### 2. Run Naming Convention

```bash
# Include key info in run names
--mlflow-run-name "pretrain-{date}-{config}-{version}"
--mlflow-run-name "pretrain-20260129-large-v1"
```

### 3. Model Versioning

```python
# Use semantic versioning in model names
uc_model_name = "exmedbert_pretrained_v1_0_0"
uc_model_name = "exmedbert_finetuned_cardiology_v2_1_0"
```

### 4. Model Aliases

```python
# Workflow: New Model → Challenger → Champion → Production
# 1. Train new model (version 5)
# 2. Set as Challenger
client.set_registered_model_alias(name=model_name, alias="Challenger", version=5)

# 3. Evaluate against Champion
# 4. If better, promote to Champion
client.set_registered_model_alias(name=model_name, alias="Champion", version=5)

# 5. Deploy to production
client.set_registered_model_alias(name=model_name, alias="Production", version=5)
```

### 5. Tagging Models

```python
# Add descriptive tags
client.set_model_version_tag(
    name=model_name,
    version=5,
    key="use_case",
    value="cardiovascular_risk_prediction"
)
```

---

## Troubleshooting

### Issue 1: "No active MLflow run"

**Error:** `MlflowException: No active run to log to`

**Solution:** This shouldn't happen with the new implementation, but if it does:
```python
# Make sure report_to=["mlflow"] is set in TrainingArguments
training_config = TrainingArguments(
    ...,
    report_to=["mlflow"],
)
```

### Issue 2: Model Registration Fails

**Error:** `MlflowException: Model registration failed`

**Solutions:**
1. Check Unity Catalog permissions:
   ```sql
   -- In Databricks SQL
   GRANT CREATE MODEL ON SCHEMA main.exmedbert TO `your_user`;
   ```

2. Verify catalog and schema exist:
   ```sql
   SHOW CATALOGS;
   SHOW SCHEMAS IN main;
   ```

3. Disable registration temporarily:
   ```bash
   --no-register-model
   ```

### Issue 3: "Databricks authentication required"

**Error:** `ValueError: cannot configure default credentials`

**Solution:**
```bash
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"
```

### Issue 4: Metrics Not Showing Up

**Problem:** Training runs but no metrics in MLflow

**Solutions:**
1. Check `report_to` parameter is set correctly
2. Verify MLflow experiment is set before training
3. Check logs for MLflow errors
4. Ensure `logging_steps` > 0

### Issue 5: Can't Find Model in UC

**Problem:** Model trained but not visible in Unity Catalog

**Solutions:**
1. Check if registration was enabled: `--register-model`
2. Verify you have permissions to the catalog/schema
3. Check MLflow UI → Artifacts tab for model
4. Try registering manually:
   ```python
   mlflow.register_model(
       f"runs:/{run_id}/model",
       "main.exmedbert.my_model"
   )
   ```

---

## Advanced Usage

### Custom Metrics Callback

If you need to log additional custom metrics:

```python
from transformers import TrainerCallback
import mlflow

class CustomMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Log custom metrics
            if "custom_metric" in logs:
                mlflow.log_metric(
                    "custom_metric",
                    logs["custom_metric"],
                    step=state.global_step
                )

# Add to trainer
trainer = MedBertStreamingTrainer(
    ...,
    callbacks=[CustomMetricsCallback()],
)
```

### Compare Runs Programmatically

```python
import mlflow
import pandas as pd

# Get experiment
experiment = mlflow.get_experiment_by_name("/Users/me/exmedbert")

# Get all runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.eval_loss ASC"],
    max_results=10
)

# Compare key metrics
comparison = runs[[
    "run_id",
    "start_time",
    "metrics.eval_loss",
    "metrics.eval_code_accuracy",
    "params.learning_rate",
    "params.train_batch_size"
]]
print(comparison)
```

### Export Run Data

```python
# Export run to pandas DataFrame
run = mlflow.get_run("run_id_here")

# Metrics
metrics_df = pd.DataFrame([run.data.metrics])

# Parameters
params_df = pd.DataFrame([run.data.params])

# Save to CSV
metrics_df.to_csv("run_metrics.csv")
params_df.to_csv("run_params.csv")
```

---

## References

- [Databricks MLflow Documentation](https://docs.databricks.com/mlflow/index.html)
- [Unity Catalog Model Registry](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [HuggingFace Trainer MLflow Integration](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.report_to)

---

**Need Help?**

- Check Databricks UI → Experiments → Your Run → Logs tab
- Review training logs in `output/*/output.log`
- Check MLflow tracking logs in Databricks

**For more information:** See [MOSAIC_STREAMING_MIGRATION.md](./MOSAIC_STREAMING_MIGRATION.md)

