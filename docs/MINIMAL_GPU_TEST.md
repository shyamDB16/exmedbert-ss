# Minimal GPU Test for ExMed-BERT Pre-training

This guide lets you **smoke-test** pre-training on a Databricks GPU cluster (or any machine with GPUs) so you can validate the repo before sharing with a customer. Two paths: **quick (no streaming)** and **streaming**.

---

## Quick run via SSH tunnel (multi-GPU cluster)

If you use the **Databricks SSH tunnel** and have a **multi-GPU** cluster:

1. **Connect:** Remote-SSH to your cluster and open the repo folder (e.g. `exmedbert-ss` or its parent under `/Workspace/Users/<you>/...`).

2. **One-time setup** (from repo root):
   ```bash
   pip install -r requirements.txt
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   python scripts/create_dummy_dataset.py --output-dir pretrain_data
   ```

3. **Run the minimal test** (5 steps, uses all visible GPUs):
   ```bash
   TRAIN_PT=pretrain_data/dummy_train.pt VAL_PT=pretrain_data/dummy_val.pt \
     ./scripts/run_minimal_pretrain_test.sh
   ```

4. **Optional – persist checkpoints** across restarts: set before the run:
   ```bash
   export OUTPUT_DIR=/Workspace/Users/$(whoami)/exmedbert-ss/output/minimal_test_pretrain
   export OUTPUT_DATA_DIR=/Workspace/Users/$(whoami)/exmedbert-ss/output/minimal_test_pretrain_data
   ```

**Verify:** `output/minimal_test_pretrain` (or your `OUTPUT_DIR`) should contain checkpoints and logs; `nvidia-smi` or cluster GPU metrics should show utilization. The script auto-detects multiple GPUs (e.g. "Using 4 gpus").

---

## Do we have everything to run training?

**Yes.** The repo is self-contained for training:

| Need | In repo |
|------|--------|
| Pre-training code | `scripts/pretrain-exmed-bert-clinvec.py`, `scripts/pretrain-exmed-bert-streaming.py`, `exmed_bert/*` |
| Entrypoints | `scripts/train_entrypoint.py`, `scripts/train_entrypoint_streaming.py` |
| Minimal test | `scripts/run_minimal_pretrain_test.sh`, `scripts/save_embeddings_from_pt.py` |
| Dependencies list | `requirements.txt` (typer, mosaicml-streaming, etc.) |
| Docs | `docs/MINIMAL_GPU_TEST.md`, `docs/STREAMING_QUICKSTART.md`, `docs/MLFLOW_DATABRICKS_GUIDE.md` |

**You must provide:** (1) **Databricks cluster** with GPU + ML runtime (PyTorch, transformers, MLflow are on the runtime). (2) **Sample data:** use the **dummy dataset** (no real data) or your own `.pt` / MDS data.

---

## Run via Databricks SSH tunnel

If you use the [Databricks SSH tunnel](https://docs.databricks.com/aws/en/dev-tools/ssh-tunnel) (e.g. from Cursor/VS Code):

1. **Clone repo in persistent storage** (required so Git and files survive restarts):  
   In an SSH terminal on the cluster:  
   `cd /Workspace/Users/<your-username>` then `git clone <your-repo-url> ExMedBert_PRL`  
   Open that folder in your IDE (e.g. **Remote-SSH: Connect to Host** → open `/Workspace/Users/<your-username>/ExMedBert_PRL`).

2. **Install deps and set PYTHONPATH** (once per session or use a setup notebook):  
   `pip install -r requirements.txt`  
   `export PYTHONPATH="${PWD}:${PYTHONPATH}"`  
   Select the cluster’s Python interpreter (**Python: Select Interpreter** → `pythonEnv-xxx`) so you get the ML runtime libs.

3. **Run the minimal test** from repo root:  
   `./scripts/run_minimal_pretrain_test.sh`  
   (Requires `pretrain_data/komodo_panta_sample_train.pt` and `_val.pt`.)

**Tip:** Use an **output dir under `/Workspace` or `/Volumes`** if you want checkpoints to persist across cluster restarts (e.g. set `OUTPUT_DIR=/Workspace/Users/<you>/ExMedBert_PRL/output/minimal_test_pretrain` before running the script). Cluster local disk (`/home`, `/root`) is ephemeral.

---

## Prerequisites

- **Cluster:** Databricks cluster with **GPU** (e.g. single-node or multi-node with 1+ GPU per node). Use a **ML Runtime** (e.g. “13.3 LTS ML (GPU)”) so PyTorch, transformers, and MLflow are already installed.
- **Repo:** Clone/copy the repo to the cluster (e.g. workspace folder or `/Workspace/Repos/...`).
- **Sample data:** The existing sample `.pt` files used by the original entrypoint:
  - `pretrain_data/komodo_panta_sample_train.pt`
  - `pretrain_data/komodo_panta_sample_val.pt`  
  Or create dummy data (no real patients): `python scripts/create_dummy_dataset.py --output-dir pretrain_data`.

---

## 1. Environment on the cluster

On the cluster (notebook or job Python environment), install dependencies and set `PYTHONPATH`:

```bash
# From repo root
pip install -r requirements.txt
# Databricks ML runtimes already include torch, transformers, mlflow.
# You may only need: pip install typer mosaicml-streaming

export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

For a **Databricks notebook**, use a **GPU** runtime (e.g. “13.3 LTS ML (GPU)” or later) and run the same `pip install` in the first cell if needed.

---

## 2. Quick smoke test (no streaming)

Uses the **original** pre-training script with **sample .pt** data and runs for a **few steps** to confirm GPU, imports, training loop, and MLflow.

### Option A: Shell script (SSH / terminal on cluster)

From the **repo root**, create dummy data then run the test:

```bash
# One-time: create dummy train/val .pt (no real patient data)
python scripts/create_dummy_dataset.py --output-dir pretrain_data

# Run minimal test (use dummy data)
TRAIN_PT=pretrain_data/dummy_train.pt VAL_PT=pretrain_data/dummy_val.pt \
  ./scripts/run_minimal_pretrain_test.sh
```

If you already have sample `.pt` files, you can skip the first step and run `./scripts/run_minimal_pretrain_test.sh` (defaults: `pretrain_data/komodo_panta_sample_train.pt` and `_val.pt`). This runs ~5 training steps and writes to `output/minimal_test_*`.

### Option B: Direct Python (notebook or job)

From the **repo root** in Python:

```python
import os
import sys
import subprocess

os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

# Paths (adjust if your sample data lives elsewhere)
DATASET_DIR = "pretrain_data"
TRAIN_DATASET = f"{DATASET_DIR}/komodo_panta_sample_train.pt"
VAL_DATASET = f"{DATASET_DIR}/komodo_panta_sample_val.pt"
OUTPUT_DIR = "output/minimal_test_pretrain"
OUTPUT_DATA_DIR = "output/minimal_test_pretrain_data"

cmd = [
    sys.executable,
    "scripts/pretrain-exmed-bert-clinvec.py",
    TRAIN_DATASET,
    VAL_DATASET,
    OUTPUT_DIR,
    OUTPUT_DATA_DIR,
    "--train-batch-size", "8",
    "--eval-batch-size", "4",
    "--max-steps", "5",
    "--num-attention-heads", "6",
    "--num-hidden-layers", "6",
    "--hidden-size", "288",
    "--intermediate-size", "512",
    "--epochs", "1",
    "--max-seq-length", "512",
    "--logging-steps", "2",
    "--eval-steps", "5",
    "--save-steps", "5",
    "--dynamic-masking",
]
subprocess.run(cmd, check=True)
```

**What this checks:**  
Data load, model build, GPU usage, training steps, logging, and that `output/` (and optionally MLflow) get written. No MDS/streaming.

---

## 3. Optional: Smoke test with streaming

Only run this if you already have **MDS data** and an **embeddings** file (e.g. from your Spark pipeline or from the helper below).

### 3.1 Save embeddings from existing .pt (one-time)

If you have a sample `.pt` dataset (e.g. `pretrain_data/dummy_train.pt` or `komodo_panta_sample_train.pt`), extract embeddings for the streaming script:

```bash
python scripts/save_embeddings_from_pt.py \
  pretrain_data/dummy_train.pt \
  pretrain_data/embeddings.pt
```

This creates `pretrain_data/embeddings.pt` for use with streaming pre-training.

### 3.2 Create minimal MDS data

**Option A: From .pt files (no Spark)** — use this for a quick streaming smoke test:

```bash
# One-time: create dummy data and embeddings (if you don't have them)
python scripts/create_dummy_dataset.py --output-dir pretrain_data
python scripts/save_embeddings_from_pt.py pretrain_data/dummy_train.pt pretrain_data/embeddings.pt

# Convert .pt to MDS (writes pretrain_data_mds/train and pretrain_data_mds/val)
python scripts/convert_pt_to_mds.py \
  --train-pt pretrain_data/dummy_train.pt \
  --val-pt pretrain_data/dummy_val.pt \
  --output-dir pretrain_data_mds
```

**Option B: On Databricks with Spark**

- Use your existing Spark pipeline to build a **small** table (e.g. 100–500 rows) with the same schema as production.
- Run `scripts/convert_spark_to_mds.py` (or the same logic in a notebook) with that table and a small output path (e.g. `/tmp/minimal_mds` or a UC volume path).
- You’ll get something like: `{out}/train/` and `{out}/val/`. Use the same `embeddings.pt` as in 3.1.

### 3.3 Run streaming pre-training (minimal steps)

From repo root, with `train_remote` and `val_remote` pointing at your minimal MDS dirs (local or UC):

```bash
# If testing without workspace MLflow (e.g. local file tracking), unset so script uses file:.../mlruns
unset DATABRICKS_RUNTIME_VERSION

python scripts/pretrain-exmed-bert-streaming.py \
  pretrain_data_mds/train \
  pretrain_data_mds/val \
  pretrain_data/embeddings.pt \
  output/minimal_streaming_test \
  output/minimal_streaming_test_data \
  --local-cache-dir /tmp/mds_cache \
  --max-steps 5 \
  --train-batch-size 8 \
  --eval-batch-size 4 \
  --logging-steps 2 \
  --eval-steps 5 \
  --save-steps 5 \
  --no-register-model \
  --no-log-model
```

- Use `--no-register-model` to skip UC model registration.
- Use `--no-log-model` to skip logging the model artifact to MLflow (avoids shared-weights save issues on a quick test).
- If you see workspace experiment errors (e.g. `/Users/root` not found), run with `unset DATABRICKS_RUNTIME_VERSION` so MLflow uses local file tracking under the output data dir.

This checks: MDS load, StreamingDataLoader, GPU training, checkpointing, and (if enabled) MLflow logging.

---

### 3.4 Test with a pre-created MLflow experiment (workspace or UC)

If you run **directly on the workspace** (not via SSH as root), the default experiment path uses your user and works. To use a **specific experiment** (e.g. one you created in the UI or with artifacts in a Unity Catalog volume), create the experiment first, then pass its name:

1. **Create the experiment** (one of these):
   - **UI:** In Databricks, go to **Workspace** → right‑click a folder → **Create** → **MLflow experiment**. Give it a name (e.g. `exmedbert-streaming`). Optionally set **Artifact location** to a UC Volume path: `dbfs:/Volumes/<catalog>/<schema>/<volume>`.
   - **API (UC Volume artifacts):** In a notebook or script:
     ```python
     import mlflow
     EXP_NAME = "/Users/shyam.sankararaman@databricks.com/exmedbert-streaming"  # or /Shared/exmedbert-streaming
     ARTIFACT_PATH = "dbfs:/Volumes/<catalog>/<schema>/<volume>"
     mlflow.set_tracking_uri("databricks")
     if mlflow.get_experiment_by_name(EXP_NAME) is None:
         mlflow.create_experiment(name=EXP_NAME, artifact_location=ARTIFACT_PATH)
     ```

2. **Run the test** with that experiment:
   ```bash
   python scripts/pretrain-exmed-bert-streaming.py \
     pretrain_data_mds/train pretrain_data_mds/val pretrain_data/embeddings.pt \
     output/minimal_streaming output/minimal_streaming_data \
     --mlflow-experiment-name "/Users/shyam.sankararaman@databricks.com/exmedbert-streaming" \
     --max-steps 5 --no-register-model --local-cache-dir /tmp/mds_cache
   ```
   Use the exact experiment name/path you created. With this setup, **`/Users/root` is not used**; your experiment lives under your user (or Shared/UC), so there is no parent-directory error.

3. **Check:** In **Machine Learning** → **Experiments**, open the experiment and confirm a new run with params and metrics.

---

## 4. What to verify before sharing with the customer

- [ ] **Quick test (no streaming)** runs to completion (a few steps and one eval).
- [ ] **GPU** is used (e.g. `nvidia-smi` or Databricks GPU utilization).
- [ ] **Output dir** contains checkpoint(s) and logs (e.g. `output/minimal_test_pretrain/`).
- [ ] **MLflow:** If tracking is enabled, one run appears with metrics/params (and optionally one model artifact).
- [ ] (Optional) **Streaming test** runs with minimal MDS + embeddings and writes checkpoints + dataloader state.

---

## 5. Minimal setup summary for the customer

You can share the repo and this guidance:

1. **Requirements:** GPU cluster (Databricks or other), sample `.pt` data under `pretrain_data/` (or their own paths).
2. **Install:** `pip install -r requirements.txt` and set `PYTHONPATH` to repo root.
3. **Smoke test:** Run `./scripts/run_minimal_pretrain_test.sh` (or the Python snippet in 2.B) from repo root.
4. **Streaming:** Only if they use MDS: create minimal MDS + embeddings, then run `pretrain-exmed-bert-streaming.py` with `--max-steps 5` and `--no-register-model` for a quick check.

No need to run full epochs or large data for this minimal validation.

---

## Pre-training on a different dataset

If you don’t have the sample patient `.pt` files and want to run pre-training on **another dataset**, rework is **small to moderate** as long as your data can be mapped to the **same data contract** the model and trainer expect.

### What the repo assumes (data contract)

Every batch must be a dict with at least:

| Key            | Shape / type        | Meaning |
|----------------|---------------------|--------|
| `codes`        | `(batch, seq_len)` long | Token IDs (e.g. medical codes). |
| `ages`         | `(batch, seq_len)` long | Age token per position. |
| `visits`       | `(batch, seq_len)` long | Visit index per position. |
| `segments`     | `(batch, seq_len)` long | Segment/type IDs. |
| `sex`          | `(batch,)` long     | Sex token per sample. |
| `state`        | `(batch,)` long     | Region/state token per sample. |
| `attention_mask` | `(batch, seq_len)` bool | Valid (non-pad) positions. |
| `code_labels`  | `(batch, seq_len)` long | MLM labels (-100 where not predicted). |
| `code_mask`    | `(batch, seq_len)` bool | Positions to predict (optional if labels use -100). |
| `plos_label`   | `(batch,)` float or long | Scalar label (e.g. length-of-stay); can be dummy if unused. |

Vocabularies: **CodeDict**, **AgeDict**, **SexDict**, **StateDict** (and **EndpointDict** for fine-tuning). They are just id ↔ entity mappings; you build them from your own code lists / demographics.

### What can stay as-is

- **Model and config** (`exmed_bert/models/`) – no change.
- **Trainer and training scripts** – they only need a Dataset that returns the dict above.
- **Metrics, MLflow, checkpointing** – unchanged.
- **Streaming / MDS** – same; your Spark (or other) pipeline just writes the same column names and types.

### What you need to add or change

1. **Data adapter**  
   A way to go from your raw data → the batch dict above. Options:
   - Implement a small **PyTorch Dataset** that loads your format and returns that dict (no need to use `Patient` or `PatientDataset`).
   - Or, if you keep the same “sequence of codes + demographics” idea: build **Patient**-like objects from your schema and reuse `PatientDataset`; then only the code that **creates** those objects from your source changes.

2. **Vocabulary construction**  
   From your code lists and demographics, build:
   - **CodeDict** (and optionally AgeDict, SexDict, StateDict) with the same interface as in `exmed_bert/data/encoding.py` (they’re generic dicts + special tokens like PAD, MASK).
   - Save them (e.g. in the same `.pt` as the dataset or a separate `embeddings.pt`) and pass them into the training script.

3. **Optional: no “length of stay”**  
   If you don’t have a scalar target: pass a dummy `plos_label` (e.g. zeros). The PLOS head will still run; you can later disable or reweight that loss with a small config/trainer change if needed.

4. **Streaming (if you use MDS)**  
   Your Spark (or other) job should output columns that match the MDS schema in `scripts/convert_spark_to_mds.py` (codes, ages, visits, segments, sex, state, code_labels, code_mask, plos_label, etc.). Often this is just column renaming in Spark.

### Effort summary

| Situation | Rework |
|-----------|--------|
| New data source, same “codes + demographics + optional scalar label” | **Low–moderate**: new data loader + vocab building; model and training scripts unchanged. |
| Same structure but different code system (e.g. other vocabularies) | **Low**: new vocabs + adapter from your codes to token IDs. |
| Completely different modality (e.g. free text, images) | **High**: different model and input pipeline; not just a data swap. |

So for “an entirely different dataset”: if it can be expressed as **sequences of discrete tokens plus demographics and optional scalar labels**, you mainly add a **data adapter** and **vocabularies**; the rest of the repo can stay.
