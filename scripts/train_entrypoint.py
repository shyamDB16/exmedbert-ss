# train_entrypoint.py
import os
import sys
import subprocess


def main():
    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

    # Debug flags (use only for debugging; disable for performance)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    print("==========================================")
    print("ExMed-BERT Pretraining on Synthetic Data")
    print("==========================================")

    # ------------------------------------------------------------------
    # Paths (datasets already exist)
    # ------------------------------------------------------------------
    DATASET_DIR = "pretrain_data"
    TRAIN_DATASET = f"{DATASET_DIR}/komodo_panta_sample_train.pt"
    VAL_DATASET = f"{DATASET_DIR}/komodo_panta_sample_val.pt"

    OUTPUT_DIR = "output/komodo_panta_sample_pretrain"
    OUTPUT_DATA_DIR = "output/komodo_panta_sample_pretrain_data"

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------
    print("\nStarting ExMed-BERT pretraining...")
    print(f"  Training data: {TRAIN_DATASET}")
    print(f"  Validation data: {VAL_DATASET}")
    print(f"  Output: {OUTPUT_DIR}\n")

    cmd = [
        sys.executable,
        "scripts/pretrain-exmed-bert-clinvec.py",
        TRAIN_DATASET,
        VAL_DATASET,
        OUTPUT_DIR,
        OUTPUT_DATA_DIR,
        "--train-batch-size", "124",
        "--eval-batch-size", "5",
        "--num-attention-heads", "6",
        "--num-hidden-layers", "6",
        "--hidden-size", "288",
        "--intermediate-size", "512",
        "--epochs", "1",
        "--learning-rate", "3e-5",
        "--max-seq-length", "512",
        "--seed", "201214",
        "--logging-steps", "5",
        "--eval-steps", "50",
        "--save-steps", "50",
        "--dynamic-masking",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()