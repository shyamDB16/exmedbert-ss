# train_entrypoint_streaming.py
"""
Entry point for multi-GPU training with Mosaic Streaming.

This is the updated version of train_entrypoint.py that uses Mosaic Streaming
for efficient data loading from cloud storage.

Usage:
    # For local MDS data:
    python scripts/train_entrypoint_streaming.py

    # For Unity Catalog volumes (on Databricks):
    export DATABRICKS_HOST="https://your-workspace.databricks.com"
    export DATABRICKS_TOKEN="your-token"
    python scripts/train_entrypoint_streaming.py --use-uc-volumes
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def setup_environment():
    """Configure environment variables for training."""
    os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"
    
    # Debug flags (disable for production)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"


def main():
    parser = argparse.ArgumentParser(description="ExMed-BERT Streaming Training")
    parser.add_argument(
        "--use-uc-volumes",
        action="store_true",
        help="Use Unity Catalog volumes (requires Databricks auth)"
    )
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default="pretrain_data_mds",
        help="Local directory with MDS data"
    )
    parser.add_argument(
        "--uc-catalog",
        type=str,
        default="main",
        help="Unity Catalog catalog name"
    )
    parser.add_argument(
        "--uc-schema",
        type=str,
        default="exmedbert",
        help="Unity Catalog schema name"
    )
    parser.add_argument(
        "--uc-volume",
        type=str,
        default="training_data",
        help="Unity Catalog volume name"
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default="pretrain_data/embeddings.pt",
        help="Path to embeddings file"
    )
    parser.add_argument(
        "--local-cache-dir",
        type=str,
        default="/tmp/mds_cache",
        help="Local cache directory for MDS shards"
    )
    # MLflow arguments
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--no-log-model",
        action="store_true",
        help="Disable logging model to MLflow"
    )
    # Unity Catalog Model Registry arguments
    parser.add_argument(
        "--no-register-model",
        action="store_true",
        help="Disable model registration to Unity Catalog"
    )
    parser.add_argument(
        "--uc-model-catalog",
        type=str,
        default="main",
        help="Unity Catalog catalog name for model"
    )
    parser.add_argument(
        "--uc-model-schema",
        type=str,
        default="exmedbert",
        help="Unity Catalog schema name for model"
    )
    parser.add_argument(
        "--uc-model-name",
        type=str,
        default="exmedbert_streaming",
        help="Model name in Unity Catalog"
    )
    parser.add_argument(
        "--model-alias",
        type=str,
        default=None,
        help="Alias to set for model version (e.g., 'Champion', 'Production')"
    )
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    print("=" * 60)
    print("ExMed-BERT Pretraining with Mosaic Streaming")
    print("=" * 60)
    
    # Configure data paths
    if args.use_uc_volumes:
        print("Using Unity Catalog volumes for data storage")
        base_path = f"/Volumes/{args.uc_catalog}/{args.uc_schema}/{args.uc_volume}"
        train_remote = f"{base_path}/mds_train"
        val_remote = f"{base_path}/mds_val"
        
        # Check Databricks auth
        if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
            print("\nWARNING: Databricks authentication not configured!")
            print("Set these environment variables:")
            print("  export DATABRICKS_HOST='https://your-workspace.databricks.com'")
            print("  export DATABRICKS_TOKEN='your-token'")
            sys.exit(1)
    else:
        print("Using local MDS data")
        train_remote = f"{args.local_data_dir}/train"
        val_remote = f"{args.local_data_dir}/val"
        
        # Check if data exists
        if not Path(train_remote).exists():
            print(f"\nERROR: Training data not found at {train_remote}")
            print("Please run the Spark to MDS conversion first:")
            print("  python scripts/convert_spark_to_mds.py")
            sys.exit(1)
    
    # Output directories
    output_dir = "output/streaming_pretrain"
    output_data_dir = "output/streaming_pretrain_data"
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_data_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Training data: {train_remote}")
    print(f"  Validation data: {val_remote}")
    print(f"  Embeddings: {args.embeddings_path}")
    print(f"  Local cache: {args.local_cache_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"\nMLflow & Model Registry:")
    print(f"  Experiment: {args.mlflow_experiment_name or '(auto-generated)'}")
    print(f"  Log model: {not args.no_log_model}")
    print(f"  Register to UC: {not args.no_register_model}")
    if not args.no_register_model:
        print(f"  Model name: {args.uc_model_catalog}.{args.uc_model_schema}.{args.uc_model_name}")
        if args.model_alias:
            print(f"  Model alias: {args.model_alias}")
    print()
    
    # Build training command
    cmd = [
        sys.executable,
        "scripts/pretrain-exmed-bert-streaming.py",
        train_remote,
        val_remote,
        args.embeddings_path,
        output_dir,
        output_data_dir,
        "--local-cache-dir", args.local_cache_dir,
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
        "--predownload", "256",  # Predownload for performance
    ]
    
    # Add Databricks auth if using UC volumes
    if args.use_uc_volumes:
        if os.environ.get("DATABRICKS_HOST"):
            cmd.extend(["--databricks-host", os.environ["DATABRICKS_HOST"]])
        if os.environ.get("DATABRICKS_TOKEN"):
            cmd.extend(["--databricks-token", os.environ["DATABRICKS_TOKEN"]])
    
    # Add MLflow parameters
    if args.mlflow_experiment_name:
        cmd.extend(["--mlflow-experiment-name", args.mlflow_experiment_name])
    if args.mlflow_run_name:
        cmd.extend(["--mlflow-run-name", args.mlflow_run_name])
    if args.no_log_model:
        cmd.append("--no-log-model")
    
    # Add Unity Catalog Model Registry parameters
    if args.no_register_model:
        cmd.append("--no-register-model")
    else:
        cmd.extend(["--uc-model-catalog", args.uc_model_catalog])
        cmd.extend(["--uc-model-schema", args.uc_model_schema])
        cmd.extend(["--uc-model-name", args.uc_model_name])
        if args.model_alias:
            cmd.extend(["--model-alias", args.model_alias])
    
    print("Starting training...")
    print(f"Command: {' '.join(cmd[:3])} [...]")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()

