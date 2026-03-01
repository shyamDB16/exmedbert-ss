#!/usr/bin/env python
"""
ExMed-BERT Pretraining with Mosaic Streaming.

This is an updated version of pretrain-exmed-bert-clinvec.py that uses
Mosaic Streaming for efficient data loading from cloud storage.

Key differences:
- Uses StreamingDataset instead of PatientDataset
- Supports loading from Unity Catalog volumes, S3, DBFS
- Better memory efficiency and faster startup
- Built-in support for distributed training
"""

import os
import logging
import math
import sys
from pathlib import Path
from typing import Optional, List, Dict

import mlflow
import transformers
import typer
from typer import Option
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import exmed_bert.models as MM
from exmed_bert.data.streaming_dataset import create_streaming_datasets
from exmed_bert.models.streaming_trainer import MedBertStreamingTrainer
from exmed_bert.utils.metrics import compute_metrics_merged_model
from exmed_bert.utils.mlflow_utils import (
    setup_mlflow_tracking,
    log_streaming_params,
    log_model_architecture_params,
    log_training_config_params,
    log_system_info,
    register_model_to_uc,
    set_model_alias,
    create_model_tags,
    generate_run_name,
)

logger = logging.getLogger(__name__)


def setup_databricks_auth(db_host: Optional[str] = None, db_token: Optional[str] = None):
    """
    Configure Databricks authentication for Unity Catalog volumes.
    
    Required when loading data from UC volumes in distributed training.
    Set these environment variables on all worker nodes.
    
    Args:
        db_host: Databricks workspace URL
        db_token: Databricks access token
    """
    # Try to get from environment if not provided
    db_host = db_host or os.environ.get("DATABRICKS_HOST")
    db_token = db_token or os.environ.get("DATABRICKS_TOKEN")
    
    if db_host and db_token:
        os.environ["DATABRICKS_HOST"] = db_host
        os.environ["DATABRICKS_TOKEN"] = db_token
        logger.info(f"Configured Databricks authentication for host: {db_host}")
    else:
        logger.warning(
            "Databricks authentication not configured. "
            "This is required for Unity Catalog volumes. "
            "Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables."
        )


def pretrain(
    train_remote: Path,
    val_remote: Path,
    embeddings_path: Path,
    output_dir: Path,
    output_data_dir: Path,
    local_cache_dir: Path = Path("/tmp/mds_cache"),
    model_dir: Optional[Path] = None,
    train_batch_size: int = 256,
    eval_batch_size: int = 256,
    num_attention_heads: int = 6,
    num_hidden_layers: int = 6,
    hidden_size: int = 288,
    intermediate_size: int = 512,
    epochs: Optional[int] = None,
    max_steps: int = -1,
    learning_rate: float = 3e-5,
    gradient_accumulation_steps: int = 1,
    max_seq_length: int = 512,
    seed: int = 201214,
    num_workers: int = 0,
    logging_steps: int = 100,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    warmup_steps: int = 10_000,
    warmup_ratio: Optional[float] = None,
    initialization: str = "orthogonal",
    dynamic_masking: bool = True,
    max_masked: Optional[int] = None,
    plos: bool = True,
    # Mosaic Streaming specific
    predownload: int = 256,
    shuffle_seed: Optional[int] = None,
    # ClinVec integration parameters
    use_clinvec: bool = False,
    clinvec_dir: Optional[Path] = None,
    vocab_types: str = Option("icd10cm,atc", help="Comma-separated list of vocabulary types"),
    use_hierarchical_init: bool = True,
    resize_if_needed: bool = True,
    resize_method: str = Option("auto", help="Method for resizing embeddings"),
    # Databricks authentication
    databricks_host: Optional[str] = None,
    databricks_token: Optional[str] = None,
    # MLflow parameters
    mlflow_experiment_name: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
    log_model: bool = Option(True, "--log-model/--no-log-model"),
    # Unity Catalog Model Registry parameters
    register_model: bool = Option(True, "--register-model/--no-register-model"),
    uc_model_catalog: str = "main",
    uc_model_schema: str = "exmedbert",
    uc_model_name: str = "exmedbert_streaming",
    model_alias: Optional[str] = None,
):
    """
    Pretrain ExMed-BERT using Mosaic Streaming for data loading.
    
    Args:
        train_remote: Remote path to training MDS data (S3, DBFS, UC volume)
        val_remote: Remote path to validation MDS data
        embeddings_path: Path to embeddings.pt file with vocabularies
        output_dir: Directory for model checkpoints
        output_data_dir: Directory for training logs and metrics
        local_cache_dir: Local directory for MDS cache
        model_dir: Path to pretrained model (for continued training)
        train_batch_size: Training batch size per device
        eval_batch_size: Evaluation batch size per device
        predownload: Number of samples to predownload for performance
        shuffle_seed: Seed for data shuffling (defaults to main seed)
        databricks_host: Databricks workspace URL (for UC authentication)
        databricks_token: Databricks access token (for UC authentication)
        mlflow_experiment_name: Name of MLflow experiment (auto-generated if None)
        mlflow_run_name: Name of MLflow run (auto-generated if None)
        log_model: Whether to log model to MLflow
        register_model: Whether to register model to Unity Catalog
        uc_model_catalog: Unity Catalog catalog name for model registration
        uc_model_schema: Unity Catalog schema name for model registration
        uc_model_name: Model name in Unity Catalog
        model_alias: Optional alias to set for model version (e.g., "Champion")
        ... (other args same as original pretrain script)
    """
    # Setup
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/output.log"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Setup Databricks auth if needed
    setup_databricks_auth(databricks_host, databricks_token)
    
    # Set seeds
    transformers.set_seed(seed)
    shuffle_seed = shuffle_seed or seed
    
    # Setup MLflow tracking (auto-detects Databricks)
    experiment_name = setup_mlflow_tracking(
        output_data_dir=str(output_data_dir),
        experiment_name=mlflow_experiment_name,
    )
    
    logger.info("=" * 60)
    logger.info("ExMed-BERT Pretraining with Mosaic Streaming")
    logger.info("=" * 60)
    logger.info(f"MLflow Experiment: {experiment_name}")
    logger.info(f"Train data: {train_remote}")
    logger.info(f"Val data: {val_remote}")
    logger.info(f"Embeddings: {embeddings_path}")
    logger.info(f"Local cache: {local_cache_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Load datasets using Mosaic Streaming
    logger.info("Initializing Streaming datasets...")
    train_data, val_data = create_streaming_datasets(
        train_remote=str(train_remote),
        val_remote=str(val_remote),
        local_base=str(local_cache_dir),
        embeddings_path=str(embeddings_path),
        batch_size=train_batch_size,
        max_length=max_seq_length,
        dynamic_masking=dynamic_masking,
        max_masked=max_masked or 20,
        predownload=predownload,
        shuffle_seed=shuffle_seed,
    )
    
    logger.info(f"Loaded train dataset length: {len(train_data)}")
    logger.info(f"Loaded val dataset length: {len(val_data)}")
    logger.info(f"Train batch size: {train_batch_size}")
    
    # Get embeddings from dataset
    code_embed = train_data.code_embed
    age_embed = train_data.age_embed
    sex_embed = train_data.sex_embed
    state_embed = train_data.state_embed
    
    # Model configuration
    config = MM.ExMedBertConfig(
        age_vocab_size=len(age_embed),
        code_vocab_size=len(code_embed),
        graph_heads=4,
        graph_hidden_size=int(hidden_size / 4),
        hidden_size=hidden_size,
        initialization=initialization,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_seq_length,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        number_of_codes=len(code_embed.entity_to_id),
        pad_token_id=code_embed("PAD"),
        plos_weight=(0.25, 1.0),
        predict_codes=True,
        predict_los=plos,
        region_vocab_size=len(state_embed),
        sex_vocab_size=len(sex_embed),
    )
    
    # Initialize or load model
    if model_dir is not None:
        logger.info(f"Loading pretrained model from {model_dir}")
        model = MM.ExMedBertForMaskedLM.from_pretrained(
            model_dir, config=config, code_embed=code_embed
        )
    else:
        logger.info("Initializing new model")
        model = MM.ExMedBertForMaskedLM(config, code_embed=code_embed)
    
    # ClinVec Integration
    if use_clinvec and clinvec_dir is not None:
        logger.info("=" * 60)
        logger.info("INTEGRATING CLINVEC PRE-TRAINED EMBEDDINGS")
        logger.info("=" * 60)
        
        vocab_types_list = [v.strip() for v in vocab_types.split(",") if v.strip()]
        
        try:
            from exmed_bert.utils.clinvec_integration import integrate_clinvec_with_exmedbert
            
            stats = integrate_clinvec_with_exmedbert(
                model=model,
                code_dict=code_embed,
                clinvec_dir=str(clinvec_dir),
                vocab_types=vocab_types_list,
                resize_if_needed=resize_if_needed,
                resize_method=resize_method,
                use_hierarchical_init=use_hierarchical_init,
                verbose=True
            )
            
            total_loaded = sum(stats.values())
            logger.info(f"Successfully loaded {total_loaded} ClinVec embeddings")
            for vocab_type, count in stats.items():
                logger.info(f"  {vocab_type}: {count} embeddings")
                
        except Exception as e:
            logger.error(f"Failed to load ClinVec embeddings: {e}")
            logger.warning("Continuing with standard random initialization")
    else:
        logger.info("Using standard random embedding initialization")
    
    # Log parameter counts
    logger.info(
        f"Trainable parameters: {model.num_parameters(only_trainable=True, exclude_embeddings=False):,}"
    )
    logger.info(
        f"Trainable parameters (excl. embeddings): "
        f"{model.num_parameters(only_trainable=True, exclude_embeddings=True):,}"
    )
    
    # Calculate warmup steps
    if warmup_ratio is not None and epochs is not None:
        warmup_steps = int(warmup_ratio * (len(train_data) / train_batch_size) * epochs)
    elif warmup_ratio is not None and max_steps > 0:
        warmup_steps = int(warmup_ratio * max_steps)
    elif epochs is None and warmup_steps is None:
        raise ValueError("Must specify epochs or max_steps")
    
    logger.info(f"Warmup steps: {warmup_steps}")
    
    # Calculate epochs if using max_steps
    if epochs is None and max_steps > 0:
        epochs = int(max_steps / math.ceil(len(train_data) / train_batch_size))
    
    if epochs is None:
        raise ValueError("Must specify either epochs or max_steps")
    
    # Training configuration
    training_config = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_eval_batch_size=eval_batch_size,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        eval_strategy=transformers.IntervalStrategy.STEPS,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        label_names=["code_labels", "plos_label"],
        dataloader_num_workers=num_workers,
        logging_dir=f"{output_data_dir}/logs",
        fp16=False,
        save_steps=save_steps,
        save_strategy="steps" if save_steps > 0 else "no",
        ignore_data_skip=True,
        dataloader_pin_memory=True,  # Better performance with streaming
        remove_unused_columns=False,  # Important for custom data format
        # MLflow integration - log metrics to MLflow
        report_to=["mlflow"],  # Enable MLflow logging
        load_best_model_at_end=True if eval_steps > 0 else False,
        metric_for_best_model="eval_loss",
        save_total_limit=3,  # Keep only last 3 checkpoints
    )
    
    logger.info(f"Using {training_config.n_gpu} GPU(s)")
    logger.info(f"World size: {training_config.world_size}")
    
    # Generate run name if not provided
    if mlflow_run_name is None:
        mlflow_run_name = generate_run_name(prefix="streaming-pretrain")
    
    logger.info(f"Starting MLflow run: {mlflow_run_name}")
    
    # Start MLflow run for complete tracking
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Log system information
        log_system_info()
        
        # Log streaming-specific parameters
        log_streaming_params(
            train_remote=str(train_remote),
            val_remote=str(val_remote),
            embeddings_path=str(embeddings_path),
            local_cache_dir=str(local_cache_dir),
            predownload=predownload,
            shuffle_seed=shuffle_seed,
            train_dataset_size=len(train_data),
            val_dataset_size=len(val_data),
        )
        
        # Log model architecture parameters
        log_model_architecture_params(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_seq_length=max_seq_length,
            code_vocab_size=len(code_embed),
            age_vocab_size=len(age_embed),
            sex_vocab_size=len(sex_embed),
            state_vocab_size=len(state_embed),
        )
        
        # Log training configuration parameters
        log_training_config_params(
            learning_rate=learning_rate,
            batch_size=train_batch_size,
            epochs=epochs,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=0.01,
            max_steps=max_steps,
            seed=seed,
            dynamic_masking=dynamic_masking,
            max_masked=max_masked or 20,
        )
        
        # Log ClinVec parameters if used
        if use_clinvec:
            mlflow.log_params({
                "clinvec_enabled": True,
                "clinvec_vocab_types": vocab_types,
                "clinvec_hierarchical_init": use_hierarchical_init,
                "clinvec_resize_method": resize_method,
            })
        
        # Initialize trainer with streaming support
        logger.info("Initializing trainer with StreamingDataLoader support...")
        trainer = MedBertStreamingTrainer(
            model=model,
            args=training_config,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics_merged_model,
            model_init=None,
            use_streaming_dataloader=True,  # Enable StreamingDataLoader
        )
        
        # Training (HF Trainer auto-logs metrics to MLflow)
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            logger.info("Starting training from scratch")
            trainer.train()
        
        # Evaluation
        logger.info("Running final evaluation...")
        eval_result = trainer.evaluate(eval_dataset=val_data)
        
        # Save evaluation results
        eval_path = os.path.join(output_data_dir, "eval_results.txt")
        with open(eval_path, "w") as writer:
            logger.info("***** Evaluation Results *****")
            for key, value in sorted(eval_result.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
        
        # Save model
        save_path = model_dir if model_dir is not None else output_data_dir / "pretrained-model"
        logger.info(f"Saving model to {save_path}")
        trainer.model.save_pretrained(save_path, safe_serialization=False)
        
        # Log model to MLflow
        if log_model:
            logger.info("Logging model to MLflow...")
            try:
                mlflow.transformers.log_model(
                    transformers_model=trainer.model,
                    artifact_path="model",
                )
                logger.info("Model successfully logged to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")
        
        # Register model to Unity Catalog
        if register_model:
            logger.info("Registering model to Unity Catalog...")
            
            # Create model tags
            model_tags = create_model_tags(
                train_remote=str(train_remote),
                streaming_enabled=True,
                dynamic_masking=dynamic_masking,
                use_clinvec=use_clinvec,
            )
            
            # Register the model
            model_version = register_model_to_uc(
                run_id=run.info.run_id,
                model_name=uc_model_name,
                catalog=uc_model_catalog,
                schema=uc_model_schema,
                tags=model_tags,
                await_registration=False,
            )
            
            # Set model alias if requested
            if model_version and model_alias:
                set_model_alias(
                    model_name=uc_model_name,
                    version=model_version.version,
                    alias=model_alias,
                    catalog=uc_model_catalog,
                    schema=uc_model_schema,
                )
        
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        if register_model:
            logger.info(f"Model: {uc_model_catalog}.{uc_model_schema}.{uc_model_name}")
        logger.info("=" * 60)


if __name__ == "__main__":
    typer.run(pretrain)

