"""
MLflow utilities for Databricks and Unity Catalog integration.

This module provides helper functions for:
- Detecting Databricks environment
- Setting up MLflow tracking with Unity Catalog
- Logging streaming-specific parameters
- Model registration to Unity Catalog Model Registry
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def is_databricks_environment() -> bool:
    """
    Detect if running in Databricks environment.
    
    Returns:
        True if running on Databricks, False otherwise
    """
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))


def get_databricks_user() -> Optional[str]:
    """
    Get current Databricks username.
    
    Returns:
        Username if available, None otherwise
    """
    # Try various environment variables
    user = (
        os.getenv("DATABRICKS_USER") or
        os.getenv("USER") or
        os.getenv("USERNAME")
    )
    return user


def setup_mlflow_tracking(
    output_data_dir: str,
    experiment_name: Optional[str] = None,
    use_databricks: Optional[bool] = None,
) -> str:
    """
    Setup MLflow tracking URI and experiment.
    
    Args:
        output_data_dir: Directory for local MLflow runs (used if not on Databricks)
        experiment_name: Name of MLflow experiment (if None, auto-generated)
        use_databricks: Force Databricks tracking (if None, auto-detect)
        
    Returns:
        The experiment name/path that was set
    """
    import mlflow
    
    # Determine if we should use Databricks tracking
    if use_databricks is None:
        use_databricks = is_databricks_environment()
    
    # Set tracking URI
    if use_databricks:
        mlflow.set_tracking_uri("databricks")
        logger.info("Using Databricks MLflow tracking")
        
        # Generate experiment name if not provided
        if experiment_name is None:
            user = get_databricks_user()
            if user:
                experiment_name = f"/Users/{user}/exmedbert-streaming"
            else:
                experiment_name = "/Shared/exmedbert-streaming"
    else:
        tracking_uri = f"file:{output_data_dir}/mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Using local MLflow tracking: {tracking_uri}")
        
        # Use simple name for local experiments
        if experiment_name is None:
            experiment_name = "exmedbert-streaming"
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")
    
    return experiment_name


def log_streaming_params(
    train_remote: str,
    val_remote: str,
    embeddings_path: str,
    local_cache_dir: str,
    predownload: int,
    shuffle_seed: int,
    train_dataset_size: int,
    val_dataset_size: int,
    **kwargs
) -> None:
    """
    Log Mosaic Streaming specific parameters to MLflow.
    
    Args:
        train_remote: Remote training data path
        val_remote: Remote validation data path
        embeddings_path: Path to embeddings file
        local_cache_dir: Local cache directory
        predownload: Number of samples to predownload
        shuffle_seed: Seed for data shuffling
        train_dataset_size: Size of training dataset
        val_dataset_size: Size of validation dataset
        **kwargs: Additional parameters to log
    """
    import mlflow
    
    params = {
        "data_train_remote": train_remote,
        "data_val_remote": val_remote,
        "data_embeddings_path": embeddings_path,
        "streaming_local_cache": local_cache_dir,
        "streaming_predownload": predownload,
        "streaming_shuffle_seed": shuffle_seed,
        "dataset_train_size": train_dataset_size,
        "dataset_val_size": val_dataset_size,
    }
    
    # Parse Unity Catalog paths if applicable
    if train_remote.startswith("/Volumes/"):
        uc_parts = train_remote.split("/")[2:5]  # catalog, schema, volume
        if len(uc_parts) == 3:
            params["uc_catalog"] = uc_parts[0]
            params["uc_schema"] = uc_parts[1]
            params["uc_volume"] = uc_parts[2]
    
    # Add any additional parameters
    params.update(kwargs)
    
    # Log to MLflow
    mlflow.log_params(params)
    logger.info(f"Logged {len(params)} streaming parameters to MLflow")


def log_model_architecture_params(
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_seq_length: int,
    code_vocab_size: int,
    age_vocab_size: int,
    sex_vocab_size: int,
    state_vocab_size: int,
    **kwargs
) -> None:
    """
    Log model architecture parameters to MLflow.
    
    Args:
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Feedforward layer size
        max_seq_length: Maximum sequence length
        code_vocab_size: Size of medical code vocabulary
        age_vocab_size: Size of age vocabulary
        sex_vocab_size: Size of sex vocabulary
        state_vocab_size: Size of state/region vocabulary
        **kwargs: Additional architecture parameters
    """
    import mlflow
    
    params = {
        "model_hidden_size": hidden_size,
        "model_num_layers": num_hidden_layers,
        "model_num_heads": num_attention_heads,
        "model_intermediate_size": intermediate_size,
        "model_max_seq_length": max_seq_length,
        "model_code_vocab_size": code_vocab_size,
        "model_age_vocab_size": age_vocab_size,
        "model_sex_vocab_size": sex_vocab_size,
        "model_state_vocab_size": state_vocab_size,
    }
    
    params.update(kwargs)
    mlflow.log_params(params)
    logger.info(f"Logged {len(params)} model architecture parameters to MLflow")


def log_training_config_params(
    learning_rate: float,
    batch_size: int,
    epochs: int,
    warmup_steps: int,
    gradient_accumulation_steps: int,
    weight_decay: float,
    max_steps: int,
    seed: int,
    **kwargs
) -> None:
    """
    Log training configuration parameters to MLflow.
    
    Args:
        learning_rate: Learning rate
        batch_size: Training batch size
        epochs: Number of epochs
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Gradient accumulation steps
        weight_decay: Weight decay value
        max_steps: Maximum training steps
        seed: Random seed
        **kwargs: Additional training parameters
    """
    import mlflow
    
    params = {
        "train_learning_rate": learning_rate,
        "train_batch_size": batch_size,
        "train_epochs": epochs,
        "train_warmup_steps": warmup_steps,
        "train_grad_accum_steps": gradient_accumulation_steps,
        "train_weight_decay": weight_decay,
        "train_max_steps": max_steps,
        "train_seed": seed,
    }
    
    params.update(kwargs)
    mlflow.log_params(params)
    logger.info(f"Logged {len(params)} training configuration parameters to MLflow")


def register_model_to_uc(
    run_id: str,
    model_name: str,
    catalog: str = "main",
    schema: str = "exmedbert",
    tags: Optional[Dict[str, str]] = None,
    await_registration: bool = False,
) -> Any:
    """
    Register trained model to Unity Catalog Model Registry.
    
    Args:
        run_id: MLflow run ID
        model_name: Name for the model
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
        tags: Optional tags for the model version
        await_registration: Wait for registration to complete
        
    Returns:
        ModelVersion object
    """
    import mlflow
    from mlflow.exceptions import MlflowException
    
    # Construct full model name in UC format
    full_model_name = f"{catalog}.{schema}.{model_name}"
    
    try:
        logger.info(f"Registering model to Unity Catalog: {full_model_name}")
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=full_model_name,
            tags=tags,
            await_registration_for=300 if await_registration else None,  # 5 min timeout
        )
        
        logger.info(
            f"Model registered successfully: {full_model_name} "
            f"(version {model_version.version})"
        )
        
        return model_version
        
    except MlflowException as e:
        logger.error(f"Failed to register model to Unity Catalog: {e}")
        logger.warning("Model training completed but not registered to UC")
        return None


def set_model_alias(
    model_name: str,
    version: int,
    alias: str,
    catalog: str = "main",
    schema: str = "exmedbert",
) -> None:
    """
    Set an alias for a model version in Unity Catalog.
    
    Common aliases: "Champion", "Challenger", "Staging", "Production"
    
    Args:
        model_name: Name of the model
        version: Model version number
        alias: Alias to set (e.g., "Champion", "Production")
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
    """
    import mlflow
    from mlflow.exceptions import MlflowException
    
    full_model_name = f"{catalog}.{schema}.{model_name}"
    
    try:
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=full_model_name,
            alias=alias,
            version=version
        )
        logger.info(f"Set alias '{alias}' for {full_model_name} version {version}")
        
    except MlflowException as e:
        logger.warning(f"Failed to set model alias: {e}")


def log_model_with_signature(
    model: Any,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
) -> None:
    """
    Log model with inferred signature to MLflow.
    
    Args:
        model: The model to log
        artifact_path: Path within the run's artifact directory
        registered_model_name: If provided, register model with this name
    """
    import mlflow
    
    try:
        # For HuggingFace transformers models
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
        logger.info(f"Model logged to MLflow at '{artifact_path}'")
        
    except Exception as e:
        logger.warning(f"Failed to log model with mlflow.transformers: {e}")
        logger.info("Attempting to log as PyTorch model...")
        
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
            logger.info(f"Model logged as PyTorch model at '{artifact_path}'")
        except Exception as e2:
            logger.error(f"Failed to log model to MLflow: {e2}")


def generate_run_name(
    prefix: str = "streaming-pretrain",
    include_timestamp: bool = True,
) -> str:
    """
    Generate a descriptive run name.
    
    Args:
        prefix: Prefix for the run name
        include_timestamp: Whether to include timestamp
        
    Returns:
        Generated run name
    """
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{prefix}-{timestamp}"
    return prefix


def log_system_info() -> None:
    """Log system and environment information to MLflow."""
    import mlflow
    import platform
    import torch
    
    info = {
        "system_platform": platform.platform(),
        "system_python_version": platform.python_version(),
        "system_pytorch_version": torch.__version__,
        "system_cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["system_cuda_version"] = torch.version.cuda
        info["system_gpu_count"] = torch.cuda.device_count()
        info["system_gpu_name"] = torch.cuda.get_device_name(0)
    
    if is_databricks_environment():
        info["system_databricks_runtime"] = os.getenv("DATABRICKS_RUNTIME_VERSION", "unknown")
    
    mlflow.log_params(info)
    logger.info("Logged system information to MLflow")


def create_model_tags(
    train_remote: str,
    streaming_enabled: bool = True,
    dynamic_masking: bool = False,
    use_clinvec: bool = False,
    **kwargs
) -> Dict[str, str]:
    """
    Create tags for model registration.
    
    Args:
        train_remote: Training data source
        streaming_enabled: Whether streaming was used
        dynamic_masking: Whether dynamic masking was used
        use_clinvec: Whether ClinVec embeddings were used
        **kwargs: Additional tags
        
    Returns:
        Dictionary of tags
    """
    tags = {
        "data_source": train_remote,
        "streaming_enabled": str(streaming_enabled),
        "dynamic_masking": str(dynamic_masking),
        "clinvec_enabled": str(use_clinvec),
        "training_framework": "huggingface_transformers",
        "created_by": get_databricks_user() or "unknown",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
    }
    
    if is_databricks_environment():
        tags["trained_on"] = "databricks"
        tags["databricks_runtime"] = os.getenv("DATABRICKS_RUNTIME_VERSION", "unknown")
    
    tags.update(kwargs)
    return tags

