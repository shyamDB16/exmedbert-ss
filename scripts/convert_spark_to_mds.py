#!/usr/bin/env python
"""
Convert Spark DataFrame to Mosaic Data Shard (MDS) format for ExMedBert training.

This script demonstrates how to convert patient data from a Spark DataFrame
to MDS format that can be efficiently loaded during distributed training.

Requirements:
    - Databricks Runtime 15.2 ML or higher (includes Mosaic Streaming)
    - Unity Catalog volume for persistent storage (or local path for testing)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, struct, array, to_json
from streaming.base.converters import dataframe_to_mds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_patient_dataframe(df: DataFrame) -> DataFrame:
    """
    Transform raw patient data into format expected by ExMedBert.
    
    Expected schema after transformation:
    - patient_id: string
    - codes: array<int> - tokenized medical codes
    - ages: array<int> - tokenized ages
    - visits: array<int> - visit indices
    - segments: array<int> - segment embeddings
    - sex: int - tokenized sex
    - birth_year: int
    - state: int - tokenized state/region
    - code_labels: array<int> - labels for masked LM
    - code_mask: array<bool> - mask indicating which codes to predict
    - plos_label: float - length of stay label
    - endpoint_labels: array<float> - endpoint labels for fine-tuning
    - unpadded_length: int - actual sequence length before padding
    
    Args:
        df: Raw Spark DataFrame with patient data
        
    Returns:
        Transformed DataFrame ready for MDS conversion
    """
    # This is a template - adjust based on your actual Spark schema
    # The transformation should match what PatientDataset.__getitem__ returns
    
    logger.info("Transforming patient data for ExMedBert format...")
    
    # Example transformations (adjust to your actual data):
    transformed_df = df.select(
        col("patient_id").cast("string"),
        col("medical_codes").alias("codes"),  # Should be tokenized already
        col("age_tokens").alias("ages"),
        col("visit_indices").alias("visits"),
        col("segment_tokens").alias("segments"),
        col("sex_token").cast("int").alias("sex"),
        col("birth_year").cast("int"),
        col("state_token").cast("int").alias("state"),
        col("masked_code_labels").alias("code_labels"),
        col("code_mask").alias("code_mask"),
        col("length_of_stay").cast("float").alias("plos_label"),
        col("endpoint_labels"),
        col("sequence_length").cast("int").alias("unpadded_length")
    )
    
    logger.info(f"Transformed {transformed_df.count()} patients")
    return transformed_df


def convert_to_mds(
    df: DataFrame,
    output_path: str,
    compression: str = "zstd",
    hashes: Optional[list] = None,
    size_limit: str = "100mb",
    num_workers: int = 8,
) -> None:
    """
    Convert Spark DataFrame to MDS format.
    
    Args:
        df: Prepared DataFrame with ExMedBert format
        output_path: Path to save MDS files (local or Unity Catalog volume)
        compression: Compression algorithm ('zstd', 'gzip', 'bz2', or None)
        hashes: List of hash algorithms for data validation (e.g., ['sha256'])
        size_limit: Target size per shard file
        num_workers: Number of workers for parallel conversion
    """
    logger.info(f"Converting DataFrame to MDS format at: {output_path}")
    logger.info(f"  Compression: {compression}")
    logger.info(f"  Size limit: {size_limit}")
    logger.info(f"  Workers: {num_workers}")
    
    # Define column encodings for MDS
    # This tells MDS how to serialize each column type
    columns = {
        "patient_id": "str",
        "codes": "ndarray:int32",
        "ages": "ndarray:int32",
        "visits": "ndarray:int32",
        "segments": "ndarray:int32",
        "sex": "int32",
        "birth_year": "int32",
        "state": "int32",
        "code_labels": "ndarray:int32",
        "code_mask": "ndarray:bool",
        "plos_label": "float32",
        "endpoint_labels": "ndarray:float32",
        "unpadded_length": "int32",
    }
    
    try:
        dataframe_to_mds(
            df=df,
            merge_index=True,
            mds_kwargs={
                "out": output_path,
                "columns": columns,
                "compression": compression,
                "hashes": hashes or ["sha256"],
                "size_limit": size_limit,
            },
            udf_iterable=None,  # Use if you need additional preprocessing
        )
        logger.info(f"Successfully converted data to MDS format")
        logger.info(f"Output location: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during MDS conversion: {e}")
        raise


def save_embeddings(
    code_embed: Any,
    age_embed: Any,
    sex_embed: Any,
    state_embed: Any,
    endpoint_dict: Any,
    output_dir: str,
) -> None:
    """
    Save embedding dictionaries separately from the MDS data.
    
    These need to be loaded alongside the MDS dataset during training.
    
    Args:
        code_embed: CodeDict instance
        age_embed: AgeDict instance
        sex_embed: SexDict instance
        state_embed: StateDict instance
        endpoint_dict: EndpointDict instance
        output_dir: Directory to save embedding files
    """
    import torch
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    embeddings = {
        "code_embed": code_embed,
        "age_embed": age_embed,
        "sex_embed": sex_embed,
        "state_embed": state_embed,
        "endpoint_dict": endpoint_dict,
    }
    
    torch.save(embeddings, output_path / "embeddings.pt")
    logger.info(f"Saved embeddings to {output_path / 'embeddings.pt'}")


def main(
    input_table: Optional[str] = None,
    input_path: Optional[str] = None,
    output_mds_path: str = "/Volumes/catalog/schema/volume/mds_train",
    embeddings_path: str = "/Volumes/catalog/schema/volume/embeddings",
    train_test_split: float = 0.8,
):
    """
    Main conversion pipeline from Spark to MDS.
    
    Args:
        input_table: Spark table name (e.g., 'catalog.schema.patients')
        input_path: Alternative: path to parquet/delta files
        output_mds_path: Output path for MDS files (supports UC volumes)
        embeddings_path: Output path for embedding dictionaries
        train_test_split: Ratio for train/val split
    """
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("ExMedBert-Spark-to-MDS") \
        .getOrCreate()
    
    try:
        # Load data
        if input_table:
            logger.info(f"Loading data from table: {input_table}")
            df = spark.table(input_table)
        elif input_path:
            logger.info(f"Loading data from path: {input_path}")
            df = spark.read.parquet(input_path)
        else:
            raise ValueError("Must provide either input_table or input_path")
        
        logger.info(f"Loaded {df.count()} records")
        
        # Prepare data
        prepared_df = prepare_patient_dataframe(df)
        
        # Split train/val
        train_df, val_df = prepared_df.randomSplit(
            [train_test_split, 1 - train_test_split],
            seed=201214
        )
        
        logger.info(f"Split: {train_df.count()} train, {val_df.count()} val")
        
        # Convert train set
        convert_to_mds(
            df=train_df,
            output_path=f"{output_mds_path}/train",
            compression="zstd",
            size_limit="100mb",
        )
        
        # Convert validation set
        convert_to_mds(
            df=val_df,
            output_path=f"{output_mds_path}/val",
            compression="zstd",
            size_limit="100mb",
        )
        
        logger.info("=" * 60)
        logger.info("MDS Conversion Complete!")
        logger.info("=" * 60)
        logger.info(f"Train data: {output_mds_path}/train")
        logger.info(f"Val data: {output_mds_path}/val")
        logger.info(f"Next steps:")
        logger.info(f"  1. Save embedding dictionaries separately")
        logger.info(f"  2. Update training script to use StreamingDataset")
        logger.info(f"  3. Configure Databricks authentication if using UC volumes")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    import typer
    typer.run(main)

