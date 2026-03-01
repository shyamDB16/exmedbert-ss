"""
Mosaic Streaming adapter for ExMedBert training.

This module provides a wrapper around Mosaic's StreamingDataset that maintains
compatibility with the existing PatientDataset interface while enabling efficient
streaming from cloud storage.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from streaming import StreamingDataset
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logging.warning("Mosaic Streaming not available. Install with: pip install mosaicml-streaming")

from exmed_bert.data.encoding import AgeDict, CodeDict, EndpointDict, SexDict, StateDict
from exmed_bert.data.data_exceptions import UndefinedEndpoint

logger = logging.getLogger(__name__)


class StreamingPatientDataset(Dataset):
    """
    Adapter that wraps Mosaic StreamingDataset for ExMedBert training.
    
    This class maintains API compatibility with PatientDataset while using
    Mosaic Streaming for efficient data loading from cloud storage.
    
    Key differences from PatientDataset:
    - Data loaded from MDS format instead of .pt files
    - Streaming from cloud storage (S3, DBFS, Unity Catalog volumes)
    - Better memory efficiency for large datasets
    - Built-in support for distributed training
    """
    
    def __init__(
        self,
        remote: str,
        local: str,
        code_embed: CodeDict,
        age_embed: AgeDict,
        sex_embed: SexDict,
        state_embed: StateDict,
        endpoint_dict: EndpointDict,
        max_length: int = 512,
        do_eval: bool = True,
        mask_substances: bool = True,
        dynamic_masking: bool = False,
        min_unmasked: int = 1,
        max_masked: int = 20,
        masked_lm_prob: float = 0.15,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        shuffle_seed: int = 201214,
        num_canonical_nodes: Optional[int] = None,
        predownload: Optional[int] = None,
    ):
        """
        Initialize StreamingPatientDataset.
        
        Args:
            remote: Remote path to MDS dataset (S3, DBFS, UC volume)
            local: Local cache directory for downloaded shards
            code_embed: CodeDict instance for medical code vocabularies
            age_embed: AgeDict instance for age encoding
            sex_embed: SexDict instance for sex encoding
            state_embed: StateDict instance for state/region encoding
            endpoint_dict: EndpointDict instance for endpoint labels
            max_length: Maximum sequence length
            do_eval: Whether this is an evaluation dataset
            mask_substances: Whether to mask substance/drug codes
            dynamic_masking: Whether to apply dynamic masking
            min_unmasked: Minimum number of unmasked tokens
            max_masked: Maximum number of masked tokens
            masked_lm_prob: Probability of masking a token
            batch_size: Batch size for optimal shard loading
            shuffle: Whether to shuffle data
            shuffle_seed: Random seed for shuffling
            num_canonical_nodes: Number of nodes for distributed training
            predownload: Number of samples to predownload (for performance)
        """
        super().__init__()
        
        if not STREAMING_AVAILABLE:
            raise ImportError(
                "Mosaic Streaming is required. Install with: pip install mosaicml-streaming"
            )
        
        self.remote = remote
        self.local = local
        
        # Embeddings and vocabularies
        self.code_embed = code_embed
        self.age_embed = age_embed
        self.sex_embed = sex_embed
        self.state_embed = state_embed
        self.endpoint_dict = endpoint_dict
        
        # Configuration
        self.max_length = max_length
        self.eval = do_eval
        self.mask_substances = mask_substances
        self.dynamic_masking = dynamic_masking
        self.min_unmasked = min_unmasked
        self.max_masked = max_masked
        self.masked_lm_prob = masked_lm_prob
        
        # Endpoint configuration (for fine-tuning)
        self.endpoint_index = None
        self.selected_endpoint = None
        
        # Initialize Mosaic StreamingDataset
        logger.info(f"Initializing StreamingDataset from {remote}")
        logger.info(f"Local cache: {local}")
        
        self.streaming_dataset = StreamingDataset(
            remote=remote,
            local=local,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            num_canonical_nodes=num_canonical_nodes,
            predownload=predownload or 256,  # Predownload for better performance
            keep_zip=False,  # Don't keep compressed files
            download_retry=2,
            download_timeout=60,
            validate_hash=None,  # Validate during conversion, not runtime
            shuffle_algo="naive",  # or "py1b", "py1br", "py1e", "py2s"
        )
        
        logger.info(f"Loaded streaming dataset with {len(self.streaming_dataset)} samples")
    
    def state_dict(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Return dataset state for checkpointing (delegates to underlying StreamingDataset).
        Enables exact data-index resume when used with StreamingDataLoader.
        """
        if hasattr(self.streaming_dataset, "state_dict"):
            return self.streaming_dataset.state_dict(*args, **kwargs)
        return None
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore dataset state from checkpoint (delegates to underlying StreamingDataset).
        """
        if hasattr(self.streaming_dataset, "load_state_dict"):
            self.streaming_dataset.load_state_dict(state_dict)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.streaming_dataset)
    
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """
        Load a patient sample and return formatted data.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with patient data tensors
            
        Raises:
            UndefinedEndpoint: If endpoint not set for fine-tuning
        """
        if not self.eval and self.endpoint_index is None:
            raise UndefinedEndpoint("Set endpoint index first for fine-tuning.")
        
        # Load sample from streaming dataset
        sample = self.streaming_dataset[index]
        
        # Convert numpy arrays to tensors (MDS column names)
        codes = torch.from_numpy(sample["codes"]).long()
        ages = torch.from_numpy(sample["ages"]).long()
        visits = torch.from_numpy(sample["visits"]).long()
        segments = torch.from_numpy(sample["segments"]).long()
        sex = torch.tensor(sample["sex"], dtype=torch.long)
        state = torch.tensor(sample["state"], dtype=torch.long)
        pad_id = self.code_embed("PAD")
        attention_mask = sample["codes"] != pad_id
        if isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask)
        # Build dict with both MDS names and model-expected keys (ExMedBert: input_ids, entity_ids, position_ids, etc.)
        patient_data = {
            "codes": codes,
            "ages": ages,
            "visits": visits,
            "segments": segments,
            "sex": sex,
            "state": state,
            "attention_mask": attention_mask,
            "input_ids": codes,
            "entity_ids": segments,
            "position_ids": visits,
            "age_ids": ages,
            "sex_ids": sex.unsqueeze(0).expand_as(codes) if codes.dim() > 0 else sex.unsqueeze(0),
            "state_ids": state.unsqueeze(0).expand_as(codes) if codes.dim() > 0 else state.unsqueeze(0),
        }
        
        # Add labels for pretraining
        if self.eval:
            # For evaluation/pretraining, use provided labels
            patient_data["code_labels"] = torch.from_numpy(sample["code_labels"]).long()
            patient_data["code_mask"] = torch.from_numpy(sample["code_mask"]).bool()
            patient_data["plos_label"] = torch.tensor(sample["plos_label"], dtype=torch.float)
            
            # Apply dynamic masking if enabled
            if self.dynamic_masking:
                patient_data = self._apply_dynamic_masking(patient_data)
            # Keep input_ids in sync with codes after dynamic masking
            patient_data["input_ids"] = patient_data["codes"]
        else:
            # For fine-tuning, return endpoint labels
            endpoint_labels = torch.from_numpy(sample["endpoint_labels"]).float()
            if self.endpoint_index is not None:
                patient_data["labels"] = endpoint_labels.index_select(
                    dim=-1, index=self.endpoint_index
                )
        # Return only keys the model expects (avoid passing extra keys into forward())
        for key in ("codes", "ages", "visits", "segments", "sex", "state", "code_mask"):
            patient_data.pop(key, None)
        return patient_data
    
    def _apply_dynamic_masking(self, patient_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply dynamic masking to the input sequence.
        
        This replicates the dynamic masking logic from the original PatientDataset.
        
        Args:
            patient_data: Dictionary with patient tensors
            
        Returns:
            Updated patient_data with new masks
        """
        codes = patient_data["codes"]
        attention_mask = patient_data["attention_mask"]
        
        # Get valid positions (non-padding)
        valid_positions = torch.where(attention_mask)[0]
        
        if len(valid_positions) == 0:
            return patient_data
        
        # Determine how many tokens to mask
        num_to_mask = min(
            self.max_masked,
            max(self.min_unmasked, int(len(valid_positions) * self.masked_lm_prob))
        )
        
        # Filter maskable positions (exclude special tokens, optionally drugs)
        maskable_positions = []
        for pos in valid_positions:
            code_id = codes[pos].item()
            code_str = self.code_embed.ids_to_entity.get(code_id, "")
            
            # Skip special tokens
            if code_str in ["PAD", "MASK", "CLS", "SEP"]:
                continue
            
            # Skip substances if configured
            if not self.mask_substances and code_str.startswith("ATC"):
                continue
            
            maskable_positions.append(pos)
        
        if len(maskable_positions) == 0:
            return patient_data
        
        # Randomly select positions to mask
        num_to_mask = min(num_to_mask, len(maskable_positions))
        indices = torch.randperm(len(maskable_positions))[:num_to_mask]
        masked_positions = [maskable_positions[i] for i in indices]
        
        # Create new mask and labels
        new_mask = torch.zeros_like(codes, dtype=torch.bool)
        new_labels = torch.full_like(codes, -100)  # -100 = ignore in loss
        
        for pos in masked_positions:
            new_mask[pos] = True
            new_labels[pos] = codes[pos]
            
            # MLM strategy: 80% MASK, 10% random, 10% unchanged
            rand = torch.rand(1).item()
            if rand < 0.8:
                codes[pos] = self.code_embed("MASK")
            elif rand < 0.9:
                # Replace with random code
                random_code = torch.randint(0, len(self.code_embed), (1,)).item()
                codes[pos] = random_code
            # else: keep original (10%)
        
        patient_data["codes"] = codes
        patient_data["input_ids"] = codes
        patient_data["code_mask"] = new_mask
        patient_data["code_labels"] = new_labels

        return patient_data
    
    def endpoints(self, selected_endpoints: List[str]):
        """
        Set the endpoints for fine-tuning.
        
        Args:
            selected_endpoints: List of endpoint names to use
        """
        available_endpoints = self.endpoint_dict.endpoints
        endpoint_index = torch.tensor(
            [i for i, x in enumerate(available_endpoints) if x in selected_endpoints]
        )
        
        if len(endpoint_index) == 1:
            self.selected_endpoint = selected_endpoints[0]
            self.endpoint_index = endpoint_index
        else:
            raise UndefinedEndpoint("Must specify exactly one endpoint")
    
    @classmethod
    def from_embeddings_file(
        cls,
        remote: str,
        local: str,
        embeddings_path: str,
        **kwargs
    ) -> "StreamingPatientDataset":
        """
        Create dataset by loading embeddings from a file.
        
        Args:
            remote: Remote MDS path
            local: Local cache path
            embeddings_path: Path to saved embeddings.pt file
            **kwargs: Additional arguments for __init__
            
        Returns:
            Initialized StreamingPatientDataset
        """
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings = torch.load(embeddings_path, weights_only=False)
        
        return cls(
            remote=remote,
            local=local,
            code_embed=embeddings["code_embed"],
            age_embed=embeddings["age_embed"],
            sex_embed=embeddings["sex_embed"],
            state_embed=embeddings["state_embed"],
            endpoint_dict=embeddings["endpoint_dict"],
            **kwargs
        )


def create_streaming_datasets(
    train_remote: str,
    val_remote: str,
    local_base: str,
    embeddings_path: str,
    batch_size: int = 256,
    **kwargs
) -> tuple:
    """
    Convenience function to create train and validation streaming datasets.
    
    Args:
        train_remote: Remote path to training MDS data
        val_remote: Remote path to validation MDS data
        local_base: Base directory for local cache
        embeddings_path: Path to embeddings file
        batch_size: Batch size for optimal shard loading
        **kwargs: Additional arguments passed to StreamingPatientDataset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_local = os.path.join(local_base, "train")
    val_local = os.path.join(local_base, "val")
    
    # Create directories
    Path(train_local).mkdir(parents=True, exist_ok=True)
    Path(val_local).mkdir(parents=True, exist_ok=True)
    
    train_dataset = StreamingPatientDataset.from_embeddings_file(
        remote=train_remote,
        local=train_local,
        embeddings_path=embeddings_path,
        batch_size=batch_size,
        shuffle=True,
        do_eval=True,
        **kwargs
    )
    
    val_dataset = StreamingPatientDataset.from_embeddings_file(
        remote=val_remote,
        local=val_local,
        embeddings_path=embeddings_path,
        batch_size=batch_size,
        shuffle=False,
        do_eval=True,
        **kwargs
    )
    
    return train_dataset, val_dataset

