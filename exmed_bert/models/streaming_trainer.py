# imports ----------------------------------------------------------------------

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import logging

# global vars ------------------------------------------------------------------

logger = logging.get_logger(__name__)

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
else:
    _is_native_amp_available = False


# Check if Mosaic Streaming is available
try:
    from streaming import StreamingDataset, StreamingDataLoader
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    StreamingDataset = None
    StreamingDataLoader = None


def _get_rank() -> int:
    """Return current process rank (0 if not distributed)."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class StreamingDataLoaderResumeCallback(TrainerCallback):
    """
    Restores StreamingDataLoader state after checkpoint load so training resumes
    at the exact data index (deterministic resume).
    Runs on_train_begin when resuming; the train dataloader already exists then.
    """

    def __init__(self, trainer: Optional[Any] = None):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        resume_from_checkpoint = (
            getattr(self.trainer, "_resume_from_checkpoint_path", None)
            or getattr(args, "resume_from_checkpoint", None)
            or getattr(state, "resume_from_checkpoint", None)
        )
        if not resume_from_checkpoint:
            return

        trainer = self.trainer
        if trainer is None:
            return
        if not STREAMING_AVAILABLE or not hasattr(trainer, "train_dataloader"):
            return
        if not isinstance(trainer.train_dataloader, StreamingDataLoader):
            return

        rank = _get_rank()
        state_path = os.path.join(resume_from_checkpoint, f"dataloader_state_rank{rank}.pt")
        if not os.path.isfile(state_path):
            logger.warning(
                "StreamingDataLoader state not found at %s (rank %s). "
                "Resuming without exact data-index restore.",
                state_path,
                rank,
            )
            return

        try:
            state_dict = torch.load(state_path, weights_only=True)
            trainer.train_dataloader.load_state_dict(state_dict)
            logger.info("Restored StreamingDataLoader state from %s (rank %s)", state_path, rank)
        except Exception as e:
            logger.warning(
                "Failed to restore StreamingDataLoader state: %s. Resuming without exact data-index restore.",
                e,
            )


# class definitions ------------------------------------------------------------


class MedBertStreamingTrainer(Trainer):
    """
    Enhanced Trainer for Med-BERT that supports both regular and streaming datasets.
    
    Key features:
    - Compatible with Mosaic StreamingDataset
    - Uses StreamingDataLoader when appropriate for better checkpointing
    - Falls back to standard behavior for non-streaming datasets
    - Maintains custom prediction_step for Med-BERT evaluation
    """

    def __init__(self, *args, use_streaming_dataloader: bool = True, **kwargs):
        """
        Initialize trainer.
        
        Args:
            use_streaming_dataloader: If True, use StreamingDataLoader for StreamingDatasets
            *args, **kwargs: Passed to parent Trainer class
        """
        super().__init__(*args, **kwargs)
        self.use_streaming_dataloader = use_streaming_dataloader and STREAMING_AVAILABLE
        self._resume_from_checkpoint_path = None

        if self.use_streaming_dataloader:
            logger.info("StreamingDataLoader support enabled")
            self.add_callback(StreamingDataLoaderResumeCallback(self))
        elif use_streaming_dataloader and not STREAMING_AVAILABLE:
            logger.warning(
                "StreamingDataLoader requested but Mosaic Streaming not available. "
                "Install with: pip install mosaicml-streaming"
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader.
        
        If train_dataset is a StreamingDataset, returns a StreamingDataLoader
        for better checkpoint/resume support. Otherwise, returns standard DataLoader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        
        # Check if we should use StreamingDataLoader (StreamingDataset or our wrapper)
        uses_streaming = (
            self.use_streaming_dataloader
            and STREAMING_AVAILABLE
            and (
                isinstance(train_dataset, StreamingDataset)
                or (
                    hasattr(train_dataset, "streaming_dataset")
                    and isinstance(getattr(train_dataset, "streaming_dataset"), StreamingDataset)
                )
            )
        )
        if uses_streaming:
            logger.info("Using StreamingDataLoader for training")
            return self._create_streaming_dataloader(
                dataset=train_dataset,
                batch_size=self.args.train_batch_size,
                shuffle=True,
            )
        else:
            # Use default DataLoader from parent class
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation DataLoader.
        
        If eval_dataset is a StreamingDataset, returns a StreamingDataLoader.
        Otherwise, returns standard DataLoader.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
            
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # Check if we should use StreamingDataLoader (StreamingDataset or our wrapper)
        uses_streaming = (
            self.use_streaming_dataloader
            and STREAMING_AVAILABLE
            and (
                isinstance(eval_dataset, StreamingDataset)
                or (
                    hasattr(eval_dataset, "streaming_dataset")
                    and isinstance(getattr(eval_dataset, "streaming_dataset"), StreamingDataset)
                )
            )
        )
        if uses_streaming:
            
            logger.info("Using StreamingDataLoader for evaluation")
            return self._create_streaming_dataloader(
                dataset=eval_dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
            )
        else:
            # Use default DataLoader from parent class
            return super().get_eval_dataloader(eval_dataset)

    def _create_streaming_dataloader(
        self,
        dataset: "StreamingDataset",
        batch_size: int,
        shuffle: bool,
    ) -> "StreamingDataLoader":
        """
        Create a StreamingDataLoader with appropriate settings.
        
        Args:
            dataset: StreamingDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle
            
        Returns:
            StreamingDataLoader instance
        """
        if not STREAMING_AVAILABLE:
            raise ImportError("Mosaic Streaming not available")
            
        return StreamingDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            timeout=0,
            persistent_workers=self.args.dataloader_num_workers > 0,
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on model using inputs.
        
        This is the same custom prediction step used in the original MedBertTrainer,
        adapted for Med-BERT's multi-task pretraining (masked LM + length of stay).
        
        Args:
            model: The model to evaluate
            inputs: The inputs and targets
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore in output
            
        Returns:
            Tuple of (loss, logits, labels)
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        
        if ignore_keys is not None:
            raise NotImplementedError("ignore_keys not supported in MedBertStreamingTrainer")

        with torch.no_grad():
            outputs = model(**inputs)
            
            if has_labels:
                if self.label_smoother is not None and "labels" in inputs:
                    logger.error("Label smoother is not implemented")
                    raise NotImplementedError
                else:
                    loss = outputs["loss"].mean().detach()
                    logits = tuple(
                        v.argmax(2) for k, v in outputs.items() if k in ["logits"]
                    )
                    if "plos_logits" in outputs.keys():
                        logits += (outputs["plos_logits"].argmax(1),)
            else:
                loss = None
                logits = tuple(
                    v.argmax(2) for k, v in outputs.items() if k in ["logits"]
                )
                if "plos_logits" in outputs.keys():
                    logits += (outputs["plos_logits"].argmax(1),)
                    
            if self.args.past_index >= 0:
                logger.error("Past index has been removed in this implementation")
                raise NotImplementedError

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = tuple(inputs.get(name) for name in self.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return loss, logits, labels

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint with StreamingDataLoader state if applicable (per rank).
        Enables exact data-index resume when training is restarted with same GPU count.
        """
        checkpoint_folder = super()._save_checkpoint(model, trial, metrics=metrics)

        if not (
            self.use_streaming_dataloader
            and STREAMING_AVAILABLE
            and hasattr(self, "train_dataloader")
            and isinstance(self.train_dataloader, StreamingDataLoader)
        ):
            return checkpoint_folder

        try:
            rank = _get_rank()
            state_dict_path = os.path.join(checkpoint_folder, f"dataloader_state_rank{rank}.pt")
            # Mosaic StreamingDataLoader.state_dict() may take samples_in_epoch; try no-arg first
            state_dict = self.train_dataloader.state_dict()
            if state_dict is not None:
                torch.save(state_dict, state_dict_path)
                logger.info("Saved StreamingDataLoader state to %s (rank %s)", state_dict_path, rank)
        except Exception as e:
            logger.warning("Failed to save StreamingDataLoader state: %s", e)

        return checkpoint_folder

    def _load_from_checkpoint(self, resume_from_checkpoint):
        """
        Load training state only. StreamingDataLoader state is restored in
        StreamingDataLoaderResumeCallback.on_train_begin (after dataloader exists).
        """
        self._resume_from_checkpoint_path = resume_from_checkpoint
        return super()._load_from_checkpoint(resume_from_checkpoint)


# Backwards compatibility: alias to the original MedBertTrainer behavior
class MedBertTrainer(MedBertStreamingTrainer):
    """
    Alias for MedBertStreamingTrainer with streaming disabled by default.
    
    This maintains backwards compatibility with existing code that uses MedBertTrainer.
    """
    def __init__(self, *args, **kwargs):
        # Default to not using streaming for backwards compatibility
        kwargs.setdefault('use_streaming_dataloader', False)
        super().__init__(*args, **kwargs)

