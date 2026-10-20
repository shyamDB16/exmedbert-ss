"""Attention-mask-aware ExMed-BERT model construction and transfer learning."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from exmed_bert.models.config import ExMedBertConfig
from exmed_bert.models.model import ExMedBertModel

from .checkpoint import load_model_for_transfer
from .modeling import create_model as create_mlm_model
from .vocabulary import VocabularyArtifacts


def create_config(
    vocabulary: VocabularyArtifacts,
    pretrain_manifest: dict[str, Any],
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_sequence_length: int,
) -> ExMedBertConfig:
    if hidden_size % num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    return ExMedBertConfig(
        age_vocab_size=int(pretrain_manifest["vocabulary"]["age_size"]),
        code_vocab_size=vocabulary.code_size,
        hidden_size=hidden_size,
        initialization="normal",
        intermediate_size=intermediate_size,
        max_position_embeddings=max_sequence_length,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        number_of_codes=int(pretrain_manifest["vocabulary"]["entity_size"]),
        pad_token_id=0,
        predict_codes=False,
        predict_los=False,
        region_vocab_size=int(pretrain_manifest["vocabulary"]["state_size"]),
        sex_vocab_size=int(pretrain_manifest["vocabulary"]["sex_size"]),
        freeze_code_embeddings=False,
    )


class ExMedBertForBinaryEndpoint(nn.Module):
    """ExMed-BERT encoder with masked mean pooling and one binary logit."""

    def __init__(
        self,
        config: ExMedBertConfig,
        vocabulary: VocabularyArtifacts,
        *,
        positive_weight: float,
    ):
        super().__init__()
        self.bert = ExMedBertModel(
            config,
            add_pooling_layer=False,
            code_embed=vocabulary.build_code_dict(),
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()
        self.register_buffer("positive_weight", torch.tensor([positive_weight], dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        age_ids: torch.Tensor,
        sex_ids: torch.Tensor,
        state_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        endpoint_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden = self.bert(
            input_ids=input_ids,
            entity_ids=entity_ids,
            age_ids=age_ids,
            sex_ids=sex_ids,
            state_ids=state_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if endpoint_labels is not None:
            loss = binary_cross_entropy_with_logits(
                logits,
                endpoint_labels.float(),
                pos_weight=self.positive_weight,
            )
        return {"loss": loss, "logits": logits}


def create_classifier(
    vocabulary: VocabularyArtifacts,
    pretrain_manifest: dict[str, Any],
    *,
    positive_weight: float,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_sequence_length: int,
) -> ExMedBertForBinaryEndpoint:
    config = create_config(
        vocabulary,
        pretrain_manifest,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_sequence_length=max_sequence_length,
    )
    return ExMedBertForBinaryEndpoint(config, vocabulary, positive_weight=positive_weight)


def transfer_pretrained_encoder(
    classifier: ExMedBertForBinaryEndpoint,
    vocabulary: VocabularyArtifacts,
    pretrain_manifest: dict[str, Any],
    checkpoint_path: str,
    *,
    pretrain_manifest_sha256: str,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_sequence_length: int,
) -> None:
    source = create_mlm_model(
        vocabulary,
        pretrain_manifest,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_sequence_length=max_sequence_length,
    )
    load_model_for_transfer(
        checkpoint_path,
        source,
        expected_manifest_sha256=pretrain_manifest_sha256,
    )
    classifier.bert.load_state_dict(source.bert.state_dict(), strict=True)


def freeze_encoder(model: ExMedBertForBinaryEndpoint) -> None:
    for parameter in model.bert.parameters():
        parameter.requires_grad = False
