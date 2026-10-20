"""Construct the repository's ExMed-BERT masked-language model for AIR."""

from __future__ import annotations

from typing import Any

from exmed_bert.models.config import ExMedBertConfig
from exmed_bert.models.model import ExMedBertForMaskedLM

from .vocabulary import VocabularyArtifacts


def create_model(
    vocabulary: VocabularyArtifacts,
    manifest: dict[str, Any],
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_sequence_length: int,
) -> ExMedBertForMaskedLM:
    if hidden_size % num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    code_dict = vocabulary.build_code_dict()
    config = ExMedBertConfig(
        age_vocab_size=int(manifest["vocabulary"]["age_size"]),
        code_vocab_size=vocabulary.code_size,
        hidden_size=hidden_size,
        initialization="normal",
        intermediate_size=intermediate_size,
        max_position_embeddings=max_sequence_length,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        number_of_codes=int(manifest["vocabulary"]["entity_size"]),
        pad_token_id=0,
        predict_codes=True,
        predict_los=False,
        plos_weight=None,
        region_vocab_size=int(manifest["vocabulary"]["state_size"]),
        sex_vocab_size=int(manifest["vocabulary"]["sex_size"]),
        freeze_code_embeddings=False,
    )
    return ExMedBertForMaskedLM(config, code_embed=code_dict)
