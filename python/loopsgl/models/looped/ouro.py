from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from loopsgl.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
)
from loopsgl.utils import nvtx_annotate

from ..utils import GatedMLP, RopeAttn
from .base import BaseLoopedModel

if TYPE_CHECKING:
    from ..config import ModelConfig


class OuroDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self.self_attn = RopeAttn(config, layer_id)
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        cache_layer_id: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x, cache_layer_id=cache_layer_id)
        x = self.input_layernorm_2.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        x = self.post_attention_layernorm_2.forward(x)
        return x, residual


class OuroModel(BaseOP):
    """Layer and weight container for staged Ouro execution."""

    def __init__(self, config: ModelConfig) -> None:
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [OuroDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.early_exit_gate = LinearReplicated(config.hidden_size, 1, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Execute Ouro through OuroForCausalLM's loop stages.")


class OuroForCausalLM(BaseLoopedModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(loop_steps=config.loop_steps, num_core_layers=config.num_layers)
        self.model = OuroModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def prelude(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.model.embed_tokens.forward(input_ids), None

    def core_step(
        self, hidden_states: torch.Tensor, loop_step: int, injection: torch.Tensor | None
    ) -> torch.Tensor:
        x = hidden_states
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.model.layers.op_list):
            x, residual = layer.forward(
                x, residual, cache_layer_id=self.core_cache_layer_index(layer_id, loop_step)
            )
        return self.model.norm.forward(x, residual)[0]

    def coda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head.forward(hidden_states)


__all__ = ["OuroForCausalLM"]
