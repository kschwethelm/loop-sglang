from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from loopsgl.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
)
from loopsgl.utils import init_logger, nvtx_annotate

from ..utils import GatedMLP, RopeAttn
from .base import BaseLoopedForCausalLM, BaseLoopedModel
from .config import HuginnConfig
from .exit_gates import relative_state_change

if TYPE_CHECKING:
    from ..config import ModelConfig


logger = init_logger(__name__)


class HuginnAttention(RopeAttn):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        hf_config = cast(HuginnConfig, config.hf_config)
        super().__init__(config, layer_id, has_attn_bias=hf_config.attention_bias)

    @nvtx_annotate("MHA")
    def forward(
        self,
        x: torch.Tensor,
        *,
        cache_layer_id: int | None = None,
        core_metadata_depth: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Huginn adds its query/key bias after rounding the projection to the model dtype.
        qkv = F.linear(x, self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            qkv.add_(self.qkv_proj.bias)
        o = self.attn.forward(
            qkv, cache_layer_id=cache_layer_id, core_metadata_depth=core_metadata_depth
        )
        return self.o_proj.forward(o)


class HuginnDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self.self_attn = HuginnAttention(config, layer_id)
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        *,
        cache_layer_id: int | None = None,
        core_metadata_depth: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(
            x, cache_layer_id=cache_layer_id, core_metadata_depth=core_metadata_depth
        )
        x = self.input_layernorm_2.forward(x, residual)[0]
        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)
        x = self.post_attention_layernorm_2.forward(x, residual)[0]
        return x


class HuginnModel(BaseLoopedModel):
    def __init__(self, config: ModelConfig) -> None:
        hf_config = cast(HuginnConfig, config.hf_config)
        super().__init__(
            loop_steps=config.loop_steps,
            num_core_layers=hf_config.num_core_layers,
            num_prelude_layers=hf_config.num_prelude_layers,
            num_coda_layers=hf_config.num_coda_layers,
            exit_threshold=config.loop_params.exit_threshold,
        )
        threshold = self.loop_params.exit_threshold
        if threshold is not None and (not isfinite(threshold) or threshold < 0):
            raise ValueError("Huginn exit_threshold must be finite and nonnegative")
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.prelude = OPList(
            [
                HuginnDecoderLayer(config, layer_id)
                for layer_id in range(hf_config.num_prelude_layers)
            ]
        )
        self.adapter = LinearReplicated(
            2 * config.hidden_size, config.hidden_size, has_bias=hf_config.bias
        )
        self.layers = OPList(
            [
                HuginnDecoderLayer(config, hf_config.num_prelude_layers + layer_id)
                for layer_id in range(hf_config.num_core_layers)
            ]
        )
        coda_layer_offset = hf_config.num_prelude_layers + hf_config.num_core_layers
        self.coda = OPList(
            [
                HuginnDecoderLayer(config, coda_layer_offset + layer_id)
                for layer_id in range(hf_config.num_coda_layers)
            ]
        )
        self.norm = RMSNorm(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._embed_scale = hf_config.init_values["embed_scale"]
        self._state_init_std = hf_config.init_values["std"]
        self._state_init = hf_config.state_init
        if self._state_init != "zero" and self._state_init_std > 0:
            logger.warning_rank0(
                "Huginn uses random loop-state initialization. "
                "This makes the forward pass non-deterministic. "
                "This affects prefix caching and retraction with recomputation."
            )

    def cdb_state_spec(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Describe shapes and dtypes of per-token states retained between CDB execution stages."""
        weight = self.embed_tokens.weight
        hidden = ((weight.shape[-1],), weight.dtype)
        return {"hidden": hidden, "injection": hidden, "exit": hidden}

    def forward_prelude(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        injection = self.embed_tokens.forward(input_ids)
        injection = injection * self._embed_scale
        for layer_id, layer in enumerate(self.prelude.op_list):
            injection = layer.forward(
                injection, cache_layer_id=self.prelude_cache_layer_index(layer_id)
            )
        if self._state_init != "zero" and self._state_init_std > 0:
            state = torch.empty_like(injection)
            std = self._state_init_std
            torch.nn.init.trunc_normal_(state, std=std, a=-3 * std, b=3 * std)
            state.mul_(self._embed_scale)
        else:
            state = torch.zeros_like(injection)
        return state, injection

    def core_step(
        self,
        hidden_states: torch.Tensor,
        loop_step: int | torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        assert injection is not None
        x = self.adapter.forward(torch.cat([hidden_states, injection], dim=-1))
        for layer_id, layer in enumerate(self.layers.op_list):
            x = layer.forward(
                x,
                cache_layer_id=self.core_cache_layer_index(layer_id, loop_step),
                core_metadata_depth=(
                    loop_step if layer_id == 0 and self.loop_cache_policy == "last_exited" else None
                ),
            )
        return x

    def init_exit_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Use the initial recurrent state as the first comparison's previous state."""
        return hidden_states

    def exit_signal(
        self,
        hidden_states: torch.Tensor,
        exit_state: torch.Tensor,
        loop_step: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Threshold relative state change to request an exit."""
        return (
            relative_state_change(hidden_states, exit_state) < self.loop_params.exit_threshold,
            hidden_states,
        )

    def forward_coda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(hidden_states)
        for layer_id, layer in enumerate(self.coda.op_list):
            x = layer.forward(x, cache_layer_id=self.coda_cache_layer_index(layer_id))
        x = self.norm.forward(x)
        return x


class HuginnForCausalLM(BaseLoopedForCausalLM):
    def __init__(self, config: ModelConfig) -> None:
        self.model = HuginnModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )


__all__ = ["HuginnForCausalLM"]
