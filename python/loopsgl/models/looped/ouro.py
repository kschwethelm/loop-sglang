from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Tuple

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
from .base import BaseLoopedForCausalLM, BaseLoopedModel
from .exit_gates import cumulative_exit_probability

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
        core_metadata_depth: int | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(
            x, cache_layer_id=cache_layer_id, core_metadata_depth=core_metadata_depth
        )
        x = self.input_layernorm_2.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        x = self.post_attention_layernorm_2.forward(x)
        return x, residual


class OuroModel(BaseLoopedModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(
            loop_steps=config.loop_steps,
            num_core_layers=config.num_layers,
            exit_threshold=config.loop_params.exit_threshold,
        )
        threshold = self.loop_params.exit_threshold
        if threshold is not None and not 0 < threshold <= 1:
            raise ValueError("Ouro exit_threshold must be in (0, 1]")
        self.min_loop_steps = getattr(config.hf_config, "min_loop_steps", 1)
        if (
            isinstance(self.min_loop_steps, bool)
            or not isinstance(self.min_loop_steps, int)
            or not 1 <= self.min_loop_steps <= self.loop_steps
        ):
            raise ValueError("Ouro min_loop_steps must be an integer in [1, loop_steps]")
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
        self._exit_gate_type = getattr(config.hf_config, "exit_gate_type", "early_exit")
        if self._exit_gate_type not in ("early_exit", "lookahead"):
            raise ValueError("Ouro exit_gate_type must be 'early_exit' or 'lookahead'")

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """Load all model weights after mapping the selected gate weights to 'early_exit_gate' keys."""
        early_prefix = f"{prefix}.early_exit_gate" if prefix else "early_exit_gate"
        lookahead_prefix = f"{prefix}.lookahead_exit_gate" if prefix else "lookahead_exit_gate"
        for parameter in ("weight", "bias"):
            lookahead_key = f"{lookahead_prefix}.{parameter}"
            if self._exit_gate_type == "lookahead":
                state_dict[f"{early_prefix}.{parameter}"] = state_dict.pop(lookahead_key)
            else:
                state_dict.pop(lookahead_key, None)
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

    def cdb_state_spec(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Describe shapes and dtypes of per-token states retained between CDB execution stages."""
        weight = self.embed_tokens.weight
        hidden = ((weight.shape[-1],), weight.dtype)
        return {"hidden": hidden, "exit": ((), torch.float32)}

    def phase_uses_attention(self, phase: Literal["prelude", "core", "coda"]) -> bool:
        return phase == "core"

    def forward_prelude(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.embed_tokens.forward(input_ids), None

    def core_step(
        self,
        hidden_states: torch.Tensor,
        loop_step: int | torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        x = hidden_states
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(
                x,
                residual,
                cache_layer_id=self.core_cache_layer_index(layer_id, loop_step),
                core_metadata_depth=(
                    loop_step if layer_id == 0 and self.loop_cache_policy == "last_exited" else None
                ),
            )
        return self.norm.forward(x, residual)[0]

    def init_exit_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Start each token's cumulative exit probability at zero in FP32."""
        return torch.zeros(hidden_states.shape[0], dtype=torch.float32, device=hidden_states.device)

    def exit_signal(
        self,
        hidden_states: torch.Tensor,
        exit_state: torch.Tensor,
        loop_step: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate exit probability and threshold it to request an exit."""
        logits = self.early_exit_gate.forward(hidden_states).squeeze(-1)
        next_exit_state = cumulative_exit_probability(logits, exit_state)
        exit_depth = loop_step + 1 + (self._exit_gate_type == "lookahead")
        signal = (next_exit_state >= self.loop_params.exit_threshold) & (
            exit_depth >= self.min_loop_steps
        )
        return signal, next_exit_state

    def forward_coda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class OuroForCausalLM(BaseLoopedForCausalLM):
    def __init__(self, config: ModelConfig) -> None:
        self.model = OuroModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )


__all__ = ["OuroForCausalLM"]
