from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

from loopsgl.core import get_global_ctx

from ..base import BaseLLMModel

if TYPE_CHECKING:
    import torch


class BaseLoopedModel(BaseLLMModel):
    def __init__(
        self,
        *,
        loop_steps: int,
        num_core_layers: int,
        num_prelude_layers: int = 0,
        num_coda_layers: int = 0,
    ) -> None:
        if isinstance(loop_steps, bool) or not isinstance(loop_steps, int) or loop_steps <= 0:
            raise ValueError("loop_steps must be a positive integer")
        self.loop_steps = loop_steps
        self.num_prelude_layers = num_prelude_layers
        self.num_core_layers = num_core_layers
        self.num_coda_layers = num_coda_layers
        self.loop_cache_policy: Literal["depth_indexed", "shared"] = "depth_indexed"

    @property
    def num_kv_layers(self) -> int:
        """Cache layers for one prelude, recurrent core slots, and one coda."""
        slots = self.loop_steps if self.loop_cache_policy == "depth_indexed" else 1
        return self.num_prelude_layers + slots * self.num_core_layers + self.num_coda_layers

    def prelude_cache_layer_index(self, layer_id: int) -> int:
        return layer_id

    def core_cache_layer_index(self, layer_id: int, loop_step: int) -> int:
        slot = loop_step if self.loop_cache_policy == "depth_indexed" else 0
        return self.num_prelude_layers + slot * self.num_core_layers + layer_id

    def coda_cache_layer_index(self, layer_id: int) -> int:
        return self.num_kv_layers - self.num_coda_layers + layer_id

    @abstractmethod
    def prelude(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    @abstractmethod
    def core_step(
        self, hidden_states: torch.Tensor, loop_step: int, injection: torch.Tensor | None
    ) -> torch.Tensor: ...

    @abstractmethod
    def coda(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def forward(self) -> torch.Tensor:
        hidden_states, injection = self.prelude(get_global_ctx().batch.input_ids)
        for loop_step in range(self.loop_steps):
            hidden_states = self.core_step(hidden_states, loop_step, injection)
        return self.coda(hidden_states)
