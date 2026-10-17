from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from loopsgl.core import LoopParams, get_global_ctx
from loopsgl.distributed import try_get_tp_info
from loopsgl.layers import BaseOP
from loopsgl.utils import init_logger

from ..base import BaseLLMModel

if TYPE_CHECKING:
    import torch


logger = init_logger(__name__)


class BaseLoopedModel(ABC, BaseOP):
    def __init__(
        self,
        *,
        loop_steps: int,
        num_core_layers: int,
        num_prelude_layers: int = 0,
        num_coda_layers: int = 0,
        exit_threshold: float | None = None,
    ) -> None:
        if loop_steps is None:
            raise ValueError("loop_steps must be a positive integer")
        self.loop_params = LoopParams(loop_steps=loop_steps, exit_threshold=exit_threshold)
        self.num_prelude_layers = num_prelude_layers
        self.num_core_layers = num_core_layers
        self.num_coda_layers = num_coda_layers
        self._loop_cache_policy: Literal["depth_indexed", "shared", "last_exited"] = "depth_indexed"

    @property
    def loop_steps(self) -> int:
        assert self.loop_params.loop_steps is not None
        return self.loop_params.loop_steps

    @property
    def loop_cache_policy(self) -> Literal["depth_indexed", "shared", "last_exited"]:
        return self._loop_cache_policy

    @loop_cache_policy.setter
    def loop_cache_policy(self, policy: Literal["depth_indexed", "shared", "last_exited"]) -> None:
        if policy == "shared" and policy != self._loop_cache_policy and self.loop_steps > 1:
            tp_info = try_get_tp_info()
            if tp_info is None or tp_info.is_primary():
                logger.warning(
                    "Shared KV cache leads to different attention patterns during prefill and decode. "
                    "Prefill tokens attends to the same loop step, while decode tokens attend to the last processed loop step. "
                    "Retraction with recomputation and prefix reuse may change cached KV and subsequent outputs."
                )
        self._loop_cache_policy = policy

    @property
    def num_kv_layers(self) -> int:
        """Count the number of KV cache slot indices needed by the model.

        Prelude and coda run once, so each attention layer needs one cache slot.
        Core layers might need multiple cache slots depending on which cache policy is used:
        - Depth-indexed gives each loop step its own cache slot, using the existing cache layout.
        - Shared KV gives only one KV slot that is shared (and overwritten) across loop steps.
        - Last-exited KV gives each loop step its own cache slot, but uses a different implementation method
          that does not rely on slot indices. So we count one slot.
        """
        slots = self.loop_steps if self.loop_cache_policy == "depth_indexed" else 1
        return self.num_prelude_layers + slots * self.num_core_layers + self.num_coda_layers

    def prelude_cache_layer_index(self, layer_id: int) -> int:
        return layer_id

    def core_cache_layer_index(self, layer_id: int, loop_step: int | torch.Tensor) -> int:
        """Return the KV cache layer index for a core layer.

        Add the prelude offset to place core layers after the prelude layers.
        For depth-indexed KV, also add loop_step * num_core_layers. This gives each depth its own cache indices without changing the cache implementation.
        Last-exited KV resolves depth-specific addresses before attention, so we don't need to add anything here.
        """
        if self.loop_cache_policy == "depth_indexed":
            if not isinstance(loop_step, int):
                raise ValueError("Depth-indexed KV requires a scalar loop step")
            return self.num_prelude_layers + loop_step * self.num_core_layers + layer_id

        return self.num_prelude_layers + layer_id

    def coda_cache_layer_index(self, layer_id: int) -> int:
        return self.num_kv_layers - self.num_coda_layers + layer_id

    def cdb_state_spec(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Describe shapes and dtypes of per-token states retained between CDB execution stages."""
        raise NotImplementedError(f"{type(self).__name__} does not support staged state pools")

    def phase_uses_attention(self, phase: Literal["prelude", "core", "coda"]) -> bool:
        """Return whether a phase needs attention metadata, positions, or KV locations.

        Fully looped models can override this to return True only for the core.
        """
        return True

    @abstractmethod
    def forward_prelude(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    @abstractmethod
    def core_step(
        self,
        hidden_states: torch.Tensor,
        loop_step: int | torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor: ...

    def init_exit_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Initialize per-token exit state from the recurrent state returned by prelude.

        The result is a device tensor with tokens along its first dimension.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement init_exit_state for adaptive decoding"
        )

    def exit_signal(
        self,
        hidden_states: torch.Tensor,
        exit_state: torch.Tensor,
        loop_step: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (exit requested, updated per-token exit state).

        The scheduler determines whether the decision takes effect immediately or after one more core step.
        loop_step is the zero-based core step just executed, either a scalar or one device-tensor entry per token.
        Hidden states, exit state, and per-token loop steps refer to the same tokens in the same row order.
        The signal has shape [batch_size], dtype torch.bool, and the states' device.
        The updated exit state retains its input shape and dtype.
        Both exit methods must support CUDA graph capture without host synchronization.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement exit_signal for adaptive decoding"
        )

    @abstractmethod
    def forward_coda(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Execute the model's full forward pass.

        This is used by prefill of both CDB and CB, and for CB decoding.
        """
        hidden_states, injection = self.forward_prelude(input_ids)
        for loop_step in range(self.loop_steps):
            hidden_states = self.core_step(hidden_states, loop_step, injection)
        return self.forward_coda(hidden_states)


class BaseLoopedForCausalLM(BaseLLMModel):
    model: BaseLoopedModel

    def forward(self) -> torch.Tensor:
        hidden_states = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(hidden_states)

    def forward_coda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the transformer coda and return vocabulary logits."""
        return self.lm_head.forward(self.model.forward_coda(hidden_states))
