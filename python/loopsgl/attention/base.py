from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch

    from loopsgl.core import ForwardBatch


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: ForwardBatch
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: ForwardBatch) -> None: ...

    def prepare_core_metadata(self, batch: ForwardBatch, depths: int | torch.Tensor) -> None:
        raise ValueError("This attention backend does not support last-exited KV")

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: ForwardBatch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: ForwardBatch) -> None: ...


class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: ForwardBatch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: ForwardBatch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def prepare_core_metadata(self, batch: ForwardBatch, depths: int | torch.Tensor) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        backend.prepare_core_metadata(batch, depths)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: ForwardBatch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: ForwardBatch) -> None:
        self.decode_backend.prepare_for_replay(batch)
