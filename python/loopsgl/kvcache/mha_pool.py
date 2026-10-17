# CPU KV copy methods adapted from SGLang's memory_pool.py.
# Copyright 2023-2024 SGLang Team.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/sglang.txt in the repository root.

from __future__ import annotations

import torch

from loopsgl.distributed import get_tp_info
from loopsgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._kv_buffer[:, :, 0].zero_()
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        self._row_bytes = local_kv_heads * head_dim * dtype.itemsize
        self._cpu_offloading_chunk_size = max(1, (64 * 1024**2) // self._row_bytes)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from loopsgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
            row_bytes=self._row_bytes,
        )

    def cpu_copy_size(self, num_tokens: int) -> int:
        return 2 * self.num_layers * num_tokens * self._row_bytes

    def get_cpu_copy(self, indices: torch.Tensor) -> torch.Tensor:
        """Snapshot every KV layer in token order, with bounded device staging space."""
        indices = indices.to(device=self.device, dtype=torch.int64)
        shape = (2, self.num_layers, len(indices), *self._storage_shape[1:])
        cpu_tensors = torch.empty(shape, dtype=self.dtype, device="cpu")
        chunk_size = self._cpu_offloading_chunk_size
        for kv_index, buffer in enumerate((self._k_buffer, self._v_buffer)):
            for layer_id in range(self.num_layers):
                storage = buffer[layer_id].view(self._storage_shape)
                for start in range(0, len(indices), chunk_size):
                    end = start + chunk_size
                    rows = storage.index_select(0, indices[start:end])
                    cpu_tensors[kv_index, layer_id, start:end].copy_(rows, non_blocking=False)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        return cpu_tensors

    def load_cpu_copy(self, cpu_tensors: torch.Tensor, indices: torch.Tensor) -> None:
        """Restore exact KV values into new token locations before releasing the backup."""
        expected_shape = (2, self.num_layers, len(indices), *self._storage_shape[1:])
        if (
            not cpu_tensors.is_cpu
            or cpu_tensors.dtype != self.dtype
            or tuple(cpu_tensors.shape) != expected_shape
        ):
            raise ValueError("CPU KV backup does not match the destination pool and token count.")
        indices = indices.to(device=self.device, dtype=torch.int64)
        chunk_size = self._cpu_offloading_chunk_size
        for kv_index, buffer in enumerate((self._k_buffer, self._v_buffer)):
            for layer_id in range(self.num_layers):
                storage = buffer[layer_id].view(self._storage_shape)
                for start in range(0, len(indices), chunk_size):
                    end = start + chunk_size
                    rows = cpu_tensors[kv_index, layer_id, start:end].to(self.device)
                    storage.index_copy_(0, indices[start:end], rows)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
