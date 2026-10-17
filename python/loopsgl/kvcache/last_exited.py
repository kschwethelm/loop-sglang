from __future__ import annotations

from math import gcd

import torch
import triton
import triton.language as tl

from loopsgl.distributed import get_tp_info
from loopsgl.utils import div_even

from .base import BaseKVCachePool


@triton.jit
def _reserve_versions(
    handles,
    restore_depths,
    versions,
    depths,
    stack,
    counter,
    restore: tl.constexpr,
    width: tl.constexpr,
    steps: tl.constexpr,
    fixed_groups: tl.constexpr,
    core_groups: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    handle = tl.load(handles + token)
    slots = tl.arange(0, block)
    depth = tl.load(restore_depths + token) if restore else steps
    count = fixed_groups + depth * core_groups
    start = tl.atomic_add(counter, -count) - count
    # Uncomputed depths share the last computed version of each core group.
    source = tl.where(
        slots < fixed_groups,
        slots,
        fixed_groups
        + tl.minimum((slots - fixed_groups) // core_groups, depth - 1) * core_groups
        + (slots - fixed_groups) % core_groups,
    )
    mask = slots < fixed_groups + steps * core_groups
    rows = tl.load(stack + start + source, mask, 0)
    tl.store(versions + slots * width + handle, rows, mask)
    tl.store(depths + handle, depth)


@triton.jit
def _release_versions(
    handles,
    exits,
    versions,
    depths,
    stack,
    counter,
    width: tl.constexpr,
    steps: tl.constexpr,
    fixed_groups: tl.constexpr,
    core_groups: tl.constexpr,
    alias: tl.constexpr,
    exit_stride: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    handle = tl.load(handles + token)
    old_depth = tl.load(depths + handle)
    depth = tl.load(exits + token * exit_stride) if alias else 0
    slots = tl.arange(0, block)
    end = fixed_groups + old_depth * core_groups
    begin = fixed_groups + depth * core_groups if alias else 0
    rows = tl.load(versions + slots * width + handle, slots < end, 0)
    start = tl.atomic_add(counter, end - begin)
    tl.store(stack + start + slots - begin, rows, (slots >= begin) & (slots < end))
    if alias:
        final = fixed_groups + (depth - 1) * core_groups + (slots - fixed_groups) % core_groups
        final_row = tl.load(versions + final * width + handle)
        tl.store(
            versions + slots * width + handle,
            final_row,
            (slots >= begin) & (slots < fixed_groups + steps * core_groups),
        )
    tl.store(depths + handle, depth)


class LastExitedKVCache(BaseKVCachePool):
    """Share physical KV slots between fixed layers and recurrent depth versions.

    A slot reserves one row in each layer lane, with contiguous rows within each lane.
    Fixed layers and core depths allocate from the same free list.
    Logical token handles remain stable across prefix insertion and eviction.
    Completion aliases skipped depths to the final computed version of each core group.
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
        core_start: int,
        core_end: int,
        loop_steps: int,
    ) -> None:
        if page_size != 1:
            raise ValueError("Last-exited KV storage requires page_size=1.")
        if not 0 <= core_start < core_end <= num_layers or loop_steps < 1:
            raise ValueError("Invalid recurrent KV layer range or loop depth.")
        if num_pages < 1:
            raise ValueError("KV pools must include at least the padding row.")
        local_heads = div_even(num_kv_heads, get_tp_info().size, allow_replicate=True)
        self.core_start = core_start
        self.core_end = core_end
        self.loop_steps = loop_steps
        self._num_layers = num_layers
        self._device = device
        self._dtype = dtype
        num_core_layers = core_end - core_start
        num_fixed_layers = num_layers - num_core_layers
        self.layers_per_slot = gcd(num_core_layers, num_fixed_layers)
        self.core_groups = num_core_layers // self.layers_per_slot
        self.fixed_groups = num_fixed_layers // self.layers_per_slot
        self.slots_per_token = self.fixed_groups + loop_steps * self.core_groups
        # Preserve the nominal page budget while sharing all physical storage.
        self.num_kv_slots = (num_pages - 1) * (num_layers // self.layers_per_slot) + 1
        self.available_slots = self.num_kv_slots - 1
        self.slot_map = torch.zeros(
            (self.slots_per_token, num_pages), dtype=torch.int32, device=device
        )
        self.fixed_map = self.slot_map[: self.fixed_groups].unsqueeze(0)
        self.version_map = self.slot_map[self.fixed_groups :].view(
            loop_steps, self.core_groups, num_pages
        )
        self.completed_depth = torch.zeros(num_pages, dtype=torch.int32, device=device)
        self._free_slots = torch.arange(1, self.num_kv_slots, dtype=torch.int32, device=device)
        self._free_count = torch.tensor([self.available_slots], dtype=torch.int32, device=device)
        self._row_shape = (local_heads, head_dim)
        self._row_bytes = local_heads * head_dim * dtype.itemsize
        self._kv_buffer = torch.empty(
            (2, self.layers_per_slot, self.num_kv_slots, local_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self._kv_buffer[:, :, 0].zero_()
        self.layer_groups: list[int] = []
        self._buffers: list[torch.Tensor] = []
        for layer in range(num_layers):
            ordinal = (
                layer - core_start
                if self.is_core_layer(layer)
                else layer if layer < core_start else layer - num_core_layers
            )
            group, lane = divmod(ordinal, self.layers_per_slot)
            self.layer_groups.append(group)
            self._buffers.append(self._kv_buffer[:, lane].unsqueeze(2))

    @staticmethod
    def memory_per_page(
        num_layers: int, num_core_layers: int, loop_steps: int, row_bytes: int
    ) -> int:
        """Bound storage and allocator metadata bytes per nominal one-token page."""
        layers_per_slot = gcd(num_core_layers, num_layers - num_core_layers)
        core_groups = num_core_layers // layers_per_slot
        fixed_groups = (num_layers - num_core_layers) // layers_per_slot
        mappings = fixed_groups + loop_steps * core_groups
        # Account for slot mappings, completed depths, and the shared free-slot stack.
        return (
            2 * num_layers * row_bytes
            + (mappings + 1 + fixed_groups + core_groups) * torch.int32.itemsize
        )

    def restore_slots(self, backup: torch.Tensor, num_tokens: int) -> int:
        """Count exact fixed-layer and recurrent storage required by a CPU snapshot."""
        depths = self.restore_depths(backup, num_tokens)
        return num_tokens * self.fixed_groups + int(depths.sum()) * self.core_groups

    def reserve_versions(self, handles: torch.Tensor) -> None:
        needed = len(handles) * self.slots_per_token
        if needed > self.available_slots:
            raise RuntimeError(
                f"KV pool exhausted: {needed} slots requested, {self.available_slots} available."
            )
        if len(handles):
            _reserve_versions[(len(handles),)](
                handles,
                handles,
                self.slot_map,
                self.completed_depth,
                self._free_slots,
                self._free_count,
                False,
                self.slot_map.shape[1],
                self.loop_steps,
                self.fixed_groups,
                self.core_groups,
                triton.next_power_of_2(self.slots_per_token),
            )
            self.available_slots -= needed

    def alias_versions(
        self, handles: torch.Tensor, depths: torch.Tensor, released_versions: int
    ) -> None:
        if len(handles) != len(depths) or not 0 <= released_versions <= len(handles) * (
            self.loop_steps - 1
        ):
            raise ValueError("Invalid exit metadata.")
        if len(handles):
            _release_versions[(len(handles),)](
                handles,
                depths,
                self.slot_map,
                self.completed_depth,
                self._free_slots,
                self._free_count,
                self.slot_map.shape[1],
                self.loop_steps,
                self.fixed_groups,
                self.core_groups,
                True,
                depths.stride(0),
                triton.next_power_of_2(self.slots_per_token),
            )
            self.available_slots += released_versions * self.core_groups

    def free_versions(self, handles: torch.Tensor, released_slots: int) -> None:
        if not 0 <= released_slots <= len(handles) * self.slots_per_token:
            raise ValueError("Invalid freed KV slot count.")
        if len(handles):
            _release_versions[(len(handles),)](
                handles,
                handles,
                self.slot_map,
                self.completed_depth,
                self._free_slots,
                self._free_count,
                self.slot_map.shape[1],
                self.loop_steps,
                self.fixed_groups,
                self.core_groups,
                False,
                1,
                triton.next_power_of_2(self.slots_per_token),
            )
            self.available_slots += released_slots

    def is_core_layer(self, index: int) -> bool:
        return self.core_start <= index < self.core_end

    def restore_depths(self, backup: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if (
            backup.dtype != torch.uint8
            or not backup.is_cpu
            or backup.nbytes != self.cpu_copy_size(num_tokens)
        ):
            raise ValueError("Invalid last-exited CPU KV backup.")
        depths = (
            backup[-num_tokens * 4 :].view(torch.int32)
            if num_tokens
            else torch.empty(0, dtype=torch.int32)
        )
        if bool((depths < 1).any()) or bool((depths > self.loop_steps).any()):
            raise ValueError("CPU KV backup contains invalid completed depths.")
        return depths

    def reserve_restore_versions(self, handles: torch.Tensor, backup: torch.Tensor) -> int:
        depths = self.restore_depths(backup, len(handles))
        needed = len(handles) * self.fixed_groups + int(depths.sum()) * self.core_groups
        if needed > self.available_slots:
            raise RuntimeError("Recurrent KV pool lacks space for the exact CPU snapshot.")
        if len(handles):
            _reserve_versions[(len(handles),)](
                handles,
                depths.to(self.device),
                self.slot_map,
                self.completed_depth,
                self._free_slots,
                self._free_count,
                True,
                self.slot_map.shape[1],
                self.loop_steps,
                self.fixed_groups,
                self.core_groups,
                triton.next_power_of_2(self.slots_per_token),
            )
            self.available_slots -= needed
        return len(handles) * self.loop_steps - int(depths.sum())

    def k_cache(self, index: int) -> torch.Tensor:
        return self._buffers[index][0]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._buffers[index][1]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from loopsgl.kernel import store_cache

        buffer = self._buffers[layer_id]
        store_cache(
            k_cache=buffer[0].view(-1, *self._row_shape),
            v_cache=buffer[1].view(-1, *self._row_shape),
            indices=out_loc,
            k=k,
            v=v,
            row_bytes=self._row_bytes,
        )

    @property
    def _expanded_layers(self) -> int:
        return self.num_layers + (self.core_end - self.core_start) * (self.loop_steps - 1)

    def cpu_copy_size(self, num_tokens: int) -> int:
        return 2 * self._expanded_layers * num_tokens * self._row_bytes + num_tokens * 4

    def get_cpu_copy(self, indices: torch.Tensor) -> torch.Tensor:
        """Pack exact KV bytes followed by one int32 completed depth per token."""
        indices = indices.to(device=self.device, dtype=torch.int64)
        count = len(indices)
        shape = (2, self._expanded_layers, count, *self._row_shape)
        backup = torch.empty(self.cpu_copy_size(count), dtype=torch.uint8, device="cpu")
        value_bytes = backup.numel() - count * 4
        rows = backup[:value_bytes].view(self.dtype).view(shape)
        expanded = 0
        for layer, buffer in enumerate(self._buffers):
            steps = self.loop_steps if self.core_start <= layer < self.core_end else 1
            for depth in range(steps):
                locations = (
                    self.version_map[depth, self.layer_groups[layer], indices].long()
                    if steps > 1 or self.core_start <= layer < self.core_end
                    else self.fixed_map[0, self.layer_groups[layer], indices].long()
                )
                chunk_size = max(1, (64 * 1024**2) // self._row_bytes)
                for kv in range(2):
                    for start in range(0, count, chunk_size):
                        end = start + chunk_size
                        rows[kv, expanded, start:end].copy_(
                            buffer[kv, :, 0].index_select(0, locations[start:end])
                        )
                expanded += 1
        backup[value_bytes:].view(torch.int32).copy_(self.completed_depth[indices])
        return backup

    def load_cpu_copy(self, cpu_tensors: torch.Tensor, indices: torch.Tensor) -> None:
        """Restore a snapshot after its exact live versions have been reserved."""
        count = len(indices)
        self.restore_depths(cpu_tensors, count)
        value_bytes = cpu_tensors.numel() - count * 4
        values = (
            cpu_tensors[:value_bytes]
            .view(self.dtype)
            .view(2, self._expanded_layers, count, *self._row_shape)
        )
        expanded = 0
        for layer, buffer in enumerate(self._buffers):
            steps = self.loop_steps if self.core_start <= layer < self.core_end else 1
            for depth in range(steps):
                locations = (
                    self.version_map[depth, self.layer_groups[layer], indices].long()
                    if self.core_start <= layer < self.core_end
                    else self.fixed_map[0, self.layer_groups[layer], indices].long()
                )
                chunk_size = max(1, (64 * 1024**2) // self._row_bytes)
                for kv in range(2):
                    for start in range(0, count, chunk_size):
                        end = start + chunk_size
                        buffer[kv, :, 0].index_copy_(
                            0, locations[start:end], values[kv, expanded, start:end].to(self.device)
                        )
                expanded += 1
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
