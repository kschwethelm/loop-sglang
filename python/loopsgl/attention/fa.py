from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch

from loopsgl.core import ForwardBatch, get_global_ctx
from loopsgl.utils import init_logger, is_sm100_supported

from .base import BaseAttnBackend, BaseAttnMetadata
from .metadata import prepare_last_exited_metadata
from .utils import BaseCaptureData, get_metadata_buffer, prepare_paged_metadata

if TYPE_CHECKING:
    from loopsgl.models import ModelConfig

logger = init_logger(__name__)


@dataclass
class FACaptureData(BaseCaptureData):
    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor
    core_page_table: torch.Tensor | None = None
    core_out_loc: torch.Tensor | None = None
    fixed_page_table: torch.Tensor | None = None
    fixed_out_loc: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.core_start = getattr(self.kvcache, "core_start", 0)
        self.core_end = getattr(self.kvcache, "core_end", 0)
        self.capture: FACaptureData | None = None
        self.eager: BaseCaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.core_eager_table: torch.Tensor | None = None
        self.core_eager_out: torch.Tensor | None = None
        self.core_capture_table: torch.Tensor | None = None
        self.core_capture_out: torch.Tensor | None = None
        self.fixed_eager_table: torch.Tensor | None = None
        self.fixed_eager_out: torch.Tensor | None = None
        self.fixed_capture_table: torch.Tensor | None = None
        self.fixed_capture_out: torch.Tensor | None = None
        self.scale = config.head_dim**-0.5
        self.version = 4 if is_sm100_supported() else 3
        logger.info_rank0(f"Using FlashAttention backend: FA{self.version}")

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: ForwardBatch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        out_loc, page_table = batch.out_loc, metadata.page_table
        if hasattr(self.kvcache, "slot_map"):
            core = self.core_start <= layer_id < self.core_end
            locations = metadata.core_out_loc if core else metadata.fixed_out_loc
            tables = metadata.core_page_table if core else metadata.fixed_page_table
            assert locations is not None and tables is not None
            group = self.kvcache.layer_groups[layer_id]
            out_loc, page_table = locations[group], tables[group]
        self.kvcache.store_kv(k, v, out_loc, layer_id)
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )

    def prepare_core_metadata(self, batch: ForwardBatch, depths: int | torch.Tensor) -> None:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        assert metadata.core_page_table is not None and metadata.core_out_loc is not None
        prepare_last_exited_metadata(
            metadata.page_table,
            metadata.core_page_table,
            self.kvcache.version_map,
            metadata.cache_seqlens,
            metadata.cu_seqlens_q,
            batch.out_loc,
            metadata.core_out_loc,
            depths,
        )

    def _attach_kv_buffers(self, batch: ForwardBatch, *, capture: bool) -> None:
        if not hasattr(self.kvcache, "slot_map"):
            return
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        for kind, groups in (
            ("core", self.kvcache.core_groups),
            ("fixed", self.kvcache.fixed_groups),
        ):
            if not groups:
                continue
            table_name = f"{kind}_{'capture' if capture else 'eager'}_table"
            out_name = f"{kind}_{'capture' if capture else 'eager'}_out"
            table, out = getattr(self, table_name), getattr(self, out_name)
            shape = (groups, *metadata.page_table.shape)
            out_size = batch.out_loc.numel()
            if not capture:
                if table is None or any(old < new for old, new in zip(table.shape, shape)):
                    assert self.eager is not None
                    table = torch.empty(
                        (groups, *self.eager.page_table.shape),
                        dtype=torch.int32,
                        device=self.kvcache.device,
                    )
                    setattr(self, table_name, table)
                if out is None or out.shape[1] < out_size:
                    out = torch.empty(
                        (groups, 1 << (out_size - 1).bit_length()),
                        dtype=torch.int32,
                        device=self.kvcache.device,
                    )
                    setattr(self, out_name, out)
            assert table is not None and out is not None
            physical_table = table[:, : shape[1], : shape[2]]
            physical_out = out[:, :out_size]
            setattr(metadata, f"{kind}_page_table", physical_table)
            setattr(metadata, f"{kind}_out_loc", physical_out)
            if kind == "fixed":
                prepare_last_exited_metadata(
                    metadata.page_table,
                    physical_table,
                    self.kvcache.fixed_map,
                    metadata.cache_seqlens,
                    metadata.cu_seqlens_q,
                    batch.out_loc,
                    physical_out,
                    0,
                )

    def prepare_metadata(self, batch: ForwardBatch) -> None:
        self.eager = get_metadata_buffer(batch, self.eager, self.page_size)
        prepare_paged_metadata(batch, self.eager, self.page_size)
        bs = batch.padded_size
        num_pages = (max(batch.seq_lens_cpu) + self.page_size - 1) // self.page_size
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=self.eager.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=self.eager.cu_seqlens_q[: bs + 1],
            cache_seqlens=self.eager.seq_lens[:bs],
            max_seqlen_k=max(batch.seq_lens_cpu),
            max_seqlen_q=max(batch.extend_lens_cpu),
            page_table=self.eager.page_table[:bs, :num_pages],
        )

        self._attach_kv_buffers(batch, capture=False)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(
            max_bs, (max_seq_len + self.page_size - 1) // self.page_size, self.kvcache.device
        )
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)
        if hasattr(self.kvcache, "slot_map"):
            for kind, groups in (
                ("core", self.kvcache.core_groups),
                ("fixed", self.kvcache.fixed_groups),
            ):
                setattr(
                    self,
                    f"{kind}_capture_table",
                    torch.zeros(
                        (groups, *capture.page_table.shape),
                        dtype=torch.int32,
                        device=self.kvcache.device,
                    ),
                )
                setattr(
                    self,
                    f"{kind}_capture_out",
                    torch.zeros((groups, max_bs), dtype=torch.int32, device=self.kvcache.device),
                )

    def prepare_for_capture(self, batch: ForwardBatch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata
        self._attach_kv_buffers(batch, capture=True)

    def prepare_for_replay(self, batch: ForwardBatch) -> None:
        bs = batch.padded_size
        assert self.capture is not None and bs in self.capture_bs
        prepare_paged_metadata(batch, self.capture, self.page_size)
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=self.capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=self.capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=self.capture.seq_lens[:bs],
            max_seqlen_k=self.capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,
            page_table=self.capture.page_table[:bs],
        )

        self._attach_kv_buffers(batch, capture=True)


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,  # TODO: support FA4 on blackwell
    )
