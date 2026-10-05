from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
import torch
from loopsgl.core import Batch, Req, SamplingParams
from loopsgl.distributed import DistributedInfo
from loopsgl.engine.config import EngineConfig
from loopsgl.engine.engine import Engine
from loopsgl.models.looped.base import BaseLoopedModel
from loopsgl.models.looped.config import OuroConfig

LoopCachePolicy = Literal["depth_indexed", "shared"]


class LayoutModel(BaseLoopedModel):
    def prelude(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return input_ids, None

    def core_step(
        self, hidden_states: torch.Tensor, loop_step: int, injection: torch.Tensor | None
    ) -> torch.Tensor:
        return hidden_states

    def coda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


@pytest.mark.parametrize(
    ("policy", "num_kv_layers", "core_indices", "coda_indices", "ouro_kv_layers"),
    [
        ("depth_indexed", 36, [2, 6, 10, 14, 18, 22, 26, 30], [34, 35], 96),
        ("shared", 8, [2] * 8, [6, 7], 24),
    ],
)
def test_loop_cache_layout(
    policy: LoopCachePolicy,
    num_kv_layers: int,
    core_indices: list[int],
    coda_indices: list[int],
    ouro_kv_layers: int,
) -> None:
    """Prelude and coda stay distinct while core slots follow the selected policy."""
    model = LayoutModel(loop_steps=8, num_prelude_layers=2, num_core_layers=4, num_coda_layers=2)
    model.loop_cache_policy = policy
    prelude = [model.prelude_cache_layer_index(i) for i in range(2)]
    core = [[model.core_cache_layer_index(i, step) for i in range(4)] for step in range(8)]
    coda = [model.coda_cache_layer_index(i) for i in range(2)]
    assert model.num_kv_layers == num_kv_layers
    assert prelude == [0, 1]
    assert [indices[0] for indices in core] == core_indices
    assert coda == coda_indices
    core_slots = {index for indices in core for index in indices}
    assert len(core_slots) == num_kv_layers - len(prelude) - len(coda)
    assert set(prelude) | core_slots | set(coda) == set(range(num_kv_layers))

    ouro = LayoutModel(loop_steps=4, num_core_layers=24)
    ouro.loop_cache_policy = policy
    assert ouro.num_kv_layers == ouro_kv_layers
    assert ouro.core_cache_layer_index(0, 0) == 0
    assert ouro.core_cache_layer_index(23, 3) == ouro_kv_layers - 1


@pytest.mark.skipif(
    not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available(),
    reason="Ouro integration check requires CUDA in a SLURM allocation",
)
@pytest.mark.parametrize("policy", ["depth_indexed", "shared"])
def test_ouro_loop_cache(policy: LoopCachePolicy, tmp_path: Path) -> None:
    """Check cached prefill/decode and graph replay in a fresh engine process."""
    # Engine initialization requires a fresh CUDA and distributed context.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), policy, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@torch.inference_mode()
def _check_ouro_loop_cache(policy: LoopCachePolicy, checkpoint: Path) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "GPU checks require a SLURM allocation"
    config = OuroConfig(
        total_ut_steps=3,
        architectures=["OuroForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=1e6,
        max_position_embeddings=128,
    )
    config.save_pretrained(checkpoint)
    engine = Engine(
        EngineConfig(
            model_path=str(checkpoint),
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            use_dummy_weight=True,
            use_pynccl=False,
            max_running_req=1,
            num_page_override=16,
            cuda_graph_bs=[2],
            loop_cache_policy=policy,
        )
    )
    try:
        num_layers = 6 if policy == "depth_indexed" else 2
        expected_indices = list(range(6)) if policy == "depth_indexed" else [0, 1] * 3
        cache = engine.kv_cache
        assert engine.model.loop_steps == 3
        assert engine.model.num_kv_layers == engine.num_kv_layers == cache.num_layers == num_layers
        buffers = [
            buffer for i in range(num_layers) for buffer in (cache.k_cache(i), cache.v_cache(i))
        ]
        assert torch.count_nonzero(engine.page_table[engine.dummy_req.table_idx]) == 0
        for buffer in buffers:
            assert torch.count_nonzero(buffer[0]) == 0
        for name, weight in engine.model.state_dict().items():
            if "layernorm" in name or name == "model.norm.weight":
                weight.fill_(1)
            else:
                weight.mul_(0.05)
        for buffer in buffers:
            buffer.zero_()
        engine.page_table[0, :8] = torch.arange(1, 9, device=engine.device, dtype=torch.int32)

        writes: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        store_kv = cache.store_kv

        def record_kv(
            k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
        ) -> None:
            writes.append((layer_id, k.clone().view(-1, 2, 64), v.clone().view(-1, 2, 64)))
            store_kv(k, v, out_loc, layer_id)

        cache.store_kv = record_kv
        tokens = [1, 2, 3]
        for step in range(3):
            cached_len = 0 if step == 0 else len(tokens) - 1
            req = Req(
                torch.tensor(tokens, dtype=torch.int32), 0, cached_len, 2, 0, SamplingParams(), None
            )  # type: ignore[arg-type]
            batch = Batch([req], "prefill" if step == 0 else "decode")
            batch.padded_reqs = batch.reqs
            batch.input_ids = torch.tensor(
                tokens[cached_len:], device=engine.device, dtype=torch.int32
            )
            batch.positions = torch.arange(
                cached_len, len(tokens), device=engine.device, dtype=torch.int32
            )
            batch.out_loc = engine.page_table[0, cached_len : len(tokens)]
            if step > 0:
                batch.padded_reqs = batch.reqs + [engine.dummy_req]
                for name in ("input_ids", "positions", "out_loc"):
                    value = getattr(batch, name)
                    setattr(batch, name, torch.cat([value, value.new_zeros(1)]))
            engine.attn_backend.prepare_metadata(batch)
            before = [buffer.clone() for buffer in buffers]
            writes.clear()
            with engine.ctx.forward_batch(batch):
                eager = engine.model.forward().float()[: batch.size].clone()
            assert torch.isfinite(eager).all()
            assert [index for index, _, _ in writes] == expected_indices
            # Shared slots retain the last write; depth-indexed slots retain every step.
            saved = {index: (k, v) for index, k, v in writes}
            valid = batch.out_loc != 0
            for index, (k, v) in saved.items():
                torch.testing.assert_close(
                    cache.k_cache(index).view(-1, 2, 64)[batch.out_loc[valid].long()], k[valid]
                )
                torch.testing.assert_close(
                    cache.v_cache(index).view(-1, 2, 64)[batch.out_loc[valid].long()], v[valid]
                )
            for buffer, previous in zip(buffers, before):
                torch.testing.assert_close(
                    buffer.view(-1, 2, 64)[: cached_len + 1],
                    previous.view(-1, 2, 64)[: cached_len + 1],
                )
            if step > 0:
                after = [buffer.clone() for buffer in buffers]
                for buffer, previous in zip(buffers, before):
                    buffer.copy_(previous)
                engine.attn_backend.prepare_metadata(batch)
                with engine.ctx.forward_batch(batch):
                    graphed = engine.graph_runner.replay(batch).clone()
                torch.testing.assert_close(graphed, eager, rtol=0.02, atol=0.02)
                assert torch.equal(graphed.argmax(-1), eager.argmax(-1))
                for buffer, expected in zip(buffers, after):
                    torch.testing.assert_close(buffer, expected, rtol=0.02, atol=0.02)
            for buffer in buffers:
                assert torch.count_nonzero(buffer[0]) == 0
            tokens.append(6 + step)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    _check_ouro_loop_cache(sys.argv[1], Path(sys.argv[2]))  # type: ignore[arg-type]
