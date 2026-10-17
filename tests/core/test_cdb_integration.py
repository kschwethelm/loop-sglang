from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
import torch
from loopsgl.core import Req, SamplingParams
from loopsgl.managers.cdb_scheduler import CDBDecodeToken, CDBScheduler
from loopsgl.managers.schedule_batch import ScheduleBatch

_GPU_REQUIRED = pytest.mark.skipif(
    not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available(),
    reason="Scheduler integration checks require CUDA in a SLURM allocation",
)


def _run_worker(checkpoint: Path, mode: str, policy: str, tp_size: int) -> None:
    """Bound GPU and distributed execution in fresh subprocesses."""
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(checkpoint),
            mode,
            policy,
            str(tp_size),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=240)
    finally:
        # The TP ranks inherit this process group, including when its launcher exits first.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    assert process.returncode == 0, output


@_GPU_REQUIRED
@pytest.mark.parametrize(
    ("model", "policy"), [("ouro", "shared"), ("ouro", "last_exited"), ("huginn", "shared")]
)
def test_cdb_numerical_correctness(
    model: str, policy: str, tiny_checkpoint: Callable[[str], Path]
) -> None:
    """Mixed-depth refill and padded graph replay preserve isolated eager logits and logical KV."""
    _run_worker(tiny_checkpoint(model), "numerical", policy, 1)


@_GPU_REQUIRED
@pytest.mark.parametrize(("policy", "tp_size"), [("shared", 1), ("last_exited", 2)])
def test_retraction_preserves_state(
    policy: str, tp_size: int, tiny_checkpoint: Callable[[str], Path]
) -> None:
    """Real allocation pressure preserves exact offloaded KV and subsequent computation at the same TP size."""
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} allocated GPUs")
    _run_worker(tiny_checkpoint("ouro"), "pressure", policy, tp_size)


class _ObservedScheduler(CDBScheduler):
    """Observe production execution while supplying a fixed token history for comparisons."""

    def _run_prefill(self, batch: ScheduleBatch) -> None:
        self.sampling_reqs = tuple(batch.reqs)
        super()._run_prefill(batch)

    def _run_coda(self, tokens: Sequence[CDBDecodeToken]) -> None:
        self.sampling_reqs = tuple(token.req for token in tokens)
        self.exits.extend((token.req.uid, token.position, token.depth) for token in tokens)
        super()._run_coda(tokens)

    def _free_req_resources(self, req: Req) -> None:
        if req.finished_reason == "eos" and self.draining_for_memory:
            self.eos_during_drain += 1
        self.model_runner.stream.synchronize()
        indices = self.cache_manager.page_table[req.kv.req_pool_idx, : req.kv.kv_committed_len]
        self.snapshots[req.uid] = self.model_runner.kv_cache.get_cpu_copy(indices)
        super()._free_req_resources(req)


def _assert_kv_close(actual: torch.Tensor, expected: torch.Tensor, token_count: int) -> None:
    """Compare logical KV values and retain exact last-exited depth metadata."""
    if actual.dtype == torch.uint8:
        torch.testing.assert_close(
            actual[-token_count * 4 :].view(torch.int32),
            expected[-token_count * 4 :].view(torch.int32),
            rtol=0,
            atol=0,
        )
        actual, expected = actual[: -token_count * 4].view(torch.bfloat16), expected[
            : -token_count * 4
        ].view(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


@torch.inference_mode()
def _gpu_worker(
    rank: int, checkpoint: str, mode: str, policy: str, tp_size: int, port: int
) -> None:
    """Run isolated references and production batched scheduling with controlled model histories."""
    from loopsgl.core import LoopParams
    from loopsgl.distributed import DistributedInfo
    from loopsgl.managers.io_struct import TokenizedGenerateReqInput
    from loopsgl.managers.scheduler_components.config import CDBConfig, SchedulerConfig
    from loopsgl.model_executor.model_runner import ModelRunner

    assert os.environ.get("SLURM_JOB_ID"), "GPU execution requires SLURM"
    original_load = ModelRunner._load_weight_state_dict

    def stable_weights(runner: ModelRunner, config: Any) -> dict[str, torch.Tensor]:
        weights = original_load(runner, config)
        for name, value in weights.items():
            if "norm" in name and name.endswith("weight"):
                value.fill_(1)
            elif "early_exit_gate" in name:
                value.zero_()
            else:
                value.mul_(0.02)
        return weights

    ModelRunner._load_weight_state_dict = stable_weights
    pages = (30 if policy == "shared" else 120) if mode == "pressure" else 512
    config = SchedulerConfig(
        model_path=checkpoint,
        tp_info=DistributedInfo(rank, tp_size),
        dtype=torch.bfloat16,
        scheduler="cdb",
        loop_cache_policy=policy,
        loop_params=LoopParams(loop_steps=4, exit_threshold=0.5),
        cdb_config=CDBConfig(refill=True, min_coda_batch_size=1),
        max_running_req=4,
        max_extend_tokens=32,
        max_seq_len_override=128,
        num_page_override=pages,
        attention_backend="fa",
        cuda_graph_bs=[1, 2, 4],
        cuda_graph_max_bs=4,
        cuda_graph_backend_prefill="disabled",
        use_dummy_weight=True,
        use_pynccl=False,
        nccl_port=port,
        disable_radix_cache=True,
        retraction_mode="offload",
    )
    scheduler = _ObservedScheduler(config)
    runner = scheduler.model_runner
    scheduler.logits, scheduler.snapshots, scheduler.exits = {}, {}, []
    scheduler.receive_msg = lambda blocking=False: []
    scheduler.send_result = lambda reply: None
    graph_enabled = runner.graph_runner.can_use_cuda_graph
    replay = runner.graph_runner.replay_phase
    sample = runner._sample_logits
    stats = {"graphs": 0, "padded": 0, "mixed_depths": 0, "restored": 0}
    replay_exits = True
    eos_early = False

    def observed_replay(phase: str, batch: Any, slots: torch.Tensor, depths: torch.Tensor) -> Any:
        stats["graphs"] += 1
        stats["padded"] += batch.padded_size > batch.size
        if phase == "core":
            stats["mixed_depths"] += len(set(depths.tolist())) > 1
        return replay(phase, batch, slots, depths)

    def prescribed_tokens(logits: torch.Tensor, args: Any) -> torch.Tensor:
        assert torch.isfinite(logits).all()
        result = sample(logits, args)
        for index, req in enumerate(scheduler.sampling_reqs):
            position = max(0, req.device_len - req.origin_input_len)
            scheduler.logits[req.uid, position] = logits[index].float().cpu().clone()
            result[index] = 2 if eos_early and req.uid == 0 and position == 2 else 8 + req.uid
        return result

    restore = scheduler.retraction_manager.retraction_restore

    def checked_restore(req: Req) -> None:
        backup = req.kv.retraction_backup
        expected, length = backup.cpu_tensors.clone(), backup.cached_len
        restore(req)
        runner.stream.synchronize()
        actual = runner.kv_cache.get_cpu_copy(
            scheduler.cache_manager.page_table[req.kv.req_pool_idx, :length]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        stats["restored"] += 1

    runner._sample_logits = prescribed_tokens
    runner.graph_runner.replay_phase = observed_replay
    scheduler.retraction_manager.retraction_restore = checked_restore
    prompts = [[1] + [3 + uid] * (7 if mode == "pressure" else uid + 1) for uid in range(3)]
    budgets = [8, 12, 20] if mode == "pressure" else [4, 6, 5]

    def run(uids: list[int], graphs: bool) -> tuple[dict, dict]:
        scheduler.reset_generation()
        scheduler.logits, scheduler.snapshots, scheduler.exits = {}, {}, []
        scheduler.eos_during_drain = 0
        runner.graph_runner.can_use_cuda_graph = graph_enabled if graphs else lambda batch: False
        for uid in uids:
            depths = [(2, 4, 3)[(index + uid) % 3] for index in range(budgets[uid] - 1)]
            scheduler._process_one_msg(
                TokenizedGenerateReqInput(
                    uid,
                    torch.tensor(prompts[uid], dtype=torch.int32),
                    SamplingParams(
                        max_tokens=budgets[uid],
                        ignore_eos=not eos_early,
                        replay_exit_depths=depths if replay_exits else None,
                    ),
                )
            )
        for _ in range(5000):
            scheduler._event_loop_iteration()
            if scheduler.is_fully_idle():
                break
        else:
            raise AssertionError("Scheduler failed to make progress")
        runner.stream.synchronize()
        for uid in uids:
            exits = [
                (position, depth) for owner, position, depth in scheduler.exits if owner == uid
            ]
            count = 3 if eos_early and uid == 0 else budgets[uid]
            if replay_exits:
                assert exits == [
                    (len(prompts[uid]) + index, (2, 4, 3)[(index + uid) % 3])
                    for index in range(count - 1)
                ]
            else:
                assert len(exits) == count - 1
                assert all(2 <= depth <= 4 for _, depth in exits)
            assert sum(owner == uid for owner, _ in scheduler.logits) == count
        scheduler.cache_manager.clear()
        assert scheduler.table_manager.allocated_size == 0
        assert (
            not scheduler.retraction_manager.runnable
            and scheduler.retraction_manager.cpu_bytes == 0
        )
        assert not any(runner._gate_leased)
        return scheduler.logits.copy(), scheduler.snapshots.copy()

    try:
        for replay_exits in (True, False):
            eos_early = mode == "pressure" and not replay_exits
            reference_logits, reference_kv, reference_exits = {}, {}, []
            for uid in range(3):
                logits, snapshots = run([uid], graphs=False)
                reference_logits.update(logits)
                reference_kv.update(snapshots)
                reference_exits.extend(scheduler.exits)
            actual_logits, actual_kv = run([0, 1, 2], graphs=True)
            assert actual_logits.keys() == reference_logits.keys()
            assert sorted(scheduler.exits) == sorted(reference_exits)
            for key, expected in reference_logits.items():
                torch.testing.assert_close(actual_logits[key], expected, rtol=0.02, atol=0.002)
            for uid, expected in reference_kv.items():
                count = 3 if eos_early and uid == 0 else budgets[uid]
                _assert_kv_close(actual_kv[uid], expected, len(prompts[uid]) + count - 1)
            if eos_early and policy == "shared":
                assert scheduler.eos_during_drain > 0
            if mode == "pressure":
                if replay_exits:
                    assert scheduler.num_retracted_reqs > 0 and stats["restored"] > 0
                elif policy == "last_exited":
                    assert scheduler.num_retracted_reqs == 0
                assert (
                    scheduler.retraction_manager.offloaded_kv_bytes
                    == scheduler.retraction_manager.restored_kv_bytes
                )
            else:
                assert scheduler.num_retracted_reqs == 0
        assert stats["graphs"] > 0 and stats["padded"] > 0 and stats["mixed_depths"] > 0
        print(f"PASS {mode} {policy} TP{tp_size} rank={rank}: {stats}", flush=True)
    finally:
        scheduler.shutdown()
        if rank == 0:
            for channel in (
                scheduler.recv_from_tokenizer,
                scheduler.send_to_tokenizer,
                scheduler.send_to_detokenizer,
            ):
                channel.stop()
            for addr in (config.zmq_backend_addr, config.zmq_detokenizer_addr):
                Path(addr.removeprefix("ipc://")).unlink(missing_ok=True)


if __name__ == "__main__":
    from loopsgl.utils import get_free_port

    checkpoint, mode, policy, tp = sys.argv[1:]
    size = int(tp)
    arguments = (checkpoint, mode, policy, size, get_free_port())
    if size == 1:
        _gpu_worker(0, *arguments)
    else:
        torch.multiprocessing.spawn(_gpu_worker, args=arguments, nprocs=size, join=True)
