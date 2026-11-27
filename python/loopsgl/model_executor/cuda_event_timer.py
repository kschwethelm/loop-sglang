from __future__ import annotations

from collections import defaultdict, deque
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Iterator

import torch


class CudaEventTimer:
    """Sum elapsed time for execution phases on one CUDA stream.

    Events mark each phase's start and end without waiting for the GPU.
    Completed event pairs are read and reused during execution to reduce event allocations.
    Intervals include gaps between launches and delays caused by work on other streams.
    Work on other streams is not timed directly.
    """

    def __init__(self, stream: torch.cuda.Stream, *, enabled: bool) -> None:
        self.stream = stream
        self.enabled = enabled
        self._free = [self._new_pair() for _ in range(64)] if enabled else []
        self._pending: deque[tuple[str, torch.cuda.Event, torch.cuda.Event]] = deque()
        self._seconds: dict[str, float] = defaultdict(float)
        self._noop = nullcontext()

    @staticmethod
    def _new_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def measure(self, phase: str) -> AbstractContextManager[None]:
        """Return a context manager that times the phase when timing is enabled."""
        return self._measure(phase) if self.enabled else self._noop

    @contextmanager
    def _measure(self, phase: str) -> Iterator[None]:
        if not self._free:
            self._collect()
        start, end = self._free.pop() if self._free else self._new_pair()
        start.record(self.stream)
        try:
            yield
        finally:
            end.record(self.stream)
            self._pending.append((phase, start, end))

    def _collect(self) -> None:
        """Read completed timings and recycle their events without waiting for pending phases."""
        while self._pending and self._pending[0][2].query():
            phase, start, end = self._pending.popleft()
            self._seconds[phase] += start.elapsed_time(end) / 1000
            self._free.append((start, end))

    def snapshot(self, phases: tuple[str, ...]) -> dict[str, float]:
        """Return phase totals in seconds without clearing them.

        The caller must wait for all recorded phases to finish first.
        The decode_s total sums all requested phases except prefill.
        """
        self._collect()
        if self._pending:
            raise RuntimeError("CUDA event timing requires a drained compute stream.")
        result = {f"{phase}_s": self._seconds.get(phase, 0.0) for phase in phases}
        result["decode_s"] = sum(
            self._seconds.get(phase, 0.0) for phase in phases if phase != "prefill"
        )
        return result

    def reset(self) -> None:
        """Clear accumulated times after all recorded phases have finished.

        Keep allocated event pairs for reuse in the next measurement.
        """
        self.snapshot(())
        self._seconds.clear()
