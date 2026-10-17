# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGLang-style segmented CUDA graphs for single-stream model execution.

Tensor arguments alias graph-pool storage without retaining every intermediate.
Eager breaks write their results into caller-owned tensors allocated in a segment.
The segment graphs keep the pool alive until all replay closures are released.
"""

from __future__ import annotations

import functools
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, TypeVar

import torch

_CAPTURE: ContextVar[BreakableCUDAGraphCapture | None] = ContextVar(
    "breakable_capture", default=None
)
_WARMUP: ContextVar[bool] = ContextVar("prefill_graph_warmup", default=False)
F = TypeVar("F", bound=Callable[..., None])


def is_prefill_graph_warmup() -> bool:
    """Return whether prefill is preparing graphs before KV allocation."""
    return _WARMUP.get()


@contextmanager
def prefill_graph_warmup() -> Iterator[None]:
    """Use attention and initialization stubs while preparing prefill graphs."""
    token = _WARMUP.set(True)
    try:
        yield
    finally:
        _WARMUP.reset(token)


def _weak_ref_if_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or value.device.type == "cpu":
            return value
        from sgl_kernel import weak_ref_tensor

        return weak_ref_tensor(value)
    if isinstance(value, tuple):
        return tuple(_weak_ref_if_tensor(item) for item in value)
    if isinstance(value, list):
        return [_weak_ref_if_tensor(item) for item in value]
    return value


def eager_on_graph(enable: bool = True) -> Callable[[F], F]:
    """Run an in-place operation eagerly between captured graph segments."""

    def decorate(inner: F) -> F:
        if not enable:
            return inner

        @functools.wraps(inner)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            capture = _CAPTURE.get()
            if capture is None:
                return inner(*args, **kwargs)
            capture.end_segment()
            if capture.barrier_fn is not None:
                capture.barrier_fn()
            result = inner(*args, **kwargs)
            if result is not None:
                raise TypeError("Eager graph breaks must write their outputs in place")
            capture.graph.break_fns.append(
                functools.partial(
                    inner,
                    *tuple(_weak_ref_if_tensor(arg) for arg in args),
                    **{key: _weak_ref_if_tensor(value) for key, value in kwargs.items()},
                )
            )
            capture.begin_segment()

        return wrapper  # type: ignore[return-value]

    return decorate


class BreakableCUDAGraph:
    """CUDA graph segments separated by recorded eager calls."""

    def __init__(self) -> None:
        self.segments: list[torch.cuda.CUDAGraph] = []
        self.break_fns: list[Callable[[], None]] = []

    def replay(self) -> None:
        for index, segment in enumerate(self.segments):
            segment.replay()
            if index < len(self.break_fns):
                self.break_fns[index]()

    def reset(self) -> None:
        self.break_fns.clear()
        for segment in self.segments:
            segment.reset()
        self.segments.clear()


class BreakableCUDAGraphCapture:
    """Capture segments sharing a pool on the caller's current CUDA stream."""

    def __init__(
        self,
        graph: BreakableCUDAGraph,
        *,
        pool: Any,
        barrier_fn: Callable[[], None] | None = None,
    ) -> None:
        self.graph = graph
        self.pool = pool
        self.barrier_fn = barrier_fn
        self._segment: torch.cuda.CUDAGraph | None = None
        self._token: Any = None

    def __enter__(self) -> BreakableCUDAGraphCapture:
        if _CAPTURE.get() is not None:
            raise RuntimeError("Breakable CUDA graph captures cannot nest")
        self._token = _CAPTURE.set(self)
        try:
            self.begin_segment()
        except BaseException:
            _CAPTURE.reset(self._token)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if self._segment is not None:
                self.end_segment()
        finally:
            _CAPTURE.reset(self._token)

    def begin_segment(self) -> None:
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
        self._segment = graph

    def end_segment(self) -> None:
        if self._segment is None:
            raise RuntimeError("No CUDA graph segment is being captured")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The CUDA Graph is empty")
            self._segment.capture_end()
        self.graph.segments.append(self._segment)
        self._segment = None
