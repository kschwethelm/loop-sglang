from __future__ import annotations

import functools
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

from loopsgl.env import ENV

if TYPE_CHECKING:
    import torch

_NULL_CONTEXT = nullcontext()


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def profile_range(debug_name: str | None) -> AbstractContextManager[None]:
    """Return a named NVTX range when enabled, otherwise a shared no-op context."""
    if debug_name is None or not ENV.ENABLE_NVTX:
        return _NULL_CONTEXT
    from torch.cuda.nvtx import range

    return range(debug_name)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
