"""Public Python interface for Loop-SGLang inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loopsgl.entrypoints.engine import Engine

__all__ = ["Engine"]


def __getattr__(name: str) -> type[Engine]:
    """Load the public engine on demand without initializing runtime imports for submodules."""
    if name != "Engine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from loopsgl.entrypoints.engine import Engine

    globals()[name] = Engine
    return Engine
