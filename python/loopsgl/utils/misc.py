from __future__ import annotations

import resource

from .logger import init_logger

logger = init_logger(__name__)


def set_ulimit(target_soft_limit: int = 65535) -> None:
    """Raise the open-file soft limit to support concurrent serving connections."""
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit != resource.RLIM_INFINITY and soft_limit < target_soft_limit:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft_limit, hard_limit))
        except (OSError, ValueError) as error:
            logger.warning(f"Failed to set RLIMIT_NOFILE: {error}")


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


class Unset:
    pass


UNSET = Unset()
