from __future__ import annotations

import socket


def get_free_port() -> int:
    """Return an available localhost TCP port without reserving it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
