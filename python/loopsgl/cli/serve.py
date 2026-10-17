"""Launch inference serving through the shared server argument parser."""

from __future__ import annotations

import argparse

from loopsgl.launch_server import run_server
from loopsgl.server_args import parse_args


def serve(args: argparse.Namespace, extra_argv: list[str]) -> None:
    """Resolve serving arguments and run the HTTP or interactive shell frontend."""
    server_args, run_shell = parse_args(extra_argv, prog="loopsgl serve")
    run_server(server_args, run_shell=run_shell)
