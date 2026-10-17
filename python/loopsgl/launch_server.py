"""Launch the Loop-SGLang inference server from command-line arguments."""

from __future__ import annotations

import sys

from loopsgl.server_args import ServerArgs, parse_args


def run_server(server_args: ServerArgs, *, run_shell: bool = False) -> None:
    """Run the HTTP server or interactive shell with resolved server arguments."""
    from loopsgl.entrypoints.http_server import launch_server

    launch_server(server_args, run_shell=run_shell)


def main(run_shell: bool = False) -> None:
    """Parse command-line arguments and launch the selected frontend."""
    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    run_server(server_args, run_shell=run_shell)


if __name__ == "__main__":
    main()
