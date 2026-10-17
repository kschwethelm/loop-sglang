"""Dispatch Loop-SGLang command-line subcommands."""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse the subcommand and forward its arguments to the selected entrypoint."""
    parser = argparse.ArgumentParser(prog="loopsgl", description="Loop-SGLang command line.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser("serve", help="Launch an inference server.", add_help=False)
    args, extra_argv = parser.parse_known_args()
    if args.subcommand == "serve":
        from loopsgl.cli.serve import serve

        serve(args, extra_argv)
