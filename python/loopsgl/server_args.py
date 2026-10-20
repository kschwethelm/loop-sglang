from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, fields
from typing import Any, List, Tuple

from loopsgl.core import LoopParams
from loopsgl.distributed import DistributedInfo
from loopsgl.managers.scheduler_components.config import CDBConfig, SchedulerConfig
from loopsgl.utils import (
    cached_load_hf_config,
    get_and_verify_dtype,
    init_logger,
    parse_model_override_args,
)


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    silent_output: bool = False
    default_chat_template_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.default_chat_template_kwargs is not None and not isinstance(
            self.default_chat_template_kwargs, dict
        ):
            raise ValueError("--default-chat-template-kwargs must decode to a JSON object")

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add server and engine options to a shared argument parser."""
        _add_cli_args(parser)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> ServerArgs:
        """Resolve server and engine options from a parsed namespace."""
        return _from_cli_args(args)[0]


def _add_cli_args(parser: argparse.ArgumentParser) -> None:
    from loopsgl.attention import validate_attn_backend
    from loopsgl.moe import SUPPORTED_MOE_BACKENDS

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--json-model-override-args",
        type=str,
        default=ServerArgs.json_model_override_args,
        help="A dictionary in JSON string format used to override default model configurations.",
    )

    parser.add_argument(
        "--default-chat-template-kwargs",
        type=json.loads,
        default=ServerArgs.default_chat_template_kwargs,
        help="Default chat template kwargs as a JSON object (e.g. enable_thinking). "
        "Per-request chat_template_kwargs takes precedence.",
    )

    # --- Looped model specific arguments ---
    parser.add_argument(
        "--scheduler",
        choices=["cb", "cdb"],
        default=ServerArgs.scheduler,
        help="Batching policy. Defaults to CDB when --exit-threshold is supplied, otherwise CB. "
        "CDB allows depth-adaptive inference for looped models.",
    )

    parser.add_argument(
        "--loop-steps",
        type=int,
        default=LoopParams.loop_steps,
        help="Number of recurrent core steps. Defaults to the model's configured depth.",
    )

    parser.add_argument(
        "--loop-cache-policy",
        choices=["depth_indexed", "shared", "last_exited"],
        default=ServerArgs.loop_cache_policy,
        help="KV cache policy for recurrent core layers of looped models. "
        "Defaults to depth_indexed for CB and last_exited for CDB. "
        "Prelude and coda layers keep one KV cache slot each. "
        "Depth-indexed gives each loop step its own KV cache slot (only for full-depth CB). "
        "Shared-cache gives each loop step only one KV cache slot that gets overwritten by each loop step (for CDB and CB). "
        "Last-exited gives each loop step its own KV cache slot, skipped depths from early-exits free memory and later steps attend to the last exited loop cache (only for CDB).",
    )

    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=LoopParams.exit_threshold,
        help="Exit threshold for depth-adaptive looped models (CDB only). Disabled unless explicitly supplied.",
    )

    # CDB scheduler arguments.
    parser.add_argument(
        "--disable-lookahead-exit",
        action="store_true",
        default=CDBConfig.disable_lookahead_exit,
        help="Debugging: disable one-step exit lookahead. This is substantially slower because "
        "the GPU waits for CPU gate evaluation after every loop step.",
    )

    parser.add_argument(
        "--refill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refill mode of the CDB paper. Keeps recurrent core batch size large and is usually faster than no-refill.",
    )

    parser.add_argument(
        "--min-coda-batch-size",
        type=int,
        default=CDBConfig.min_coda_batch_size,
        help="Minimum CDB-refill coda batch before executing the coda.",
    )
    # --- End of looped model specific arguments ---

    parser.add_argument(
        "--dtype",
        type=str.lower,
        default="auto",
        choices=["auto", "half", "float16", "float", "float32", "bfloat16"],
        help="Data type for model weights and activations. 'auto' uses the checkpoint dtype, "
        "converting FP32 or missing dtype metadata to FP16. "
        "'half' and 'float' are aliases for FP16 and FP32.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--min-free-slots-delay",
        type=int,
        default=ServerArgs.min_free_slots_delay,
        help="Hold new prefills until at least N request slots are available.",
    )

    parser.add_argument(
        "--watermark",
        type=float,
        default=ServerArgs.watermark,
        help="Fraction of physical KV cache capacity to keep available when admitting new requests.",
    )

    parser.add_argument(
        "--retraction-mode",
        choices=["none", "recompute", "offload"],
        default=ServerArgs.retraction_mode,
        help="Handle KV pressure by CPU offloading, recomputation, or failing with none. "
        "For CDB note: Recompute retraction can compute a different KV cache entry for shared KV cache policies.",
    )

    parser.add_argument(
        "--retraction-cpu-memory-gb",
        type=float,
        default=ServerArgs.retraction_cpu_memory_gb,
        help="Maximum CPU KV backup memory per worker in GiB.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="Fraction of initially free GPU memory to budget for model weights, runtime buffers, "
        "and KV cache. Remaining initialization allocations and decode graphs need extra headroom.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--nccl-port",
        type=int,
        default=ServerArgs.nccl_port,
        help="The port for distributed initialization. Defaults to an available localhost port.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--cuda-graph-backend-prefill",
        choices=["breakable", "disabled"],
        default=ServerArgs.cuda_graph_backend_prefill,
        help="Prefill graph backend. Breakable graphs keep attention and logits processing eager.",
    )
    parser.add_argument(
        "--cuda-graph-bs-prefill",
        nargs="+",
        type=int,
        default=ServerArgs.cuda_graph_bs_prefill,
        help="Total query-token buckets to capture for prefill.",
    )
    parser.add_argument(
        "--cuda-graph-max-bs-prefill",
        type=int,
        default=ServerArgs.cuda_graph_max_bs_prefill,
        help="Maximum prefill graph query-token count. Defaults to the prefill token budget.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=ServerArgs.max_total_tokens,
        help="KV cache token capacity. For last-exited KV, capacity assumes full depth; "
        "early exits allow more tokens to fit. Defaults to automatic memory-based sizing.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--disable-radix-cache",
        action="store_true",
        default=ServerArgs.disable_radix_cache,
        help="Disable prefix caching.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )


def _from_cli_args(args: argparse.Namespace, run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    attrs = {attr.name for attr in fields(ServerArgs)}
    attrs.update({"shell_mode", "model_source", "tensor_parallel_size"})
    kwargs = {key: value for key, value in vars(args).items() if key in attrs}
    kwargs["loop_params"] = LoopParams(
        loop_steps=args.loop_steps, exit_threshold=args.exit_threshold
    )
    kwargs["cdb_config"] = CDBConfig(
        refill=args.refill,
        min_coda_batch_size=args.min_coda_batch_size,
        disable_lookahead_exit=args.disable_lookahead_exit,
    )

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    kwargs["dtype"] = get_and_verify_dtype(
        cached_load_hf_config(
            kwargs["model_path"], parse_model_override_args(kwargs["json_model_override_args"])
        ),
        kwargs["dtype"],
    )
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell


def parse_args(
    args: List[str], run_shell: bool = False, *, prog: str | None = None
) -> Tuple[ServerArgs, bool]:
    """Parse server options and resolve interactive shell settings."""
    parser = argparse.ArgumentParser(prog=prog, description="LoopSGL Server Arguments")
    ServerArgs.add_cli_args(parser)
    return _from_cli_args(parser.parse_args(args), run_shell=run_shell)
