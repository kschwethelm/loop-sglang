# Features of Loop-SGLang

## Online Serving

Loop-SGLang supports online serving with an OpenAI-compatible API server.
It provides the standard `/v1/chat/completions` endpoint, allowing seamless integration with existing tools and clients.
For detailed command-line arguments and configuration options, run `loopsgl serve --help`.

## Interactive Shell Mode

For demonstration and testing purposes, an interactive shell mode is available.
In this mode, users can input prompts directly, and the LLM will generate responses in real-time.
The shell automatically caches chat history to maintain context.
To clear the conversation history and start a new session, use the `/reset` command.

Example:

```bash
loopsgl serve --model "KristianS7/Ouro-1.4B" --shell
```

## Distributed Serving

To scale performance across multiple GPUs, Loop-SGLang supports Tensor Parallelism (TP).
You can enable distributed serving by specifying the number of GPUs with the `--tp n` argument, where `n` is the degree of parallelism.

## Supported Models

Our framework currently supports the following looped model architectures:
- [`Ouro`](https://huggingface.co/KristianS7/Ouro-1.4B)
- [`Huginn`](https://huggingface.co/KristianS7/huginn-0125)

And the following standard model architectures:

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) series
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) series (including MoE)
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) series

## Chunked Prefill

Chunked Prefill, a technique introduced by [Sarathi-Serve](https://arxiv.org/abs/2403.02310), is enabled by default.
This feature splits long prompts into smaller chunks during the prefill phase, significantly reducing peak memory usage and preventing Out-Of-Memory (OOM) errors in long-context serving.
The chunk size can be configured using `--max-prefill-length n`.
Note that setting `n` to a very small value (e.g., 128) is not recommended as it may significantly degrade performance.

## Page Size

You can specify the page size of the system using the `--page-size` argument.

## Attention Backends

Loop-SGLang integrates high-performance attention kernels, including [`FlashAttention`](https://github.com/Dao-AILab/flash-attention) (`fa`), [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer) (`fi`) and [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM) (`trtllm`).
It supports using different backends for the prefill and decode phases to maximize efficiency.
The default `auto` selection checks GPU architecture, model dimensions, dtype, and backend availability.
On NVIDIA Hopper GPUs, it selects `FlashAttention 3` for both prefill and decode when compatible.

You can specify the backend using the `--attn` argument.
If two values are provided (e.g., `--attn fa,fi`), the first specifies the prefill backend and the second the decode backend.
Automatic selection may adjust the page size for compatibility; `trtllm` supports page sizes 16, 32, and 64, while FA4 requires 128.

## CUDA Graph

To minimize CPU launch overhead during decoding, Loop-SGLang supports capturing and replaying CUDA graphs.
This feature is enabled by default.
The maximum batch size for CUDA graph capture can be set with `--cuda-graph-max-bs n`.
Setting `n` to `0` disables this feature.

Prefill uses SGLang-style breakable CUDA graphs by default.
The transformer body replays in total query-token buckets, while attention and final-row logits processing use the current batch's real lengths.
Padded rows never reach the KV cache.

Use `--cuda-graph-backend-prefill disabled` to run prefill eagerly.
Use `--cuda-graph-max-bs-prefill n` to cap captured query-token counts or `--cuda-graph-bs-prefill 128 256 512` to select explicit buckets.
The default cap is the prefill token budget.
Batches above the cap, or requiring more than twice their real token count in padding, run eagerly.

## Radix Cache

Adopting the original design from [SGLang](https://github.com/sgl-project/sglang.git), Loop-SGLang implements a Radix Cache to manage the Key-Value (KV) cache.
This allows the reuse of KV cache for shared prefixes across requests, reducing redundant computation.
This feature is enabled by default and can be disabled with `--disable-radix-cache`.

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Illustration of Radix Attention from [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/).*

## Overlap Scheduling

To further reduce CPU overhead, Loop-SGLang employs overlap scheduling, a technique proposed in [NanoFlow](https://arxiv.org/abs/2408.12757).
This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*Illustration of Overlap Scheduling from [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).*

## Retraction (Preemption)

When we run out of memory (i.e., cannot allocate KV cache for another token), we retract running sequences through CPU offloading or recomputation.
We first process outstanding outputs so finished sequences can free their memory.
If we still need more memory, we retract sequences until all remaining sequences can generate their next token.
We preferentially retract sequences with shorter outputs and longer inputs.
Retracted sequences resume later, either by restoring their KV cache from CPU memory or by recomputing it from their token history.
