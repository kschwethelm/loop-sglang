# Loop-SGLang

**Lightweight serving engine for looped LMs built on Mini-SGLang.**

> [!WARNING]
> **Work in progress.**
> Full-depth continuous batching for Ouro and Huginn is implemented.
> Depth-adaptive inference and continuous depth batching remain under development.

## Project Scope

Looped language models reuse a recurrent core across multiple loop steps.
Depth-adaptive inference lets each token exit that core after a different number of steps, allocating compute according to the token's needs.
Continuous depth batching (CDB) forms new batches between loop steps, removing exited tokens and optionally refilling freed slots with new tokens to keep GPU execution efficient.

Loop-SGLang aims to bring this scheduling approach into a lightweight serving engine built on Mini-SGLang.
The next steps include scheduling between loop steps and early exit for depth-adaptive looped models.

For the research implementation and experimental results, see [Continuous Depth Batching for Looped Language Models](https://github.com/LoopedLMs/looped-lm-continuous-batching) and the paper, [Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching](https://arxiv.org/abs/2608.09444).

## Progress

- Full-depth CB support for [Ouro](https://huggingface.co/KristianS7/Ouro-1.4B) and [Huginn](https://huggingface.co/KristianS7/huginn-0125).
  Use our linked model forks for compatibility with Loop-SGLang.
- Depth-indexed and shared looped LM KV caching (note: shared caching changes model behavior)
- Generation-based accuracy evaluation with lm-eval

## Attribution

This independent codebase is based on [Mini-SGLang](https://github.com/sgl-project/mini-sglang), a compact implementation of [SGLang](https://github.com/sgl-project/sglang).
The original MIT license and copyright notice are retained in [LICENSE](LICENSE).
The baseline documentation and benchmark results below originate from Mini-SGLang and do not demonstrate CDB performance.

## ✨ Inherited Baseline Features

- **High Performance**: Achieves state-of-the-art throughput and latency with advanced optimizations.
- **Lightweight & Readable**: A clean, modular, and fully type-annotated codebase that is easy to understand and modify.
- **Advanced Optimizations**:
  - **Radix Cache**: Reuses KV cache for shared prefixes across requests.
  - **Chunked Prefill**: Reduces peak memory usage for long-context serving.
  - **Overlap Scheduling**: Hides CPU scheduling overhead with GPU computation.
  - **Tensor Parallelism**: Scales inference across multiple GPUs.
  - **Optimized Kernels**: Integrates **FlashAttention** and **FlashInfer** for maximum efficiency.
  - ...

## 🚀 Quick Start

> **⚠️ Platform Support**: Mini-SGLang currently supports **Linux only** (x86_64 and aarch64). Windows and macOS are not supported due to dependencies on Linux-specific CUDA kernels (`sgl-kernel`, `flashinfer`). We recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows or Docker for cross-platform compatibility.

### 1. Environment Setup

We recommend using `uv` for a fast and reliable installation (note that `uv` does not conflict with `conda`).

```bash
# Create a virtual environment (Python 3.10+ recommended)
uv venv --python=3.12
source .venv/bin/activate
```

**Prerequisites**: Mini-SGLang relies on CUDA kernels that are JIT-compiled. Ensure you have the **NVIDIA CUDA Toolkit** installed and that its version matches your driver's version. You can check your driver's CUDA capability with `nvidia-smi`.

### 2. Installation

Install Loop-SGLang directly from source.
The Python package is `loopsgl`, and the CLI runs with `python -m loopsgl`.

```bash
git clone https://github.com/kschwethelm/loop-sglang.git
cd loop-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 Installing on Windows (WSL2)</b></summary>

Since Mini-SGLang requires Linux-specific dependencies, Windows users should use WSL2:

1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Install CUDA on WSL2**:
   - Follow [NVIDIA's WSL2 CUDA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - Ensure your Windows GPU drivers support WSL2

3. **Install Loop-SGLang in WSL2**:
   ```bash
   # Inside WSL2 terminal
   git clone https://github.com/kschwethelm/loop-sglang.git
   cd loop-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **Access from Windows**: The server will be accessible at `http://localhost:8000` from Windows browsers and applications.

</details>

<details>
<summary><b>🐳 Running with Docker</b></summary>

**Prerequisites**:
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **Build the Docker image**:
   ```bash
   docker build -t loopsgl .
   ```

2. **Run the server**:
   ```bash
   docker run --gpus all -p 1919:1919 \
       loopsgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **Run in interactive shell mode**:
   ```bash
   docker run -it --gpus all \
       loopsgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **Using Docker Volumes for persistent caches** (recommended for faster subsequent startups):
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       loopsgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. Online Serving

Launch an OpenAI-compatible API server with a single command.

```bash
# Deploy Ouro or Huginn with the default depth-indexed loop cache
python -m loopsgl --model "KristianS7/Ouro-1.4B"
python -m loopsgl --model "KristianS7/huginn-0125"

# Deploy meta-llama/Llama-3.1-70B-Instruct on 4 GPUs with Tensor Parallelism, on port 30000
python -m loopsgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

Once the server is running, you can send requests using standard tools like `curl` or any OpenAI-compatible client.

### 4. Interactive Shell

Chat with your model directly in the terminal by adding the `--shell` flag.

```bash
python -m loopsgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

You can also use `/reset` to clear the chat history.

### 5. Accuracy Evaluation

Install the optional `eval` dependencies and run generation tasks such as GSM8K through the offline engine.
Log-likelihood and perplexity evaluation are not yet supported.

```bash
uv run --extra eval python -m loopsgl.evaluation run --model loopsgl \
    --model_args pretrained=KristianS7/Ouro-1.4B,loop_cache_policy=depth_indexed,max_gen_toks=256 \
    --tasks gsm8k_cot --num_fewshot 3 --batch_size auto
```

## Upstream Baseline Benchmarks

These benchmark results are inherited from Mini-SGLang.
They have not been reproduced for Loop-SGLang and do not evaluate depth-adaptive looped models.

### Offline inference

See [bench.py](./benchmark/offline/bench.py) for more details.
Set `LOOPSGL_DISABLE_OVERLAP_SCHEDULING=1` for ablation study on overlap scheduling.

Test Configuration:

- Hardware: 1xH200 GPU.
- Model: Qwen3-0.6B, Qwen3-14B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100-1024 tokens
- Output Length: Randomly sampled between 100-1024 tokens

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### Online inference

See [benchmark_qwen.py](./benchmark/online/bench_qwen.py) for more details.

Test Configuration:

- Hardware: 4xH200 GPU, connected by NVLink.
- Model: Qwen3-32B
- Dataset: [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl), replaying first 1000 requests.

Launch command:

```bash
# Loop-SGLang
python -m loopsgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **Note**: If you encounter network issues when downloading models from HuggingFace, try using `--model-source modelscope` to download from ModelScope instead:
> ```bash
> python -m loopsgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 Learn More

- **[Detailed Features](./docs/features.md)**: Explore all available features and command-line arguments.
- **[System Architecture](./docs/structures.md)**: Dive deep into the design and data flow of Mini-SGLang.

## Citation

If you use continuous depth batching in your work, please cite:

```bibtex
@misc{schwethelm2026cdb,
      title={Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching},
      author={Kristian Schwethelm and Daniel Rueckert and Georgios Kaissis},
      year={2026},
      eprint={2608.09444},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.09444},
}
```
