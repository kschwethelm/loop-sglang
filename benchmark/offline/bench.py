# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import argparse
import time
from random import randint, seed

from loopsgl.core import SamplingParams
from loopsgl.llm import LLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark offline generation.")
    parser.add_argument("--model", default="KristianS7/Ouro-1.4B", help="Model ID or local path.")
    parser.add_argument(
        "--loop-cache-policy", choices=("depth_indexed", "shared"), default="depth_indexed"
    )
    args = parser.parse_args()
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # align the hyperparameters
    llm = LLM(
        args.model,
        loop_cache_policy=args.loop_cache_policy,
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        cuda_graph_max_bs=256,
        page_size=256,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
