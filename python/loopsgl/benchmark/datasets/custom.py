# Adapted from SGLang v0.5.20: python/sglang/benchmark/datasets/custom.py.
# Licensed under the Apache License, Version 2.0. See LICENSES/sglang.txt.

import json
import os
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from loopsgl.benchmark.datasets.common import (
    ASSISTANT_SUFFIX,
    BaseDataset,
    DatasetRow,
)
from loopsgl.benchmark.utils import remove_suffix


@dataclass
class CustomDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    context_len: Optional[int]
    prompt_suffix: str
    apply_chat_template: bool
    shuffle: bool = True
    input_format: str = "text"

    @classmethod
    def from_args(cls, args: Namespace) -> "CustomDataset":
        assert not getattr(args, "tokenize_prompt", False)
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
            shuffle=getattr(args, "shuffle", True),
            input_format=getattr(args, "custom_input_format", "text"),
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id: Optional[str] = None
    ) -> List[DatasetRow]:
        return sample_custom_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            context_len=self.context_len,
            prompt_suffix=self.prompt_suffix,
            apply_chat_template=self.apply_chat_template,
            shuffle=self.shuffle,
            input_format=self.input_format,
        )


def _load_token_id_row(data: Dict[str, Any], fixed_output_len: Optional[int]) -> DatasetRow:
    """Read stored prompt IDs and a generation budget without tokenizing text."""
    for key in ("input_ids", "output_ids"):
        if key == "output_ids" and key not in data:
            continue
        token_ids = data.get(key)
        if not isinstance(token_ids, list) or any(
            type(token_id) is not int or token_id < 0 for token_id in token_ids
        ):
            raise ValueError(f"{key} must be a list of nonnegative integer token IDs.")

    prompt_len = len(data["input_ids"])
    if "input_len" in data and (
        type(data["input_len"]) is not int or data["input_len"] != prompt_len
    ):
        raise ValueError("input_len must match the number of input_ids.")

    stored_output_len = data.get("output_len")
    output_ids = data.get("output_ids")
    if "output_len" in data and (type(stored_output_len) is not int or stored_output_len <= 0):
        raise ValueError("output_len must be a positive integer.")
    if output_ids is not None and stored_output_len is not None:
        if stored_output_len != len(output_ids):
            raise ValueError("output_len must match the number of output_ids.")

    if fixed_output_len is not None:
        output_len = fixed_output_len
    elif stored_output_len is not None:
        output_len = stored_output_len
    elif output_ids is not None:
        output_len = len(output_ids)
    else:
        raise ValueError("Token ID records require output_len or output_ids.")

    exit_depths = data.get("exit_depths")
    if exit_depths is not None and (
        not isinstance(exit_depths, list)
        or len(exit_depths) != output_len - 1
        or any(type(depth) is not int or depth < 1 for depth in exit_depths)
    ):
        raise ValueError("exit_depths must contain output_len - 1 positive integer depths.")
    return DatasetRow(
        prompt=data["input_ids"],
        prompt_len=prompt_len,
        output_len=output_len,
        exit_depths=exit_depths,
    )


def sample_custom_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template: bool = False,
    shuffle: bool = True,
    input_format: str = "text",
) -> List[DatasetRow]:
    """Sample text or stored token IDs from a custom JSONL dataset.

    Text records use the first two turns of 'conversations' or 'conversation'.
    Token ID records use 'input_ids' and 'output_len' or the length of 'output_ids'.
    Optional stored lengths must match their corresponding token ID arrays.
    Reference output IDs determine the generation budget and are never model inputs.
    """
    if input_format not in {"text", "token-ids"}:
        raise ValueError(f"Unknown custom input format: {input_format}")
    if input_format == "token-ids" and (prompt_suffix or apply_chat_template):
        raise ValueError("Token ID input cannot use prompt_suffix or apply_chat_template.")
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset
    dataset = []
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if line:  # skip empty lines
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    if input_format == "token-ids":
                        raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
                    continue  # skip lines with JSON errors

    if input_format == "text":
        # Filter out the conversations with less than 2 turns.
        processed_dataset = []
        for data in dataset:
            convs = data.get("conversations", data.get("conversation", []))
            if len(convs) >= 2:
                user_turn = convs[0].get("content", convs[0].get("value", ""))
                assist_turn = convs[1].get("content", convs[1].get("value", ""))
                processed_dataset.append((user_turn, assist_turn))
        dataset = processed_dataset
    if shuffle:
        random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[DatasetRow] = []

    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        if input_format == "token-ids":
            if not isinstance(dataset[i], dict):
                raise ValueError("Token ID records must be JSON objects.")
            try:
                row = _load_token_id_row(dataset[i], fixed_output_len)
            except ValueError as exc:
                record_id = dataset[i].get("id", i)
                raise ValueError(f"Invalid token ID record {record_id!r}: {exc}") from exc
        else:
            # Tokenize the prompts and completions.
            prompt = dataset[i][0]

            if prompt_suffix:
                prompt = remove_suffix(prompt, ASSISTANT_SUFFIX) + prompt_suffix + ASSISTANT_SUFFIX

            if apply_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    return_dict=False,
                )
                if tokenizer.bos_token:
                    prompt = prompt.replace(tokenizer.bos_token, "")

            prompt_token_ids = tokenizer.encode(prompt)
            completion = dataset[i][1]
            completion_token_ids = tokenizer.encode(completion)
            row = DatasetRow(
                prompt=prompt,
                prompt_len=len(prompt_token_ids),
                output_len=(
                    len(completion_token_ids) if fixed_output_len is None else fixed_output_len
                ),
            )

        if row.prompt_len < 2 or row.output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and row.prompt_len + row.output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append(row)

    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset
