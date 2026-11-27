"""Prepare fixed token sequences for forced-decoding paper workload recordings."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


@contextmanager
def output_file(destination: Path) -> Iterator[Path]:
    """Publish a complete JSONL file without replacing an existing output."""
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output file already exists: {destination}")
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as name:
        path = Path(name) / destination.name
        yield path
        destination.hardlink_to(path)


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON array of records.")
        yield from records
    else:
        raise ValueError(f"{path}: expected a JSON, JSONL, or Parquet file.")


def text_pair(record: dict[str, Any], dataset: str) -> tuple[str, str]:
    """Extract each dataset's paper prompt and reference continuation."""
    if dataset == "sharegpt":
        turns = record.get("conversations") or []
        if len(turns) < 2 or str(turns[0].get("from", "")).lower() not in {"human", "user"}:
            return "", ""
        if str(turns[1].get("from", "")).lower() not in {"gpt", "assistant"}:
            return "", ""
        return str(turns[0].get("value", "")).strip(), str(turns[1].get("value", "")).strip()
    if dataset == "alpaca":
        instruction = str(record.get("instruction", "")).strip()
        extra_input = str(record.get("input", "")).strip()
        prompt = f"{instruction}\n{extra_input}" if extra_input else instruction
        return prompt.strip(), str(record.get("output", "")).strip()
    article = str(record.get("article") or "").strip()
    prompt = f"Summarize the following scientific article in an abstract.\n\n{article}\n\nAbstract:"
    return prompt if article else "", str(record.get("abstract") or "").strip()


def prepare_workload(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.max_model_len < 8:
        raise ValueError("--max-model-len must allow at least four tokens per side.")
    tokenizer_names = list(dict.fromkeys([args.tokenizer, *args.filter_tokenizers]))
    with output_file(args.output_path) as path:
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in tokenizer_names]
        rows = []
        for ordinal, record in enumerate(read_records(args.data_path)):
            if not isinstance(record, dict):
                raise ValueError(f"Source row {ordinal}: expected an object.")
            prompt, reference = text_pair(record, args.dataset)
            if not prompt or not reference:
                continue
            encoded = [
                (
                    tokenizer.encode(prompt, add_special_tokens=True),
                    tokenizer.encode(reference, add_special_tokens=False),
                )
                for tokenizer in tokenizers
            ]
            if any(
                min(len(inputs), len(outputs)) < 4
                or len(inputs) + len(outputs) > args.max_model_len
                for inputs, outputs in encoded
            ):
                continue
            inputs, outputs = encoded[0]
            rows.append({"id": str(ordinal), "input_ids": inputs, "output_ids": outputs})
        if not rows or (args.limit is not None and len(rows) < args.limit):
            raise ValueError(
                f"Only {len(rows)} requests survived filtering. Requested {args.limit}."
            )
        random.Random(args.seed).shuffle(rows)
        if args.limit is not None:
            rows = rows[: args.limit]
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Tokenized {len(rows)} requests to {args.output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["sharegpt", "alpaca", "arxiv"])
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--filter-tokenizers",
        nargs="+",
        default=["KristianS7/Ouro-1.4B", "KristianS7/huginn-0125"],
        help="Keep only requests accepted by every listed tokenizer.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=16384)
    args = parser.parse_args()
    try:
        prepare_workload(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
