# Adapted from SGLang v0.5.20: python/sglang/benchmark/datasets/__init__.py.
# Licensed under the Apache License, Version 2.0. See LICENSES/sglang.txt.

from typing import Any, Dict, List, Optional, Type

from loopsgl.benchmark.datasets.common import BaseDataset, DatasetRow
from loopsgl.benchmark.datasets.custom import CustomDataset
from loopsgl.benchmark.datasets.random import RandomDataset
from loopsgl.benchmark.datasets.sharegpt import ShareGPTDataset

DATASET_MAPPING: Dict[str, Type[BaseDataset]] = {
    "sharegpt": ShareGPTDataset,
    "custom": CustomDataset,
    "random": RandomDataset,
    "random-ids": RandomDataset,
}


def get_dataset(args: Any, tokenizer: Any, model_id: Optional[str] = None) -> List[DatasetRow]:
    dataset_name = args.dataset_name
    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_cls = DATASET_MAPPING[dataset_name]
    dataset = dataset_cls.from_args(args)
    return dataset.load(tokenizer=tokenizer, model_id=model_id)


__all__ = ["DATASET_MAPPING", "DatasetRow", "get_dataset"]
