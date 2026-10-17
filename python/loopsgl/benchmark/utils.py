# Adapted from SGLang v0.5.20: python/sglang/benchmark/utils.py.
# Licensed under the Apache License, Version 2.0. See LICENSES/sglang.txt.

import json
import os


def remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text


def download_and_cache_hf_file(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Download a file from Hugging Face and cache it locally."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def is_file_valid_json(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"{path} exists but json loading fails ({e=}), thus treat as invalid file")
        return False
