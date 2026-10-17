import functools
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase
from transformers.utils.hub import cached_file

from .logger import init_logger

logger = init_logger(__name__)


# Adapted from SGLang v0.5.20: python/sglang/srt/configs/model_config.py.
# Licensed under the Apache License, Version 2.0. See LICENSES/sglang.txt.
_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def get_and_verify_dtype(
    config: Union[PretrainedConfig, Dict[str, Any]],
    dtype: Union[str, torch.dtype],
) -> torch.dtype:
    """Resolve inference dtype from checkpoint metadata and explicit overrides."""
    # Checkpoint dtype metadata can be absent or None.
    if isinstance(config, dict):
        config_dtype = config.get("dtype", None) or config.get("torch_dtype", None)
    else:
        config_dtype = getattr(config, "dtype", None)
    if isinstance(config_dtype, str):
        config_dtype = _STR_DTYPE_TO_TORCH_DTYPE.get(config_dtype, None)
    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                # Following the common practice, we use float16 for float32 models.
                torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            logger.debug("Upcasting %s to %s.", config_dtype, torch_dtype)
        elif config_dtype == torch.float32:
            logger.debug("Downcasting %s to %s.", config_dtype, torch_dtype)
        else:
            logger.debug("Casting %s to %s.", config_dtype, torch_dtype)

    return torch_dtype


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str, json_model_override_args: str = "{}") -> PretrainedConfig:
    config = AutoConfig.from_pretrained(model_path)
    model_override_args = parse_model_override_args(json_model_override_args)
    # Adapted from SGLang v0.5.20: python/sglang/srt/utils/hf_transformers/config.py.
    # Licensed under the Apache License, Version 2.0. See LICENSES/sglang.txt.
    for key, value in model_override_args.items():
        current = getattr(config, key, None)
        if isinstance(value, dict) and isinstance(current, PretrainedConfig):
            current.update(value)
        else:
            setattr(config, key, value)
    return config


def parse_model_override_args(value: str) -> Dict[str, Any]:
    """Parse SGLang-style JSON overrides as a model configuration object."""
    overrides = json.loads(value)
    if not isinstance(overrides, dict):
        raise ValueError("json_model_override_args must contain a JSON object")
    return overrides


def cached_load_hf_config(
    model_path: str, model_override_args: Dict[str, Any] | None = None
) -> PretrainedConfig:
    # Initialize local config registrations before loading checkpoint metadata.
    import loopsgl.models  # noqa: F401

    overrides = json.dumps(model_override_args or {})
    # SGLang gives each model an isolated copy of the cached overridden config.
    return deepcopy(_load_hf_config(model_path, overrides))


def load_eos_token_ids(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    hf_config: PretrainedConfig | None = None,
) -> set[int]:
    """Combine model, generation-config, and tokenizer stopping token IDs."""
    config = hf_config if hf_config is not None else cached_load_hf_config(model_path)
    config_eos_token_id = getattr(config, "eos_token_id", None)
    path = cached_file(
        model_path, "generation_config.json", _raise_exceptions_for_missing_entries=False
    )
    generation_eos_token_id = None
    if path is not None:
        with Path(path).open(encoding="utf-8") as file:
            generation_eos_token_id = json.load(file).get("eos_token_id")

    eos_token_ids: set[int] = set()
    for token_ids in (
        config_eos_token_id,
        generation_eos_token_id,
        tokenizer.eos_token_id,
        getattr(tokenizer, "additional_stop_token_ids", None),
    ):
        if isinstance(token_ids, int):
            eos_token_ids.add(token_ids)
        else:
            eos_token_ids.update(token_ids or [])
    return eos_token_ids


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
