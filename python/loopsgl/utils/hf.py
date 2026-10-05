import functools
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase
from transformers.utils.hub import cached_file


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
def _load_hf_config(model_path: str) -> PretrainedConfig:
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    # Initialize local config registrations before loading checkpoint metadata.
    import loopsgl.models  # noqa: F401

    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def load_eos_token_ids(model_path: str, tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Combine model, generation-config, and tokenizer stopping token IDs."""
    config_eos_token_id = getattr(cached_load_hf_config(model_path), "eos_token_id", None)
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
