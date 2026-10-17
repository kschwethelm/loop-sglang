from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from loopsgl.models.looped.config import HuginnConfig, OuroConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Callable[[str], Path]:
    """Create local looped-model configs and a tokenizer without downloading weights."""

    def create(model: str = "ouro") -> Path:
        path = tmp_path / model
        config = {
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "rope_theta": 10000,
            "max_position_embeddings": 128,
            "tie_word_embeddings": False,
            "eos_token_id": 2,
        }
        if model == "huginn":
            hf_config = HuginnConfig(
                **config,
                architectures=["RavenForCausalLM"],
                num_prelude_layers=1,
                num_core_layers=1,
                num_coda_layers=1,
                loop_steps=4,
                bias=False,
                attention_bias=True,
                state_init="zero",
                init_values={"embed_scale": 1.4, "std": 0.1},
            )
        else:
            hf_config = OuroConfig(**config, architectures=["OuroForCausalLM"], total_ut_steps=4)
        hf_config.save_pretrained(path)
        tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="t0", eos_token="t2"
        ).save_pretrained(path)
        return path

    return create
