# Adapted from EleutherAI/lm-evaluation-harness v0.4.13:
# lm_eval/models/sglang_causallms.py.
# Copyright (c) 2020 EleutherAI. See LICENSES/lm-evaluation-harness.txt.

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple, cast

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, handle_stop_sequences, postprocess_generated_text
from loopsgl.core import SamplingParams
from tqdm import tqdm

GenerationRequest = Tuple[Tuple[str, List[int]], Dict[str, Any]]


@register_model("loopsgl")
class LoopSGLangLM(TemplateLM):
    """Generation-only lm-eval backend using the offline Loop-SGLang scheduler."""

    def __init__(
        self,
        pretrained: str,
        batch_size: int | str = 1,
        max_model_len: int | None = None,
        max_gen_toks: int = 256,
        add_bos_token: bool = False,
        dtype: str | torch.dtype = "bfloat16",
        device: str | None = "cuda",
        think_end_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if device not in (None, "cuda", "cuda:0"):
            raise ValueError("The offline Loop-SGLang backend supports a single CUDA GPU.")
        self.batch_size = "auto" if batch_size == "auto" else int(batch_size)
        if self.batch_size != "auto" and self.batch_size <= 0:
            raise ValueError("batch_size must be positive or 'auto'.")
        if max_gen_toks <= 0:
            raise ValueError("max_gen_toks must be positive.")
        if max_model_len is not None:
            if max_model_len <= 1:
                raise ValueError("max_model_len must leave room for context and generation.")
            if "max_seq_len_override" in kwargs:
                raise ValueError("Specify only one of max_model_len and max_seq_len_override.")
            kwargs["max_seq_len_override"] = max_model_len
        if isinstance(dtype, str):
            dtypes = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            if dtype not in dtypes:
                raise ValueError(f"Unsupported dtype: {dtype}. Choose from {tuple(dtypes)}.")
            dtype = dtypes[dtype]

        from loopsgl.llm import LLM

        torch_seed = torch.initial_seed()
        self.model = LLM(pretrained, dtype=dtype, **kwargs)
        # Engine initialization seeds Torch; restore the evaluator's sampling seed.
        torch.manual_seed(torch_seed)
        self.tokenizer = self.model.tokenizer
        self._max_length = self.model.engine.max_seq_len
        self._max_gen_toks = max_gen_toks
        self.add_bos_token = add_bos_token or "gemma" in pretrained.lower()
        self.think_end_token = think_end_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> int:
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    def tok_encode(
        self,
        string: str | List[str],
        left_truncate_len: int | None = None,
        add_special_tokens: bool | None = None,
        truncation: bool = False,
    ) -> List[int] | List[List[int]]:
        if add_special_tokens is None:
            add_special_tokens = self.add_bos_token
        encoding = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids
        if left_truncate_len:
            if isinstance(string, str):
                encoding = encoding[-left_truncate_len:]
            else:
                encoding = [enc[-left_truncate_len:] for enc in encoding]
        return encoding

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self,
        chat_history: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        if not requests:
            return []
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding = cast(
            List[List[int]], self.tok_encode(list(context), add_special_tokens=self.add_bos_token)
        )
        encoded_requests = [
            ((text, tokens), kwargs)
            for text, tokens, kwargs in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(request: GenerationRequest) -> Tuple[int, str]:
            return -len(request[0][1]), request[0][0]

        re_ords = Collator(encoded_requests, _collate_gen, group_by=None)
        chunks = re_ords.get_batched(n=0 if self.batch_size == "auto" else self.batch_size)
        res = []
        eos = self.tok_decode([self.eot_token_id])
        with tqdm(
            total=len(requests),
            disable=disable_tqdm or self.rank != 0,
            desc="Running generate_until requests",
        ) as pbar:
            for chunk in chunks:
                context_and_encoding, all_gen_kwargs = zip(*chunk)
                context, context_encoding = zip(*context_and_encoding)
                context_encoding_truncated = []
                sampling_params = []
                stop_sequences = []
                for tokens, gen_kwargs in zip(context_encoding, all_gen_kwargs):
                    if not isinstance(gen_kwargs, dict):
                        raise ValueError("Generation arguments must be a dictionary.")
                    kwargs = copy.deepcopy(gen_kwargs)
                    until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
                    if any(not isinstance(term, str) for term in until):
                        raise ValueError("Stop sequences must be strings.")
                    max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)
                    if not isinstance(max_gen_toks, int) or not 0 < max_gen_toks < self.max_length:
                        raise ValueError("max_gen_toks must be positive and below max_length.")
                    max_ctx_len = self.max_length - max_gen_toks
                    context_encoding_truncated.append(
                        tokens[-max_ctx_len:] or [self.prefix_token_id]
                    )
                    stop_sequences.append(until)
                    if kwargs.pop("spaces_between_special_tokens", False):
                        raise ValueError("spaces_between_special_tokens=True is unsupported.")
                    kwargs = self.modify_gen_kwargs(kwargs)
                    kwargs["stop"] = until
                    sampling_params.append(SamplingParams(max_tokens=max_gen_toks, **kwargs))

                cont = self._model_generate(context_encoding_truncated, sampling_params)
                for output, text, gen_kwargs, until in zip(
                    cont, context, all_gen_kwargs, stop_sequences
                ):
                    generated_text = cast(str, output["text"])
                    generated_text = postprocess_generated_text(
                        generated_text, until, self.think_end_token
                    )
                    res.append(generated_text)
                    self.cache_hook.add_partial(
                        "generate_until", (text, gen_kwargs), generated_text
                    )
                    pbar.update(1)
        return re_ords.get_original(res)

    def _model_generate(
        self, requests: List[List[int]], sampling_params: List[SamplingParams]
    ) -> List[Dict[str, str | List[int]]]:
        return self.model.generate(prompts=requests, sampling_params=sampling_params)

    @staticmethod
    def modify_gen_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False:
            kwargs["temperature"] = 0.0
            kwargs["top_p"] = 1.0
        else:
            kwargs.setdefault("temperature", 1.0 if do_sample else 0.0)
        return kwargs

    def loglikelihood(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loop-SGLang currently supports only generate_until tasks.")

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        **kwargs: Any,
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loop-SGLang currently supports only generate_until tasks.")

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError("Loop-SGLang currently supports only generate_until tasks.")
