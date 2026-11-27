# Adapted from EleutherAI/lm-evaluation-harness v0.4.13:
# lm_eval/models/sglang_causallms.py.
# Copyright (c) 2020 EleutherAI. See LICENSES/lm-evaluation-harness.txt.

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple, cast

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import CacheHook, TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, handle_stop_sequences, postprocess_generated_text
from tqdm import tqdm

from loopsgl.core import LoopParams, SamplingParams

eval_logger = logging.getLogger(__name__)

GenerationRequest = Tuple[Tuple[str, List[int]], Dict[str, Any], str]
LikelihoodRequest = Tuple[Tuple[str, str] | None, List[int], List[int]]


@register_model("loopsgl")
class LoopSGLangLM(TemplateLM):
    """Generation and likelihood evaluation using the offline Loop-SGLang scheduler."""

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
        loop_steps: int | None = None,
        exit_threshold: float | None = None,
        refill: bool = True,
        min_coda_batch_size: int = 1,
        disable_lookahead_exit: bool = False,
        score_first_token_in_decode: bool = False,
        exit_depth_output: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._exit_depth_output = Path(exit_depth_output) if exit_depth_output is not None else None
        if self._exit_depth_output is not None and self._exit_depth_output.exists():
            raise FileExistsError(f"Exit-depth output already exists: {self._exit_depth_output}")
        self._exit_depth_counts: dict[str, Counter[int]] = {}
        if device not in (None, "cuda", "cuda:0"):
            raise ValueError("Use device=cuda and configure tensor parallelism through tp_size.")
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

        from loopsgl import Engine
        from loopsgl.managers.scheduler_components.config import CDBConfig

        kwargs.setdefault("retraction_mode", "none")
        kwargs.setdefault("disable_radix_cache", True)
        kwargs.setdefault("chunked_prefill_size", -1)
        self._warned_settings: set[str] = set()
        torch_seed = torch.initial_seed()
        self.model = Engine(
            pretrained,
            dtype=dtype,
            loop_params=LoopParams(loop_steps=loop_steps, exit_threshold=exit_threshold),
            cdb_config=CDBConfig(
                refill=refill,
                min_coda_batch_size=min_coda_batch_size,
                disable_lookahead_exit=disable_lookahead_exit,
            ),
            **kwargs,
        )
        # Keep the evaluator's caller-side sampling seed unchanged.
        torch.manual_seed(torch_seed)
        self.tokenizer = self.model.tokenizer
        self._max_length = self.model.max_seq_len
        self._max_gen_toks = max_gen_toks
        self.add_bos_token = add_bos_token or "gemma" in pretrained.lower()
        self.think_end_token = think_end_token
        self.score_first_token_in_decode = score_first_token_in_decode

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
        chat_template_kwargs: Dict[str, Any] | None = None,
    ) -> str:
        return self.model.tokenizer_manager.apply_chat_template(
            chat_history,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )

    def set_cache_hook(self, cache_hook: CacheHook) -> None:
        if self._exit_depth_output is not None and cache_hook.dbdict is not None:
            raise ValueError("Exit-depth recording requires running lm-eval without --use_cache.")
        super().set_cache_hook(cache_hook)

    def _write_exit_depth_counts(self, counts: dict[str, Counter[int]]) -> None:
        path = self._exit_depth_output
        if path is None or not counts:
            return
        merged = self._exit_depth_counts | {
            task: self._exit_depth_counts.get(task, Counter()) + histogram
            for task, histogram in counts.items()
        }
        report = {
            "tasks": {
                task: {
                    "decode_tokens": histogram.total(),
                    "exit_depth_counts": dict(sorted(histogram.items())),
                }
                for task, histogram in sorted(merged.items())
            }
        }
        self._write_report(path, report, replace=bool(self._exit_depth_counts))
        self._exit_depth_counts = merged

    @staticmethod
    def _write_report(path: Path, report: dict[str, Any], *, replace: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            try:
                file.write(json.dumps(report, indent=2) + "\n")
                file.close()
                if replace:
                    temporary.replace(path)
                else:
                    path.hardlink_to(temporary)
            finally:
                temporary.unlink(missing_ok=True)

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        if not requests:
            return []
        self._warn_generation_config()
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding = cast(
            List[List[int]], self.tok_encode(list(context), add_special_tokens=self.add_bos_token)
        )
        collect_depths = self._exit_depth_output is not None
        task_names = [req.task_name or "unknown" for req in requests] if collect_depths else []
        depth_counts: dict[str, Counter[int]] = {task: Counter() for task in task_names}
        encoded_requests = [
            ((text, tokens), kwargs, task_names[i] if collect_depths else "")
            for i, (text, tokens, kwargs) in enumerate(
                zip(context, context_encoding, all_gen_kwargs, strict=True)
            )
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
                context_and_encoding, all_gen_kwargs, chunk_tasks = zip(*chunk)
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
                    if collect_depths:
                        kwargs["return_loop_steps"] = True
                    sampling_params.append(SamplingParams(max_new_tokens=max_gen_toks, **kwargs))

                cont = self._model_generate(context_encoding_truncated, sampling_params)
                for output, text, gen_kwargs, until, task in zip(
                    cont, context, all_gen_kwargs, stop_sequences, chunk_tasks, strict=True
                ):
                    if collect_depths:
                        depth_counts[task].update(
                            depth for _, depth in output["meta_info"]["decode_token_loop_steps"]
                        )
                    generated_text = cast(str, output["text"])
                    generated_text = postprocess_generated_text(
                        generated_text, until, self.think_end_token
                    )
                    res.append(generated_text)
                    self.cache_hook.add_partial(
                        "generate_until", (text, gen_kwargs), generated_text
                    )
                    pbar.update(1)
        self._write_exit_depth_counts(depth_counts)
        return re_ords.get_original(res)

    def _model_generate(
        self, requests: List[List[int]], sampling_params: List[SamplingParams]
    ) -> List[Dict[str, Any]]:
        return self.model.generate(input_ids=requests, sampling_params=sampling_params)

    @staticmethod
    def modify_gen_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False:
            kwargs["temperature"] = 0.0
            kwargs["top_p"] = 1.0
        else:
            kwargs.setdefault("temperature", 1.0 if do_sample else 0.0)
        return kwargs

    def _warn_generation_config(self) -> None:
        config = self.model.server_args
        shared = config.loop_cache_policy == "shared" and config.model_config.loop_steps > 1
        chunked = config.chunked_prefill_size > 0
        warnings = {
            "prefix caching": (
                not config.disable_radix_cache and shared,
                "cached prompt state can change subsequent outputs",
            ),
            "chunked prefill": (
                chunked and shared,
                "prompt state can depend on chunk boundaries",
            ),
            "recomputation": (
                config.retraction_mode == "recompute"
                and (shared or config.loop_params.exit_threshold is not None),
                "rebuilt states can differ from uninterrupted decoding",
            ),
        }
        for setting, (enabled, reason) in warnings.items():
            if enabled and setting not in self._warned_settings:
                eval_logger.warning(
                    "Generation evaluation enables %s with %s KV: %s.",
                    setting,
                    config.loop_cache_policy,
                    reason,
                )
                self._warned_settings.add(setting)

    def _likelihood_uses_forcing(self) -> bool:
        config = self.model.server_args
        if config.scheduler == "cdb" and config.loop_params.exit_threshold is None:
            raise ValueError(
                "Initial likelihood support excludes full-depth CDB; use scheduler='cb'."
            )
        if (
            not config.disable_radix_cache
            or config.retraction_mode != "none"
            or config.chunked_prefill_size > 0
        ):
            raise ValueError(
                "Likelihood requires disable_radix_cache=True, retraction_mode='none', "
                "and chunked_prefill_size=-1; generation overrides do not support scoring yet."
            )
        return (
            self.score_first_token_in_decode
            or config.loop_params.exit_threshold is not None
            or config.loop_cache_policy == "shared"
        )

    def loglikelihood(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        prepared = []
        for req in requests:
            context, continuation = req.args
            if not continuation:
                context_enc, continuation_enc = [], []
            elif context == "":
                continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
                if continuation_enc and continuation_enc[0] == self.prefix_token_id:
                    context_enc, continuation_enc = continuation_enc[:1], continuation_enc[1:]
                else:
                    context_enc = [self.prefix_token_id]
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
            prepared.append(((context, continuation), context_enc, continuation_enc))
        task_names = (
            [req.task_name or "unknown" for req in requests]
            if self._exit_depth_output is not None
            else None
        )
        return self._loglikelihood_tokens(
            prepared, disable_tqdm=disable_tqdm, task_names=task_names
        )

    def _loglikelihood_tokens(
        self,
        requests: List[LikelihoodRequest],
        disable_tqdm: bool = False,
        *,
        task_names: List[str] | None = None,
        **kwargs: Any,
    ) -> List[Tuple[float, bool]]:
        if not requests:
            return []
        forcing = self._likelihood_uses_forcing() if any(req[2] for req in requests) else False
        collect_depths = self._exit_depth_output is not None
        if task_names is None:
            task_names = ["unknown" if collect_depths else ""] * len(requests)
        depth_counts: dict[str, Counter[int]] = (
            {task: Counter() for task in task_names} if collect_depths else {}
        )
        tagged_requests = list(zip(requests, task_names, strict=True))

        def _collate_likelihood(row: Tuple[LikelihoodRequest, str]) -> Tuple[int, Tuple[int, ...]]:
            (_, context, continuation), _ = row
            tokens = context + continuation
            return -len(tokens), tuple(tokens)

        re_ord = Collator(tagged_requests, sort_fn=_collate_likelihood)
        chunks = re_ord.get_batched(n=0 if self.batch_size == "auto" else int(self.batch_size))
        result = []
        with tqdm(
            total=len(requests),
            disable=disable_tqdm or self.rank != 0,
            desc="Running loglikelihood requests",
        ) as progress:
            for chunk in chunks:
                answers = [(0.0, True)] * len(chunk)
                inputs, params, score_rows = [], [], []
                for i, ((_, context, continuation), _) in enumerate(chunk):
                    if not continuation:
                        continue
                    reserve = len(continuation) + (0 if forcing else 1)
                    available = self.max_length - reserve
                    if available < 1:
                        raise ValueError(
                            "Continuation leaves no room for a conditioning token and the scoring budget."
                        )
                    context = (context or [self.prefix_token_id])[-available:]
                    if self.score_first_token_in_decode:
                        if len(context) < 2:
                            raise ValueError(
                                "score_first_token_in_decode requires at least two context tokens "
                                "after truncation."
                            )
                        # The first forced output seeds decode at the last context position.
                        prompt, tokens, score_start = context[:-1], context[-1:] + continuation, 1
                    elif forcing:
                        prompt, tokens, score_start = context, continuation, 0
                    else:
                        prompt = tokens = context + continuation
                        score_start = len(context)
                    inputs.append(prompt)
                    score_rows.append((i, tokens, score_start))
                    params.append(
                        SamplingParams(
                            temperature=0.0,
                            max_new_tokens=len(tokens) if forcing else 1,
                            ignore_eos=forcing,
                            trace_decode_token_ids=list(tokens) if forcing else None,
                            return_loop_steps=collect_depths and forcing,
                        )
                    )
                if inputs:
                    outputs = self.model.generate(
                        input_ids=inputs,
                        sampling_params=params,
                        return_logprob=True,
                        logprob_start_len=-1 if forcing else 0,
                        top_logprobs_num=2,
                    )
                    for (i, tokens, score_start), output in zip(score_rows, outputs, strict=True):
                        answers[i] = self._parse_logprobs(
                            tokens, output, score_start, output_logprobs=forcing
                        )
                        if collect_depths and forcing:
                            depth_counts[chunk[i][1]].update(
                                depth for _, depth in output["meta_info"]["decode_token_loop_steps"]
                            )
                for ((cache_key, _, _), _), answer in zip(chunk, answers, strict=True):
                    result.append(answer)
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                    progress.update(1)
        self._write_exit_depth_counts(depth_counts)
        return re_ord.get_original(result)

    @staticmethod
    def _parse_logprobs(
        tokens: List[int], outputs: Dict[str, Any], ctxlen: int, *, output_logprobs: bool = False
    ) -> Tuple[float, bool]:
        prefix = "output" if output_logprobs else "input"
        meta = outputs.get("meta_info", {})
        entries = meta.get(f"{prefix}_token_logprobs", [])
        top_rows = meta.get(f"{prefix}_top_logprobs", [])
        expected = tokens[ctxlen:]
        scores = entries[ctxlen:]
        tops = top_rows[ctxlen:]
        token_start = 0 if output_logprobs else ctxlen
        if (
            len(entries) != len(tokens)
            or len(top_rows) != len(tokens)
            or [entry[1] for entry in entries[token_start:]] != tokens[token_start:]
        ):
            raise ValueError(
                "Likelihood response does not cover the complete supplied continuation."
            )
        if any(entry[0] is None for entry in scores) or any(not row for row in tops):
            raise ValueError("Likelihood response contains missing scores or greedy predictions.")
        return (
            sum(entry[0] for entry in scores),
            all(
                max(row, key=lambda item: item[0])[1] == token
                for row, token in zip(tops, expected, strict=True)
            ),
        )

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError("Loop-SGLang loglikelihood_rolling is not implemented.")
