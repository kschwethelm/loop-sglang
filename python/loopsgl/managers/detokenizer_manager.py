from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

from loopsgl.managers.io_struct import (
    BaseReq,
    BatchStrOutput,
    BatchTokenIDOutput,
    StrOutput,
    TokenIDOutput,
)
from loopsgl.utils import (
    ZmqPullQueue,
    ZmqPushQueue,
    cached_load_hf_config,
    load_eos_token_ids,
    load_tokenizer,
)

if TYPE_CHECKING:
    from loopsgl.server_args import ServerArgs


def _is_chinese_char(cp: int) -> bool:
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str) -> str:
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    decoded_ids: List[int]
    decoded_str: str
    read_offset: int  # length of read ids
    surr_offset: int  # length of surr ids
    sent_offset: int  # length of sent out string
    loop_step_text: str = ""
    loop_steps: List[int] | None = None
    pending_loop_steps: int = 0

    def update_loop_steps(self, text: str, current_loop_steps: int | None = None) -> None:
        """Retain character attribution across buffered decoding and final text cleanup."""
        if self.loop_steps is None:
            return
        prefix = min(len(text), len(self.loop_step_text))
        if not text.startswith(self.loop_step_text[:prefix]):
            prefix = 0
            for before, after in zip(self.loop_step_text, text):
                if before != after:
                    break
                prefix += 1
        # A character assembled from several tokens uses their largest loop count.
        depth = (
            current_loop_steps
            or self.pending_loop_steps
            or (self.loop_steps[-1] if self.loop_steps else 1)
        )
        self.loop_steps[prefix:] = [depth] * (len(text) - prefix)
        if prefix < len(text):
            self.loop_steps[prefix] = max(depth, self.pending_loop_steps)
        self.loop_step_text = text


class DetokenizerManager:
    """Decode scheduler outputs in a process independent of frontend tokenization."""

    def __init__(self, server_args: ServerArgs) -> None:
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = load_tokenizer(server_args.model_path)
        hf_config = cached_load_hf_config(server_args.model_path, server_args.model_override_args)
        self.eos_token_ids = load_eos_token_ids(
            server_args.model_path, self.tokenizer, hf_config=hf_config
        )
        self.recv_from_scheduler = ZmqPullQueue(
            server_args.zmq_detokenizer_addr, create=False, decoder=BaseReq.decoder
        )
        self.send_to_tokenizer = ZmqPushQueue(
            server_args.zmq_frontend_addr, create=False, encoder=BaseReq.encoder
        )

    def event_loop(self) -> None:
        """Decode scheduler messages and return incremental or completed text."""
        while True:
            message = self.recv_from_scheduler.get()
            messages = message.data if isinstance(message, BatchTokenIDOutput) else [message]
            replies = self.detokenize(messages)
            self.send_to_tokenizer.put(
                replies[0] if len(replies) == 1 else BatchStrOutput(data=replies)
            )

    def shutdown(self) -> None:
        """Close the IPC channels owned by this process."""
        self.recv_from_scheduler.stop()
        self.send_to_tokenizer.stop()

    def detokenize(self, msgs: List[TokenIDOutput]) -> List[StrOutput]:
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            if not msg.skip_token:
                s.decoded_ids.append(msg.next_token)
                if msg.loop_steps is not None:
                    if s.loop_steps is None:
                        s.loop_steps = []
                    if (
                        not msg.skip_special_tokens
                        or msg.next_token not in self.tokenizer.all_special_ids
                    ):
                        s.pending_loop_steps = max(s.pending_loop_steps, msg.loop_steps)
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        read_texts, surr_texts = [""] * len(msgs), [""] * len(msgs)
        for skip in (False, True):
            indices = [i for i, msg in enumerate(msgs) if msg.skip_special_tokens == skip]
            if indices:
                decoded = self.tokenizer.batch_decode(
                    [read_ids[i] for i in indices], skip_special_tokens=skip
                )
                surrounded = self.tokenizer.batch_decode(
                    [surr_ids[i] for i in indices], skip_special_tokens=skip
                )
                for i, text, surround in zip(indices, decoded, surrounded, strict=True):
                    read_texts[i], surr_texts[i] = text, surround

        replies: List[StrOutput] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]
            has_incomplete_char = new_text.endswith("�")
            if s.loop_steps is not None:
                previous_length = len(s.loop_step_text)
                # Attribute stable characters even while an incomplete word is withheld.
                s.update_loop_steps(s.decoded_str + new_text.rstrip("�"), msg.loop_steps)
                if has_incomplete_char and len(s.loop_step_text) > previous_length:
                    s.pending_loop_steps = msg.loop_steps or s.pending_loop_steps
            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not has_incomplete_char:
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            if msg.stop_text is not None:
                output_str = msg.stop_text
            elif msg.finished:
                output_str = self.tokenizer.decode(
                    s.decoded_ids, skip_special_tokens=msg.skip_special_tokens
                )
            elif msg.stop_buffer_length:
                output_str = output_str[: max(0, len(output_str) - msg.stop_buffer_length)]
            if msg.finished:
                s.update_loop_steps(output_str)
            incremental_output = output_str[s.sent_offset :]
            incremental_loop_steps = (
                s.loop_steps[s.sent_offset : len(output_str)] if s.loop_steps is not None else None
            )
            s.sent_offset = len(output_str)
            if not has_incomplete_char:
                s.pending_loop_steps = 0
            replies.append(
                StrOutput(
                    uid=msg.uid,
                    incremental_output=incremental_output,
                    finished=msg.finished,
                    token_ids=list(s.decoded_ids) if msg.finished else None,
                    text=output_str if msg.finished else None,
                    incremental_loop_steps=incremental_loop_steps,
                )
            )
            if msg.finished:
                del self.decode_map[msg.uid]

        return replies


def run_detokenizer_process(server_args: ServerArgs, ack_queue: mp.Queue) -> None:
    """Initialize the detokenizer and serve outputs until the owning runtime exits."""
    import traceback

    ready_sent = False
    try:
        manager = DetokenizerManager(server_args)
        try:
            ack_queue.put("Detokenizer is ready")
            ready_sent = True
            manager.event_loop()
        except KeyboardInterrupt:
            pass
        finally:
            manager.shutdown()
    except Exception:
        if not ready_sent:
            ack_queue.put({"error": traceback.format_exc()})
        raise
