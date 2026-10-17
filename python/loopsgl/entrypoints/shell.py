"""Format terminal output by the per-token loop steps."""

from __future__ import annotations

import re
from collections.abc import Sequence
from colorsys import hsv_to_rgb
from itertools import groupby

from prompt_toolkit.formatted_text import FormattedText


def loop_step_style(steps: int | None, max_steps: int, *, thinking: bool = False) -> str:
    """Map loop counts from blue through green and yellow to red."""
    if steps is None:
        return "fg:#888888 italic" if thinking else ""
    fraction = (steps - 1) / max(1, max_steps - 1)
    rgb = hsv_to_rgb((2 / 3) * (1 - fraction), 0.65, 0.75 if thinking else 1.0)
    style = "fg:#{:02x}{:02x}{:02x}".format(*(round(channel * 255) for channel in rgb))
    return style + " italic" if thinking else style


def loop_step_legend(max_steps: int) -> FormattedText:
    """Label every loop count with the same color used for generated text."""
    return FormattedText(
        [("", "Loop steps used to produce each token: ")]
        + [(loop_step_style(depth, max_steps), f"{depth} ") for depth in range(1, max_steps + 1)]
    )


def format_loop_steps(
    text: str,
    steps: Sequence[int | None] | None,
    max_steps: int,
    *,
    thinking: bool = False,
) -> FormattedText:
    """Group adjacent characters with the same loop count into terminal fragments."""
    if steps is None:
        return FormattedText([(loop_step_style(None, max_steps, thinking=thinking), text)])
    if len(text) != len(steps):
        raise ValueError("Loop counts must align with the emitted characters.")
    return FormattedText(
        [
            (
                loop_step_style(depth, max_steps, thinking=thinking),
                "".join(char for char, _ in chars),
            )
            for depth, chars in groupby(zip(text, steps), key=lambda item: item[1])
        ]
    )


class ShellOutputFormatter:
    """Style streamed thinking sections while preserving text and token loop counts."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.thinking = False
        self._text = ""
        self._steps: list[int | None] = []

    def feed(
        self, text: str, steps: Sequence[int] | None, *, finished: bool = False
    ) -> FormattedText:
        """Buffer partial think tags and restore normal styling after closing tags."""
        if steps is not None and len(text) != len(steps):
            raise ValueError("Loop counts must align with the emitted characters.")
        self._text += text
        self._steps.extend([None] * len(text) if steps is None else steps)
        # Retain only a suffix that could become a complete tag in the next chunk.
        keep = (
            0
            if finished
            else max(
                (
                    size
                    for tag in ("<think>", "</think>")
                    for size in range(1, len(tag))
                    if self._text.endswith(tag[:size])
                ),
                default=0,
            )
        )
        end = len(self._text) - keep
        visible, counts = self._text[:end], self._steps[:end]
        self._text, self._steps = self._text[end:], self._steps[end:]
        fragments = FormattedText([])
        cursor = 0
        for match in re.finditer(r"</?think>", visible):
            fragments.extend(
                format_loop_steps(
                    visible[cursor : match.start()],
                    counts[cursor : match.start()],
                    self.max_steps,
                    thinking=self.thinking,
                )
            )
            fragments.extend(
                format_loop_steps(
                    match.group(),
                    counts[match.start() : match.end()],
                    self.max_steps,
                    thinking=True,
                )
            )
            self.thinking = match.group() == "<think>"
            cursor = match.end()
        fragments.extend(
            format_loop_steps(
                visible[cursor:], counts[cursor:], self.max_steps, thinking=self.thinking
            )
        )
        return fragments
