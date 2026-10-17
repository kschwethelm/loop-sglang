from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loopsgl.layers import BaseOP

if TYPE_CHECKING:
    import torch

    from loopsgl.layers import ParallelLMHead


class BaseLLMModel(ABC, BaseOP):
    model: BaseOP
    lm_head: ParallelLMHead

    @abstractmethod
    def forward(self) -> torch.Tensor: ...
