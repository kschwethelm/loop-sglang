from typing import Any

from transformers import PretrainedConfig


class OuroConfig(PretrainedConfig):
    model_type = "ouro"

    def __init__(self, total_ut_steps: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.total_ut_steps = total_ut_steps

    @property
    def loop_steps(self) -> int:
        return self.total_ut_steps


class HuginnConfig(PretrainedConfig):
    model_type = "huginn_raven"
