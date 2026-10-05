from transformers import AutoConfig

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .looped.config import HuginnConfig, OuroConfig
from .register import get_model_class
from .weight import load_weight

AutoConfig.register(OuroConfig.model_type, OuroConfig)
AutoConfig.register(HuginnConfig.model_type, HuginnConfig)


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
