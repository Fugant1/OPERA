"""Models package for OPERA."""

from src.models.lora import get_lora_config
from src.models.qwen_omni import QwenOmniLoader

__all__ = ["QwenOmniLoader", "get_lora_config"]
