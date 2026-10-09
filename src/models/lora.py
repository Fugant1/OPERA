"""LoRA adapter configuration for Qwen 2.5 Omni thinker modules."""

from typing import Any, Optional

from src.config import ModelConfig
from src.utils.logging import get_logger

logger = get_logger("models.lora")


def get_lora_config(config: Optional[ModelConfig] = None) -> Any:
    """Build PEFT LoraConfig targeting thinker attention and MLP projections.

    Args:
        config: ModelConfig instance (uses defaults if None).

    Returns:
        peft.LoraConfig instance.
    """
    cfg = config or ModelConfig()
    try:
        from peft import LoraConfig
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            task_type=None,
            target_modules=cfg.target_modules_regex,
        )
        logger.info(
            "LoRA Config initialized: r=%d, alpha=%d, target_modules='%s'",
            cfg.lora_r,
            cfg.lora_alpha,
            cfg.target_modules_regex,
        )
        return peft_cfg
    except ImportError:
        logger.warning("peft library not installed. Returning configuration dict.")
        return {
            "r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "bias": cfg.lora_bias,
            "target_modules": cfg.target_modules_regex,
        }
