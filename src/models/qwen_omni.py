"""Model and processor initialization for Qwen 2.5 Omni with 4-bit quantization and LoRA."""

from typing import Any, Optional, Tuple

from src.config import ModelConfig
from src.models.lora import get_lora_config
from src.utils.logging import get_logger

logger = get_logger("models.qwen_omni")


class QwenOmniLoader:
    """Manages processor loading, 4-bit BitsAndBytes quantization, and LoRA adaptation for Qwen 2.5 Omni."""

    @staticmethod
    def get_quantization_config(config: Optional[ModelConfig] = None) -> Any:
        """Build BitsAndBytesConfig for 4-bit NormalFloat (NF4) quantization."""
        cfg = config or ModelConfig()
        try:
            import torch
            from transformers import BitsAndBytesConfig

            dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
            return BitsAndBytesConfig(
                load_in_4bit=cfg.load_in_4bit,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            )
        except ImportError:
            logger.warning("torch or transformers not available for BitsAndBytesConfig.")
            return None

    @classmethod
    def load_processor(cls, model_id: str = "Qwen/Qwen2.5-Omni-7B") -> Any:
        """Load and configure AutoProcessor disabling talker audio output."""
        from transformers import AutoProcessor

        logger.info("Loading AutoProcessor for %s...", model_id)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(processor, "generation_config") and processor.generation_config is not None:
            processor.generation_config.return_audio = False
        return processor

    @classmethod
    def load_model(
        cls,
        config: Optional[ModelConfig] = None,
        peft_config: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Load quantized model, disable audio talker, and apply LoRA adapters.

        Returns:
            Tuple of (model, processor).
        """
        cfg = config or ModelConfig()
        processor = cls.load_processor(cfg.model_id)

        import torch
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import Qwen2_5OmniForConditionalGeneration

        bnb_config = cls.get_quantization_config(cfg)
        logger.info("Loading Qwen2.5-Omni model with 4-bit quantization...")

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            cfg.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Disable speech synthesis / talker module
        model.disable_talker()
        model.generation_config.return_audio = False
        if hasattr(model, "talker_config"):
            model.config.enable_talker = False

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Apply LoRA adapters
        lora_cfg = peft_config or get_lora_config(cfg)
        model = get_peft_model(model, lora_cfg)

        logger.info("Trainable parameters:")
        model.print_trainable_parameters()

        return model, processor
