"""OPERA: Omni-modal Preference Extraction and Ranking Architecture.

A modular research framework for Speech Emotion Recognition (SER)
using Group Relative Policy Optimization (GRPO) and physical acoustic grounding.
"""

__version__ = "0.2.0"

# Expose core configurations and utilities unconditionally
from src.config import (
    AcousticsConfig,
    DataConfig,
    ExperimentConfig,
    GRPOConfig,
    ModelConfig,
    RewardConfig,
)
from src.prompts import COT_PROMPT, QWEN_PERSONA, build_chat_prompt, build_messages
from src.utils import get_audio_duration, get_logger

# Domain-specific exports with graceful dependency handling
try:
    from src.rewards import (
        AcousticInventoryReward,
        ClassWeightedAccuracyReward,
        CompletionParser,
        FormatReward,
        RewardManager,
        compute_emotion_weights,
        compute_format_reward,
        extract_acoustic_inventory,
        extract_answer_content,
        extract_reasoning,
        reward_label_accuracy_weighted,
    )
except ImportError:
    pass

try:
    from src.data import (
        ArchiveExtractor,
        BalancedSampler,
        MELDLoader,
        SERGRPODataset,
        load_meld,
        sample_balanced_meld,
        split_train_val,
    )
except ImportError:
    pass

try:
    from src.acoustics import (
        DEFAULT_ACOUSTIC_CUES,
        AcousticAnalyzer,
        AcousticVisualizer,
        SpeakerDiagnostics,
        eGeMAPSExtractor,
    )
except ImportError:
    pass

try:
    from src.models import QwenOmniLoader, get_lora_config
except ImportError:
    pass

try:
    from src.training import RewardLoggingCallback, SERGRPOTrainer
except ImportError:
    pass

try:
    from src.pipeline import SERGRPOPipeline
except ImportError:
    pass
