"""Rewards package for OPERA SER-GRPO multi-objective reinforcement learning."""

from src.rewards.accuracy import ClassWeightedAccuracyReward, compute_emotion_weights, reward_label_accuracy_weighted
from src.rewards.acoustic import (
    AcousticInventoryReward,
    compute_top_speaker_acoustic_quantiles,
    reward_acoustic_inventory,
    score_acoustic_piece,
)
from src.rewards.composite import RewardManager
from src.rewards.format import FormatReward, compute_format_reward
from src.rewards.parser import (
    REQUIRED_INVENTORY_KEYS,
    VALID_EMOTIONS,
    CompletionParser,
    extract_acoustic_inventory,
    extract_answer_content,
    extract_reasoning,
)

__all__ = [
    "AcousticInventoryReward",
    "ClassWeightedAccuracyReward",
    "CompletionParser",
    "FormatReward",
    "REQUIRED_INVENTORY_KEYS",
    "RewardManager",
    "VALID_EMOTIONS",
    "compute_emotion_weights",
    "compute_format_reward",
    "compute_top_speaker_acoustic_quantiles",
    "extract_acoustic_inventory",
    "extract_answer_content",
    "extract_reasoning",
    "reward_acoustic_inventory",
    "reward_label_accuracy_weighted",
    "score_acoustic_piece",
]
