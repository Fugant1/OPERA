"""Rewards package for OPERA SER-GRPO multi-objective reinforcement learning."""

from src.rewards.accuracy import ClassWeightedAccuracyReward, compute_emotion_weights, reward_label_accuracy_weighted
from src.rewards.acoustic import (
    ALL_ACOUSTIC_FEATURES,
    CUE_FEATURE_MAP,
    VOICE_QUALITY_CHOICES,
    VOICE_QUALITY_FEATURE_MAP,
    AcousticInventoryReward,
    build_acoustic_audio_lookup,
    compute_top_speaker_acoustic_quantiles,
    determine_voice_quality_ground_truth,
    reward_acoustic_inventory,
    score_acoustic_piece,
    score_voice_quality_piece,
)
from src.rewards.composite import RewardManager, unified_gated_grpo_reward, unified_grpo_reward
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
    "ALL_ACOUSTIC_FEATURES",
    "AcousticInventoryReward",
    "CUE_FEATURE_MAP",
    "ClassWeightedAccuracyReward",
    "CompletionParser",
    "FormatReward",
    "REQUIRED_INVENTORY_KEYS",
    "RewardManager",
    "VALID_EMOTIONS",
    "VOICE_QUALITY_CHOICES",
    "VOICE_QUALITY_FEATURE_MAP",
    "build_acoustic_audio_lookup",
    "compute_emotion_weights",
    "compute_format_reward",
    "compute_top_speaker_acoustic_quantiles",
    "determine_voice_quality_ground_truth",
    "extract_acoustic_inventory",
    "extract_answer_content",
    "extract_reasoning",
    "reward_acoustic_inventory",
    "reward_label_accuracy_weighted",
    "score_acoustic_piece",
    "score_voice_quality_piece",
    "unified_gated_grpo_reward",
    "unified_grpo_reward",
]
