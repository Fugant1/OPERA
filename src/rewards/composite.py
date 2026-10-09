"""Composite reward registry and orchestrator for GRPO training."""

from typing import Any, Callable, Dict, List, Optional

from src.rewards.accuracy import ClassWeightedAccuracyReward, compute_emotion_weights
from src.rewards.acoustic import AcousticInventoryReward
from src.rewards.format import FormatReward


class RewardManager:
    """Manages multi-objective reward functions for GRPO rollouts."""

    def __init__(
        self,
        class_weights: Optional[Dict[str, float]] = None,
        quantiles_ref: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.r1_accuracy = ClassWeightedAccuracyReward(class_weights=class_weights)
        self.r2_acoustic = AcousticInventoryReward(quantiles_ref=quantiles_ref)
        self.r3_format = FormatReward()

    def get_reward_functions(self) -> List[Callable]:
        """Returns list of callable reward functions compatible with TRL GRPOTrainer.

        TRL GRPOTrainer expects: List[Callable[[prompts, completions, ...], List[float]]]
        """
        return [
            self.r1_accuracy,
            self.r2_acoustic,
            self.r3_format,
        ]

    def update_class_weights(self, class_weights: Dict[str, float]) -> None:
        """Update R1 class rarity weights."""
        self.r1_accuracy.set_weights(class_weights)

    def update_quantiles(self, quantiles_ref: Dict[str, Dict[str, float]]) -> None:
        """Update R2 empirical quantile thresholds."""
        self.r2_acoustic.quantiles_ref = quantiles_ref
