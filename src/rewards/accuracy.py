"""R1: Smoothed inverse class-frequency weighted accuracy reward."""

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    import pandas as pd

from src.rewards.parser import LABEL_SYNONYMS, extract_answer_content
from src.utils.logging import get_logger

logger = get_logger("rewards.accuracy")


def compute_emotion_weights(
    train_set: Any,
    power: float = 0.4,
) -> Dict[str, float]:
    """Computes smoothed inverse class-frequency weights for GRPO rewards.

    Uses w_c = (max_count / count) ** power so the majority class (neutral)
    is anchored at 1.0, and minority classes receive an advantage boost
    without blowing up gradient variance.

    Args:
        train_set: Pandas DataFrame, HuggingFace Dataset, or iterable with 'emotion'.
        power: Smoothing exponent. 0.0 = uniform (all 1.0), 0.4-0.5 = smoothed
          inverse (recommended for RLVR), 1.0 = raw inverse frequency.

    Returns:
        Dictionary mapping normalized emotion names to positive float weights.
    """
    if hasattr(train_set, "__getitem__") and "emotion" in train_set:
        emotions = list(train_set["emotion"])
    elif isinstance(train_set, (list, tuple)):
        emotions = [item.get("emotion", item.get("label", "")) for item in train_set]
    else:
        emotions = []

    emotions = [str(e).lower().strip() for e in emotions if str(e).strip()]
    counts = Counter(emotions)

    if not counts:
        return {}

    max_count = max(counts.values())
    weights = {
        emotion: round(float((max_count / count) ** power), 3)
        for emotion, count in counts.items()
    }
    logger.info("Computed class rarity weights: %s", weights)
    return weights


class ClassWeightedAccuracyReward:
    """Evaluates prediction exact match against target label with rarity weighting."""

    def __init__(self, class_weights: Optional[Dict[str, float]] = None):
        self.class_weights = class_weights or {}

    def set_weights(self, class_weights: Dict[str, float]) -> None:
        """Update active class weights."""
        self.class_weights = class_weights

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        label: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Compute accuracy rewards for a batch of GRPO rollouts.

        Returns:
            List of float rewards: weight of target class on match, 0.0 otherwise.
        """
        rewards: List[float] = []
        for comp, target in zip(completions, label):
            pred_raw = extract_answer_content(comp)
            pred_norm = LABEL_SYNONYMS.get(pred_raw, pred_raw)

            target_norm = str(target).lower().strip()
            target_norm = LABEL_SYNONYMS.get(target_norm, target_norm)

            if pred_norm == target_norm and pred_norm != "":
                weight = self.class_weights.get(target_norm, 1.0)
                rewards.append(float(weight))
            else:
                rewards.append(0.0)

        return rewards


def reward_label_accuracy_weighted(
    prompts: List[str],
    completions: List[str],
    label: List[str],
    class_weights: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> List[float]:
    """Functional wrapper for class-weighted accuracy reward."""
    rewarder = ClassWeightedAccuracyReward(class_weights=class_weights)
    return rewarder(prompts, completions, label, **kwargs)
