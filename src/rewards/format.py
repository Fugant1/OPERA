"""R3: Structural XML reasoning tag and slot completion format reward."""

from typing import Any, Dict, List, Optional

from src.rewards.parser import (
    REQUIRED_INVENTORY_KEYS,
    VALID_EMOTIONS,
    extract_acoustic_inventory,
    extract_answer_content,
    extract_reasoning,
    parse_completion,
)


def compute_format_reward_from_parsed(p: Dict[str, Any]) -> float:
    """Computes a graded format reward in [0.0, 1.0] from pre-parsed completion structure.

    Components:
        1. Correct tag enclosure order (<think> ... </think> <answer> ... </answer>): +0.20
        2. Slot completion for all 5 required acoustic inventory keys: +0.35 * (valid / 5)
        3. Prosodic reasoning non-empty and 5-150 words: +0.25 (or +0.10 if > 150 words)
        4. Valid predicted emotion class: +0.20
    """
    reward = 0.0

    # 1. XML Tag ordering
    if p["valid_tag_order"]:
        reward += 0.20

    # 2. Slot completion
    reward += 0.35 * (p["valid_slots_count"] / 5.0)

    # 3. Reasoning quality and length
    clean_reasoning = p["reasoning"].strip()
    placeholder = "(Deduce the emotion from the cues above)"

    if len(clean_reasoning) >= 20 and clean_reasoning != placeholder:
        words = len(clean_reasoning.split())
        if 5 <= words <= 150:
            reward += 0.25
        elif words > 150:
            reward += 0.10

    # 4. Valid emotion token
    if p["is_valid_emotion"]:
        reward += 0.20

    return round(reward, 3)


def reward_format(
    prompts: List[str],
    completions: List[str],
    **kwargs: Any,
) -> List[float]:
    """Batch format reward function for TRL GRPOTrainer."""
    parsed_list = kwargs.get("parsed_completions") or [parse_completion(c) for c in completions]
    return [compute_format_reward_from_parsed(p) for p in parsed_list]


def compute_format_reward(*args: Any, **kwargs: Any) -> Any:
    """Hybrid format reward: supports batch (GRPOTrainer) and single-completion calls."""
    if "prompts" in kwargs or "completions" in kwargs or (
        len(args) >= 2 and isinstance(args[0], list) and isinstance(args[1], list)
    ):
        kw = dict(kwargs)
        prompts = kw.pop("prompts", args[0] if len(args) > 0 else [])
        completions = kw.pop("completions", args[1] if len(args) > 1 else [])
        return reward_format(prompts=prompts, completions=completions, **kw)

    # Single-completion evaluation
    if len(args) == 1 and isinstance(args[0], str):
        p = parse_completion(args[0])
        return compute_format_reward_from_parsed(p)
    elif len(args) >= 4:
        p = parse_completion(args[0])
        return compute_format_reward_from_parsed(p)
    elif "completion" in kwargs:
        p = parse_completion(kwargs["completion"])
        return compute_format_reward_from_parsed(p)
    else:
        return 0.0


class FormatReward:
    """Callable format reward wrapper for TRL GRPOTrainer."""

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Compute format rewards for a batch of rollouts in single-pass."""
        return reward_format(prompts=prompts, completions=completions, **kwargs)
