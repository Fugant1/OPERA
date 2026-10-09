"""R3: Structural XML reasoning tag and slot completion format reward."""

from typing import Any, Dict, List, Optional

from src.rewards.parser import (
    REQUIRED_INVENTORY_KEYS,
    VALID_EMOTIONS,
    extract_acoustic_inventory,
    extract_answer_content,
    extract_reasoning,
)


def compute_format_reward(
    completion: str,
    parsed_inventory: Dict[str, Optional[str]],
    extracted_reasoning: str,
    extracted_answer: str,
) -> float:
    """Computes a graded format reward in [0.0, 1.0] to guide early GRPO rollouts

    without incentivizing reward hacking.

    Components:
        1. Correct tag enclosure order (<think> ... </think> <answer> ... </answer>): +0.20
        2. Slot completion for all 5 required acoustic inventory keys: +0.35 * (valid / 5)
        3. Prosodic reasoning non-empty and 5-150 words: +0.25 (or +0.10 if > 150 words)
        4. Valid predicted emotion class: +0.20
    """
    reward = 0.0

    # 1. XML Tag ordering
    has_think_open = "<think>" in completion
    has_think_close = "</think>" in completion
    has_answer_open = "<answer>" in completion
    has_answer_close = "</answer>" in completion

    if has_think_open and has_think_close and has_answer_open and has_answer_close:
        t_open = completion.find("<think>")
        t_close = completion.find("</think>")
        a_open = completion.find("<answer>")
        a_close = completion.find("</answer>")
        if t_open < t_close < a_open < a_close:
            reward += 0.20

    # 2. Slot completion
    valid_slots = sum(
        1 for key in REQUIRED_INVENTORY_KEYS if parsed_inventory.get(key) is not None
    )
    reward += 0.35 * (valid_slots / 5.0)

    # 3. Reasoning quality and length
    clean_reasoning = extracted_reasoning.strip()
    placeholder = "(Deduce the emotion from the cues above)"

    if len(clean_reasoning) >= 20 and clean_reasoning != placeholder:
        words = clean_reasoning.split()
        if 5 <= len(words) <= 150:
            reward += 0.25
        elif len(words) > 150:
            reward += 0.10

    # 4. Valid emotion token
    if extracted_answer in VALID_EMOTIONS:
        reward += 0.20

    return round(reward, 3)


class FormatReward:
    """Callable format reward wrapper for TRL GRPOTrainer."""

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Compute format rewards for a batch of rollouts."""
        rewards: List[float] = []
        for comp in completions:
            inventory = extract_acoustic_inventory(comp)
            reasoning = extract_reasoning(comp)
            answer = extract_answer_content(comp)
            score = compute_format_reward(comp, inventory, reasoning, answer)
            rewards.append(score)
        return rewards
