"""Training callbacks for monitoring GRPO rewards and emotion distributions."""

from typing import Any, Dict

from src.utils.logging import get_logger

logger = get_logger("training.callbacks")


class RewardLoggingCallback:
    """Logs detailed reward diagnostics during GRPO training."""

    def __init__(self):
        self.step = 0

    def on_step_end(self, metrics: Dict[str, Any]) -> None:
        """Invoked at each training step with active metrics."""
        self.step += 1
        rewards = {k: v for k, v in metrics.items() if "reward" in k.lower()}
        if rewards:
            logger.info("Step %d | Rewards: %s", self.step, rewards)
