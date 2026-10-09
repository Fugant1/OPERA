"""GRPO Trainer orchestration for Speech Emotion Recognition."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.config import GRPOConfig
from src.rewards.composite import RewardManager
from src.utils.logging import get_logger

logger = get_logger("training.trainer")


class SERGRPOTrainer:
    """Orchestrates Group Relative Policy Optimization (GRPO) training for SER."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        reward_funcs: Optional[List[Callable]] = None,
        config: Optional[GRPOConfig] = None,
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reward_funcs = reward_funcs or []
        self.config = config or GRPOConfig()
        self.trainer: Optional[Any] = None

    def build_trainer(self) -> Any:
        """Instantiate TRL GRPOTrainer with configured hyperparameters."""
        try:
            from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

            output_dir = str(self.config.output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            training_args = TRLGRPOConfig(
                output_dir=output_dir,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_generations=self.config.num_generations,
                max_prompt_length=self.config.max_prompt_length,
                max_completion_length=self.config.max_completion_length,
                temperature=self.config.temperature,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                max_steps=self.config.max_steps,
            )

            self.trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=self.reward_funcs,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )
            logger.info("TRL GRPOTrainer initialized successfully.")
            return self.trainer
        except ImportError as e:
            logger.error("trl library is required for GRPOTrainer: %s", e)
            raise

    def train(self) -> Any:
        """Run GRPO reinforcement learning training loop."""
        if self.trainer is None:
            self.build_trainer()
        logger.info("Starting GRPO training for SER...")
        return self.trainer.train()
