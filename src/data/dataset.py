"""Dataset preparation and formatting for SER GRPO training."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("data.dataset")


def split_train_val(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    stratify_col: str = "emotion",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform deterministic stratified train/validation split.

    Args:
        df: Input DataFrame.
        test_size: Proportion for validation set.
        seed: Random seed.
        stratify_col: Column used for stratification.

    Returns:
        (train_df, val_df)
    """
    try:
        from sklearn.model_selection import train_test_split
        stratify = df[stratify_col] if stratify_col in df.columns else None
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ImportError:
        # Fallback if scikit-learn is not available
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_idx = int(len(shuffled) * (1.0 - test_size))
        train_df = shuffled.iloc[:split_idx].copy()
        val_df = shuffled.iloc[split_idx:].copy()

    logger.info(
        "Train/Val split: %d train, %d val (Stratified by '%s')",
        len(train_df),
        len(val_df),
        stratify_col,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class SERGRPODataset:
    """Prepares structured records for GRPO trainer rollouts."""

    def __init__(self, df: pd.DataFrame, prompt_template: str):
        self.df = df.copy()
        self.prompt_template = prompt_template

    def to_records(self) -> List[Dict[str, Any]]:
        """Convert DataFrame into HuggingFace/TRL GRPOTrainer compliant records."""
        records = []
        for _, row in self.df.iterrows():
            record = {
                "prompt": self.prompt_template,
                "label": str(row.get("emotion", "")).lower().strip(),
                "audio_path": str(row.get("audio_path", "")),
                "speaker": str(row.get("speaker", "")),
                "text": str(row.get("text", "")),
            }
            # Carry over acoustic feature ground truth if present
            for col in self.df.columns:
                if col.startswith("F0") or col.startswith("loudness") or "sma3" in col or "Voiced" in col:
                    record[col] = row[col]
            records.append(record)
        return records

    def to_hf_dataset(self) -> Any:
        """Converts records into a datasets.Dataset if HuggingFace datasets is installed."""
        try:
            from datasets import Dataset
            return Dataset.from_list(self.to_records())
        except ImportError:
            return self.to_records()
