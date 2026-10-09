"""Deterministic emotion balancing and audio duration filtering for MELD."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.audio import get_audio_duration
from src.utils.logging import get_logger

logger = get_logger("data.sampler")


class BalancedSampler:
    """Creates a deterministic, emotion-balanced MELD subset with duration filtering."""

    def __init__(
        self,
        target_total: int = 2000,
        max_neutral_pct: float = 0.15,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        seed: int = 42,
    ):
        self.target_total = target_total
        self.max_neutral_pct = max_neutral_pct
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.seed = seed

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample and balance a DataFrame containing 'audio_path' and 'emotion'.

        Filtering:
            - Removes missing/invalid audio paths.
            - Keeps audio duration within [min_duration, max_duration].

        Balancing:
            - Caps neutral at max_neutral_pct.
            - Distributes remaining samples across non-neutral emotions.
            - Redistributes unused quotas from underrepresented classes.

        Returns:
            Balanced and filtered DataFrame preserving all original columns.
        """
        df = df.copy()

        # 1. Deterministic initial ordering
        id_cols = [c for c in ["dialogue", "utterance"] if c in df.columns]
        if id_cols:
            df = df.sort_values(id_cols)
        elif "audio_path" in df.columns:
            df = df.sort_values("audio_path")
        df = df.reset_index(drop=True)

        # 2. Validate audio file existence
        if "audio_path" not in df.columns:
            raise ValueError("DataFrame must contain 'audio_path'.")

        df = df[
            df["audio_path"].notna()
            & df["audio_path"].apply(lambda p: Path(str(p)).is_file())
        ].copy()
        logger.info("After file existence validation: %d samples", len(df))

        # 3. Audio duration filtering
        if "duration" not in df.columns:
            logger.info("Computing audio durations...")
            df["duration"] = df["audio_path"].apply(get_audio_duration)

        failed_duration = df["duration"].isna().sum()
        if failed_duration > 0:
            logger.warning("Could not determine duration for %d files.", failed_duration)

        df = df[df["duration"].notna()].copy()
        df = df[
            df["duration"].between(self.min_duration, self.max_duration, inclusive="both")
        ].copy()
        logger.info(
            "Duration filter [%.1fs, %.1fs]: %d samples remaining",
            self.min_duration,
            self.max_duration,
            len(df),
        )

        # 4. Normalize emotion labels
        df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
        df = df[df["emotion"].ne("nan")].copy()
        emotions = sorted(df["emotion"].unique())

        if not emotions:
            raise ValueError("No valid emotion classes remain after filtering.")

        # 5. Cap neutral class
        max_neutral_count = int(self.target_total * self.max_neutral_pct)
        non_neutral = [e for e in emotions if e != "neutral"]
        neutral_target = (
            min(max_neutral_count, len(df[df["emotion"] == "neutral"]))
            if "neutral" in emotions
            else 0
        )
        remaining_slots = self.target_total - neutral_target

        # 6. Initial non-neutral quotas
        quotas: Dict[str, int] = {}
        if non_neutral:
            base = remaining_slots // len(non_neutral)
            remainder = remaining_slots % len(non_neutral)
            for i, emotion in enumerate(non_neutral):
                quotas[emotion] = base + (1 if i < remainder else 0)

        if "neutral" in emotions:
            quotas["neutral"] = neutral_target

        # 7. Iteratively redistribute unavailable samples from saturated classes
        while True:
            deficit = 0
            saturated = set()

            for emotion, quota in quotas.items():
                available = df["emotion"].eq(emotion).sum()
                if available < quota:
                    deficit += quota - available
                    quotas[emotion] = available
                    saturated.add(emotion)

            if deficit == 0:
                break

            eligible = [
                emotion
                for emotion in non_neutral
                if emotion not in saturated and quotas[emotion] < df["emotion"].eq(emotion).sum()
            ]

            if not eligible:
                break

            base_extra = deficit // len(eligible)
            remainder = deficit % len(eligible)
            for i, emotion in enumerate(eligible):
                quotas[emotion] += base_extra + (1 if i < remainder else 0)

        # 8. Deterministic sampling per class
        sampled: List[pd.DataFrame] = []
        for emotion in sorted(quotas):
            n = quotas[emotion]
            if n == 0:
                continue
            class_df = df[df["emotion"] == emotion]
            sampled.append(
                class_df.sample(n=n, random_state=self.seed, replace=False)
            )

        if not sampled:
            raise ValueError("No samples available after balancing.")

        result = pd.concat(sampled, ignore_index=True)

        # 9. Final deterministic ordering
        if id_cols:
            result = result.sort_values(id_cols)
        result = result.reset_index(drop=True)

        logger.info(
            "Balanced Sampling Complete: Requested %d, Returned %d",
            self.target_total,
            len(result),
        )
        return result


def sample_balanced_meld(
    df: pd.DataFrame,
    target_total: int = 2000,
    max_neutral_pct: float = 0.15,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Functional wrapper for BalancedSampler."""
    sampler = BalancedSampler(
        target_total=target_total,
        max_neutral_pct=max_neutral_pct,
        min_duration=min_duration,
        max_duration=max_duration,
        seed=seed,
    )
    return sampler.sample(df)
