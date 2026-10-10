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


TOP_6_SPEAKERS = ["Joey", "Ross", "Rachel", "Phoebe", "Monica", "Chandler"]

STRICTLY_POSITIVE_ACOUSTIC_FEATURES = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "loudness_sma3_percentile50.0",
    "VoicedSegmentsPerSec",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
]

ALL_ACOUSTIC_COLS = STRICTLY_POSITIVE_ACOUSTIC_FEATURES + [
    "HNRdBACF_sma3nz_amean",
    "hammarbergIndexV_sma3nz_amean",
]


class TopSpeakerBalancedSampler:
    """Samples emotion-balanced speech utterances exclusively from calibrated top speakers,

    guaranteeing strict exclusion of validation audios, temporal bounds (1s-20s),
    and non-zero physical eGeMAPS acoustic functionals.
    """

    def __init__(
        self,
        top_speakers: Optional[List[str]] = None,
        min_duration: float = 1.0,
        max_duration: float = 20.0,
        max_neutral_pct: float = 0.15,
        target_total: Optional[int] = None,
        seed: int = 42,
    ):
        self.top_speakers = top_speakers or TOP_6_SPEAKERS
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_neutral_pct = max_neutral_pct
        self.target_total = target_total
        self.seed = seed

    def sample(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Execute filtering and balanced sampling across top speakers."""
        df = train_df.copy()

        # 1. Excluir estritamente áudios do conjunto de validação
        if val_df is not None and "audio_path" in val_df.columns and "audio_path" in df.columns:
            val_paths = set(val_df["audio_path"].dropna().astype(str))
            initial_count = len(df)
            df = df[~df["audio_path"].astype(str).isin(val_paths)].copy()
            logger.info("Validation exclusion: removed %d overlapping audio files", initial_count - len(df))

        # 2. Filtrar apenas os top speakers calibrados
        if "speaker" in df.columns:
            df = df[df["speaker"].isin(self.top_speakers)].copy()
            logger.info("Filtered to %d top speakers: %d samples remaining", len(self.top_speakers), len(df))

        # 3. Recorte temporal (1.0s a 20.0s)
        dur_col = "duration" if "duration" in df.columns else ("time" if "time" in df.columns else None)
        if dur_col is not None:
            df = df[df[dur_col].notna()].copy()
            df = df[df[dur_col].between(self.min_duration, self.max_duration, inclusive="both")].copy()
            logger.info("Duration bounds [%.1fs, %.1fs]: %d samples remaining", self.min_duration, self.max_duration, len(df))

        # 4. Recorte de eGeMAPS: métricas de áudio não podem estar zeradas nem nulas
        for col in ALL_ACOUSTIC_COLS:
            if col in df.columns:
                df = df[df[col].notna() & (df[col] != 0.0)].copy()
                if col in STRICTLY_POSITIVE_ACOUSTIC_FEATURES:
                    df = df[df[col] > 0.0].copy()
        logger.info("Non-zero acoustic filter: %d samples remaining", len(df))

        # 5. Normalizar rótulos de emoção
        df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
        emotions = sorted(df["emotion"].unique())
        non_neutral = [e for e in emotions if e != "neutral"]

        target_total = self.target_total
        if target_total is None or target_total > len(df):
            counts = df["emotion"].value_counts()
            min_non_neutral = min([counts.get(e, 0) for e in non_neutral]) if non_neutral else 0
            slots_non_neutral = min_non_neutral * len(non_neutral)
            neutral_avail = counts.get("neutral", 0)
            neutral_target = min(neutral_avail, int(slots_non_neutral * self.max_neutral_pct / (1.0 - self.max_neutral_pct)))
            target_total = slots_non_neutral + neutral_target

        max_neutral = int(target_total * self.max_neutral_pct)
        neutral_count = min(max_neutral, df["emotion"].eq("neutral").sum()) if "neutral" in emotions else 0
        remaining_slots = target_total - neutral_count

        quotas: Dict[str, int] = {}
        if non_neutral:
            base = remaining_slots // len(non_neutral)
            rem = remaining_slots % len(non_neutral)
            for i, emo in enumerate(non_neutral):
                quotas[emo] = base + (1 if i < rem else 0)
        if "neutral" in emotions:
            quotas["neutral"] = neutral_count

        # Redistribuição iterativa de cotas
        while True:
            deficit = 0
            saturated = set()
            for emo, quota in quotas.items():
                avail = df["emotion"].eq(emo).sum()
                if avail < quota:
                    deficit += (quota - avail)
                    quotas[emo] = avail
                    saturated.add(emo)
            if deficit == 0:
                break
            eligible = [e for e in non_neutral if e not in saturated and quotas[e] < df["emotion"].eq(e).sum()]
            if not eligible:
                break
            base_extra = deficit // len(eligible)
            rem_extra = deficit % len(eligible)
            for i, emo in enumerate(eligible):
                quotas[emo] += base_extra + (1 if i < rem_extra else 0)

        # 6. Amostragem determinística balanceada entre emoções e locutores
        sampled_dfs = []
        for emo, q in quotas.items():
            if q == 0:
                continue
            sub_emo = df[df["emotion"] == emo]
            spk_groups = []
            spk_quota = q // len(self.top_speakers)
            spk_rem = q % len(self.top_speakers)

            for s_idx, spk in enumerate(sorted(self.top_speakers)):
                spk_df = sub_emo[sub_emo["speaker"] == spk]
                needed = spk_quota + (1 if s_idx < spk_rem else 0)
                take = min(needed, len(spk_df))
                if take > 0:
                    spk_groups.append(spk_df.sample(n=take, random_state=self.seed, replace=False))

            sampled_emo = pd.concat(spk_groups) if spk_groups else pd.DataFrame()
            if len(sampled_emo) < q:
                missing = q - len(sampled_emo)
                already_idx = set(sampled_emo.index)
                remaining_pool = sub_emo[~sub_emo.index.isin(already_idx)]
                if len(remaining_pool) > 0:
                    fill = remaining_pool.sample(n=min(missing, len(remaining_pool)), random_state=self.seed, replace=False)
                    sampled_emo = pd.concat([sampled_emo, fill])

            sampled_dfs.append(sampled_emo)

        result = pd.concat(sampled_dfs, ignore_index=True)
        id_cols = [c for c in ["dialogue", "utterance"] if c in result.columns]
        if id_cols:
            result = result.sort_values(id_cols)
        elif "audio_path" in result.columns:
            result = result.sort_values("audio_path")
        result = result.reset_index(drop=True)

        logger.info("Top speaker balanced sampling completed: %d samples returned", len(result))
        return result


def sample_balanced_top_speakers(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    top_speakers: Optional[List[str]] = None,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    max_neutral_pct: float = 0.15,
    target_total: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience functional wrapper for TopSpeakerBalancedSampler."""
    sampler = TopSpeakerBalancedSampler(
        top_speakers=top_speakers,
        min_duration=min_duration,
        max_duration=max_duration,
        max_neutral_pct=max_neutral_pct,
        target_total=target_total,
        seed=seed,
    )
    return sampler.sample(train_df=train_df, val_df=val_df)
