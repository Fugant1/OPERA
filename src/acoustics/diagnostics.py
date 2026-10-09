"""Speaker quantile validity, volume distribution, and acoustic bandwidth diagnostics."""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("acoustics.diagnostics")


class SpeakerDiagnostics:
    """Evaluates the statistical validity of speaker-conditioned quantiles and emotional diversity."""

    def __init__(
        self,
        min_utterances_threshold: int = 8,
        label_col: str = "emotion",
        f0_col: str = "F0semitoneFrom27.5Hz_sma3nz_amean",
    ):
        self.min_utterances_threshold = min_utterances_threshold
        self.label_col = label_col
        self.f0_col = f0_col

    def _shannon_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy with base 2."""
        probs = probabilities[probabilities > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def diagnose(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Perform comprehensive speaker volume, bias, and acoustic bandwidth diagnosis.

        Returns:
            Dictionary with:
            - 'volume_distribution': Summary of speaker tail distribution.
            - 'emotional_diversity': Per-speaker dominant emotion, percentage, and normalized entropy.
            - 'bandwidth_collapse': Pitch dynamic range bandwidth check for eligible speakers.
        """
        if "speaker" not in df.columns:
            raise ValueError("DataFrame must contain 'speaker' column.")

        total_samples = len(df)
        spk_counts = df["speaker"].value_counts()
        num_unique_speakers = len(spk_counts)

        # 1. Volume / Long-tail volume analysis
        bins = [0, 1, 3, 7, 20, np.inf]
        bin_labels = [
            "1 utterance (Degenerate)",
            "2-3 utterances (Critical)",
            f"4-{self.min_utterances_threshold - 1} utterances (Below threshold)",
            f"{self.min_utterances_threshold}-20 utterances (Borderline)",
            "> 20 utterances (Statistically stable)",
        ]

        binned_speakers = pd.cut(spk_counts, bins=bins, labels=bin_labels)
        spk_dist = pd.DataFrame({
            "Speaker Count": binned_speakers.value_counts(sort=False),
            "% of Speakers": (binned_speakers.value_counts(sort=False, normalize=True) * 100).round(2),
        })

        utterance_sums = []
        for interval_label in bin_labels:
            matching_spks = spk_counts[binned_speakers == interval_label].index
            utterance_sums.append(df[df["speaker"].isin(matching_spks)].shape[0])

        spk_dist["Utterances Affected"] = utterance_sums
        spk_dist["% of Corpus"] = ((spk_dist["Utterances Affected"] / total_samples) * 100).round(2)

        speakers_below_threshold = (spk_counts < self.min_utterances_threshold).sum()
        pct_fallback = (
            df["speaker"].isin(spk_counts[spk_counts < self.min_utterances_threshold].index).sum()
            / total_samples
        ) * 100

        logger.info(
            "Speaker Volume: %d total unique speakers | %d (%.1f%%) below %d utterances | Fallback impact: %.2f%%",
            num_unique_speakers,
            speakers_below_threshold,
            (speakers_below_threshold / num_unique_speakers * 100),
            self.min_utterances_threshold,
            pct_fallback,
        )

        # 2. Emotional diversity / sampling bias
        label_col = self.label_col if self.label_col in df.columns else ("label" if "label" in df.columns else None)
        df_div = pd.DataFrame()

        if label_col:
            num_classes = df[label_col].nunique()
            max_entropy = math.log2(num_classes) if num_classes > 1 else 1.0

            diversity_records = []
            for spk in spk_counts.index:
                sub = df[df["speaker"] == spk]
                counts = sub[label_col].value_counts()
                probs = counts.values / len(sub)
                shannon_ent = self._shannon_entropy(probs) / max_entropy if max_entropy > 0 else 0.0

                dominant_emo = counts.index[0]
                dominant_pct = (counts.iloc[0] / len(sub)) * 100

                diversity_records.append({
                    "speaker": spk,
                    "total_utterances": len(sub),
                    "dominant_emotion": dominant_emo,
                    "dominant_pct": round(dominant_pct, 1),
                    "normalized_entropy": round(shannon_ent, 3),
                    "status_calibration": (
                        "Individual" if len(sub) >= self.min_utterances_threshold else "Global Fallback"
                    ),
                })
            df_div = pd.DataFrame(diversity_records)

        # 3. Bandwidth collapse verification in pitch
        df_iqr = pd.DataFrame()
        if self.f0_col in df.columns:
            eligible_speakers = spk_counts[spk_counts >= self.min_utterances_threshold].index
            iqr_records = []
            for spk in eligible_speakers[:15]:
                sub_f0 = df[df["speaker"] == spk][self.f0_col].dropna()
                if len(sub_f0) < 2:
                    continue
                q33 = sub_f0.quantile(0.33)
                q66 = sub_f0.quantile(0.66)
                band_width = q66 - q33
                iqr_records.append({
                    "speaker": spk,
                    "n_utterances": len(sub_f0),
                    "Q33_semitones": round(q33, 2),
                    "Q66_semitones": round(q66, 2),
                    "bandwidth_moderate": round(band_width, 2),
                    "is_collapsed": band_width < 1.0,
                })
            df_iqr = pd.DataFrame(iqr_records)

        return {
            "volume_distribution": spk_dist,
            "emotional_diversity": df_div,
            "bandwidth_collapse": df_iqr,
        }
