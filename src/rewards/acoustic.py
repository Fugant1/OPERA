"""R2: Acoustic Inventory physical grounding reward.

Evaluates whether the discrete acoustic cues generated during Chain-of-Thought
reasoning accurately ground to empirical physical eGeMAPS functionals using
speaker-calibrated quartiles for the top speakers.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import pandas as pd

from src.rewards.parser import extract_acoustic_inventory
from src.utils.logging import get_logger

logger = get_logger("rewards.acoustic")

# Mapping of the 4 continuous acoustic descriptors in the prompt to eGeMAPSv02 features
CUE_FEATURE_MAP: Dict[str, str] = {
    "pitch_height": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "pitch_dynamics": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "vocal_energy": "loudness_sma3_percentile50.0",
    "speaking_rate": "VoicedSegmentsPerSec",
}

# Normalization of prompt vocabulary to the 3 ordinal tiers (low, moderate, high)
PRED_TIER_MAP: Dict[str, Dict[str, str]] = {
    "pitch_height": {
        "low": "low",
        "moderate": "moderate",
        "elevated": "high",
        "high": "high",
    },
    "pitch_dynamics": {
        "narrow": "low",
        "low": "low",
        "moderate": "moderate",
        "wide": "high",
        "high": "high",
    },
    "vocal_energy": {
        "quiet": "low",
        "low": "low",
        "moderate": "moderate",
        "loud": "high",
        "high": "high",
    },
    "speaking_rate": {
        "slow": "low",
        "low": "low",
        "moderate": "moderate",
        "fast": "high",
        "high": "high",
    },
}


def compute_top_speaker_acoustic_quantiles(
    df: Any,
    top_n: int = 6,
    cues_map: Dict[str, str] = CUE_FEATURE_MAP,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Computes the 4 quantiles (quartiles: Q25 and Q75) for the top N most frequent speakers.

    Quantile divisions:
      - 1st quantile (x <= Q25): Low
      - 2nd & 3rd quantiles (Q25 < x <= Q75): Moderate
      - 4th quantile (x > Q75): High

    Includes '__GLOBAL__' as fallback for speakers outside top N.
    """
    spk_counts = df["speaker"].value_counts()
    top_speakers = spk_counts.head(top_n).index.tolist()

    quantiles_dict: Dict[str, Dict[str, Dict[str, float]]] = {}

    for spk in top_speakers:
        sub = df[df["speaker"] == spk]
        quantiles_dict[spk] = {}
        for cue_name, col_name in cues_map.items():
            if col_name in sub.columns:
                series = sub[col_name].dropna()
                if len(series) > 0:
                    quantiles_dict[spk][cue_name] = {
                        "q25": float(series.quantile(0.25)),
                        "q75": float(series.quantile(0.75)),
                    }

    # Global fallback for speakers with few utterances or out-of-top-N
    quantiles_dict["__GLOBAL__"] = {}
    for cue_name, col_name in cues_map.items():
        if col_name in df.columns:
            series = df[col_name].dropna()
            if len(series) > 0:
                quantiles_dict["__GLOBAL__"][cue_name] = {
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                }

    logger.info("Computed acoustic quartiles for top %d speakers: %s", len(top_speakers), top_speakers)
    return quantiles_dict


def score_acoustic_piece(truth_tier: str, pred_tier: Optional[str]) -> float:
    """Computes the reward score for an individual acoustic dimension according to the payoff matrix:

      - Truth High/Low ; Predicted High/Low -> +1.0
      - Truth High/Low ; Predicted Moderate -> +0.0
      - Truth Moderate ; Predicted Moderate -> +0.5
      - Truth High     ; Predicted Low      -> -1.0
      - Truth Low      ; Predicted High     -> -1.0
      - Truth Moderate ; Predicted High/Low -> 0.0
      - Predicted Missing / None            -> -1.0
    """
    if pred_tier is None:
        return -1.0

    t = truth_tier.lower()
    p = pred_tier.lower()

    if t in ("high", "low"):
        if p == t:
            return 1.0
        elif p == "moderate":
            return 0.0
        else:
            return -1.0
    elif t == "moderate":
        if p == "moderate":
            return 0.5
        else:
            return 0.0

    return 0.0


class AcousticInventoryReward:
    """R2 Reward function evaluating Acoustic Inventory grounding against speaker quartiles."""

    def __init__(
        self,
        quantiles_ref: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        audio_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        normalize: bool = False,
    ):
        self.quantiles_ref = quantiles_ref or {}
        self.audio_lookup = audio_lookup or {}
        self.normalize = normalize

    def set_quantiles(self, quantiles_ref: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Update active speaker quantiles reference."""
        self.quantiles_ref = quantiles_ref

    def set_audio_lookup(self, audio_lookup: Dict[str, Dict[str, Any]]) -> None:
        """Update audio_path -> feature cache lookup."""
        self.audio_lookup = audio_lookup

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Compute R2 acoustic inventory rewards for a batch of GRPO rollouts.

        Returns:
            List of float rewards per completion.
        """
        rewards: List[float] = []

        speakers = kwargs.get("speaker", [None] * len(completions))
        audio_paths = kwargs.get("audio_path", [None] * len(completions))

        for idx, comp in enumerate(completions):
            parsed_inv = extract_acoustic_inventory(comp)

            spk = speakers[idx] if idx < len(speakers) else None
            audio_p = audio_paths[idx] if idx < len(audio_paths) else None

            lookup_data = self.audio_lookup.get(audio_p, {}) if audio_p else {}
            if spk is None:
                spk = lookup_data.get("speaker", "__GLOBAL__")

            spk_q = self.quantiles_ref.get(
                spk, self.quantiles_ref.get("__GLOBAL__", {})
            )

            cue_scores: Dict[str, float] = {}

            for cue_name, feat_col in CUE_FEATURE_MAP.items():
                val = None
                if feat_col in kwargs and idx < len(kwargs[feat_col]):
                    val = kwargs[feat_col][idx]
                elif feat_col in lookup_data:
                    val = lookup_data[feat_col]

                if val is None or cue_name not in spk_q:
                    continue

                q25 = spk_q[cue_name]["q25"]
                q75 = spk_q[cue_name]["q75"]

                # Physical truth quartile classification
                if val <= q25:
                    truth_tier = "low"
                elif val <= q75:
                    truth_tier = "moderate"
                else:
                    truth_tier = "high"

                # Parse and map model prediction
                raw_pred = parsed_inv.get(cue_name)
                pred_tier = None
                if raw_pred:
                    raw_clean = str(raw_pred).lower().strip()
                    pred_tier = PRED_TIER_MAP.get(cue_name, {}).get(raw_clean)

                cue_scores[cue_name] = score_acoustic_piece(truth_tier, pred_tier)

            if cue_scores:
                total = sum(cue_scores.values())
                score = (total / len(cue_scores)) if self.normalize else total
                rewards.append(round(float(score), 3))
            else:
                rewards.append(0.0)

        return rewards


def reward_acoustic_inventory(
    prompts: List[str],
    completions: List[str],
    quantiles_ref: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    audio_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    normalize: bool = False,
    **kwargs: Any,
) -> List[float]:
    """Functional wrapper for AcousticInventoryReward."""
    rewarder = AcousticInventoryReward(
        quantiles_ref=quantiles_ref,
        audio_lookup=audio_lookup,
        normalize=normalize,
    )
    return rewarder(prompts, completions, **kwargs)
