"""R2: Acoustic Inventory physical grounding reward.

Evaluates whether the discrete acoustic cues generated during Chain-of-Thought
reasoning accurately ground to empirical physical eGeMAPS functionals.
"""

from typing import Any, Dict, List, Optional

from src.rewards.parser import extract_acoustic_inventory
from src.utils.logging import get_logger

logger = get_logger("rewards.acoustic")

# Literature-derived acoustic profiles for speech emotions (Scherer et al., Juslin et al.)
PHONETIC_PROFILES: Dict[str, Dict[str, List[str]]] = {
    "anger": {
        "pitch_height": ["elevated"],
        "pitch_dynamics": ["wide", "moderate"],
        "vocal_energy": ["loud"],
        "speaking_rate": ["fast", "moderate"],
        "voice_quality": ["pressed/tense", "harsh/creaky"],
    },
    "joy": {
        "pitch_height": ["elevated", "moderate"],
        "pitch_dynamics": ["wide"],
        "vocal_energy": ["loud", "moderate"],
        "speaking_rate": ["fast", "moderate"],
        "voice_quality": ["modal/normal", "breathy"],
    },
    "sadness": {
        "pitch_height": ["low", "moderate"],
        "pitch_dynamics": ["narrow"],
        "vocal_energy": ["quiet"],
        "speaking_rate": ["slow"],
        "voice_quality": ["breathy", "modal/normal"],
    },
    "fear": {
        "pitch_height": ["elevated"],
        "pitch_dynamics": ["wide", "narrow"],
        "vocal_energy": ["moderate", "loud"],
        "speaking_rate": ["fast"],
        "voice_quality": ["breathy", "pressed/tense"],
    },
    "disgust": {
        "pitch_height": ["low", "moderate"],
        "pitch_dynamics": ["narrow", "moderate"],
        "vocal_energy": ["moderate", "quiet"],
        "speaking_rate": ["slow", "moderate"],
        "voice_quality": ["harsh/creaky", "pressed/tense"],
    },
    "surprise": {
        "pitch_height": ["elevated"],
        "pitch_dynamics": ["wide"],
        "vocal_energy": ["loud", "moderate"],
        "speaking_rate": ["fast", "moderate"],
        "voice_quality": ["modal/normal", "breathy"],
    },
    "neutral": {
        "pitch_height": ["moderate"],
        "pitch_dynamics": ["moderate", "narrow"],
        "vocal_energy": ["moderate"],
        "speaking_rate": ["moderate"],
        "voice_quality": ["modal/normal"],
    },
}


class AcousticInventoryReward:
    """Computes physical acoustic consistency reward in [0.0, 1.0].

    Can operate in two complementary modes:
    1. Direct physical grounding: compares predicted discrete cues against
       empirical eGeMAPS quantiles (Q33 and Q66 thresholds).
    2. Phonetic coherence: compares predicted discrete cues against target emotion
       acoustic literature profiles.
    """

    def __init__(
        self,
        quantiles_ref: Optional[Dict[str, Dict[str, float]]] = None,
        weight_per_cue: float = 0.20,
    ):
        self.quantiles_ref = quantiles_ref or {}
        self.weight_per_cue = weight_per_cue

    def _score_single(
        self,
        parsed_inventory: Dict[str, Optional[str]],
        target_emotion: Optional[str] = None,
        f0_val: Optional[float] = None,
        loudness_val: Optional[float] = None,
        rate_val: Optional[float] = None,
    ) -> float:
        """Score a single rollout's acoustic inventory."""
        score = 0.0

        # Mode A: Grounded against continuous eGeMAPS values if available
        if f0_val is not None and "Pitch Height (st)" in self.quantiles_ref:
            q_f0 = self.quantiles_ref["Pitch Height (st)"]
            expected_f0 = (
                "low" if f0_val < q_f0.get("Q33", 25.0)
                else ("elevated" if f0_val > q_f0.get("Q66", 35.0) else "moderate")
            )
            if parsed_inventory.get("pitch_height") == expected_f0:
                score += self.weight_per_cue

        if loudness_val is not None and "Vocal Energy (Sones)" in self.quantiles_ref:
            q_loud = self.quantiles_ref["Vocal Energy (Sones)"]
            expected_loud = (
                "quiet" if loudness_val < q_loud.get("Q33", 0.3)
                else ("loud" if loudness_val > q_loud.get("Q66", 0.7) else "moderate")
            )
            if parsed_inventory.get("vocal_energy") == expected_loud:
                score += self.weight_per_cue

        if rate_val is not None and "Speaking Rate (seg/s)" in self.quantiles_ref:
            q_rate = self.quantiles_ref["Speaking Rate (seg/s)"]
            expected_rate = (
                "slow" if rate_val < q_rate.get("Q33", 2.0)
                else ("fast" if rate_val > q_rate.get("Q66", 3.5) else "moderate")
            )
            if parsed_inventory.get("speaking_rate") == expected_rate:
                score += self.weight_per_cue

        # Mode B: Fallback or augment with phonetic profile coherence
        if score == 0.0 and target_emotion and target_emotion in PHONETIC_PROFILES:
            profile = PHONETIC_PROFILES[target_emotion]
            matches = 0
            for cue_key, expected_opts in profile.items():
                if parsed_inventory.get(cue_key) in expected_opts:
                    matches += 1
            score = round(matches * self.weight_per_cue, 3)

        return min(1.0, round(score, 3))

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        label: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Compute acoustic inventory grounding rewards for rollout batch."""
        rewards: List[float] = []
        labels = label if label is not None else [None] * len(completions)

        f0_list = kwargs.get("F0semitoneFrom27.5Hz_sma3nz_amean", [None] * len(completions))
        loud_list = kwargs.get("loudness_sma3_percentile50.0", [None] * len(completions))
        rate_list = kwargs.get("VoicedSegmentsPerSec", [None] * len(completions))

        for idx, (comp, target) in enumerate(zip(completions, labels)):
            inventory = extract_acoustic_inventory(comp)
            f0 = f0_list[idx] if idx < len(f0_list) else None
            loud = loud_list[idx] if idx < len(loud_list) else None
            rate = rate_list[idx] if idx < len(rate_list) else None

            target_norm = str(target).lower().strip() if target else None
            score = self._score_single(
                inventory,
                target_emotion=target_norm,
                f0_val=f0,
                loudness_val=loud,
                rate_val=rate,
            )
            rewards.append(score)

        return rewards
