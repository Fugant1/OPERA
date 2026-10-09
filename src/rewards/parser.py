"""Robust output parsing for multi-turn Chain-of-Thought rollouts."""

import re
from typing import Dict, List, Optional, Set

VALID_EMOTIONS: Set[str] = {
    "anger",
    "joy",
    "sadness",
    "surprise",
    "fear",
    "disgust",
    "neutral",
}

LABEL_SYNONYMS: Dict[str, str] = {
    "happy": "joy",
    "happiness": "joy",
    "sad": "sadness",
    "mad": "anger",
    "angry": "anger",
    "scared": "fear",
    "fearful": "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
}

REQUIRED_INVENTORY_KEYS: List[str] = [
    "pitch_height",
    "pitch_dynamics",
    "vocal_energy",
    "speaking_rate",
    "voice_quality",
]


class CompletionParser:
    """Parses structural elements from model completions:

    - XML reasoning <think> and decision <answer> tags
    - [Acoustic Inventory] physical cues
    - [Prosodic Reasoning] rationale
    - <answer> predicted emotion label
    """

    @staticmethod
    def extract_acoustic_inventory(completion: str) -> Dict[str, Optional[str]]:
        """Extract and normalize all 5 physical acoustic cue descriptors."""
        section_match = re.search(
            r"\[Acoustic Inventory\](.*?)(?:\[Prosodic Reasoning\]|</think>|$)",
            completion,
            re.DOTALL | re.IGNORECASE,
        )
        target_text = section_match.group(1) if section_match else completion

        patterns = {
            "pitch_height": (
                r"Pitch Height\s*:\s*(low|moderate|elevated)\b",
                {"low": "low", "moderate": "moderate", "elevated": "elevated"},
            ),
            "pitch_dynamics": (
                r"Pitch Dynamics\s*:\s*(narrow|moderate|wide)\b",
                {"narrow": "narrow", "moderate": "moderate", "wide": "wide"},
            ),
            "vocal_energy": (
                r"Vocal Energy\s*:\s*(quiet|moderate|loud)\b",
                {"quiet": "quiet", "moderate": "moderate", "loud": "loud"},
            ),
            "speaking_rate": (
                r"Speaking Rate\s*:\s*(slow|moderate|fast)\b",
                {"slow": "slow", "moderate": "moderate", "fast": "fast"},
            ),
            "voice_quality": (
                r"Voice Quality\s*:\s*(pressed/tense|pressed|tense|breathy|harsh/creaky|harsh|creaky|modal/normal|modal|normal)\b",
                {
                    "pressed/tense": "pressed/tense",
                    "pressed": "pressed/tense",
                    "tense": "pressed/tense",
                    "breathy": "breathy",
                    "harsh/creaky": "harsh/creaky",
                    "harsh": "harsh/creaky",
                    "creaky": "harsh/creaky",
                    "modal/normal": "modal/normal",
                    "modal": "modal/normal",
                    "normal": "modal/normal",
                },
            ),
        }

        parsed: Dict[str, Optional[str]] = {}
        for key, (pattern, canonical_map) in patterns.items():
            match = re.search(pattern, target_text, re.IGNORECASE)
            if match:
                raw_val = match.group(1).lower().strip()
                parsed[key] = canonical_map.get(raw_val, raw_val)
            else:
                parsed[key] = None

        return parsed

    @staticmethod
    def extract_reasoning(completion: str) -> str:
        """Extract text within [Prosodic Reasoning] excluding placeholder instructions."""
        match = re.search(
            r"\[Prosodic Reasoning\](.*?)(?:</think>|<answer>|$)",
            completion,
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return ""

        reasoning = match.group(1).strip()
        placeholder = "(Deduce the emotion from the cues above)"
        if reasoning.startswith(placeholder):
            reasoning = reasoning[len(placeholder) :].strip()

        return reasoning

    @staticmethod
    def extract_answer_content(completion: str) -> str:
        """Extract and clean predicted emotion label from <answer> tags."""
        match = re.search(
            r"<answer>(.*?)(?:</answer>|$)",
            completion,
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return ""

        raw_text = match.group(1).strip().lower()
        cleaned = re.sub(r"[\[\]\"\'\*\.]", "", raw_text).strip()

        if cleaned in VALID_EMOTIONS:
            return cleaned

        for emotion in VALID_EMOTIONS:
            if re.search(rf"\b{re.escape(emotion)}\b", cleaned):
                return emotion

        return cleaned


# Global convenience functional aliases
extract_acoustic_inventory = CompletionParser.extract_acoustic_inventory
extract_reasoning = CompletionParser.extract_reasoning
extract_answer_content = CompletionParser.extract_answer_content
