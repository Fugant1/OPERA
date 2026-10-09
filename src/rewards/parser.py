"""Robust output parsing for multi-turn Chain-of-Thought rollouts."""

import re
from typing import Any, Dict, List, Optional, Set

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


import functools

@functools.lru_cache(maxsize=8192)
def parse_completion(completion: str) -> Dict[str, Any]:
    """Parse estruturado completo da completion em UMA ÚNICA PASSAGEM com cache LRU.

    Centraliza a extração de tags, inventário acústico, raciocínio prosódico e resposta,
    evitando que múltiplas funções de recompensa repitam regex sobre a mesma string.
    """
    if not isinstance(completion, str):
        completion = str(completion) if completion is not None else ""

    t_open = completion.find("<think>")
    t_close = completion.find("</think>")
    a_open = completion.find("<answer>")
    a_close = completion.find("</answer>")

    has_think = (t_open != -1 and t_close != -1)
    has_answer = (a_open != -1 and a_close != -1)
    valid_tag_order = (has_think and has_answer and t_open < t_close < a_open < a_close)

    inventory = extract_acoustic_inventory(completion)
    valid_slots_count = sum(1 for v in inventory.values() if v is not None)

    reasoning = extract_reasoning(completion)

    answer_raw = extract_answer_content(completion)
    answer_norm = LABEL_SYNONYMS.get(answer_raw, answer_raw) if "LABEL_SYNONYMS" in globals() else answer_raw
    is_valid_emotion = answer_norm in VALID_EMOTIONS

    return {
        "raw": completion,
        "has_think": has_think,
        "has_answer": has_answer,
        "valid_tag_order": valid_tag_order,
        "inventory": inventory,
        "valid_slots_count": valid_slots_count,
        "reasoning": reasoning,
        "answer_raw": answer_raw,
        "answer_norm": answer_norm,
        "is_valid_emotion": is_valid_emotion,
    }

