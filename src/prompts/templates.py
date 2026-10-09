"""Prompt templates, personas, and Chain-of-Thought (CoT) format definitions."""

from typing import Any, Dict, List

QWEN_PERSONA = (
    "You are an expert in speech emotion recognition and acoustic phonetics. "
    "You evaluate vocal affect strictly through physical acoustic properties."
)

COT_PROMPT = """Listen to the audio clip and predict the speaker's emotional state solely from acoustics.

Allowed emotions: anger, joy, sadness, surprise, fear, disgust, neutral

Instructions:
1. Complete [Acoustic Inventory] selecting one option per line.
2. In [Prosodic Reasoning], deduce which emotion best matches these physical cues and eliminate conflicting states.
3. Output only the predicted class inside <answer>.

Output strictly in this format:
<think>
[Acoustic Inventory]
- Pitch Height: low | moderate | elevated
- Pitch Dynamics: narrow | moderate | wide
- Vocal Energy: quiet | moderate | loud
- Speaking Rate: slow | moderate | fast
- Voice Quality: pressed/tense | breathy | harsh/creaky | modal/normal

[Prosodic Reasoning]
(Deduce the emotion from the cues above)
</think>
<answer>emotion</answer>"""


def build_chat_prompt(persona: str = QWEN_PERSONA, instruction: str = COT_PROMPT) -> str:
    """Build a complete formatted prompt combining persona and instruction."""
    return f"{persona}\n\n{instruction}"


def build_messages(
    audio_path: str,
    persona: str = QWEN_PERSONA,
    instruction: str = COT_PROMPT,
) -> List[Dict[str, Any]]:
    """Build multi-modal chat messages for Qwen Omni processor."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": persona}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": instruction},
            ],
        },
    ]
