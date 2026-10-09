"""Audio utility functions for duration extraction and validation."""

from pathlib import Path
from typing import Optional


def get_audio_duration(audio_path: str | Path) -> Optional[float]:
    """Return audio duration in seconds, or None if the file cannot be read.

    Tries librosa first, falling back to soundfile or wave if available.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds, or None if failed.
    """
    path_str = str(audio_path)
    if not Path(path_str).is_file():
        return None

    try:
        import librosa
        return float(librosa.get_duration(path=path_str))
    except Exception:
        pass

    try:
        import soundfile as sf
        info = sf.info(path_str)
        return float(info.duration)
    except Exception:
        pass

    try:
        import wave
        with wave.open(path_str, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return float(frames) / float(rate)
    except Exception:
        return None
