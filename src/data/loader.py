"""MELD Dataset loader, archive extraction, and path resolution.

Loads audio files and sentiment/emotion metadata for the MELD corpus.
"""

from pathlib import Path
import subprocess
import tarfile
from typing import Optional, Union

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("data.loader")


class ArchiveExtractor:
    """Manages archive extraction and dataset discovery on disk."""

    @staticmethod
    def extract_tar(archive_path: Path, destination: Path) -> None:
        """Extract a .tar archive to destination."""
        archive_path = Path(archive_path)
        destination = Path(destination)

        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        destination.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting %s -> %s...", archive_path, destination)

        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(destination)

        logger.info("Extraction complete: %s", destination)

    @staticmethod
    def dataset_info(path: Path) -> None:
        """Print basic disk statistics for an extracted dataset."""
        path = Path(path)
        n_files = sum(1 for p in path.rglob("*") if p.is_file())
        try:
            size_out = subprocess.run(
                ["du", "-sh", str(path)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()[0]
        except Exception:
            size_out = "unknown"

        logger.info("Dataset Directory: %s | Files: %d | Total Size: %s", path, n_files, size_out)

    @staticmethod
    def find_file(root: Path, pattern: str) -> Path:
        """Find the first file matching a glob pattern."""
        matches = list(Path(root).rglob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching '{pattern}' in {root}")
        return matches[0]


class MELDLoader:
    """Loads and standardizes MELD dialogue and acoustic emotion data."""

    def __init__(
        self,
        archive_path: Optional[Path] = None,
        dataset_dir: Optional[Path] = None,
    ):
        self.archive_path = Path(archive_path) if archive_path else None
        self.dataset_dir = Path(dataset_dir) if dataset_dir else Path("./data/MELD")

    def load_split(self, split: str = "train") -> pd.DataFrame:
        """Load a single MELD split and resolve absolute audio paths.

        Expected files inside the split directory:
            {split}_preprocessed.csv
            {split}_sent_emo.csv
            *.wav

        Args:
            split: Split name (e.g., 'train', 'dev', 'test').

        Returns:
            Normalized pandas DataFrame with columns:
            ['dialogue', 'utterance', 'text', 'emotion', 'speaker', 'audio_path']
        """
        split_dir = self.dataset_dir / f"preprocessed_{split}_splits"
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        audio_files = list(split_dir.rglob("*.wav"))
        preprocessed_csv = ArchiveExtractor.find_file(split_dir, f"{split}_preprocessed.csv")
        emotion_csv = ArchiveExtractor.find_file(split_dir, f"{split}_sent_emo.csv")

        df = pd.read_csv(preprocessed_csv)
        df_emo = pd.read_csv(emotion_csv)

        # Resolve audio filenames -> absolute paths
        audio_map = {path.name: str(path.resolve()) for path in audio_files}

        df["audio_path"] = (
            df["audio_path"]
            .astype(str)
            .map(lambda x: Path(x).name)
            .map(audio_map)
        )

        # Normalize merge keys
        df["dialogue"] = df["dialogue"].astype(int)
        df["utterance"] = df["utterance"].astype(int)
        df_emo["Dialogue_ID"] = df_emo["Dialogue_ID"].astype(int)
        df_emo["Utterance_ID"] = df_emo["Utterance_ID"].astype(int)

        # Add MELD metadata
        df = df.merge(
            df_emo[["Dialogue_ID", "Utterance_ID", "Emotion", "Utterance", "Speaker"]],
            left_on=["dialogue", "utterance"],
            right_on=["Dialogue_ID", "Utterance_ID"],
            how="left",
        )

        df = (
            df.drop(columns=["Dialogue_ID", "Utterance_ID"])
            .rename(
                columns={
                    "Utterance": "text",
                    "Emotion": "emotion",
                    "Speaker": "speaker",
                }
            )
        )

        # Basic validations
        missing_audio = df["audio_path"].isna().sum()
        missing_emotion = df["emotion"].isna().sum()

        logger.info(
            "Split '%s' loaded: %d samples (Audio matched: %d/%d, Metadata matched: %d/%d)",
            split,
            len(df),
            len(df) - missing_audio,
            len(df),
            len(df) - missing_emotion,
            len(df),
        )

        return df

    def load(
        self,
        archive_path: Optional[Path] = None,
        destination: Optional[Path] = None,
        split: str = "train",
    ) -> pd.DataFrame:
        """Extract if needed and load the requested MELD split."""
        archive_path = Path(archive_path) if archive_path else self.archive_path
        destination = Path(destination) if destination else self.dataset_dir

        if not destination.exists():
            if archive_path and archive_path.is_file():
                ArchiveExtractor.extract_tar(archive_path, destination)
            else:
                raise FileNotFoundError(
                    f"Dataset directory '{destination}' not found and archive '{archive_path}' unavailable."
                )
        else:
            logger.info("Dataset directory already exists: %s", destination)

        ArchiveExtractor.dataset_info(destination)
        self.dataset_dir = destination
        return self.load_split(split=split)


def load_meld(
    archive_path: Optional[Union[str, Path]] = None,
    destination: Optional[Union[str, Path]] = None,
    split: str = "train",
) -> pd.DataFrame:
    """Convenience function to load a raw MELD dataset split."""
    loader = MELDLoader(
        archive_path=Path(archive_path) if archive_path else None,
        dataset_dir=Path(destination) if destination else None,
    )
    return loader.load(split=split)
