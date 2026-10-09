"""openSMILE eGeMAPSv02 feature extraction, caching, and acoustic inventory discretization."""

from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("acoustics.features")

# Default mapping of Acoustic Inventory descriptors to eGeMAPSv02 functionals
DEFAULT_ACOUSTIC_CUES: Dict[str, str] = {
    "Pitch Height (st)": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "Pitch Dynamics (st)": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "Vocal Energy (Sones)": "loudness_sma3_percentile50.0",
    "Speaking Rate (seg/s)": "VoicedSegmentsPerSec",
    "Spectral Tilt (dB)": "hammarbergIndexV_sma3nz_amean",
    "HNR (dB)": "HNRdBACF_sma3nz_amean",
    "Jitter Local": "jitterLocal_sma3nz_amean",
    "Shimmer Local (dB)": "shimmerLocaldB_sma3nz_amean",
}


class eGeMAPSExtractor:
    """Manages eGeMAPSv02 functional extraction and disk caching for SER corpora."""

    def __init__(
        self,
        cues_dict: Optional[Dict[str, str]] = None,
        cache_path: Optional[Path] = None,
        alt_cache_path: Optional[Path] = None,
    ):
        self.cues_dict = cues_dict or DEFAULT_ACOUSTIC_CUES
        self.target_features = list(self.cues_dict.values())
        self.cache_path = Path(cache_path) if cache_path else Path("/content/train_egemaps_features.csv")
        self.alt_cache_path = Path(alt_cache_path) if alt_cache_path else Path("/content/raw_train_egemaps_features.csv")

    def extract_or_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract missing acoustic features using openSMILE or load from precomputed cache.

        Args:
            df: DataFrame containing an 'audio_path' column.

        Returns:
            DataFrame with eGeMAPSv02 feature columns populated.
        """
        df = df.copy()
        missing = [col for col in self.target_features if col not in df.columns]

        if not missing:
            logger.info("All target eGeMAPS acoustic features already present in DataFrame.")
            return df

        # 1. Attempt loading from cache
        active_cache = None
        if self.cache_path.exists():
            active_cache = self.cache_path
        elif self.alt_cache_path.exists():
            active_cache = self.alt_cache_path

        if active_cache is not None:
            logger.info("Loading cached eGeMAPSv02 features from %s...", active_cache)
            df_cached = pd.read_csv(active_cache)
            if "audio_path" in df_cached.columns:
                cols_to_merge = ["audio_path"] + [
                    c for c in self.target_features if c in df_cached.columns and c not in df.columns
                ]
                df = df.merge(
                    df_cached[cols_to_merge].drop_duplicates(subset=["audio_path"]),
                    on="audio_path",
                    how="left",
                )
            else:
                cols_to_add = [c for c in missing if c in df_cached.columns]
                df = pd.concat(
                    [df.reset_index(drop=True), df_cached[cols_to_add].reset_index(drop=True)],
                    axis=1,
                )
            return df

        # 2. Extract using openSMILE
        logger.info("Extracting acoustic features via openSMILE eGeMAPSv02 Functionals...")
        try:
            import opensmile
        except ImportError:
            logger.warning("openSMILE not installed. Installing via pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opensmile"], check=True)
            import opensmile

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        extracted_records = []
        valid_indices = []

        try:
            from tqdm.auto import tqdm
            iterator = tqdm(df.iterrows(), total=len(df), desc="Extracting eGeMAPS")
        except ImportError:
            iterator = df.iterrows()

        for idx, row in iterator:
            p = Path(str(row["audio_path"]))
            if p.is_file():
                try:
                    feat = smile.process_file(str(p))
                    extracted_records.append(feat.iloc[0].to_dict())
                    valid_indices.append(idx)
                except Exception:
                    pass

        if extracted_records:
            df_feats = pd.DataFrame(extracted_records, index=valid_indices)
            for c in self.target_features:
                if c in df_feats.columns:
                    df.loc[valid_indices, c] = df_feats[c]

            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                df_feats["audio_path"] = df.loc[valid_indices, "audio_path"].values
                df_feats.to_csv(self.cache_path, index=False)
                df_feats.to_csv(self.alt_cache_path, index=False)
                logger.info("Acoustic features successfully cached to %s", self.cache_path)
            except Exception as e:
                logger.warning("Could not write cache file: %s", e)
        else:
            logger.warning("No audio files could be processed. Verify MELD audio path extraction.")

        return df
