"""Statistical analysis of acoustic features across emotions and speakers."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.acoustics.features import DEFAULT_ACOUSTIC_CUES
from src.utils.logging import get_logger

logger = get_logger("acoustics.analyzer")


class AcousticAnalyzer:
    """Computes global dynamic ranges, emotion-specific quantiles, and speaker amplitude coverage."""

    def __init__(
        self,
        cues_dict: Optional[Dict[str, str]] = None,
        top_n_speakers: int = 20,
    ):
        self.cues_dict = cues_dict or DEFAULT_ACOUSTIC_CUES
        self.top_n_speakers = top_n_speakers

    def analyze(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute complete distributional diagnosis of acoustic cues.

        Returns:
            Dictionary containing:
            - "global_summary": DataFrame of corpus-wide quantiles and amplitudes.
            - "emotion_quantiles": DataFrame of emotion-conditional quantiles.
            - "speaker_amplitudes": Top speakers' absolute dynamic range.
            - "speaker_relative_amplitudes": Top speakers' % coverage of global range.
        """
        active_cues = {label: col for label, col in self.cues_dict.items() if col in df.columns}
        if not active_cues:
            logger.warning("No acoustic cue columns found in DataFrame for analysis.")
            return {}

        # A. Global corpus summary
        global_rows = []
        global_amps = {}
        for label, col in active_cues.items():
            s = df[col].dropna()
            if len(s) == 0:
                continue
            q10, q25, q33, q50, q66, q75, q90 = s.quantile([0.10, 0.25, 0.33, 0.50, 0.66, 0.75, 0.90])
            min_v, max_v = s.min(), s.max()
            amp = max_v - min_v
            global_amps[label] = amp
            global_rows.append({
                "Metric": label,
                "Count": len(s),
                "Mean": round(s.mean(), 2),
                "Std": round(s.std(), 2),
                "Min": round(min_v, 2),
                "Q10": round(q10, 2),
                "Q25": round(q25, 2),
                "Q33": round(q33, 2),
                "Median (Q50)": round(q50, 2),
                "Q66": round(q66, 2),
                "Q75": round(q75, 2),
                "Q90": round(q90, 2),
                "Max": round(max_v, 2),
                "Global Amplitude": round(amp, 2),
                "Global IQR": round(q75 - q25, 2),
            })
        df_global = pd.DataFrame(global_rows).set_index("Metric")

        # B. Emotion-conditioned quantiles
        emo_rows = []
        if "emotion" in df.columns:
            for emo, emo_df in df.groupby("emotion"):
                for label, col in active_cues.items():
                    s = emo_df[col].dropna()
                    if len(s) == 0:
                        continue
                    q10, q25, q33, q50, q66, q75, q90 = s.quantile(
                        [0.10, 0.25, 0.33, 0.50, 0.66, 0.75, 0.90]
                    )
                    min_v, max_v = s.min(), s.max()
                    emo_rows.append({
                        "Emotion": emo,
                        "Metric": label,
                        "Utterances": len(s),
                        "Mean": round(s.mean(), 2),
                        "Std": round(s.std(), 2),
                        "Min": round(min_v, 2),
                        "Q25": round(q25, 2),
                        "Q33": round(q33, 2),
                        "Median (Q50)": round(q50, 2),
                        "Q66": round(q66, 2),
                        "Q75": round(q75, 2),
                        "Max": round(max_v, 2),
                        "Amplitude": round(max_v - min_v, 2),
                        "IQR": round(q75 - q25, 2),
                    })
        df_emotion_quantiles = pd.DataFrame(emo_rows)

        # C. Speaker dynamic range coverage
        df_top_amplitudes = pd.DataFrame()
        df_top_relative_amplitudes = pd.DataFrame()

        if "speaker" in df.columns:
            spk_counts = df["speaker"].value_counts()
            top_speakers = spk_counts.head(self.top_n_speakers).index.tolist()

            spk_amp_rows = []
            spk_ratio_rows = []

            for spk in top_speakers:
                sub = df[df["speaker"] == spk]
                n_utt = len(sub)
                amp_entry = {"Speaker": spk, "Utterances": n_utt}
                ratio_entry = {"Speaker": f"{spk} (n={n_utt})"}

                for label, col in active_cues.items():
                    s = sub[col].dropna()
                    g_amp = global_amps.get(label, 0)
                    if len(s) > 1:
                        s_amp = s.max() - s.min()
                        amp_entry[label] = round(s_amp, 2)
                        ratio_entry[label] = round((s_amp / g_amp) * 100, 1) if g_amp > 0 else 0.0
                    else:
                        amp_entry[label] = 0.0
                        ratio_entry[label] = 0.0

                spk_amp_rows.append(amp_entry)
                spk_ratio_rows.append(ratio_entry)

            df_top_amplitudes = pd.DataFrame(spk_amp_rows).set_index("Speaker")
            df_top_relative_amplitudes = pd.DataFrame(spk_ratio_rows).set_index("Speaker")

            metric_cols = [c for c in df_top_amplitudes.columns if c != "Utterances"]
            df_global["Mean Speaker Amp"] = [
                round(df_top_amplitudes[m].mean(), 2) for m in df_global.index if m in metric_cols
            ]
            df_global["Speaker/Global Amp (%)"] = [
                round((df_top_amplitudes[m].mean() / df_global.loc[m, "Global Amplitude"]) * 100, 1)
                for m in df_global.index if m in metric_cols
            ]

        return {
            "global_summary": df_global,
            "emotion_quantiles": df_emotion_quantiles,
            "speaker_amplitudes": df_top_amplitudes,
            "speaker_relative_amplitudes": df_top_relative_amplitudes,
        }
