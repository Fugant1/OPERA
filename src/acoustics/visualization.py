"""Research visualizer for acoustic distributions and speaker dynamic range coverage."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.acoustics.features import DEFAULT_ACOUSTIC_CUES
from src.utils.logging import get_logger

logger = get_logger("acoustics.visualization")


class AcousticVisualizer:
    """Generates publication-quality figures for acoustic variability and speaker coverage."""

    def __init__(self, cues_dict: Optional[Dict[str, str]] = None):
        self.cues_dict = cues_dict or DEFAULT_ACOUSTIC_CUES

    def plot_emotion_acoustic_distributions(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
        show: bool = False,
    ):
        """Figure 1: Grid of boxplots of acoustic cues across emotions."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib or seaborn not installed. Skipping plot generation.")
            return

        active_cues = {label: col for label, col in self.cues_dict.items() if col in df.columns}
        if not active_cues or "emotion" not in df.columns:
            logger.warning("Cannot plot distributions: missing active cues or emotion column.")
            return

        n_cues = len(active_cues)
        cols = 2
        rows = (n_cues + cols - 1) // cols

        plt.style.use(
            "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
        )
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows))
        axes = np.array(axes).flatten()
        emo_order = sorted(df["emotion"].unique())

        for idx, (label, col) in enumerate(active_cues.items()):
            ax = axes[idx]
            sns.boxplot(
                data=df,
                x="emotion",
                y=col,
                order=emo_order,
                hue="emotion",
                legend=False,
                palette="Set2",
                ax=ax,
                fliersize=2,
                linewidth=1.2,
            )
            ax.set_title(f"Acoustic Variation Across Emotions: {label}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Emotion", fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.tick_params(axis="x", rotation=20)

        for idx in range(n_cues, len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(
            "Distributions and Quantiles of Physical Acoustic Cues Across Emotions",
            fontsize=14,
            fontweight="bold",
            y=1.002,
        )
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Saved Emotion Acoustic Distributions plot to %s", output_path)

        if show:
            plt.show()
        plt.close(fig)

    def plot_speaker_amplitude_coverage(
        self,
        df_relative_amplitudes: pd.DataFrame,
        top_n_speakers: int = 20,
        output_path: Optional[Path] = None,
        show: bool = False,
    ):
        """Figure 2: Heatmap and distribution of speaker dynamic range coverage vs global amplitude."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib or seaborn not installed. Skipping plot generation.")
            return

        if df_relative_amplitudes.empty:
            logger.warning("Relative amplitudes DataFrame is empty. Skipping plot.")
            return

        plt.style.use(
            "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
        )
        fig, (ax_heat, ax_dist) = plt.subplots(
            1, 2, figsize=(22, 10), gridspec_kw={"width_ratios": [1.3, 1]}
        )

        # Subplot A: Heatmap of % global amplitude coverage
        sns.heatmap(
            df_relative_amplitudes,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            cbar_kws={"label": "Speaker Amplitude / Global Amplitude (%)"},
            vmin=20,
            vmax=100,
            ax=ax_heat,
            linewidths=0.5,
        )
        ax_heat.set_title(
            f"Top {top_n_speakers} Speakers: Dynamic Range Coverage (% of Global Amplitude)",
            fontsize=13,
            fontweight="bold",
        )
        ax_heat.set_ylabel("Speaker (Utterances)", fontsize=11)
        ax_heat.tick_params(axis="x", rotation=25)

        # Subplot B: Distribution of speaker coverage across metrics
        df_melted = df_relative_amplitudes.melt(var_name="Metric", value_name="Relative Amplitude (%)")
        sns.boxplot(
            data=df_melted,
            x="Metric",
            y="Relative Amplitude (%)",
            hue="Metric",
            legend=False,
            palette="mako",
            ax=ax_dist,
            linewidth=1.2,
        )
        sns.stripplot(
            data=df_melted,
            x="Metric",
            y="Relative Amplitude (%)",
            color="crimson",
            alpha=0.6,
            jitter=0.2,
            size=5,
            ax=ax_dist,
        )
        ax_dist.axhline(100.0, color="red", linestyle="--", linewidth=1.8, label="Global Amplitude (100%)")
        ax_dist.set_ylim(0, 110)
        ax_dist.set_title(
            "Distribution of Speaker Coverage (%) Across Acoustic Metrics",
            fontsize=13,
            fontweight="bold",
        )
        ax_dist.set_xlabel("Acoustic Metric", fontsize=11)
        ax_dist.set_ylabel("Speaker Amplitude / Global Amplitude (%)", fontsize=11)
        ax_dist.tick_params(axis="x", rotation=25)
        ax_dist.legend(loc="lower right", fontsize=10)

        fig.suptitle(
            f"Amplitude Analysis: Top {top_n_speakers} Most Frequent Speakers vs. Global Dynamic Range",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Saved Speaker Amplitude Coverage plot to %s", output_path)

        if show:
            plt.show()
        plt.close(fig)
