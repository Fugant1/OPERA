"""Visualization utilities for GRPO training dynamics and SER evaluation."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def plot_grpo_training_curves(
    log_history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> Optional[Any]:
    """Generates a 2x2 diagnostic dashboard for GRPO training:

    1. Reward Curves (R1 Accuracy, R2 Acoustic Inventory, R3 Format, Total Weighted)
    2. Convergence Dynamics (Running F1-Macro and Step Accuracy)
    3. Policy Stability (Prediction Entropy and XML Format Errors)
    4. Emotion Distribution Dynamics across the 7 MELD classes
    """
    if not log_history:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    df_logs = pd.DataFrame(log_history)
    if "step" not in df_logs.columns and "global_step" in df_logs.columns:
        df_logs["step"] = df_logs["global_step"]

    if "step" in df_logs.columns:
        df_logs["step"] = df_logs["step"].ffill().fillna(df_logs.index)
    else:
        df_logs["step"] = df_logs.index

    df_step = df_logs.groupby("step", as_index=False).last()

    def smooth(series: pd.Series, alpha: float = 0.35) -> pd.Series:
        s = series.dropna()
        return s.ewm(alpha=alpha).mean() if len(s) > 1 else s

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Evolução do Treinamento GRPO - SER Speech Emotion Recognition", fontsize=16, fontweight="bold")

    # 1. Curvas de Recompensa
    ax1 = axes[0, 0]
    has_reward = False
    reward_configs = [
        ("reward/R1_accuracy", "R1: Acurácia Ponderada", "#1f77b4"),
        ("reward/R2_acoustic", "R2: Inventário Acústico", "#2ca02c"),
        ("reward/R3_format", "R3: Formato XML/CoT", "#ff7f0e"),
        ("reward/total_weighted", "Recompensa Total Ponderada", "#9467bd"),
    ]
    for col, label, color in reward_configs:
        if col in df_step.columns:
            valid = df_step.dropna(subset=[col])
            if len(valid) > 0:
                has_reward = True
                ax1.plot(valid["step"], valid[col], alpha=0.25, color=color, linestyle="--")
                ax1.plot(valid["step"], smooth(valid[col]), label=label, color=color, linewidth=2.2)

    ax1.set_title("1. Evolução das Recompensas (Rewards Growth)", fontsize=13, fontweight="semibold")
    ax1.set_xlabel("Optimization Steps")
    ax1.set_ylabel("Score de Recompensa")
    ax1.grid(True, linestyle=":", alpha=0.6)
    if has_reward:
        ax1.legend(loc="upper left")

    # 2. Desempenho: F1-Macro e Acurácia
    ax2 = axes[0, 1]
    has_perf = False
    if "metrics/f1_macro_running" in df_step.columns:
        valid_f1 = df_step.dropna(subset=["metrics/f1_macro_running"])
        if len(valid_f1) > 0:
            has_perf = True
            ax2.plot(valid_f1["step"], valid_f1["metrics/f1_macro_running"], alpha=0.25, color="#d62728", linestyle="--")
            ax2.plot(valid_f1["step"], smooth(valid_f1["metrics/f1_macro_running"]), label="F1-Macro (Running Window)", color="#d62728", linewidth=2.5)

    if "metrics/acc_step" in df_step.columns:
        valid_acc = df_step.dropna(subset=["metrics/acc_step"])
        if len(valid_acc) > 0:
            has_perf = True
            ax2.plot(valid_acc["step"], smooth(valid_acc["metrics/acc_step"]), label="Acurácia por Lote (Suavizada)", color="#17becf", linewidth=2.0)

    ax2.set_title("2. Convergência: F1-Macro e Acurácia", fontsize=13, fontweight="semibold")
    ax2.set_xlabel("Optimization Steps")
    ax2.set_ylabel("Métrica [0.0 a 1.0]")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    if has_perf:
        ax2.legend(loc="lower right")

    # 3. Estabilidade da Política: Entropia e Erro de Formato
    ax3 = axes[1, 0]
    has_stab = False
    if "metrics/prediction_entropy" in df_step.columns:
        valid_ent = df_step.dropna(subset=["metrics/prediction_entropy"])
        if len(valid_ent) > 0:
            has_stab = True
            ax3.plot(valid_ent["step"], smooth(valid_ent["metrics/prediction_entropy"]), label="Entropia de Predição (Bits)", color="#8c564b", linewidth=2.0)

    if "dist/invalid_format" in df_step.columns:
        valid_inv = df_step.dropna(subset=["dist/invalid_format"])
        if len(valid_inv) > 0:
            has_stab = True
            ax3.plot(valid_inv["step"], smooth(valid_inv["dist/invalid_format"]), label="Taxa de Formato Inválido", color="#e377c2", linewidth=2.0)

    ax3.set_title("3. Estabilidade da Política e Formato XML", fontsize=13, fontweight="semibold")
    ax3.set_xlabel("Optimization Steps")
    ax3.set_ylabel("Valor")
    ax3.grid(True, linestyle=":", alpha=0.6)
    if has_stab:
        ax3.legend(loc="upper right")

    # 4. Proporção das Predições por Emoção
    ax4 = axes[1, 1]
    has_dist = False
    emo_colors = {
        "dist/anger": "#d62728",
        "dist/joy": "#2ca02c",
        "dist/neutral": "#7f7f7f",
        "dist/sadness": "#1f77b4",
        "dist/surprise": "#ff7f0e",
        "dist/fear": "#9467bd",
        "dist/disgust": "#8c564b",
    }
    for col, color in emo_colors.items():
        if col in df_step.columns:
            valid_d = df_step.dropna(subset=[col])
            if len(valid_d) > 0:
                has_dist = True
                label = col.replace("dist/", "").capitalize()
                ax4.plot(valid_d["step"], smooth(valid_d[col]), label=label, color=color, linewidth=1.8)

    ax4.set_title("4. Dinâmica de Predição das 7 Emoções", fontsize=13, fontweight="semibold")
    ax4.set_xlabel("Optimization Steps")
    ax4.set_ylabel("Frequência Relativa")
    ax4.set_ylim(-0.02, 1.02)
    ax4.grid(True, linestyle=":", alpha=0.6)
    if has_dist:
        ax4.legend(loc="upper right", ncol=2, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()

    return fig
