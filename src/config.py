"""Global Research and Experiment Configuration for OPERA SER-GRPO.

Omni-modal Preference Extraction and Ranking Architecture (OPERA)
Speech Emotion Recognition via Group Relative Policy Optimization (GRPO).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """Dataset paths, extraction, duration filtering, and balancing settings."""

    archive_path: Path = Path("/content/drive/MyDrive/SER_DPO/MELD/MELD.tar")
    dataset_dir: Path = Path("/content/MELD")
    cache_path: Path = Path("/content/train_egemaps_features.csv")
    alt_cache_path: Path = Path("/content/raw_train_egemaps_features.csv")

    # Audio duration filters (seconds)
    min_duration: float = 0.5
    max_duration: float = 30.0

    # Class balancing parameters
    target_total: int = 2000
    max_neutral_pct: float = 0.15
    seed: int = 42

    # Train / validation split
    test_size: float = 0.2


@dataclass
class AcousticsConfig:
    """openSMILE eGeMAPSv02 acoustic cue definition and diagnostic parameters."""

    cues: Dict[str, str] = field(
        default_factory=lambda: {
            "Pitch Height (st)": "F0semitoneFrom27.5Hz_sma3nz_amean",
            "Pitch Dynamics (st)": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
            "Vocal Energy (Sones)": "loudness_sma3_percentile50.0",
            "Speaking Rate (seg/s)": "VoicedSegmentsPerSec",
            "Spectral Tilt (dB)": "hammarbergIndexV_sma3nz_amean",
            "HNR (dB)": "HNRdBACF_sma3nz_amean",
            "Jitter Local": "jitterLocal_sma3nz_amean",
            "Shimmer Local (dB)": "shimmerLocaldB_sma3nz_amean",
        }
    )

    # Diagnostic parameters
    min_utterances_threshold: int = 8
    top_n_speakers: int = 20
    moderate_bandwidth_min_st: float = 1.0


@dataclass
class ModelConfig:
    """Pretrained model architecture, 4-bit quantization, and LoRA PEFT hyperparameters."""

    model_id: str = "Qwen/Qwen2.5-Omni-7B-Instruct"

    # 4-bit BitsAndBytes Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # LoRA / PEFT hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    target_modules_regex: str = (
        r"thinker\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(gate_proj|up_proj|down_proj))"
    )


@dataclass
class RewardConfig:
    """Multi-objective reward configuration for GRPO."""

    # Class-weighted accuracy reward
    smoothing_power: float = 0.4

    # Format reward weights
    tag_order_weight: float = 0.20
    inventory_completion_weight: float = 0.35
    reasoning_length_weight: float = 0.25
    valid_answer_weight: float = 0.20

    # Acoustic inventory physical grounding reward
    grounding_weight_per_cue: float = 0.20


@dataclass
class GRPOConfig:
    """Group Relative Policy Optimization (GRPO) training hyperparameters."""

    output_dir: Path = Path("./outputs/grpo_experiment")
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 512
    temperature: float = 0.7
    kl_penalty: float = 0.04
    max_steps: int = 250
    logging_steps: int = 10
    save_steps: int = 50
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Unified master research configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    acoustics: AcousticsConfig = field(default_factory=AcousticsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
