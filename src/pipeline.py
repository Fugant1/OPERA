"""End-to-end research orchestration pipeline for OPERA SER-GRPO."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.acoustics.analyzer import AcousticAnalyzer
from src.acoustics.diagnostics import SpeakerDiagnostics
from src.acoustics.features import eGeMAPSExtractor
from src.acoustics.visualization import AcousticVisualizer
from src.config import ExperimentConfig
from src.data.dataset import SERGRPODataset, split_train_val
from src.data.loader import MELDLoader
from src.data.sampler import BalancedSampler
from src.models.qwen_omni import QwenOmniLoader
from src.prompts.templates import COT_PROMPT
from src.rewards.accuracy import compute_emotion_weights
from src.rewards.composite import RewardManager
from src.training.trainer import SERGRPOTrainer
from src.utils.logging import get_logger

logger = get_logger("pipeline")


class SERGRPOPipeline:
    """Orchestrates the complete research workflow:

    1. Raw MELD dataset ingestion & validation.
    2. eGeMAPSv02 feature extraction & caching on raw train set.
    3. Speaker quantile validity & dynamic range diagnostics.
    4. Publication-grade acoustic cue visualization (emotion variation & speaker dynamic range).
    5. Duration-filtered deterministic class balancing & train/val split.
    6. Multi-objective reward initialization (Weighted Accuracy, Acoustic Grounding, Format).
    7. 4-bit Quantized Qwen 2.5 Omni model loading with LoRA adaptation.
    8. GRPO reinforcement learning training execution.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()

    def run_acoustic_analysis(self, raw_train_set: Any, reports_dir: Path = Path("reports/figures")):
        """Execute acoustic feature extraction, diagnostics, and plotting on raw training set."""
        logger.info("--- Step 2: Acoustic Feature Extraction (Raw Train Set) ---")
        extractor = eGeMAPSExtractor(
            cues_dict=self.config.acoustics.cues,
            cache_path=self.config.data.cache_path,
            alt_cache_path=self.config.data.alt_cache_path,
        )
        raw_train_set = extractor.extract_or_load(raw_train_set)

        logger.info("--- Step 3: Speaker Quantile Validity & Long-tail Diagnostics ---")
        diagnostics = SpeakerDiagnostics(
            min_utterances_threshold=self.config.acoustics.min_utterances_threshold
        )
        diag_results = diagnostics.diagnose(raw_train_set)

        logger.info("--- Step 4: Emotion & Speaker Dynamic Range Analysis ---")
        analyzer = AcousticAnalyzer(
            cues_dict=self.config.acoustics.cues,
            top_n_speakers=self.config.acoustics.top_n_speakers,
        )
        analysis_results = analyzer.analyze(raw_train_set)

        visualizer = AcousticVisualizer(cues_dict=self.config.acoustics.cues)
        reports_dir.mkdir(parents=True, exist_ok=True)

        visualizer.plot_emotion_acoustic_distributions(
            raw_train_set,
            output_path=reports_dir / "fig1_emotion_acoustic_distributions.png",
        )

        if "speaker_relative_amplitudes" in analysis_results:
            visualizer.plot_speaker_amplitude_coverage(
                analysis_results["speaker_relative_amplitudes"],
                top_n_speakers=self.config.acoustics.top_n_speakers,
                output_path=reports_dir / "fig2_speaker_amplitude_coverage.png",
            )

        return raw_train_set, diag_results, analysis_results

    def run(self, reports_dir: Path = Path("reports/figures"), train_model: bool = False):
        """Execute complete SER-GRPO research pipeline."""
        logger.info("================================================================================")
        logger.info("OPERA: Omni-modal Preference Extraction and Ranking Architecture (SER-GRPO)")
        logger.info("================================================================================")

        # 1. Load Raw Train Set
        logger.info("--- Step 1: Loading Raw MELD Dataset ---")
        loader = MELDLoader(
            archive_path=self.config.data.archive_path,
            dataset_dir=self.config.data.dataset_dir,
        )
        raw_train_set = loader.load(split="train")

        # 2-4. Acoustic Extraction, Diagnostics, and Plots on Raw Train Set
        raw_train_set, diag_results, analysis_results = self.run_acoustic_analysis(
            raw_train_set, reports_dir=reports_dir
        )

        # 5. Balanced Sampling and Train/Val Split
        logger.info("--- Step 5: Deterministic Balancing & Duration Filtering ---")
        sampler = BalancedSampler(
            target_total=self.config.data.target_total,
            max_neutral_pct=self.config.data.max_neutral_pct,
            min_duration=self.config.data.min_duration,
            max_duration=self.config.data.max_duration,
            seed=self.config.data.seed,
        )
        pruned_train_set = sampler.sample(raw_train_set)

        train_set, val_set = split_train_val(
            pruned_train_set,
            test_size=self.config.data.test_size,
            seed=self.config.data.seed,
            stratify_col="emotion",
        )

        # 6. Reward Engineering
        logger.info("--- Step 6: Multi-Objective Reward Configuration ---")
        class_weights = compute_emotion_weights(
            train_set, power=self.config.reward.smoothing_power
        )
        quantiles_ref = (
            analysis_results.get("global_summary", {}).to_dict(orient="index")
            if hasattr(analysis_results.get("global_summary"), "to_dict")
            else {}
        )
        reward_manager = RewardManager(
            class_weights=class_weights, quantiles_ref=quantiles_ref
        )

        logger.info("Active GRPO reward functions: R1 (Weighted Accuracy), R2 (Acoustic Grounding), R3 (Format)")

        if not train_model:
            logger.info("Pipeline data preparation, acoustic diagnostics, and reward setup completed successfully.")
            return {
                "train_set": train_set,
                "val_set": val_set,
                "class_weights": class_weights,
                "reward_manager": reward_manager,
                "diagnostics": diag_results,
                "analysis": analysis_results,
            }

        # 7. Model and LoRA Loading
        logger.info("--- Step 7: Model & Adapter Initialization ---")
        model, processor = QwenOmniLoader.load_model(config=self.config.model)

        # 8. Dataset Formatting for GRPO
        train_grpo_ds = SERGRPODataset(train_set, prompt_template=COT_PROMPT).to_hf_dataset()
        val_grpo_ds = SERGRPODataset(val_set, prompt_template=COT_PROMPT).to_hf_dataset()

        # 9. Launch GRPO Training
        logger.info("--- Step 8: Launching SER-GRPO Trainer ---")
        trainer = SERGRPOTrainer(
            model=model,
            processor=processor,
            train_dataset=train_grpo_ds,
            eval_dataset=val_grpo_ds,
            reward_funcs=reward_manager.get_reward_functions(),
            config=self.config.grpo,
        )
        trainer.train()
        logger.info("GRPO Training run completed successfully.")


if __name__ == "__main__":
    pipeline = SERGRPOPipeline()
    pipeline.run()
