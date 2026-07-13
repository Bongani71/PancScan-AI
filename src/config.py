"""
Training hyperparameters for PancScan AI.

Tune values here (or override via CLI flags in train.py) without editing
the training loop itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.data_loading import get_project_root
from src.model import DEFAULT_CHANNELS, DEFAULT_DROPOUT
from src.patching import DEFAULT_NEG, DEFAULT_PATCH_SIZE, DEFAULT_POS


@dataclass
class TrainConfig:
    """All knobs for ``src.train`` in one place."""

    # --- paths ---
    # DATA_ROOT: folder containing imagesTr/ and labelsTr/ (portable; no OS-specific paths).
    data_dir: Path = field(default_factory=lambda: get_project_root() / "data" / "Task07_Pancreas")
    split_path: Path = field(default_factory=lambda: get_project_root() / "outputs" / "patient_split.json")
    output_dir: Path = field(default_factory=lambda: get_project_root() / "outputs" / "training")
    cache_dir: Path = field(default_factory=lambda: get_project_root() / "outputs" / "cache")

    # --- data / patching ---
    # "cache" = CacheDataset (RAM), "persistent" = disk cache, "none" = recompute each epoch
    dataset_cache: str = "persistent"
    patch_size: tuple[int, int, int] = DEFAULT_PATCH_SIZE
    patches_per_volume: int = 2  # RandCrop num_samples; list_data_collate flattens these
    pos_ratio: float = float(DEFAULT_POS)
    neg_ratio: float = float(DEFAULT_NEG)
    batch_size: int = 2
    num_workers: int = 0  # 0 is safest on Windows; raise on Linux if desired

    # --- model ---
    channels: tuple[int, ...] = DEFAULT_CHANNELS
    dropout: float = DEFAULT_DROPOUT

    # --- optimization ---
    loss_name: str = "dice_ce"  # or "focal_tversky"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    # An "epoch" = this many optimizer steps (patches are randomly sampled).
    steps_per_epoch: int = 100
    # Cosine annealing over max_epochs (stepped once per epoch).
    scheduler: str = "cosine"  # "cosine" | "plateau" | "none"
    use_amp: bool = True  # mixed precision; auto-disabled on CPU

    # --- validation ---
    # Sliding-window ROI should match training patch size.
    sw_batch_size: int = 1
    sw_overlap: float = 0.25
    # Validate every N epochs (1 = every epoch). Useful to speed early debugging.
    val_every: int = 1

    # --- checkpointing / early stopping ---
    early_stopping_patience: int = 20
    resume: bool = True  # load last.pt from output_dir if present
    checkpoint_name: str = "best_tumor_dice.pt"
    last_checkpoint_name: str = "last.pt"

    # --- logging ---
    log_csv_name: str = "train_log.csv"
    seed: int = 42

    # --- smoke-test overrides (set by --smoke-test) ---
    smoke_test: bool = False
    smoke_train_cases: int = 4
    smoke_val_cases: int = 2

    # --- mid-size dry run (set by --subset N) ---
    # Use first N train patients + proportional val slice (same ratio as full split).
    subset_n: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Paths → strings for JSON serialization
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d


def smoke_config(base: TrainConfig | None = None) -> TrainConfig:
    """Short, memory-light config for an end-to-end dry run."""
    cfg = base or TrainConfig()
    cfg.smoke_test = True
    cfg.max_epochs = 2
    cfg.steps_per_epoch = 4
    cfg.batch_size = 1
    cfg.patches_per_volume = 1
    # Smaller patch for faster CPU/GPU smoke runs
    cfg.patch_size = (64, 64, 48)
    cfg.dataset_cache = "cache"
    cfg.val_every = 1
    cfg.early_stopping_patience = 5
    cfg.resume = False
    cfg.output_dir = get_project_root() / "outputs" / "training_smoke"
    return cfg
