"""
CT preprocessing for PancScan AI (Step 3).

Pipeline applied to each training case (patient-level, never slice-level):

1. Resample to a shared voxel spacing (default 1.0 × 1.0 × 2.5 mm).
   - Image: linear interpolation (preserves continuous HU values).
   - Label: nearest-neighbor (keeps discrete class IDs intact).
2. Soft-tissue HU window (−100 to 240), then rescale to [0, 1].
3. Crop a foreground ROI around pancreas + tumor so we do not waste
   GPU memory on empty background during training.

This module also:
- Counts class voxels across all 281 labeled training cases (for loss
  weighting in step 5).
- Creates a reproducible patient-level train/val/test split (70/15/15).
- Saves a before/after visualization for sanity checking.

MONAI transforms are used where they save boilerplate; the public API is a
plain class (`PreprocessingPipeline`) that `src/train.py` can call later.

Memory / runtime notes
----------------------
- Scanning all 281 label volumes for class balance is I/O heavy (~few minutes).
- Full resampling of every volume is slower; use `--skip-full-preprocess`
  when you only need the split + class-balance stats + one sanity-check case.
- Cropped volumes are smaller, but 3D U-Net training may still OOM — reduce
  patch size further in train.py if needed (e.g. 96³ → 64³).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    Spacingd,
    SpatialPadd,
)

from src.data_loading import (
    COLOR_PANCREAS,
    COLOR_TUMOR,
    LABEL_BACKGROUND,
    LABEL_PANCREAS,
    LABEL_TUMOR,
    ScanPair,
    get_default_data_dir,
    get_project_root,
    load_dataset_manifest,
    load_scan_with_label,
    resolve_training_paths,
)

# ---------------------------------------------------------------------------
# Defaults chosen to match Task07_Pancreas native geometry
# ---------------------------------------------------------------------------

# Most cases are ~0.7–1.0 mm in-plane and 2.5 mm through-plane.
TARGET_SPACING: tuple[float, float, float] = (1.0, 1.0, 2.5)

# Soft-tissue window used for model input (same as visualization window).
# Kept separate as named constants so a future bone/vessel window can coexist.
SOFT_TISSUE_HU_MIN = -100.0
SOFT_TISSUE_HU_MAX = 240.0

# Margin (voxels) around the pancreas/tumor mask when cropping.
CROP_MARGIN = 20

# Patient-level split ratios (must sum to 1.0).
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42

# Keys used inside MONAI dict transforms.
KEYS_IMAGE = "image"
KEYS_LABEL = "label"


@dataclass(frozen=True)
class PreprocessConfig:
    """Tunable knobs for the preprocessing pipeline."""

    target_spacing: tuple[float, float, float] = TARGET_SPACING
    hu_min: float = SOFT_TISSUE_HU_MIN
    hu_max: float = SOFT_TISSUE_HU_MAX
    crop_margin: int = CROP_MARGIN
    # Optional spatial size after crop; if set, pad/crop to this size for
    # batching. Leave None to keep variable cropped shapes (patch sampling
    # in train.py will handle batching later).
    spatial_size: tuple[int, int, int] | None = None


@dataclass
class PreprocessedCase:
    """Result of running the pipeline on one scan/label pair."""

    patient_id: str
    image: np.ndarray  # float32, shape (1, H, W, D), values in [0, 1]
    label: np.ndarray  # int16, shape (1, H, W, D), class IDs
    original_shape: tuple[int, int, int]
    original_spacing: tuple[float, float, float]
    processed_shape: tuple[int, int, int]
    processed_spacing: tuple[float, float, float]


@dataclass
class ClassBalanceStats:
    """Aggregate voxel counts across the labeled training set."""

    num_cases: int
    background: int
    pancreas: int
    tumor: int

    @property
    def total(self) -> int:
        return self.background + self.pancreas + self.tumor

    def as_ratios(self) -> dict[str, float]:
        t = max(self.total, 1)
        return {
            "background": self.background / t,
            "pancreas": self.pancreas / t,
            "tumor": self.tumor / t,
        }

    def to_dict(self) -> dict[str, Any]:
        ratios = self.as_ratios()
        return {
            "num_cases": self.num_cases,
            "voxels": {
                "background": self.background,
                "pancreas": self.pancreas,
                "tumor": self.tumor,
                "total": self.total,
            },
            "ratios": ratios,
            # Inverse-frequency style weights (useful for CE weighting later).
            # Normalized so they sum to 3 (one per class).
            "suggested_ce_weights": _inverse_frequency_weights(ratios),
        }


def _inverse_frequency_weights(ratios: dict[str, float]) -> dict[str, float]:
    """Compute normalized inverse-frequency class weights for CE loss."""
    eps = 1e-12
    inv = {k: 1.0 / max(v, eps) for k, v in ratios.items()}
    s = sum(inv.values())
    return {k: (v / s) * len(inv) for k, v in inv.items()}


def apply_hu_window(
    volume: np.ndarray | torch.Tensor,
    hu_min: float = SOFT_TISSUE_HU_MIN,
    hu_max: float = SOFT_TISSUE_HU_MAX,
) -> np.ndarray | torch.Tensor:
    """
    Clip CT intensities to a soft-tissue HU window and rescale to [0, 1].

    Soft tissue (−100 to 240 HU) is what matters for pancreas/tumor contrast.
    This is intentionally separate from any future bone/vessel window.
    """
    if isinstance(volume, torch.Tensor):
        clipped = torch.clamp(volume, hu_min, hu_max)
        return (clipped - hu_min) / (hu_max - hu_min)

    clipped = np.clip(volume, hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min)).astype(np.float32)


def list_training_cases(data_dir: Path | None = None) -> list[dict[str, str]]:
    """
    Return training cases as dicts with patient_id, image path, label path.

    Paths are absolute strings so the split JSON is self-contained.
    """
    root = data_dir or get_default_data_dir()
    manifest = load_dataset_manifest(root)
    cases: list[dict[str, str]] = []
    for entry in manifest["training"]:
        image_path = resolve_training_paths(root, entry["image"])
        label_path = resolve_training_paths(root, entry["label"])
        patient_id = image_path.name.replace(".nii.gz", "")
        cases.append(
            {
                "patient_id": patient_id,
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
            }
        )
    return cases


def create_patient_split(
    cases: list[dict[str, str]],
    ratios: dict[str, float] | None = None,
    seed: int = SPLIT_SEED,
) -> dict[str, list[dict[str, str]]]:
    """
    Split cases by patient ID (not by slice) to avoid data leakage.

    Default 70% train / 15% val / 15% test. The shuffle is seeded so the
    same split is produced every run when seed and case list are unchanged.
    """
    ratios = ratios or SPLIT_RATIOS
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    rng = random.Random(seed)
    shuffled = cases.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    # Assign remainder to test so counts always sum to n.
    n_test = n - n_train - n_val

    split = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    # Sanity: no patient appears in more than one split.
    ids = {s: {c["patient_id"] for c in split[s]} for s in split}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert n_train + n_val + n_test == n

    return split


def save_split(
    split: dict[str, list[dict[str, str]]],
    output_path: Path,
    meta: dict[str, Any] | None = None,
    seed: int = SPLIT_SEED,
    ratios: dict[str, float] | None = None,
) -> Path:
    """Write the patient split to JSON for reproducible training runs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "ratios": ratios or SPLIT_RATIOS,
        "counts": {k: len(v) for k, v in split.items()},
        "meta": meta or {},
        "splits": split,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def load_split(path: Path) -> dict[str, list[dict[str, str]]]:
    """Load a previously saved split JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["splits"]


def compute_class_balance(
    cases: list[dict[str, str]],
    progress_every: int = 25,
) -> ClassBalanceStats:
    """
    Count background / pancreas / tumor voxels across all labeled cases.

    Uses nibabel (not full MONAI pipeline) for speed — we only need the
    raw label arrays. This informs Dice/CE class weighting in step 5.
    """
    bg = pancreas = tumor = 0
    for i, case in enumerate(cases, start=1):
        # Load label only (much smaller than the CT volume).
        import nibabel as nib

        lbl = np.asanyarray(nib.load(case["label"]).dataobj)
        lbl = np.rint(lbl).astype(np.int16)
        bg += int(np.sum(lbl == LABEL_BACKGROUND))
        pancreas += int(np.sum(lbl == LABEL_PANCREAS))
        tumor += int(np.sum(lbl == LABEL_TUMOR))
        if i % progress_every == 0 or i == len(cases):
            print(f"  Class balance: scanned {i}/{len(cases)} labels...")

    return ClassBalanceStats(
        num_cases=len(cases),
        background=bg,
        pancreas=pancreas,
        tumor=tumor,
    )


def print_class_balance(stats: ClassBalanceStats) -> None:
    """Pretty-print voxel counts and ratios (required before saving)."""
    ratios = stats.as_ratios()
    weights = stats.to_dict()["suggested_ce_weights"]
    print("\n" + "=" * 64)
    print("CLASS BALANCE (all labeled training cases, raw NIfTI labels)")
    print("=" * 64)
    print(f"Cases scanned     : {stats.num_cases}")
    print(f"Background voxels : {stats.background:,}  ({100 * ratios['background']:.4f}%)")
    print(f"Pancreas voxels   : {stats.pancreas:,}  ({100 * ratios['pancreas']:.4f}%)")
    print(f"Tumor voxels      : {stats.tumor:,}  ({100 * ratios['tumor']:.4f}%)")
    print(f"Total voxels      : {stats.total:,}")
    print(
        "Ratio (bg : pancreas : tumor) = "
        f"1 : {stats.pancreas / max(stats.background, 1):.6f} : "
        f"{stats.tumor / max(stats.background, 1):.6f}"
    )
    print(
        "Suggested CE class weights (inverse-frequency, normalized):\n"
        f"  background={weights['background']:.3f}, "
        f"pancreas={weights['pancreas']:.3f}, "
        f"tumor={weights['tumor']:.3f}"
    )
    print("=" * 64 + "\n")


def build_deterministic_transforms(
    config: PreprocessConfig | None = None,
    *,
    as_tensor: bool = True,
) -> Compose:
    """
    MONAI Compose for load → orient → resample → HU window → foreground crop.

    Used by ``PreprocessingPipeline`` and by ``train.py`` DataLoaders.
    Does **not** include random patch cropping (that belongs in patching.py).
    """
    cfg = config or PreprocessConfig()
    keys = [KEYS_IMAGE, KEYS_LABEL]
    transforms: list[Any] = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=cfg.target_spacing,
            mode=("bilinear", "nearest"),
        ),
        _HUWindowd(keys=KEYS_IMAGE, hu_min=cfg.hu_min, hu_max=cfg.hu_max),
        CropForegroundd(
            keys=keys,
            source_key=KEYS_LABEL,
            margin=cfg.crop_margin,
            allow_smaller=True,
        ),
    ]
    if cfg.spatial_size is not None:
        transforms.append(
            SpatialPadd(keys=keys, spatial_size=cfg.spatial_size, mode="constant")
        )
        transforms.append(
            Resized(
                keys=keys,
                spatial_size=cfg.spatial_size,
                mode=("trilinear", "nearest"),
            )
        )
    transforms.append(
        EnsureTyped(keys=keys, data_type="tensor" if as_tensor else "numpy")
    )
    return Compose(transforms)


class PreprocessingPipeline:
    """
    Reusable preprocess pipeline for one CT + label pair.

    Designed so train.py can do:

        pipe = PreprocessingPipeline()
        case = pipe(image_path, label_path, patient_id="pancreas_001")

    Internally uses MONAI dict transforms for orientation, spacing, and
    foreground crop. HU windowing is applied as a simple callable so the
    soft-tissue window stays explicit and easy to change.
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self._transform = build_deterministic_transforms(self.config, as_tensor=False)

    def __call__(
        self,
        image_path: str | Path,
        label_path: str | Path,
        patient_id: str | None = None,
    ) -> PreprocessedCase:
        """Run the full pipeline on one case."""
        image_path = Path(image_path)
        label_path = Path(label_path)
        pid = patient_id or image_path.name.replace(".nii.gz", "")

        # Original metadata (for before/after comparison).
        raw = load_scan_with_label(image_path, label_path)

        out = self._transform(
            {
                KEYS_IMAGE: str(image_path),
                KEYS_LABEL: str(label_path),
            }
        )

        image = np.asarray(out[KEYS_IMAGE], dtype=np.float32)
        label = np.rint(np.asarray(out[KEYS_LABEL])).astype(np.int16)

        # MONAI Spacingd stores the new spacing in meta; fall back to config.
        meta = out.get(f"{KEYS_IMAGE}_meta_dict", {})
        spacing = meta.get("pixdim")
        if spacing is not None and len(spacing) >= 4:
            # pixdim is (1, sx, sy, sz, ...) in NIfTI convention after channel.
            processed_spacing = (
                float(spacing[1]),
                float(spacing[2]),
                float(spacing[3]),
            )
        else:
            processed_spacing = self.config.target_spacing

        return PreprocessedCase(
            patient_id=pid,
            image=image,
            label=label,
            original_shape=tuple(int(x) for x in raw.image.data.shape),
            original_spacing=raw.image.voxel_spacing,
            processed_shape=tuple(int(x) for x in image.shape[1:]),
            processed_spacing=processed_spacing,
        )


class _HUWindowd:
    """
    MONAI-style dict transform: soft-tissue HU window → [0, 1].

    Kept as a tiny class (not a lambda) so it is picklable and readable.
    """

    def __init__(self, keys: str, hu_min: float, hu_max: float) -> None:
        self.keys = keys
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        d[self.keys] = apply_hu_window(d[self.keys], self.hu_min, self.hu_max)
        return d


def _overlay_rgb(ct_slice: np.ndarray, label_slice: np.ndarray) -> np.ndarray:
    """Build RGB overlay (blue=pancreas, red=tumor) on a [0,1] CT slice."""
    # If CT is still in HU (before window), window it for display.
    if ct_slice.min() < -0.5 or ct_slice.max() > 1.5:
        display = apply_hu_window(ct_slice)
    else:
        display = ct_slice.astype(np.float32)
        display = np.clip(display, 0.0, 1.0)

    rgb = np.stack([display, display, display], axis=-1)
    for mask, color in (
        (label_slice == LABEL_PANCREAS, COLOR_PANCREAS),
        (label_slice == LABEL_TUMOR, COLOR_TUMOR),
    ):
        if np.any(mask):
            rgb[mask] = (1.0 - color[3]) * rgb[mask] + color[3] * color[:3]
    return rgb


def _pick_informative_z(label_volume: np.ndarray) -> int:
    """Pick an axial index that preferably contains tumor, else pancreas."""
    # label_volume may be (H, W, D) or (1, H, W, D)
    if label_volume.ndim == 4:
        label_volume = label_volume[0]
    tumor = np.where(np.any(label_volume == LABEL_TUMOR, axis=(0, 1)))[0]
    if tumor.size:
        return int(tumor[len(tumor) // 2])
    pancreas = np.where(np.any(label_volume == LABEL_PANCREAS, axis=(0, 1)))[0]
    if pancreas.size:
        return int(pancreas[len(pancreas) // 2])
    return label_volume.shape[2] // 2


def visualize_before_after(
    raw: ScanPair,
    processed: PreprocessedCase,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """
    Save a 2×2 figure comparing original vs preprocessed axial slices.

    Top: original CT + overlay.
    Bottom: resampled / windowed / cropped CT + overlay.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_img = raw.image.data
    raw_lbl = raw.label.data
    z_raw = _pick_informative_z(raw_lbl)

    proc_img = processed.image[0]
    proc_lbl = processed.label[0]
    z_proc = _pick_informative_z(proc_lbl)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(
        f"PancScan AI — preprocessing before/after ({processed.patient_id})",
        fontsize=13,
        fontweight="bold",
    )

    axes[0, 0].imshow(apply_hu_window(raw_img[:, :, z_raw]), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(
        f"Before — CT\nshape={raw_img.shape}, spacing={tuple(round(s, 3) for s in raw.image.voxel_spacing)}"
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_overlay_rgb(raw_img[:, :, z_raw], raw_lbl[:, :, z_raw]))
    axes[0, 1].set_title(f"Before — overlay (Z={z_raw})")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(proc_img[:, :, z_proc], cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(
        f"After — CT (windowed [0,1])\nshape={processed.processed_shape}, "
        f"spacing={tuple(round(s, 3) for s in processed.processed_spacing)}"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(_overlay_rgb(proc_img[:, :, z_proc], proc_lbl[:, :, z_proc]))
    axes[1, 1].set_title(f"After — overlay (Z={z_proc})")
    axes[1, 1].axis("off")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLOR_PANCREAS[:3],
            markersize=10,
            label="Pancreas",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLOR_TUMOR[:3],
            markersize=10,
            label="Tumor",
        ),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    root = get_project_root()
    parser = argparse.ArgumentParser(
        description="PancScan AI preprocessing: spacing, HU window, crop, split, class balance.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_default_data_dir(),
        help="Path to extracted Task07_Pancreas folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs",
        help="Directory for split JSON, class-balance JSON, and visualizations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help="Random seed for patient-level split",
    )
    parser.add_argument(
        "--viz-case",
        type=str,
        default=None,
        help="Patient ID for before/after viz (default: first case with tumor)",
    )
    parser.add_argument(
        "--skip-class-balance",
        action="store_true",
        help="Skip scanning all 281 labels (faster smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.output_dir
    explore_dir = out_dir / "data_exploration"
    explore_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory : {data_dir}")
    print(f"Output directory: {out_dir}")

    cases = list_training_cases(data_dir)
    print(f"Found {len(cases)} labeled training cases")

    # ------------------------------------------------------------------
    # 1) Patient-level split (reproducible)
    # ------------------------------------------------------------------
    split = create_patient_split(cases, seed=args.seed)
    split_path = save_split(
        split,
        out_dir / "patient_split.json",
        seed=args.seed,
        meta={
            "data_dir": str(data_dir.resolve()),
            "target_spacing_mm": list(TARGET_SPACING),
            "hu_window": [SOFT_TISSUE_HU_MIN, SOFT_TISSUE_HU_MAX],
            "crop_margin_voxels": CROP_MARGIN,
        },
    )
    print(
        f"Saved patient split -> {split_path} "
        f"(train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])})"
    )

    # ------------------------------------------------------------------
    # 2) Class balance across all 281 cases (informs loss weights later)
    # ------------------------------------------------------------------
    if args.skip_class_balance:
        print("Skipping class-balance scan (--skip-class-balance).")
        balance_stats = None
    else:
        print("Scanning label maps for class balance (this may take a few minutes)...")
        balance_stats = compute_class_balance(cases)
        print_class_balance(balance_stats)
        balance_path = out_dir / "class_balance.json"
        with balance_path.open("w", encoding="utf-8") as f:
            json.dump(balance_stats.to_dict(), f, indent=2)
        print(f"Saved class balance -> {balance_path}")

    # ------------------------------------------------------------------
    # 3) Before/after visualization on one case
    # ------------------------------------------------------------------
    pipe = PreprocessingPipeline()

    if args.viz_case:
        viz_entry = next(c for c in cases if c["patient_id"] == args.viz_case)
    else:
        # Prefer a case that actually has tumor voxels for a useful overlay.
        viz_entry = None
        for c in cases:
            import nibabel as nib

            lbl = np.asanyarray(nib.load(c["label"]).dataobj)
            if np.any(np.rint(lbl) == LABEL_TUMOR):
                viz_entry = c
                break
        if viz_entry is None:
            viz_entry = cases[0]

    print(f"Running preprocess pipeline on {viz_entry['patient_id']} for sanity check...")
    raw = load_scan_with_label(Path(viz_entry["image"]), Path(viz_entry["label"]))
    processed = pipe(
        viz_entry["image"],
        viz_entry["label"],
        patient_id=viz_entry["patient_id"],
    )

    print(
        f"  Original : shape={processed.original_shape}, "
        f"spacing={tuple(round(s, 3) for s in processed.original_spacing)}"
    )
    print(
        f"  Processed: shape={processed.processed_shape}, "
        f"spacing={tuple(round(s, 3) for s in processed.processed_spacing)}, "
        f"image range=[{processed.image.min():.3f}, {processed.image.max():.3f}], "
        f"label ids={np.unique(processed.label)}"
    )

    viz_path = explore_dir / f"{processed.patient_id}_preprocess_before_after.png"
    visualize_before_after(raw, processed, viz_path)
    print(f"Saved before/after visualization -> {viz_path}")

    # Small summary for train.py consumers.
    summary = {
        "config": asdict(PreprocessConfig()),
        "split_path": str(split_path),
        "class_balance_path": str(out_dir / "class_balance.json") if balance_stats else None,
        "viz_path": str(viz_path),
        "example_case": {
            "patient_id": processed.patient_id,
            "original_shape": list(processed.original_shape),
            "processed_shape": list(processed.processed_shape),
            "original_spacing": list(processed.original_spacing),
            "processed_spacing": list(processed.processed_spacing),
        },
    }
    summary_path = out_dir / "preprocessing_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved preprocessing summary -> {summary_path}")
    print("\nStep 3 preprocessing complete.")


if __name__ == "__main__":
    main()
