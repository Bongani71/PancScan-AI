"""
Load and inspect Medical Segmentation Decathlon Task07_Pancreas NIfTI volumes.

This module is the first step in the PancScan AI pipeline. It reads 3D CT scans
and optional label maps from disk, prints basic metadata (shape, spacing, HU range),
and saves axial-slice visualizations for sanity checking before training.

Expected dataset layout (after extracting Task07_Pancreas.tar):

    data/Task07_Pancreas/
        dataset.json
        imagesTr/   # training CT scans (.nii.gz or .nii)
        labelsTr/   # matching labels: 0=background, 1=pancreas, 2=tumor
        imagesTs/   # unlabeled test scans (.nii.gz or .nii)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Label IDs defined by the MSD Task07 dataset.
LABEL_BACKGROUND = 0
LABEL_PANCREAS = 1
LABEL_TUMOR = 2

# Prefer compressed MSD layout; fall back to plain .nii (e.g. Kaggle auto-decompress).
NIFTI_EXTENSIONS: tuple[str, ...] = (".nii.gz", ".nii")

# Colors for overlay visualization (RGBA, 0–1).
COLOR_PANCREAS = np.array([0.2, 0.6, 1.0, 0.45])  # blue
COLOR_TUMOR = np.array([1.0, 0.2, 0.2, 0.70])  # red — distinct from pancreas


@dataclass(frozen=True)
class VolumeInfo:
    """Lightweight container for a loaded NIfTI volume and its metadata."""

    data: np.ndarray
    affine: np.ndarray
    voxel_spacing: tuple[float, float, float]
    path: Path


@dataclass(frozen=True)
class ScanPair:
    """A training CT scan paired with its segmentation label map."""

    image: VolumeInfo
    label: VolumeInfo
    patient_id: str


def get_project_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parents[1]


def get_default_data_dir() -> Path:
    """Default location for the extracted MSD pancreas dataset."""
    return get_project_root() / "data" / "Task07_Pancreas"


def patient_id_from_filename(name: str | Path) -> str:
    """
    Extract patient/case id from filenames like pancreas_001.nii.gz or .nii.

    Strips known NIfTI extensions; does not invent IDs from other suffixes.
    """
    name = Path(name).name
    lower = name.lower()
    for ext in NIFTI_EXTENSIONS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def find_nifti_file(directory: Path | str, patient_id: str) -> Path:
    """
    Locate ``{patient_id}.nii.gz`` or ``{patient_id}.nii`` under ``directory``.

    Kaggle (and some extractors) auto-decompress ``.nii.gz`` to plain ``.nii``.
    Prefer ``.nii.gz`` when both exist so local MSD layouts stay unchanged.

    Raises
    ------
    FileNotFoundError
        If the directory is missing or neither extension is present.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"NIfTI directory not found: {directory}\n"
            f"(looking for patient '{patient_id}')"
        )

    candidates = [directory / f"{patient_id}{ext}" for ext in NIFTI_EXTENSIONS]
    for path in candidates:
        if path.is_file():
            return path

    tried = ", ".join(c.name for c in candidates)
    raise FileNotFoundError(
        f"No NIfTI file for patient '{patient_id}' in {directory}.\n"
        f"Tried: {tried}"
    )


def load_dataset_manifest(data_dir: Path | None = None) -> dict:
    """
    Load dataset.json shipped with Task07_Pancreas.

    The manifest lists training image/label pairs and test image paths.
    """
    root = data_dir or get_default_data_dir()
    manifest_path = root / "dataset.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset.json at {manifest_path}. "
            "Extract Task07_Pancreas.tar into data/Task07_Pancreas/."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_nifti(path: Path) -> VolumeInfo:
    """
    Load a single NIfTI file with nibabel.

    Returns the voxel array (float32), affine matrix, and voxel spacing in mm.
    Spacing is read from the header zooms (pixdim); for most MSD scans this is
    approximately isotropic ~1 mm, but it varies per case.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    nii = nib.load(str(path))
    # ascloseas_possible preserves on-disk dtype; we cast to float32 for downstream math.
    data = np.asanyarray(nii.dataobj, dtype=np.float32)
    zooms = nii.header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return VolumeInfo(
        data=data,
        affine=nii.affine,
        voxel_spacing=spacing,
        path=path,
    )


def _patient_id_from_path(path: Path) -> str:
    """Extract patient/case id from a NIfTI path (.nii or .nii.gz)."""
    return patient_id_from_filename(path)


def resolve_training_paths(data_dir: Path, relative_path: str) -> Path:
    """
    Convert a path from dataset.json (e.g. ./imagesTr/...) to a real file path.

    If the manifest points at ``.nii.gz`` but only ``.nii`` exists on disk
    (common after Kaggle auto-decompress), resolve via ``find_nifti_file``.
    """
    # Manifest paths start with "./"; strip that prefix before joining.
    clean = relative_path.lstrip("./")
    path = Path(data_dir) / clean
    if path.is_file():
        return path

    patient_id = patient_id_from_filename(Path(clean).name)
    subdir = Path(data_dir) / Path(clean).parent
    return find_nifti_file(subdir, patient_id)


def load_scan_with_label(
    image_path: Path,
    label_path: Path,
) -> ScanPair:
    """
    Load one training case: CT volume + matching label map.

    Both arrays should have identical shape. Labels use integer class IDs:
    0 background, 1 pancreas, 2 tumor (cancer).
    """
    image = load_nifti(image_path)
    label = load_nifti(label_path)

    # Labels are stored as floats in NIfTI but represent discrete classes.
    label_int = np.rint(label.data).astype(np.int16)
    label = VolumeInfo(
        data=label_int,
        affine=label.affine,
        voxel_spacing=label.voxel_spacing,
        path=label.path,
    )

    if image.data.shape != label.data.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: "
            f"image {image.data.shape} vs label {label.data.shape}"
        )

    return ScanPair(
        image=image,
        label=label,
        patient_id=_patient_id_from_path(image_path),
    )


def load_first_training_case(data_dir: Path | None = None) -> ScanPair:
    """Convenience helper: load the first case listed in dataset.json."""
    root = data_dir or get_default_data_dir()
    manifest = load_dataset_manifest(root)
    first = manifest["training"][0]
    image_path = resolve_training_paths(root, first["image"])
    label_path = resolve_training_paths(root, first["label"])
    return load_scan_with_label(image_path, label_path)


def print_scan_info(scan: ScanPair) -> None:
    """
    Print human-readable metadata for a loaded scan/label pair.

    CT values are in Hounsfield Units (HU). Typical soft-tissue range for
    abdomen is roughly -100 to +240 HU — we report the actual min/max here.
    """
    img = scan.image.data
    lbl = scan.label.data

    unique_labels, counts = np.unique(lbl, return_counts=True)
    label_stats = {
        int(u): int(c) for u, c in zip(unique_labels, counts, strict=True)
    }

    tumor_voxels = int(np.sum(lbl == LABEL_TUMOR))
    pancreas_voxels = int(np.sum(lbl == LABEL_PANCREAS))

    print(f"\n{'=' * 60}")
    print(f"Patient / case ID : {scan.patient_id}")
    print(f"Image path        : {scan.image.path}")
    print(f"Label path        : {scan.label.path}")
    print(f"{'=' * 60}")
    print(f"Volume shape (X, Y, Z) : {img.shape}")
    print(f"Voxel spacing (mm)     : {scan.image.voxel_spacing}")
    print(f"CT intensity (HU)      : min={img.min():.1f}, max={img.max():.1f}")
    print(f"Label value counts     : {label_stats}")
    print(f"Pancreas voxels        : {pancreas_voxels:,}")
    print(f"Tumor voxels           : {tumor_voxels:,}")
    if tumor_voxels == 0:
        print("  (Note: some training cases have no annotated tumor voxels.)")
    print(f"{'=' * 60}\n")


def _normalize_slice_for_display(slice_2d: np.ndarray) -> np.ndarray:
    """
    Window CT slice for display using a soft-tissue HU window.

    We clip to [-100, 240] HU then scale to [0, 1] for matplotlib imshow.
    This is visualization-only; preprocessing will reuse the same window later.
    """
    windowed = np.clip(slice_2d, -100, 240)
    return (windowed - (-100)) / (240 - (-100))


def _choose_axial_slice_indices(
    label_volume: np.ndarray,
    num_slices: int,
) -> list[int]:
    """
    Pick axial slice indices that are likely informative.

    Strategy:
    1. Prefer slices that contain tumor voxels (if any).
    2. Also include slices through the pancreas region.
    3. Fall back to evenly spaced slices through the volume center.
    """
    depth = label_volume.shape[2]
    tumor_slices = np.where(np.any(label_volume == LABEL_TUMOR, axis=(0, 1)))[0]
    pancreas_slices = np.where(np.any(label_volume == LABEL_PANCREAS, axis=(0, 1)))[0]

    candidates: list[int] = []
    if tumor_slices.size > 0:
        # Include central tumor slice plus neighbors for context.
        center = int(tumor_slices[len(tumor_slices) // 2])
        candidates.extend([center - 2, center, center + 2])

    if pancreas_slices.size > 0:
        p_center = int(pancreas_slices[len(pancreas_slices) // 2])
        candidates.extend([p_center - 5, p_center, p_center + 5])

    if not candidates:
        # No labels — sample middle third of the volume.
        start, end = depth // 3, 2 * depth // 3
        candidates = list(np.linspace(start, end, num=num_slices, dtype=int))

    # Deduplicate, clamp to valid range, keep order.
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in candidates:
        idx = int(np.clip(idx, 0, depth - 1))
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)

    # Trim or pad to requested count with evenly spaced slices.
    if len(ordered) >= num_slices:
        return ordered[:num_slices]

    extra = np.linspace(depth // 4, 3 * depth // 4, num=num_slices, dtype=int)
    for idx in extra:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
        if len(ordered) >= num_slices:
            break

    return ordered[:num_slices]


def _overlay_labels_on_slice(
    base_rgb: np.ndarray,
    label_slice: np.ndarray,
) -> np.ndarray:
    """
    Alpha-blend pancreas (blue) and tumor (red) onto a grayscale RGB slice.

    Tumor is drawn on top of pancreas so the most clinically relevant region
    stands out clearly in the visualization.
    """
    rgb = base_rgb.copy()
    pancreas_mask = label_slice == LABEL_PANCREAS
    tumor_mask = label_slice == LABEL_TUMOR

    for mask, color in ((pancreas_mask, COLOR_PANCREAS), (tumor_mask, COLOR_TUMOR)):
        if not np.any(mask):
            continue
        rgb[mask] = (1.0 - color[3]) * rgb[mask] + color[3] * color[:3]

    return rgb


def visualize_axial_slices(
    scan: ScanPair,
    output_path: Path,
    num_slices: int = 4,
    dpi: int = 150,
) -> Path:
    """
    Save a figure with axial CT slices and matching label overlays.

    Layout: two rows × num_slices columns
      - Top row    : windowed CT
      - Bottom row : CT + pancreas (blue) + tumor (red) overlay

    Axial slices are taken along the last axis (Z), which is standard for
    nibabel-loaded MSD volumes when viewed in radiological orientation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = scan.image.data
    label = scan.label.data
    slice_indices = _choose_axial_slice_indices(label, num_slices=num_slices)

    fig, axes = plt.subplots(
        2,
        num_slices,
        figsize=(4 * num_slices, 8),
        squeeze=False,
    )
    fig.suptitle(
        f"PancScan AI — {scan.patient_id} (axial slices, Z axis)",
        fontsize=14,
        fontweight="bold",
    )

    for col, z_idx in enumerate(slice_indices):
        ct_slice = image[:, :, z_idx]
        lbl_slice = label[:, :, z_idx]
        display = _normalize_slice_for_display(ct_slice)
        base_rgb = np.stack([display, display, display], axis=-1)

        # Top: CT only
        ax_ct = axes[0, col]
        ax_ct.imshow(display, cmap="gray", vmin=0, vmax=1)
        ax_ct.set_title(f"Slice Z={z_idx}")
        ax_ct.axis("off")

        # Bottom: CT + labels
        ax_ov = axes[1, col]
        overlay = _overlay_labels_on_slice(base_rgb, lbl_slice)
        ax_ov.imshow(overlay)
        ax_ov.set_title("Overlay (blue=pancreas, red=tumor)")
        ax_ov.axis("off")

    # Legend outside the grid
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_PANCREAS[:3], markersize=10, label="Pancreas"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_TUMOR[:3], markersize=10, label="Tumor"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize_multiple_cases(
    data_dir: Path,
    case_indices: Iterable[int] = (0, 1, 2),
    output_dir: Path | None = None,
    num_slices: int = 4,
) -> list[Path]:
    """
    Load several training cases and save one visualization PNG per case.

    Returns list of saved file paths.
    """
    root = data_dir
    manifest = load_dataset_manifest(root)
    out_dir = output_dir or (get_project_root() / "outputs" / "data_exploration")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for idx in case_indices:
        entry = manifest["training"][idx]
        image_path = resolve_training_paths(root, entry["image"])
        label_path = resolve_training_paths(root, entry["label"])
        scan = load_scan_with_label(image_path, label_path)
        print_scan_info(scan)

        out_path = out_dir / f"{scan.patient_id}_axial_slices.png"
        visualize_axial_slices(scan, out_path, num_slices=num_slices)
        print(f"Saved visualization -> {out_path}")
        saved.append(out_path)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Task07_Pancreas NIfTI data and save sample slice visualizations.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_default_data_dir(),
        help="Path to extracted Task07_Pancreas folder (default: ./data/Task07_Pancreas/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_project_root() / "outputs" / "data_exploration",
        help="Where to save PNG visualizations",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=3,
        help="Number of training cases to visualize (from start of dataset.json)",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=4,
        help="Axial slices per case",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Data directory : {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}\n"
            "Extract Task07_Pancreas.tar so the folder contains imagesTr/, labelsTr/, "
            "imagesTs/, and dataset.json."
        )

    case_indices = range(args.num_cases)
    saved = visualize_multiple_cases(
        data_dir=args.data_dir,
        case_indices=case_indices,
        output_dir=args.output_dir,
        num_slices=args.num_slices,
    )
    print(f"\nDone. {len(saved)} visualization(s) written to {args.output_dir}")


if __name__ == "__main__":
    main()
