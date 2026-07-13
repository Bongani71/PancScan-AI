"""
Diagnose tumor presence in a patient subset (default: first 25 train IDs).

Use this before a full training run to separate:
  - data issue  → few/small tumors in the subset
  - sampling issue → tumors exist but patches rarely hit them

Example::

    python -m src.diagnose_tumor --subset 25
    python -m src.diagnose_tumor --subset 25 --data-dir /kaggle/input/.../Task07_Pancreas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from src.config import TrainConfig
from src.data_loading import (
    LABEL_PANCREAS,
    LABEL_TUMOR,
    find_nifti_file,
    get_default_data_dir,
    get_project_root,
)
from src.preprocessing import load_split
from src.train import apply_patient_subset


def count_label_voxels(label_path: Path) -> dict[str, int]:
    """Count background / pancreas / tumor voxels in one raw label NIfTI."""
    lbl = np.asanyarray(nib.load(str(label_path)).dataobj)
    lbl = np.rint(lbl).astype(np.int16)
    return {
        "background": int(np.sum(lbl == 0)),
        "pancreas": int(np.sum(lbl == LABEL_PANCREAS)),
        "tumor": int(np.sum(lbl == LABEL_TUMOR)),
        "total": int(lbl.size),
    }


def diagnose_subset(
    patient_ids: list[str],
    data_dir: Path,
) -> list[dict]:
    """Per-patient tumor stats for the given ID list."""
    rows: list[dict] = []
    for pid in patient_ids:
        label_path = find_nifti_file(data_dir / "labelsTr", pid)
        counts = count_label_voxels(label_path)
        rows.append(
            {
                "patient_id": pid,
                "label_path": str(label_path),
                "has_tumor": counts["tumor"] > 0,
                **counts,
            }
        )
    return rows


def print_report(rows: list[dict], subset_n: int) -> None:
    print("=" * 72)
    print(f"TUMOR DIAGNOSTIC - train subset (n={len(rows)}, requested={subset_n})")
    print("=" * 72)
    print(f"{'patient_id':<16} {'has_tumor':<10} {'tumor_vox':>12} {'pancreas_vox':>14}")
    print("-" * 72)

    for r in rows:
        print(
            f"{r['patient_id']:<16} "
            f"{str(r['has_tumor']):<10} "
            f"{r['tumor']:>12,} "
            f"{r['pancreas']:>14,}"
        )

    with_tumor = [r for r in rows if r["has_tumor"]]
    without = [r for r in rows if not r["has_tumor"]]
    tumor_vox = [r["tumor"] for r in with_tumor]

    print("-" * 72)
    print(f"Patients with tumor    : {len(with_tumor)} / {len(rows)}")
    print(f"Patients without tumor : {len(without)} / {len(rows)}")
    if without:
        print(f"  No-tumor IDs: {', '.join(r['patient_id'] for r in without)}")
    if tumor_vox:
        print(
            f"Tumor voxels (among positives): "
            f"min={min(tumor_vox):,}  median={int(np.median(tumor_vox)):,}  "
            f"max={max(tumor_vox):,}  total={sum(tumor_vox):,}"
        )
    else:
        print(
            "WARNING: ZERO patients in this subset have label==2 (tumor). "
            "Tumor Dice cannot improve — try a different seed/split or larger subset."
        )

    # Rough sampling implication
    frac_with = len(with_tumor) / max(len(rows), 1)
    print(
        f"\nSampling note: with pos:neg=2:1 and tumor-keyed crops, only the "
        f"{100 * frac_with:.0f}% of volumes that contain tumor can yield "
        "true tumor-centered positive patches."
    )
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose tumor voxels in a train subset")
    p.add_argument("--subset", type=int, default=25, help="First N train patients (default 25)")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Task07_Pancreas root (default: data/Task07_Pancreas)",
    )
    p.add_argument(
        "--split",
        type=Path,
        default=None,
        help="patient_split.json path (default: outputs/patient_split.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    data_dir = args.data_dir or get_default_data_dir()
    split_path = args.split or (get_project_root() / "outputs" / "patient_split.json")

    print(f"data_dir  : {data_dir}")
    print(f"split     : {split_path}")

    split = load_split(split_path)
    train_ids, val_ids = apply_patient_subset(split["train"], split["val"], args.subset)
    print(f"Inspecting {len(train_ids)} train patients (first {args.subset} of split)...")

    rows = diagnose_subset(train_ids, data_dir)
    print_report(rows, args.subset)

    if val_ids:
        print(f"\nVal slice for --subset {args.subset} ({len(val_ids)} patients):")
        val_rows = diagnose_subset(val_ids, data_dir)
        n_t = sum(1 for r in val_rows if r["has_tumor"])
        print(f"  With tumor: {n_t}/{len(val_rows)}")
        for r in val_rows:
            print(f"  {r['patient_id']}: tumor_vox={r['tumor']:,}")


if __name__ == "__main__":
    main()
