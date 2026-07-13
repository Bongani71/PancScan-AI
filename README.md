# PancScan AI

**PancScan AI** is a research prototype for explainable pancreatic analysis on abdominal CT scans. The goal is to flag suspicious pancreatic regions, show Grad-CAM-style attention heatmaps, and produce a readable summary report — similar in spirit to radiotherapy decision-support tools. **This supports clinicians; it does not replace them.**

> **Disclaimer:** This is a research prototype, not a medical device. It is **not validated for clinical use** and must not be used for diagnosis or treatment decisions. All outputs require review by a qualified radiologist or physician.

---

## What this project does (roadmap)

| Step | Module | Status |
|------|--------|--------|
| 1 | Project setup | Done |
| 2 | Data loading & exploration | Done |
| 3 | Preprocessing | Done |
| 4 | 3D U-Net model + patching + losses | Done |
| 5 | Training | Done |
| 6 | Evaluation | Planned |
| 7 | Grad-CAM explainability | Planned |
| 8 | Report generation | Planned |

Current focus: evaluation (step 6) on the held-out test split after a full training run.

---

## Dataset

**Source:** [Medical Segmentation Decathlon — Task07_Pancreas](http://medicaldecathlon.com/)  
**Provider:** Memorial Sloan Kettering Cancer Center  
**License:** [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

After extracting `Task07_Pancreas.tar` (~11.45 GB), place the folder here:

```
data/Task07_Pancreas/
├── dataset.json
├── imagesTr/    # 281 training CT scans (.nii.gz)
├── labelsTr/    # matching labels: 0=background, 1=pancreas, 2=tumor
└── imagesTs/    # 139 unlabeled test scans
```

The `data/` directory is gitignored because of file size.

---

## Environment setup

Requires **Python 3.10+** (3.11 recommended).

```bash
# From the project root
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**GPU note:** PyTorch is listed without a CUDA-specific index. For GPU training later, install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) if `pip install torch` gives you CPU-only wheels.

---

## Step 2 — Load data & visualize

### Option A: Command line

```bash
python -m src.data_loading
```

Optional flags:

```bash
python -m src.data_loading \
  --data-dir ./data/Task07_Pancreas \
  --output-dir ./outputs/data_exploration \
  --num-cases 3 \
  --num-slices 4
```

This prints volume shape, voxel spacing, HU range, and label counts for each case, then saves PNGs to `outputs/data_exploration/`.

### Option B: Jupyter notebook

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

The notebook walks through loading a single scan, inspecting metadata, and generating overlays interactively.

---

## Step 3 — Preprocessing

Resamples to **1.0 × 1.0 × 2.5 mm**, applies soft-tissue HU window **[−100, 240] → [0, 1]**, crops a pancreas/tumor ROI with MONAI `CropForegroundd`, computes class balance, and writes a patient-level **70/15/15** split.

```bash
python -m src.preprocessing
```

**Portable split format:** `outputs/patient_split.json` stores **patient IDs only** (e.g. `"pancreas_270"`), never absolute paths. Training joins IDs with `data_dir` (`data/Task07_Pancreas` by default, or `--data-dir` on Kaggle/Colab).

**NIfTI extensions:** Path lookup accepts both `{id}.nii.gz` and `{id}.nii` (Kaggle often auto-decompresses). Prefer `.nii.gz` when both exist.

> If you still have an older `patient_split.json` that embeds Windows paths, **delete it and re-run preprocessing** before training:
> ```bash
> # Windows PowerShell
> Remove-Item outputs\patient_split.json
> python -m src.preprocessing
>
> # Linux / Kaggle
> rm outputs/patient_split.json
> python -m src.preprocessing
> ```
> Training will refuse to start and print the same instructions if it detects the legacy format.

Outputs:

| File | Purpose |
|------|---------|
| `outputs/patient_split.json` | Reproducible train/val/test **patient IDs** (format_version=2) |
| `outputs/class_balance.json` | Voxel counts + suggested CE weights |
| `outputs/preprocessing_summary.json` | Config + example before/after shapes |
| `outputs/data_exploration/*_preprocess_before_after.png` | Sanity-check visualization |

Reuse from training later:

```python
from src.preprocessing import PreprocessingPipeline, load_split, resolve_split_cases

pipe = PreprocessingPipeline()
case = pipe(image_path, label_path)
ids = load_split("outputs/patient_split.json")
train_cases = resolve_split_cases(ids["train"], data_dir="data/Task07_Pancreas")
```

---

## Step 4 — Model, patching, losses

```bash
python -m src.model   # builds UNet, checks output shape, smoke-tests losses + patches
```

| Piece | Defaults | Notes |
|-------|----------|-------|
| UNet | channels `(16,32,64,128,256)`, dropout `0.2` | ~4.8M params; scale channels up if you have more VRAM |
| Patches | `96×96×64`, pos:neg = 2:1 | pads small crops first; OOM fallback `64×64×48` |
| Loss | Dice + weighted CE | weights from `outputs/class_balance.json`; alt: `focal_tversky` |

---

## Step 5 — Training

Hyperparameters live in `src/config.py` (batch size, LR, loss, patience, patch size, …).

**Smoke test first** (2 epochs, 4 train / 2 val patients, small patches — confirms the loop runs):

```bash
python -m src.train --smoke-test
```

**Mid-size GPU dry run** (25 train / ~5 val patients, full patch size, timing for extrapolation):

```bash
python -m src.train --subset 25 --no-resume --output-dir outputs/training_subset25
```

**Full training** (after subset run looks healthy):

```bash
python -m src.train
# optional:
python -m src.train --loss focal_tversky --batch-size 1 --lr 1e-4
python -m src.train --resume          # continue from outputs/training/last.pt
# On Kaggle/Colab, point at your dataset mount:
python -m src.train --data-dir /kaggle/input/task07-pancreas/Task07_Pancreas
```

| Behavior | Detail |
|----------|--------|
| Train | Random `96×96×64` patches (pos:neg 2:1), AMP when CUDA is available |
| Val | Sliding-window over **full** preprocessed volumes (not random patches) |
| Primary metric | **Tumor Dice** (checkpoint + early stopping) |
| Logs | `outputs/training/train_log.csv` — includes `train_seconds`, `val_seconds`, `epoch_seconds` |
| Checkpoints | `best_tumor_dice.pt`, `last.pt` (resumable) |
| `--subset N` | First N train patients + proportional val (e.g. 25 → ~5 val) |
| VRAM probe | At startup on CUDA: one train step reports peak GB + batch-size hint |
| `tumor_patch_frac` | Logged each epoch — fraction of train patches with ≥1 tumor voxel |
| `pred_tumor_voxels` / `gt_tumor_voxels` | Val collapse detector — if pred → 0 while Dice is flat, model stopped predicting tumor |

**Tumor Dice stuck near 0?** Check data vs sampling first (do not change loss yet):

```bash
python -m src.diagnose_tumor --subset 25
```

Positive patch crops are keyed on **label==2 (tumor)**, not pancreas. Console/CSV report `tumor_patches=XX%` each epoch — expect roughly ~67% with pos:neg=2:1 when volumes contain tumor.

After sampling is confirmed (~90%+ tumor patches), try Focal Tversky and watch **both** Dice and predicted tumor voxel counts:

```bash
python -m src.train --subset 25 --epochs 20 --no-resume \
  --loss focal_tversky --batch-size 4 \
  --data-dir /path/to/Task07_Pancreas \
  --output-dir outputs/training_subset25_focal
```

- **Collapse:** `pred_tumor=0` (or falling toward 0) while Dice stays ~0
- **Progress:** `pred_tumor` stays nonzero and tumor Dice rises over epochs

### GPU memory (Colab T4, 16 GB)

With **batch=2**, **patch=96×96×64**, AMP, and the default ~4.8M-param U-Net, peak training VRAM is typically **~6–9 GB** — comfortable on a T4. At startup the script runs a one-step probe and prints actual peak/reserved memory on your GPU.

| Headroom after probe | Suggestion |
|----------------------|------------|
| > 6 GB free | Try `--batch-size 4` |
| 3–6 GB free | Try `--batch-size 3` |
| < 3 GB free | Keep batch=2 or use patch `64×64×48` |

Validation (sliding-window over full volumes) adds separate VRAM spikes; if val OOMs, keep `sw_batch_size=1` (default).

### Realistic expectations

This is a **small** labeled set (~197 training patients) for a hard 3D segmentation problem with extreme class imbalance (tumor ≈ 0.03% of voxels). Do not expect — or fabricate — clinical-grade Dice scores from a first research prototype. Report only the metrics the code actually measures on your run. The held-out **test** split stays untouched until evaluation (step 6).

---

## Project structure

```
PancScan AI/
├── data/                  # Dataset (gitignored)
├── notebooks/             # Exploration notebooks
├── outputs/               # Saved visualizations, models, reports
├── src/
│   ├── data_loading.py    # NIfTI I/O + slice visualization  ✓
│   ├── preprocessing.py   # Spacing, HU window, crop, split ✓
│   ├── model.py           # 3D U-Net (MONAI)                 ✓
│   ├── patching.py        # Fixed-size pos/neg patch crops   ✓
│   ├── losses.py          # Dice+CE and Focal Tversky        ✓
│   ├── config.py          # Training hyperparameters         ✓
│   ├── train.py           # Training loop                    ✓
│   ├── evaluate.py        # (step 6)
│   ├── gradcam.py         # (step 7)
│   └── visualize.py       # (step 8)
├── requirements.txt
└── README.md
```

---

## Memory & runtime expectations

3D segmentation is memory-intensive. If you hit CUDA OOM during training:

- Reduce **patch size** (e.g. `96×96×64` → `64×64×48` in `src/config.py`)
- Lower **batch size** to 1
- Use a **shallower U-Net** (`channels=(16,32,64,128)` in config)
- Prefer `dataset_cache="persistent"` so preprocessed volumes live on disk
- Train on **Google Colab** with a T4/A100 if local VRAM is tight

---

## References

- Antonelli, M., et al. (2022). The Medical Segmentation Decathlon. *Nature Communications*. [MSD paper](https://doi.org/10.1038/s41467-022-30695-9)
- Task07 dataset page: http://medicaldecathlon.com/

---

## License

Code in this repository: use and modify for research as you see fit.  
Task07_Pancreas data: **CC-BY-SA 4.0** — cite the Medical Segmentation Decathlon and MSKCC when publishing results.
