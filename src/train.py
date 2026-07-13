"""
PancScan AI — training loop (Step 5).

Trains the 3D U-Net on random patches, validates with sliding-window inference
over full preprocessed volumes, and checkpoints on **validation tumor Dice**
(not mean Dice — background would otherwise dominate).

Quick smoke test (1–2 epochs, tiny subset)::

    python -m src.train --smoke-test

Full training (after smoke test looks healthy)::

    python -m src.train

Resume from ``outputs/training/last.pt`` (default if the file exists)::

    python -m src.train --resume

Memory tips if you OOM
----------------------
- ``batch_size=1``, ``patch_size=(64,64,48)``, or shallower channels in config
- ``dataset_cache="persistent"`` to keep preprocessed volumes on disk, not RAM
- Mixed precision is on by default when CUDA is available
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    PersistentDataset,
    decollate_batch,
    list_data_collate,
)
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# AMP API differs slightly across PyTorch versions
try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

from src.config import TrainConfig, smoke_config
from src.losses import build_loss
from src.model import NUM_CLASSES, build_unet, count_parameters
from src.patching import KEYS_IMAGE, KEYS_LABEL, build_train_patch_transform
from src.preprocessing import (
    build_deterministic_transforms,
    load_split,
    resolve_split_cases,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_patient_subset(
    train_cases: list,
    val_cases: list,
    n_train: int,
) -> tuple[list, list]:
    """
    Take the first ``n_train`` patients from the train split and a proportional
    slice of validation patients (preserves the original train:val ratio).

    Works on lists of patient ID strings or path dicts.

    Example: full split is 197 train / 42 val (~4.6:1). ``--subset 25`` →
    25 train / ~5 val (at least 1 val case if any exist).
    """
    n_train = max(1, min(int(n_train), len(train_cases)))
    train_subset = train_cases[:n_train]

    if not val_cases:
        return train_subset, []

    ratio = len(val_cases) / max(len(train_cases), 1)
    n_val = max(1, int(round(n_train * ratio)))
    n_val = min(n_val, len(val_cases))
    val_subset = val_cases[:n_val]
    return train_subset, val_subset


def probe_training_vram(cfg: TrainConfig, device: torch.device) -> dict[str, Any]:
    """
    One forward+backward step to measure peak VRAM for the current batch/patch.

    Called at training start when CUDA is available so you know whether to
    raise batch size before a long run.
    """
    if device.type != "cuda":
        return {"cuda": False}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = build_unet(channels=cfg.channels, dropout=cfg.dropout).to(device)
    loss_fn = build_loss(cfg.loss_name).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    images = torch.randn(
        cfg.batch_size, 1, *cfg.patch_size, device=device, dtype=torch.float32
    )
    labels = torch.randint(
        0, NUM_CLASSES, (cfg.batch_size, 1, *cfg.patch_size), device=device
    )

    use_amp = bool(cfg.use_amp)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, use_amp):
        logits = model(images)
        loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    reserved_bytes = torch.cuda.max_memory_reserved(device)

    del model, loss_fn, optimizer, images, labels, logits, loss
    torch.cuda.empty_cache()

    peak_gb = peak_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    reserved_gb = reserved_bytes / (1024**3)
    headroom_gb = total_gb - reserved_gb

    return {
        "cuda": True,
        "device_name": torch.cuda.get_device_name(device),
        "batch_size": cfg.batch_size,
        "patch_size": list(cfg.patch_size),
        "peak_allocated_gb": round(peak_gb, 2),
        "peak_reserved_gb": round(reserved_gb, 2),
        "total_vram_gb": round(total_gb, 2),
        "headroom_gb": round(headroom_gb, 2),
    }


def print_vram_guidance(info: dict[str, Any]) -> None:
    """Print VRAM probe results and batch-size suggestions."""
    if not info.get("cuda"):
        print(
            "GPU: not available on this machine. "
            "On Colab T4 (16 GB), batch=2 patch=96x96x64 with AMP typically "
            "uses ~6-9 GB peak - comfortable. Try batch=3-4 if headroom > 6 GB; "
            "batch=8 will likely OOM."
        )
        return

    print(
        f"GPU VRAM probe ({info['device_name']}): "
        f"batch={info['batch_size']}, patch={tuple(info['patch_size'])}, AMP on"
    )
    print(
        f"  Peak allocated: {info['peak_allocated_gb']:.2f} GB | "
        f"Peak reserved: {info['peak_reserved_gb']:.2f} GB | "
        f"Total: {info['total_vram_gb']:.2f} GB | "
        f"Headroom: {info['headroom_gb']:.2f} GB"
    )

    headroom = info["headroom_gb"]
    bs = info["batch_size"]
    if headroom >= 6.0:
        suggestion = f"Comfortable - try batch={bs + 2} or {bs + 4} next"
    elif headroom >= 3.0:
        suggestion = f"Moderate headroom - try batch={bs + 1} cautiously"
    else:
        suggestion = f"Tight - keep batch={bs} or reduce patch size"
    print(f"  Suggestion: {suggestion}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _cases_to_monai_dicts(cases: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert split JSON entries to MONAI LoadImaged-friendly dicts."""
    return [
        {
            KEYS_IMAGE: c["image"],
            KEYS_LABEL: c["label"],
            "patient_id": c["patient_id"],
        }
        for c in cases
    ]


class _PatchFromCacheDataset(Dataset):
    """
    Apply random patch cropping on top of already-preprocessed volumes.

    Cache/Persistent datasets must NOT include RandCrop (or every epoch would
    replay the same cached patches). This wrapper keeps preprocessing cached
    and re-samples patches each ``__getitem__`` call.
    """

    def __init__(self, base_ds: Any, patch_transform: Compose) -> None:
        self.base_ds = base_ds
        self.patch_transform = patch_transform

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, index: int) -> Any:
        item = self.base_ds[index]
        # Keep only tensor fields the patch transform needs (+ optional id).
        sample = {
            KEYS_IMAGE: item[KEYS_IMAGE],
            KEYS_LABEL: item[KEYS_LABEL],
        }
        out = self.patch_transform(sample)
        # RandCrop with num_samples>1 returns a list; list_data_collate flattens.
        return out


def build_cached_preprocess_dataset(
    cases: list[dict[str, str]],
    cfg: TrainConfig,
) -> Any:
    """Build CacheDataset / PersistentDataset / Dataset for deterministic preprocess."""
    data = _cases_to_monai_dicts(cases)
    transform = build_deterministic_transforms(as_tensor=True)

    mode = cfg.dataset_cache.lower()
    if mode == "cache":
        return CacheDataset(data=data, transform=transform, cache_rate=1.0, num_workers=0)
    if mode == "persistent":
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        return PersistentDataset(data=data, transform=transform, cache_dir=str(cfg.cache_dir))
    return Dataset(data=data, transform=transform)


def build_train_loader(cases: list[dict[str, str]], cfg: TrainConfig) -> DataLoader:
    base = build_cached_preprocess_dataset(cases, cfg)
    patch_tf = build_train_patch_transform(
        spatial_size=cfg.patch_size,
        pos=cfg.pos_ratio,
        neg=cfg.neg_ratio,
        num_samples=cfg.patches_per_volume,
    )
    ds = _PatchFromCacheDataset(base, patch_tf)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )


def build_val_loader(cases: list[dict[str, str]], cfg: TrainConfig) -> DataLoader:
    """Validation loads full preprocessed volumes (no random patches)."""
    base = build_cached_preprocess_dataset(cases, cfg)
    # Ensure tensor dtype; base already EnsureTyped, but wrap for safety.
    return DataLoader(
        base,
        batch_size=1,  # full volumes; variable shape → must be 1
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _per_class_dice_from_metric(dice_metric: DiceMetric) -> dict[str, float]:
    """
    Aggregate DiceMetric buffer into mean per-class scores.

    Expects the metric to have been called with one-hot preds/labels and
    reduction that yields per-class values (we use reduction='mean_batch'
    after stacking, or compute manually from 'none').
    """
    # Shape after aggregate depends on reduction; use get_buffer for robustness.
    buf = dice_metric.get_buffer()  # (N, C) when reduction is none-like
    if buf is None or len(buf) == 0:
        return {"background": float("nan"), "pancreas": float("nan"), "tumor": float("nan")}

    scores = torch.nanmean(buf, dim=0).detach().cpu().numpy()
    names = ["background", "pancreas", "tumor"]
    out = {}
    for i, name in enumerate(names):
        out[name] = float(scores[i]) if i < len(scores) else float("nan")
    return out


def _autocast(device: torch.device, enabled: bool):
    """Version-tolerant autocast context."""
    if not enabled:
        from contextlib import nullcontext

        return nullcontext()
    try:
        return autocast(device_type=device.type, enabled=True)
    except TypeError:
        return autocast(enabled=True)


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    scaler: GradScaler | None,
) -> float:
    """Run ``cfg.steps_per_epoch`` optimizer steps; return mean loss."""
    model.train()
    iterator = iter(loader)
    losses: list[float] = []
    use_amp = bool(cfg.use_amp and device.type == "cuda" and scaler is not None)

    for _step in range(cfg.steps_per_epoch):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        images = batch[KEYS_IMAGE].to(device)
        labels = batch[KEYS_LABEL].to(device)
        # Labels may be float meta-tensors; loss expects integer class maps.
        if labels.dtype != torch.int64 and labels.dtype != torch.long:
            labels = labels.long()

        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
) -> tuple[float, dict[str, float]]:
    """
    Sliding-window inference on full validation volumes.

    Returns (mean_val_loss, per_class_dice_dict).
    Primary metric for checkpointing: per_class_dice_dict['tumor'].
    """
    model.eval()
    inferer = SlidingWindowInferer(
        roi_size=cfg.patch_size,
        sw_batch_size=cfg.sw_batch_size,
        overlap=cfg.sw_overlap,
        mode="gaussian",
    )
    # reduction="none" → buffer shape (N_images, C) for per-class reporting
    dice_metric = DiceMetric(
        include_background=True,
        reduction="none",
        ignore_empty=True,
        num_classes=NUM_CLASSES,
    )
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    val_losses: list[float] = []
    use_amp = bool(cfg.use_amp and device.type == "cuda")

    for batch in loader:
        images = batch[KEYS_IMAGE].to(device)
        labels = batch[KEYS_LABEL].to(device)
        if labels.dtype != torch.int64 and labels.dtype != torch.long:
            labels = labels.long()

        with _autocast(device, use_amp):
            logits = inferer(images, model)
            loss = loss_fn(logits, labels)
        val_losses.append(float(loss.detach().item()))

        # Decollate batch dim for metric transforms
        logit_list = decollate_batch(logits)
        label_list = decollate_batch(labels)
        preds = [post_pred(y) for y in logit_list]
        labs = [post_label(y) for y in label_list]
        dice_metric(y_pred=preds, y=labs)

    per_class = _per_class_dice_from_metric(dice_metric)
    dice_metric.reset()
    mean_loss = float(np.mean(val_losses)) if val_losses else float("nan")
    return mean_loss, per_class


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler | None,
    epoch: int,
    best_tumor_dice: float,
    cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_tumor_dice": best_tumor_dice,
        "config": cfg.to_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler | None,
    device: torch.device,
) -> tuple[int, float]:
    """Restore training state. Returns (start_epoch, best_tumor_dice)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best = float(ckpt.get("best_tumor_dice", float("-inf")))
    return start_epoch, best


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def append_csv_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output dir: {cfg.output_dir}")

    # Save config snapshot for reproducibility
    with (cfg.output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    split = load_split(cfg.split_path)
    train_ids = split["train"]
    val_ids = split["val"]

    if cfg.smoke_test:
        train_ids = train_ids[: cfg.smoke_train_cases]
        val_ids = val_ids[: cfg.smoke_val_cases]
        print(
            f"SMOKE TEST: using {len(train_ids)} train / {len(val_ids)} val cases, "
            f"{cfg.max_epochs} epochs x {cfg.steps_per_epoch} steps, "
            f"patch={cfg.patch_size}"
        )
    elif cfg.subset_n is not None:
        full_train, full_val = len(train_ids), len(val_ids)
        train_ids, val_ids = apply_patient_subset(train_ids, val_ids, cfg.subset_n)
        print(
            f"SUBSET: using {len(train_ids)}/{full_train} train / "
            f"{len(val_ids)}/{full_val} val patients (--subset {cfg.subset_n})"
        )

    # Reconstruct paths from configurable data_dir (portable across OS/machines).
    print(f"Data root (data_dir): {cfg.data_dir}")
    train_cases = resolve_split_cases(train_ids, cfg.data_dir)
    val_cases = resolve_split_cases(val_ids, cfg.data_dir)

    print(f"Train patients: {len(train_cases)} | Val patients: {len(val_cases)}")
    print(
        f"Loss={cfg.loss_name}, lr={cfg.learning_rate}, batch={cfg.batch_size}, "
        f"cache={cfg.dataset_cache}"
    )

    print("Building datasets (first run may preprocess & cache - can take a while)...")
    train_loader = build_train_loader(train_cases, cfg)
    val_loader = build_val_loader(val_cases, cfg)

    model = build_unet(channels=cfg.channels, dropout=cfg.dropout).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    vram_info = probe_training_vram(cfg, device)
    print_vram_guidance(vram_info)

    loss_fn = build_loss(cfg.loss_name).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler == "cosine":
        scheduler: Any = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    elif cfg.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    else:
        scheduler = None

    use_amp = bool(cfg.use_amp and device.type == "cuda")
    try:
        scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None
    except TypeError:
        # Older torch.cuda.amp.GradScaler has no device arg
        scaler = GradScaler(enabled=use_amp) if use_amp else None
    if cfg.use_amp and device.type != "cuda":
        print("Note: AMP requested but CUDA unavailable - training in full precision.")

    start_epoch = 0
    best_tumor_dice = float("-inf")
    last_path = cfg.output_dir / cfg.last_checkpoint_name
    if cfg.resume and last_path.exists():
        print(f"Resuming from {last_path}")
        start_epoch, best_tumor_dice = load_checkpoint(
            last_path, model, optimizer, scheduler, scaler, device
        )
        print(f"  Resumed at epoch {start_epoch}, best tumor Dice={best_tumor_dice:.4f}")

    log_path = cfg.output_dir / cfg.log_csv_name
    epochs_without_improve = 0
    history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, cfg.max_epochs):
        t_epoch = time.time()

        t_train = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, cfg, scaler
        )
        train_seconds = time.time() - t_train

        do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.max_epochs - 1)
        val_seconds = 0.0
        if do_val:
            t_val = time.time()
            val_loss, dice = validate(model, val_loader, loss_fn, device, cfg)
            val_seconds = time.time() - t_val
            tumor_dice = dice["tumor"]
        else:
            val_loss, dice = float("nan"), {
                "background": float("nan"),
                "pancreas": float("nan"),
                "tumor": float("nan"),
            }
            tumor_dice = float("nan")

        # LR schedule
        if isinstance(scheduler, ReduceLROnPlateau):
            if do_val and not np.isnan(tumor_dice):
                scheduler.step(tumor_dice)
        elif scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - t_epoch

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if do_val else "",
            "dice_background": round(dice["background"], 6) if do_val else "",
            "dice_pancreas": round(dice["pancreas"], 6) if do_val else "",
            "dice_tumor": round(dice["tumor"], 6) if do_val else "",
            "lr": lr_now,
            "train_seconds": round(train_seconds, 1),
            "val_seconds": round(val_seconds, 1),
            "epoch_seconds": round(epoch_seconds, 1),
        }
        append_csv_log(log_path, row)
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{cfg.max_epochs - 1} | "
            f"train_loss={train_loss:.4f} | "
            + (
                f"val_loss={val_loss:.4f} | "
                f"Dice bg={dice['background']:.3f} "
                f"pancreas={dice['pancreas']:.3f} "
                f"tumor={dice['tumor']:.3f} | "
                if do_val
                else "val=skipped | "
            )
            + f"lr={lr_now:.2e} | "
            f"time: train={train_seconds:.1f}s val={val_seconds:.1f}s "
            f"epoch={epoch_seconds:.1f}s"
        )

        # Always save last (for resume)
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_tumor_dice=best_tumor_dice,
            cfg=cfg,
        )

        if do_val and not np.isnan(tumor_dice) and tumor_dice > best_tumor_dice:
            best_tumor_dice = tumor_dice
            epochs_without_improve = 0
            best_path = cfg.output_dir / cfg.checkpoint_name
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_tumor_dice=best_tumor_dice,
                cfg=cfg,
            )
            print(f"  -> New best tumor Dice={best_tumor_dice:.4f} saved to {best_path.name}")
        elif do_val:
            epochs_without_improve += 1
            print(
                f"  -> No tumor Dice improvement "
                f"({epochs_without_improve}/{cfg.early_stopping_patience})"
            )
            if epochs_without_improve >= cfg.early_stopping_patience:
                print("Early stopping triggered.")
                break

    summary = {
        "best_tumor_dice": best_tumor_dice if best_tumor_dice > float("-inf") else None,
        "epochs_run": len(history),
        "log_csv": str(log_path),
        "best_checkpoint": str(cfg.output_dir / cfg.checkpoint_name),
        "device": str(device),
        "smoke_test": cfg.smoke_test,
        "subset_n": cfg.subset_n,
        "vram_probe": vram_info,
    }
    if history:
        mean_epoch = float(np.mean([h["epoch_seconds"] for h in history]))
        summary["mean_epoch_seconds"] = round(mean_epoch, 1)
        summary["estimated_full_run_hours"] = round(
            mean_epoch * cfg.max_epochs / 3600, 2
        )
        print(
            f"\nTiming: mean epoch={mean_epoch:.1f}s | "
            f"extrapolated {cfg.max_epochs} epochs ~ "
            f"{mean_epoch * cfg.max_epochs / 3600:.1f} h"
        )
    with (cfg.output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining finished.")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PancScan AI 3D U-Net")
    p.add_argument("--smoke-test", action="store_true", help="2-epoch tiny subset dry run")
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        metavar="N",
        help="Train on first N patients (+ proportional val slice), e.g. --subset 25",
    )
    p.add_argument("--resume", action="store_true", help="Resume from last.pt if present")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--loss", type=str, default=None, choices=["dice_ce", "focal_tversky"])
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument(
        "--cache",
        type=str,
        default=None,
        choices=["cache", "persistent", "none"],
        help="Dataset caching strategy",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--split", type=Path, default=None)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Task07_Pancreas root (contains imagesTr/, labelsTr/). "
        "Default: data/Task07_Pancreas relative to project root.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = smoke_config() if args.smoke_test else TrainConfig()

    if args.resume:
        cfg.resume = True
    if args.no_resume:
        cfg.resume = False
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.loss is not None:
        cfg.loss_name = args.loss
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.patience is not None:
        cfg.early_stopping_patience = args.patience
    if args.cache is not None:
        cfg.dataset_cache = args.cache
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.split is not None:
        cfg.split_path = args.split
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.subset is not None:
        cfg.subset_n = args.subset
        if not args.resume:
            cfg.resume = False  # fresh subset run unless --resume explicitly set

    run_training(cfg)


if __name__ == "__main__":
    main()
