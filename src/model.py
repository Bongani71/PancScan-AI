"""
PancScan AI — 3D U-Net for pancreas / tumor segmentation (Step 4).

Why 3D U-Net (not 2D)?
----------------------
Pancreatic CT is a *volume*, not a stack of independent photos. A tumor that
looks ambiguous on one axial slice is often clearer when the network can see
the slices above and below it (continuity of the duct, vessel, or mass). A 2D
U-Net would treat each slice in isolation and throw away that through-plane
context. A 3D U-Net uses 3D convolutions so every prediction is conditioned on
a local neighborhood in X, Y, *and* Z — which matches how radiologists scroll
through a study.

Why patch-based training?
-------------------------
Even after Step 3 crops around the pancreas, volumes still vary in shape
(e.g. 163×97×74) and are too large to fit several full volumes on a consumer
GPU alongside 3D feature maps. We therefore train on fixed-size 3D patches
(default 96×96×64) sampled with a positive/negative bias so most patches
contain pancreas or tumor rather than empty background. At inference time we
can slide the same patch window (or use a sliding-window inferer in train /
evaluate) across the full cropped volume.

Memory guidance
---------------
Default channels (16, 32, 64, 128, 256) are sized for a single consumer GPU
(≈8–12 GB) with patch 96×96×64 and batch size 1–2. If you have more VRAM:
  - Scale channels to (32, 64, 128, 256, 512)
  - Or increase patch size toward (128, 128, 64)
If you OOM:
  - Drop to channels (16, 32, 64, 128) with strides (2, 2, 2)  [shallower]
  - Or patch (64, 64, 48), batch size 1
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets import UNet

# Number of segmentation classes in Task07_Pancreas labels.
NUM_CLASSES = 3  # 0=background, 1=pancreas, 2=tumor

# Default encoder channel widths — lighter for consumer GPUs.
# To scale up on a larger GPU: (32, 64, 128, 256, 512)
DEFAULT_CHANNELS: tuple[int, ...] = (16, 32, 64, 128, 256)

# One stride per downsampling stage; length must be len(channels) - 1.
DEFAULT_STRIDES: tuple[int, ...] = (2, 2, 2, 2)

# Dropout helps regularize with only ~197 training patients.
DEFAULT_DROPOUT = 0.2

# Residual units per level (MONAI); 2 is a good default for stability.
DEFAULT_NUM_RES_UNITS = 2


def build_unet(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = NUM_CLASSES,
    channels: Sequence[int] = DEFAULT_CHANNELS,
    strides: Sequence[int] | None = None,
    dropout: float = DEFAULT_DROPOUT,
    num_res_units: int = DEFAULT_NUM_RES_UNITS,
    norm: str | tuple = "INSTANCE",
) -> UNet:
    """
    Build a configurable 3D U-Net (MONAI) for multi-class segmentation.

    Parameters
    ----------
    spatial_dims:
        3 for volumetric CT. Keep at 3 for this project.
    in_channels:
        1 for single-modality CT.
    out_channels:
        3 logits: background, pancreas, tumor.
    channels:
        Feature widths at each encoder level. Longer / wider = more capacity
        and more GPU memory.
    strides:
        Downsampling factors between levels. Defaults to all-2s with length
        ``len(channels) - 1``.
    dropout:
        Dropout probability inside residual blocks (0 disables).
    num_res_units:
        Residual units per level (0 = classic U-Net blocks).
    norm:
        Normalization type. Instance norm is standard for small medical batches.

    Returns
    -------
    monai.networks.nets.UNet
        Network that maps (B, 1, H, W, D) → (B, 3, H, W, D) logits.
    """
    channels = tuple(channels)
    if strides is None:
        strides = tuple(2 for _ in range(len(channels) - 1))
    else:
        strides = tuple(strides)

    if len(strides) != len(channels) - 1:
        raise ValueError(
            f"len(strides) must be len(channels)-1; "
            f"got channels={channels}, strides={strides}"
        )

    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
        act="PRELU",
    )


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_sanity_check() -> None:
    """
    Build the default model, push a dummy patch through, check output shape.

    Also smoke-tests the default loss and a single random patch crop so Step 4
    is self-contained before train.py is wired up.
    """
    from src.losses import build_loss
    from src.patching import DEFAULT_PATCH_SIZE, extract_random_patches

    print("=" * 60)
    print("PancScan AI — model sanity check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_unet().to(device)
    model.eval()
    n_params = count_parameters(model)
    print(f"UNet channels={DEFAULT_CHANNELS}, dropout={DEFAULT_DROPOUT}")
    print(f"Trainable parameters: {n_params:,}")

    batch_size = 2
    patch = DEFAULT_PATCH_SIZE  # (H, W, D) in MONAI spatial order
    # Dummy input: (B, C, H, W, D)
    x = torch.randn(batch_size, 1, *patch, device=device)
    with torch.no_grad():
        logits = model(x)

    expected = (batch_size, NUM_CLASSES, *patch)
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(logits.shape)} (expected {expected})")
    if tuple(logits.shape) != expected:
        raise AssertionError(f"Shape mismatch: {tuple(logits.shape)} != {expected}")
    print("[OK] Forward pass shape")

    # Loss smoke test: labels as (B, 1, H, W, D) integer class maps.
    y = torch.randint(0, NUM_CLASSES, (batch_size, 1, *patch), device=device)
    loss_fn = build_loss("dice_ce").to(device)
    loss = loss_fn(logits, y)
    print(f"Dice+CE loss on dummy batch: {loss.item():.4f}")
    if not torch.isfinite(loss):
        raise AssertionError("Loss is not finite")
    print("[OK] Dice+CE loss")

    loss_ft = build_loss("focal_tversky").to(device)
    loss2 = loss_ft(logits, y)
    print(f"Focal Tversky loss on dummy batch: {loss2.item():.4f}")
    if not torch.isfinite(loss2):
        raise AssertionError("Focal Tversky loss is not finite")
    print("[OK] Focal Tversky loss")

    # Patching smoke test on a synthetic variable-size crop.
    vol_shape = (1, 163, 97, 74)  # similar to pancreas_290 after preprocess
    image = torch.rand(*vol_shape)
    # Sparse foreground so pos/neg sampling has something to find.
    label = torch.zeros(*vol_shape, dtype=torch.int64)
    label[:, 40:80, 30:60, 20:50] = 1  # pancreas
    label[:, 50:65, 40:55, 30:40] = 2  # tumor
    samples = extract_random_patches(
        image=image,
        label=label,
        spatial_size=patch,
        num_samples=2,
        pos=2,
        neg=1,
    )
    assert len(samples) == 2
    for i, s in enumerate(samples):
        assert s["image"].shape == (1, *patch), s["image"].shape
        assert s["label"].shape == (1, *patch), s["label"].shape
        print(f"  Patch {i}: image={tuple(s['image'].shape)}, label={tuple(s['label'].shape)}")
    print("[OK] Patch extraction")

    # Optional: shallower config note for OOM fallback
    small = build_unet(channels=(16, 32, 64, 128), strides=(2, 2, 2))
    print(
        f"OOM fallback model params: {count_parameters(small):,} "
        f"(channels=(16,32,64,128))"
    )
    print("=" * 60)
    print("All Step 4 sanity checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_sanity_check()
