"""
Fixed-size 3D patch sampling for PancScan AI training.

Preprocessing (Step 3) produces variable-shaped pancreas crops. Training needs
a fixed spatial size so batches stack cleanly on the GPU. This module wraps
MONAI's ``RandCropByPosNegLabeld`` to sample patches.

IMPORTANT — positive sampling is keyed on **tumor (label == 2)**, not
``label > 0``. MONAI's default treats any nonzero label as foreground, which
would mostly center patches on abundant pancreas tissue and starve the rare
tumor class. We therefore build a binary tumor mask and pass that as
``label_key`` while still cropping the original multi-class label for loss.

Default patch size is 96×96×64 (H×W×D). With spacing 1.0×1.0×2.5 mm that covers
roughly 96×96×160 mm — enough local context for pancreas head/body without
blowing up 3D U-Net memory.

If you OOM during training:
  - Try PATCH_SIZE_SMALL = (64, 64, 48)
  - Or reduce num_samples / batch size in train.py
"""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    MapTransform,
    RandCropByPosNegLabeld,
    SpatialPadd,
)

from src.data_loading import LABEL_TUMOR

# Spatial size as (H, W, D) — MONAI convention for 3D volumes with channel first.
DEFAULT_PATCH_SIZE: tuple[int, int, int] = (96, 96, 64)

# Lighter fallback if the default patch OOMs on a small GPU.
PATCH_SIZE_SMALL: tuple[int, int, int] = (64, 64, 48)

# Positive : negative sampling ratio for RandCropByPosNegLabeld.
# pos=2, neg=1 → ~2/3 of patches centered on **tumor** voxels (label == 2).
DEFAULT_POS = 2
DEFAULT_NEG = 1

KEYS_IMAGE = "image"
KEYS_LABEL = "label"
# Temporary binary mask used only for crop-center sampling (not for loss).
KEYS_TUMOR_FG = "tumor_fg"


class AddTumorForegroundd(MapTransform):
    """
    Create a binary tumor foreground mask for RandCropByPosNegLabeld.

    ``tumor_fg = 1`` where ``label == 2`` (tumor), else ``0``.
    MONAI's pos/neg cropper treats any nonzero voxel as "positive", so this
    mask makes positive crops tumor-centered instead of pancreas-centered.
    """

    def __init__(self, label_key: str = KEYS_LABEL, fg_key: str = KEYS_TUMOR_FG) -> None:
        super().__init__(keys=[label_key])
        self.label_key = label_key
        self.fg_key = fg_key

    def __call__(self, data: Mapping[Hashable, object]) -> dict:
        d = dict(data)
        label = d[self.label_key]
        if isinstance(label, torch.Tensor):
            d[self.fg_key] = (label == LABEL_TUMOR).to(dtype=label.dtype)
        else:
            arr = np.asarray(label)
            d[self.fg_key] = (arr == LABEL_TUMOR).astype(arr.dtype, copy=False)
        return d


def build_train_patch_transform(
    spatial_size: Sequence[int] = DEFAULT_PATCH_SIZE,
    pos: float = DEFAULT_POS,
    neg: float = DEFAULT_NEG,
    num_samples: int = 4,
) -> Compose:
    """
    Build a MONAI transform that yields ``num_samples`` random patches.

    Expects dict samples already channel-first, e.g. from PreprocessingPipeline:
        {"image": (1, H, W, D), "label": (1, H, W, D)}

    Pads first so volumes smaller than ``spatial_size`` still produce a patch,
    then samples with a **tumor-biased** RandCropByPosNegLabeld.

    If a volume has zero tumor voxels, MONAI finds an empty FG index list and
    the "positive" draws fall back to background/image region — those cases
    cannot contribute tumor-centered patches (see diagnose_tumor.py).

    Returns
    -------
    Compose
        When called on one volume dict, returns a *list* of dicts (MONAI
        behavior when num_samples > 1), each with fixed-size image/label.
    """
    spatial_size = tuple(int(s) for s in spatial_size)
    return Compose(
        [
            # Guarantee the volume is at least as large as the patch.
            SpatialPadd(
                keys=[KEYS_IMAGE, KEYS_LABEL],
                spatial_size=spatial_size,
                mode="constant",
            ),
            # Binary tumor mask → label_key for pos/neg crop centers.
            AddTumorForegroundd(label_key=KEYS_LABEL, fg_key=KEYS_TUMOR_FG),
            SpatialPadd(
                keys=[KEYS_TUMOR_FG],
                spatial_size=spatial_size,
                mode="constant",
            ),
            RandCropByPosNegLabeld(
                keys=[KEYS_IMAGE, KEYS_LABEL],
                label_key=KEYS_TUMOR_FG,  # positive = tumor (label==2), NOT label>0
                spatial_size=spatial_size,
                pos=pos,
                neg=neg,
                num_samples=num_samples,
                image_key=KEYS_IMAGE,
                image_threshold=0.0,
                allow_smaller=False,  # we already padded above
            ),
            EnsureTyped(keys=[KEYS_IMAGE, KEYS_LABEL], data_type="tensor"),
        ]
    )


def build_val_patch_transform(
    spatial_size: Sequence[int] = DEFAULT_PATCH_SIZE,
    num_samples: int = 4,
) -> Compose:
    """
    Validation patch sampler — still random, but same API as training.

    For a fully deterministic validation metric over the whole volume, prefer
    sliding-window inference in evaluate.py later. This helper is useful when
    you want a quick epoch-end Dice on a few patches without running the
    full volume.
    """
    # Same as train but typically fewer samples; seed can be set by caller.
    return build_train_patch_transform(
        spatial_size=spatial_size,
        pos=DEFAULT_POS,
        neg=DEFAULT_NEG,
        num_samples=num_samples,
    )


def patch_contains_tumor(label: torch.Tensor | np.ndarray) -> bool:
    """Return True if a patch label map contains at least one tumor voxel."""
    if isinstance(label, torch.Tensor):
        return bool((label == LABEL_TUMOR).any().item())
    return bool(np.any(np.asarray(label) == LABEL_TUMOR))


def extract_random_patches(
    image: torch.Tensor | object,
    label: torch.Tensor | object,
    spatial_size: Sequence[int] = DEFAULT_PATCH_SIZE,
    num_samples: int = 4,
    pos: float = DEFAULT_POS,
    neg: float = DEFAULT_NEG,
) -> list[dict]:
    """
    Convenience wrapper: sample patches from one preprocessed volume.

    Parameters
    ----------
    image, label:
        Channel-first arrays/tensors shaped (1, H, W, D).
    spatial_size:
        Patch size (H, W, D).
    num_samples:
        How many patches to draw from this volume.
    pos, neg:
        Foreground/background sampling weights for RandCropByPosNegLabeld
        (foreground = tumor voxels).

    Returns
    -------
    list of dict
        Each item has ``image`` and ``label`` tensors of shape
        (1, *spatial_size).
    """
    transform = build_train_patch_transform(
        spatial_size=spatial_size,
        pos=pos,
        neg=neg,
        num_samples=num_samples,
    )
    out = transform({KEYS_IMAGE: image, KEYS_LABEL: label})
    # MONAI returns a list when num_samples > 1.
    if isinstance(out, dict):
        return [out]
    return list(out)
