"""
Fixed-size 3D patch sampling for PancScan AI training.

Preprocessing (Step 3) produces variable-shaped pancreas crops. Training needs
a fixed spatial size so batches stack cleanly on the GPU. This module wraps
MONAI's ``RandCropByPosNegLabeld`` to sample patches biased toward
pancreas/tumor (positive) voxels rather than pure background.

Default patch size is 96×96×64 (H×W×D). With spacing 1.0×1.0×2.5 mm that covers
roughly 96×96×160 mm — enough local context for pancreas head/body without
blowing up 3D U-Net memory.

If you OOM during training:
  - Try PATCH_SIZE_SMALL = (64, 64, 48)
  - Or reduce num_samples / batch size in train.py
"""

from __future__ import annotations

from typing import Sequence

import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandCropByPosNegLabeld,
    SpatialPadd,
)

# Spatial size as (H, W, D) — MONAI convention for 3D volumes with channel first.
DEFAULT_PATCH_SIZE: tuple[int, int, int] = (96, 96, 64)

# Lighter fallback if the default patch OOMs on a small GPU.
PATCH_SIZE_SMALL: tuple[int, int, int] = (64, 64, 48)

# Positive : negative sampling ratio for RandCropByPosNegLabeld.
# pos=2, neg=1 → ~2/3 of patches centered on foreground (label > 0).
DEFAULT_POS = 2
DEFAULT_NEG = 1

KEYS_IMAGE = "image"
KEYS_LABEL = "label"


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
    then samples with a foreground bias via RandCropByPosNegLabeld.

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
            RandCropByPosNegLabeld(
                keys=[KEYS_IMAGE, KEYS_LABEL],
                label_key=KEYS_LABEL,
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
        Foreground/background sampling weights for RandCropByPosNegLabeld.

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
