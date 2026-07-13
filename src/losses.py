"""
Loss functions for PancScan AI multi-class segmentation.

Default: Dice + weighted Cross-Entropy (MONAI DiceCELoss).
Alternative: Focal Tversky — better when tumor recall is the bottleneck,
given tumor voxels are only ~0.029% of the raw training set.

Class weights default to the inverse-frequency values computed in Step 3
(outputs/class_balance.json): background ≈ 0.001, pancreas ≈ 0.39, tumor ≈ 2.61.
Override via ``ce_weights=...`` or ``build_loss(..., ce_weights=...)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, TverskyLoss

from src.data_loading import get_project_root

# Fallback if class_balance.json is missing (same order as suggested weights).
DEFAULT_CE_WEIGHTS: tuple[float, float, float] = (0.001, 0.39, 2.61)


def load_ce_weights(
    path: Path | None = None,
    fallback: Sequence[float] = DEFAULT_CE_WEIGHTS,
) -> tuple[float, float, float]:
    """
    Load suggested CE weights from outputs/class_balance.json if present.

    Returns (background, pancreas, tumor) as floats.
    """
    if path is None:
        path = get_project_root() / "outputs" / "class_balance.json"
    path = Path(path)
    if not path.exists():
        return tuple(float(w) for w in fallback)  # type: ignore[return-value]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    w = data["suggested_ce_weights"]
    return (
        float(w["background"]),
        float(w["pancreas"]),
        float(w["tumor"]),
    )


def build_dice_ce_loss(
    ce_weights: Sequence[float] | None = None,
    include_background: bool = True,
    lambda_dice: float = 1.0,
    lambda_ce: float = 1.0,
) -> DiceCELoss:
    """
    Combined Dice + weighted Cross-Entropy (default training loss).

    Dice handles overlap quality; weighted CE pushes the rare tumor class
    harder than background. Labels are expected as integer maps with shape
    (B, 1, H, W, D); logits as (B, 3, H, W, D).
    """
    weights = list(ce_weights) if ce_weights is not None else list(load_ce_weights())
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    return DiceCELoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
        weight=weight_tensor,
        lambda_dice=lambda_dice,
        lambda_ce=lambda_ce,
        squared_pred=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for highly imbalanced segmentation.

    Tversky index generalizes Dice with separate false-positive (alpha) and
    false-negative (beta) weights. Setting beta > alpha penalizes misses more
    than false alarms — useful when missing a tumor is worse than over-
    calling. The focal term (1 − TI)^gamma further focuses training on hard
    examples.

    Default alpha=0.3, beta=0.7, gamma=0.75 follows common medical-imaging
    practice for rare lesions. Tune beta upward if tumor sensitivity is low.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        include_background: bool = True,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        # MONAI TverskyLoss returns the loss L = 1 − TI (higher = worse).
        self.tversky = TverskyLoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            alpha=alpha,
            beta=beta,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits:
            (B, C, H, W, D) raw network outputs.
        target:
            (B, 1, H, W, D) integer class labels.
        """
        # Focal Tversky: (1 − TI)^γ = L^γ when L = 1 − TI.
        base = self.tversky(logits, target)
        return torch.pow(base, self.gamma)


def build_focal_tversky_loss(
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    include_background: bool = True,
) -> FocalTverskyLoss:
    """Factory for Focal Tversky (alternative to Dice+CE)."""
    return FocalTverskyLoss(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        include_background=include_background,
    )


def build_loss(
    name: str = "dice_ce",
    ce_weights: Sequence[float] | None = None,
    **kwargs,
) -> nn.Module:
    """
    Select a loss by name for train.py.

    Parameters
    ----------
    name:
        ``"dice_ce"`` (default) or ``"focal_tversky"``.
    ce_weights:
        Optional override for Dice+CE class weights (length 3).
    **kwargs:
        Forwarded to the underlying builder (lambda_dice, alpha, beta, …).
    """
    key = name.lower().strip().replace("-", "_")
    if key in {"dice_ce", "dicece", "dice_ce_loss"}:
        return build_dice_ce_loss(ce_weights=ce_weights, **kwargs)
    if key in {"focal_tversky", "focaltversky", "ft"}:
        # ce_weights not used by Focal Tversky — drop if passed.
        kwargs.pop("lambda_dice", None)
        kwargs.pop("lambda_ce", None)
        return build_focal_tversky_loss(**kwargs)
    raise ValueError(
        f"Unknown loss '{name}'. Use 'dice_ce' or 'focal_tversky'."
    )
