# -*- coding: utf-8 -*-
"""Loss functions used by SpatialChord-Reg training."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    denom = torch.sqrt((pred_c.square().sum(0) * target_c.square().sum(0)).clamp_min(1e-8))
    corr = (pred_c * target_c).sum(0) / denom
    corr = torch.nan_to_num(corr, nan=0.0)
    return 1.0 - corr.mean()


def regression_loss(
    outputs: Dict[str, Optional[torch.Tensor]],
    target: torch.Tensor,
    zero_std: torch.Tensor,
    direct_head: bool,
    lambda_abund: float,
    lambda_corr: float,
) -> Dict[str, torch.Tensor]:
    pred = outputs["pred_std"]
    base = F.smooth_l1_loss(pred, target)
    corr = corr_loss(pred, target)

    if direct_head:
        total = base + lambda_corr * corr
        return {
            "loss": total,
            "loss_base": base.detach(),
            "loss_bce": target.new_tensor(0.0),
            "loss_abund": target.new_tensor(0.0),
            "loss_corr": corr.detach(),
        }

    fg_target = (target > zero_std.view(1, -1) + 1e-6).float()
    bce = F.binary_cross_entropy_with_logits(outputs["fg_logits"], fg_target)

    fg_mask = fg_target.bool()
    if fg_mask.any():
        abund = F.smooth_l1_loss(outputs["abundance_std"][fg_mask], target[fg_mask])
    else:
        abund = target.new_tensor(0.0)

    total = base + bce + lambda_abund * abund + lambda_corr * corr
    return {
        "loss": total,
        "loss_base": base.detach(),
        "loss_bce": bce.detach(),
        "loss_abund": abund.detach(),
        "loss_corr": corr.detach(),
    }
