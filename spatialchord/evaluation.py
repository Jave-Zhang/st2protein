# -*- coding: utf-8 -*-
"""Prediction-space conversion and metrics for SpatialChord-Reg."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import compute_pcc, compute_rmse, compute_spearman


def destandardize(pred_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return pred_std * std + mean


def clean_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def foreground_metrics(
    true_log: np.ndarray,
    fg_score: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if fg_score is None:
        fg_score = true_log * 0.0
    true_fg = true_log > 0
    n_proteins = true_log.shape[1]
    auroc = np.full(n_proteins, np.nan, dtype=np.float32)
    auprc = np.full(n_proteins, np.nan, dtype=np.float32)
    for j in range(n_proteins):
        labels = true_fg[:, j].astype(np.int32)
        if labels.min() == labels.max():
            continue
        auroc[j] = roc_auc_score(labels, fg_score[:, j])
        auprc[j] = average_precision_score(labels, fg_score[:, j])
    return auroc, auprc, float(np.nanmean(auroc)), float(np.nanmean(auprc))


def safe_row_corr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred_c = pred - pred.mean(axis=1, keepdims=True)
    true_c = true - true.mean(axis=1, keepdims=True)
    denom = np.sqrt((pred_c ** 2).sum(axis=1) * (true_c ** 2).sum(axis=1))
    out = np.full(pred.shape[0], np.nan, dtype=np.float32)
    valid = denom > 1e-8
    out[valid] = ((pred_c[valid] * true_c[valid]).sum(axis=1) / denom[valid]).astype(np.float32)
    return out


def evaluate_predictions(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    fg_score: Optional[np.ndarray],
    protein_names: List[str],
) -> Dict:
    pcc_prot, pcc_cell, pcc_prot_mean, pcc_cell_mean = compute_pcc(
        pred_log, true_log, protein_names
    )
    sp_prot, sp_cell, sp_prot_mean, sp_cell_mean = compute_spearman(pred_log, true_log)
    rmse_prot, rmse_cell, rmse_prot_mean, rmse_cell_mean, rmse_global = compute_rmse(
        pred_log, true_log
    )
    auroc, auprc, auroc_mean, auprc_mean = foreground_metrics(true_log, fg_score)

    per_protein = {}
    for i, name in enumerate(protein_names):
        per_protein[name] = {
            "pcc": float(pcc_prot[i]),
            "spearman": float(sp_prot[i]),
            "rmse": float(rmse_prot[i]),
            "fg_auroc": float(auroc[i]),
            "fg_auprc": float(auprc[i]),
        }

    return clean_json({
        "evaluation_space": "arcsinh_protein",
        "foreground_label": "true_arcsinh_protein_gt_0",
        "foreground_score": "model_foreground_probability",
        "pcc_protein_mean": pcc_prot_mean,
        "pcc_cell_mean": pcc_cell_mean,
        "spearman_protein_mean": sp_prot_mean,
        "spearman_cell_mean": sp_cell_mean,
        "rmse_protein_mean": rmse_prot_mean,
        "rmse_cell_mean": rmse_cell_mean,
        "rmse_global": rmse_global,
        "fg_auroc_mean": auroc_mean,
        "fg_auprc_mean": auprc_mean,
        "per_protein": per_protein,
    })
