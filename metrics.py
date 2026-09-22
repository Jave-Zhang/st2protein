# -*- coding: utf-8 -*-
"""
Evaluation metrics: PCC (Pearson), Spearman rank correlation and RMSE.
Both are computed per-protein and per-cell.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
from typing import Optional, Tuple


def compute_pcc(
    pred: np.ndarray,
    true: np.ndarray,
    protein_names: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Pearson correlation coefficient per-protein and per-cell.

    Returns:
        pcc_protein [M], pcc_cell [N], mean_protein, mean_cell
    """
    n_cells, n_proteins = pred.shape

    pcc_protein = np.array([
        pearsonr(pred[:, j], true[:, j]).statistic
        if np.std(pred[:, j]) > 1e-8 and np.std(true[:, j]) > 1e-8 else 0.0
        for j in range(n_proteins)
    ])

    pcc_cell = np.array([
        pearsonr(pred[i, :], true[i, :]).statistic
        if np.std(pred[i, :]) > 1e-8 and np.std(true[i, :]) > 1e-8 else 0.0
        for i in range(n_cells)
    ])

    return pcc_protein, pcc_cell, float(np.nanmean(pcc_protein)), float(np.nanmean(pcc_cell))


def compute_spearman(
    pred: np.ndarray,
    true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Spearman rank correlation per-protein and per-cell.

    Returns:
        sp_protein [M], sp_cell [N], mean_protein, mean_cell
    """
    n_cells, n_proteins = pred.shape

    sp_protein = np.array([
        spearmanr(pred[:, j], true[:, j]).statistic
        if np.std(pred[:, j]) > 1e-8 and np.std(true[:, j]) > 1e-8 else 0.0
        for j in range(n_proteins)
    ])

    sp_cell = np.array([
        spearmanr(pred[i, :], true[i, :]).statistic
        if np.std(pred[i, :]) > 1e-8 and np.std(true[i, :]) > 1e-8 else 0.0
        for i in range(n_cells)
    ])

    return sp_protein, sp_cell, float(np.nanmean(sp_protein)), float(np.nanmean(sp_cell))


def compute_rmse(
    pred: np.ndarray,
    true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    RMSE per-protein and per-cell.

    Note:
        In scChord evaluation pipeline, inputs are already in
        arcsinh protein space (pred_log / true_log).

    Returns:
        rmse_protein [M], rmse_cell [N], mean_protein, mean_cell, global_rmse
    """
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)

    if pred.shape != true.shape:
        raise ValueError(f"pred and true must have same shape, got {pred.shape} vs {true.shape}")

    diff = pred - true
    rmse_protein = np.sqrt(np.mean(diff * diff, axis=0))
    rmse_cell = np.sqrt(np.mean(diff * diff, axis=1))
    global_rmse = float(np.sqrt(np.mean(diff * diff)))

    return (
        rmse_protein,
        rmse_cell,
        float(np.nanmean(rmse_protein)),
        float(np.nanmean(rmse_cell)),
        global_rmse,
    )


def compute_foreground_metrics(
    true_log: np.ndarray,
    fg_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Foreground detection metrics per protein.

    Inputs are in arcsinh protein space. A raw-positive protein has
    arcsinh(raw) > 0, so foreground labels are defined by true_log > 0.
    The score can be a foreground probability or any monotonic abundance score.
    """
    true_log = np.asarray(true_log, dtype=np.float32)
    fg_score = np.asarray(fg_score, dtype=np.float32)
    n_proteins = true_log.shape[1]
    auroc = np.full(n_proteins, np.nan, dtype=np.float32)
    auprc = np.full(n_proteins, np.nan, dtype=np.float32)

    for j in range(n_proteins):
        labels = (true_log[:, j] > 0.0).astype(np.int32)
        if labels.min() == labels.max():
            continue
        auroc[j] = roc_auc_score(labels, fg_score[:, j])
        auprc[j] = average_precision_score(labels, fg_score[:, j])

    return auroc, auprc, float(np.nanmean(auroc)), float(np.nanmean(auprc))


def evaluate_protein_predictions(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    protein_names: list,
    fg_score: Optional[np.ndarray] = None,
    title: str = "EVALUATION",
) -> dict:
    """
    Shared protein prediction evaluation in arcsinh protein space.

    This matches SpatialChord-Reg metrics: per-protein/per-cell PCC and
    Spearman, RMSE, and foreground AUROC/AUPRC. Methods without an explicit
    foreground head should pass pred_log as fg_score.
    """
    if fg_score is None:
        fg_score = pred_log

    pcc_prot, pcc_cell, pcc_prot_mean, pcc_cell_mean = compute_pcc(
        pred_log, true_log, protein_names
    )
    sp_prot, sp_cell, sp_prot_mean, sp_cell_mean = compute_spearman(
        pred_log, true_log
    )
    rmse_prot, rmse_cell, rmse_prot_mean, rmse_cell_mean, rmse_global = compute_rmse(
        pred_log, true_log
    )
    auroc, auprc, auroc_mean, auprc_mean = compute_foreground_metrics(
        true_log, fg_score
    )

    print(f"\n{'='*60}")
    print(f"{title}  (arcsinh protein space)")
    print(f"{'='*60}")
    print(f"PCC      protein  mean={pcc_prot_mean:.4f}  median={np.nanmedian(pcc_prot):.4f}")
    print(f"PCC      cell     mean={pcc_cell_mean:.4f}  median={np.nanmedian(pcc_cell):.4f}")
    print(f"Spearman protein  mean={sp_prot_mean:.4f}  median={np.nanmedian(sp_prot):.4f}")
    print(f"Spearman cell     mean={sp_cell_mean:.4f}  median={np.nanmedian(sp_cell):.4f}")
    print(f"RMSE     protein  mean={rmse_prot_mean:.4f}  global={rmse_global:.4f}")
    print(f"FG AUROC          mean={auroc_mean:.4f}")
    print(f"FG AUPRC          mean={auprc_mean:.4f}")
    print(f"{'='*60}")

    per_protein = {}
    print("\nPer-protein metrics:")
    for i, name in enumerate(protein_names):
        per_protein[name] = {
            "pcc": float(pcc_prot[i]),
            "spearman": float(sp_prot[i]),
            "rmse": float(rmse_prot[i]),
            "fg_auroc": float(auroc[i]),
            "fg_auprc": float(auprc[i]),
        }
        m = per_protein[name]
        print(
            f"  {name:20s}  PCC={m['pcc']:.4f}  Spearman={m['spearman']:.4f}  "
            f"RMSE={m['rmse']:.4f}  AUROC={m['fg_auroc']:.4f}  AUPRC={m['fg_auprc']:.4f}"
        )

    return {
        "evaluation_space": "arcsinh_protein",
        "foreground_label": "true_arcsinh_protein_gt_0",
        "foreground_score": "predicted_arcsinh_protein",
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
    }
