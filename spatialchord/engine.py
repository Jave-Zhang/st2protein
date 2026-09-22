# -*- coding: utf-8 -*-
"""Training, validation, and inference loops for SpatialChord-Reg."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from spatialchord.evaluation import destandardize, foreground_metrics
from spatialchord.losses import regression_loss
from spatialchord.reg_models import SpatialChordReg


VALIDATION_PRED_METRICS = (
    "val_pcc_protein_mean",
    "val_spearman_protein_mean",
    "val_rmse_global",
    "val_fg_auprc_mean",
)


def batch_to_device(
    batch: Dict[str, torch.Tensor],
    keys: Iterable[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {k: batch[k].to(device) for k in keys}


def pyg_batch_to_dict(batch, keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    return {k: getattr(batch, k) for k in keys}


def _empty_validation_pred_metrics() -> Dict[str, float]:
    return {key: float("nan") for key in VALIDATION_PRED_METRICS}


def _mean_protein_pcc(pred_log: np.ndarray, true_log: np.ndarray) -> float:
    pred_c = pred_log - pred_log.mean(axis=0, keepdims=True)
    true_c = true_log - true_log.mean(axis=0, keepdims=True)
    denom = np.sqrt((pred_c * pred_c).sum(axis=0) * (true_c * true_c).sum(axis=0))
    pcc = np.zeros(pred_log.shape[1], dtype=np.float32)
    valid = denom > 1e-8
    pcc[valid] = ((pred_c[:, valid] * true_c[:, valid]).sum(axis=0) / denom[valid]).astype(
        np.float32
    )
    return float(np.nanmean(pcc))


def _mean_protein_spearman(pred_log: np.ndarray, true_log: np.ndarray) -> float:
    vals = []
    for j in range(pred_log.shape[1]):
        if np.std(pred_log[:, j]) <= 1e-8 or np.std(true_log[:, j]) <= 1e-8:
            vals.append(0.0)
        else:
            vals.append(float(spearmanr(pred_log[:, j], true_log[:, j]).statistic))
    return float(np.nanmean(vals))


def _finish_validation_metrics(
    totals: Dict[str, float],
    n: int,
    preds: List[np.ndarray],
    true: List[np.ndarray],
    probs: List[np.ndarray],
    data_info: Optional[Dict],
) -> Dict[str, float]:
    metrics = {k: v / max(n, 1) for k, v in totals.items()}
    metrics.update(_empty_validation_pred_metrics())
    if data_info is None or not preds:
        return metrics

    pred_std = np.concatenate(preds, axis=0)
    true_std = np.concatenate(true, axis=0)
    pred_log = destandardize(pred_std, data_info["prot_mean"], data_info["prot_std"])
    true_log = destandardize(true_std, data_info["prot_mean"], data_info["prot_std"])
    fg_score = np.concatenate(probs, axis=0) if probs else None
    _, _, _, auprc_mean = foreground_metrics(true_log, fg_score)
    diff = pred_log - true_log
    metrics["val_pcc_protein_mean"] = _mean_protein_pcc(pred_log, true_log)
    metrics["val_spearman_protein_mean"] = _mean_protein_spearman(pred_log, true_log)
    metrics["val_rmse_global"] = float(np.sqrt(np.mean(diff * diff)))
    metrics["val_fg_auprc_mean"] = float(auprc_mean)
    return metrics


def train_dense_epoch(
    model: SpatialChordReg,
    loader,
    optimizer,
    device: torch.device,
    feature_keys: List[str],
    zero_std: torch.Tensor,
    args,
) -> Dict[str, float]:
    model.train()
    totals = {k: 0.0 for k in ("loss", "loss_base", "loss_bce", "loss_abund", "loss_corr")}
    n = 0

    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        if args.max_train_batches and i >= args.max_train_batches:
            break
        x = batch_to_device(batch, feature_keys, device)
        y = batch["protein"].to(device)
        out = model(
            x,
            zero_std=zero_std,
            modality_mask_train=args.modality_mask_train,
            mask_prob=args.mask_prob,
        )
        losses = regression_loss(
            out, y, zero_std, args.direct_head, args.lambda_abund, args.lambda_corr
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        bs = y.shape[0]
        for k in totals:
            totals[k] += float(losses[k].item()) * bs
        n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate_dense(
    model: SpatialChordReg,
    loader,
    device: torch.device,
    feature_keys: List[str],
    zero_std: torch.Tensor,
    args,
    data_info: Optional[Dict] = None,
) -> Dict[str, float]:
    model.eval()
    totals = {k: 0.0 for k in ("loss", "loss_base", "loss_bce", "loss_abund", "loss_corr")}
    preds, true, probs = [], [], []
    n = 0
    for i, batch in enumerate(tqdm(loader, desc="Validate", leave=False)):
        if args.max_eval_batches and i >= args.max_eval_batches:
            break
        x = batch_to_device(batch, feature_keys, device)
        y = batch["protein"].to(device)
        out = model(x, zero_std=zero_std)
        losses = regression_loss(
            out, y, zero_std, args.direct_head, args.lambda_abund, args.lambda_corr
        )
        bs = y.shape[0]
        for k in totals:
            totals[k] += float(losses[k].item()) * bs
        if data_info is not None:
            preds.append(out["pred_std"].detach().cpu().numpy())
            true.append(y.detach().cpu().numpy())
            if out["fg_prob"] is not None:
                probs.append(out["fg_prob"].detach().cpu().numpy())
        n += bs
    return _finish_validation_metrics(totals, n, preds, true, probs, data_info)


def train_graph_epoch(
    model: SpatialChordReg,
    loader,
    optimizer,
    device: torch.device,
    feature_keys: List[str],
    zero_std: torch.Tensor,
    args,
) -> Dict[str, float]:
    model.train()
    totals = {k: 0.0 for k in ("loss", "loss_base", "loss_bce", "loss_abund", "loss_corr")}
    n = 0

    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        if args.max_train_batches and i >= args.max_train_batches:
            break
        batch = batch.to(device)
        bs = batch.batch_size
        x = pyg_batch_to_dict(batch, feature_keys)
        y = batch.protein[:bs]
        out = model(
            x,
            zero_std=zero_std,
            edge_index=batch.edge_index,
            pos=batch.pos,
            modality_mask_train=args.modality_mask_train,
            mask_prob=args.mask_prob,
            output_size=bs,
        )
        out_s = {k: (v[:bs] if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
        losses = regression_loss(
            out_s, y, zero_std, args.direct_head, args.lambda_abund, args.lambda_corr
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        for k in totals:
            totals[k] += float(losses[k].item()) * bs
        n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate_graph(
    model: SpatialChordReg,
    loader,
    device: torch.device,
    feature_keys: List[str],
    zero_std: torch.Tensor,
    args,
    data_info: Optional[Dict] = None,
) -> Dict[str, float]:
    model.eval()
    totals = {k: 0.0 for k in ("loss", "loss_base", "loss_bce", "loss_abund", "loss_corr")}
    preds, true, probs = [], [], []
    n = 0

    for i, batch in enumerate(tqdm(loader, desc="Validate", leave=False)):
        if args.max_eval_batches and i >= args.max_eval_batches:
            break
        batch = batch.to(device)
        bs = batch.batch_size
        x = pyg_batch_to_dict(batch, feature_keys)
        y = batch.protein[:bs]
        out = model(
            x,
            zero_std=zero_std,
            edge_index=batch.edge_index,
            pos=batch.pos,
            output_size=bs,
        )
        out_s = {k: (v[:bs] if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
        losses = regression_loss(
            out_s, y, zero_std, args.direct_head, args.lambda_abund, args.lambda_corr
        )
        for k in totals:
            totals[k] += float(losses[k].item()) * bs
        if data_info is not None:
            preds.append(out_s["pred_std"].detach().cpu().numpy())
            true.append(y.detach().cpu().numpy())
            if out_s["fg_prob"] is not None:
                probs.append(out_s["fg_prob"].detach().cpu().numpy())
        n += bs

    return _finish_validation_metrics(totals, n, preds, true, probs, data_info)


@torch.no_grad()
def infer_dense(
    model: SpatialChordReg,
    loader,
    device: torch.device,
    feature_keys: List[str],
    zero_std: torch.Tensor,
    args,
    missing_modalities: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    model.eval()
    preds, probs, attn_sum = [], [], None
    count = 0
    for i, batch in enumerate(tqdm(loader, desc="Inference", leave=False)):
        if args.max_eval_batches and i >= args.max_eval_batches:
            break
        x = batch_to_device(batch, feature_keys, device)
        out = model(x, zero_std=zero_std, missing_modalities=missing_modalities)
        pred = out["pred_std"].detach().cpu().numpy()
        preds.append(pred)
        if out["fg_prob"] is not None:
            probs.append(out["fg_prob"].detach().cpu().numpy())
        attn = out["scale_attention"].detach().cpu()
        attn_sum = attn.sum(0) if attn_sum is None else attn_sum + attn.sum(0)
        count += pred.shape[0]
    prob_np = np.concatenate(probs, axis=0) if probs else None
    return np.concatenate(preds, axis=0), prob_np, (attn_sum / max(count, 1)).numpy()


@torch.no_grad()
def infer_graph(
    model: SpatialChordReg,
    loader,
    device: torch.device,
    feature_keys: List[str],
    zero_std: torch.Tensor,
    n_eval_nodes: int,
    n_proteins: int,
    test_idx: np.ndarray,
    index_mode: str,
    args,
    missing_modalities: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    model.eval()
    pred_eval = np.zeros((n_eval_nodes, n_proteins), dtype=np.float32)
    prob_eval = None if args.direct_head else np.zeros((n_eval_nodes, n_proteins), dtype=np.float32)
    global_to_local = None
    if index_mode == "global":
        global_to_local = {int(gid): i for i, gid in enumerate(test_idx.tolist())}
    attn_sum, count = None, 0

    for i, batch in enumerate(tqdm(loader, desc="Inference", leave=False)):
        if args.max_eval_batches and i >= args.max_eval_batches:
            break
        batch = batch.to(device)
        bs = batch.batch_size
        x = pyg_batch_to_dict(batch, feature_keys)
        out = model(
            x,
            zero_std=zero_std,
            edge_index=batch.edge_index,
            pos=batch.pos,
            missing_modalities=missing_modalities,
            output_size=bs,
        )
        node_ids = batch.n_id[:bs].detach().cpu().numpy()
        if index_mode == "global":
            local_ids = np.asarray([global_to_local[int(gid)] for gid in node_ids], dtype=np.int64)
        elif index_mode == "local":
            local_ids = node_ids.astype(np.int64)
        else:
            raise ValueError("index_mode must be one of: global, local")
        pred_eval[local_ids] = out["pred_std"][:bs].detach().cpu().numpy()
        if prob_eval is not None:
            prob_eval[local_ids] = out["fg_prob"][:bs].detach().cpu().numpy()
        attn = out["scale_attention"][:bs].detach().cpu()
        attn_sum = attn.sum(0) if attn_sum is None else attn_sum + attn.sum(0)
        count += bs

    return (
        pred_eval,
        None if prob_eval is None else prob_eval,
        (attn_sum / max(count, 1)).numpy(),
    )
