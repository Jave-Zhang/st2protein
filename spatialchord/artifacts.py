# -*- coding: utf-8 -*-
"""Checkpoint, metadata, metric table, and array writers for SpatialChord-Reg."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from spatialchord.config import (
    DEFAULT_RNA_FEATURE,
    DEFAULT_RNA_FEATURE_SUBDIR,
    missing_suffix,
    resolve_he_scales,
    resolve_rna_var_feature,
)
from spatialchord.engine import infer_dense, infer_graph
from spatialchord.evaluation import (
    clean_json,
    destandardize,
    evaluate_predictions,
    safe_row_corr,
)
from spatialchord.reg_models import SpatialChordReg


def load_checkpoint(path: Path | str, map_location=None) -> Dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_metric_tables(
    output_dir: Path,
    results: Dict,
    pred_log: np.ndarray,
    true_log: np.ndarray,
    protein_names: List[str],
    test_idx: np.ndarray,
    suffix: str,
) -> None:
    if suffix == "none":
        protein_path = output_dir / "per_protein_metrics.csv"
        cell_path = output_dir / "per_cell_metrics.csv"
    else:
        protein_path = output_dir / f"per_protein_metrics_missing_{suffix}.csv"
        cell_path = output_dir / f"per_cell_metrics_missing_{suffix}.csv"

    with open(protein_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["protein", "pcc", "spearman", "rmse", "fg_auroc", "fg_auprc"],
        )
        writer.writeheader()
        for name in protein_names:
            row = {"protein": name}
            row.update(results["per_protein"][name])
            writer.writerow(row)

    rmse_cell = np.sqrt(np.mean((pred_log - true_log) ** 2, axis=1))
    pcc_cell = safe_row_corr(pred_log, true_log)
    with open(cell_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cell_index", "local_index", "rmse", "pcc"])
        writer.writeheader()
        for local_idx, global_idx in enumerate(np.asarray(test_idx, dtype=np.int64)):
            writer.writerow(
                {
                    "cell_index": int(global_idx),
                    "local_index": int(local_idx),
                    "rmse": float(rmse_cell[local_idx]),
                    "pcc": clean_json(float(pcc_cell[local_idx])),
                }
            )


def build_checkpoint_payload(
    model: SpatialChordReg,
    optimizer,
    scheduler,
    epoch: int,
    args,
    feature_keys: List[str],
    data_info: Dict,
    best_epoch: int,
    best_val_loss: float,
) -> Dict:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": best_val_loss if epoch == best_epoch else None,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "config": vars(args),
        "feature_keys": feature_keys,
        "token_names": model.token_names(),
        "prot_mean": np.asarray(data_info["prot_mean"]).tolist(),
        "prot_std": np.asarray(data_info["prot_std"]).tolist(),
        "protein_names": data_info["protein_names"],
    }


def maybe_resume_training(
    args,
    model: SpatialChordReg,
    optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[int, int, float]:
    if not args.resume_checkpoint:
        return 1, 0, float("inf")

    ckpt = load_checkpoint(args.resume_checkpoint, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing_keys:
        print(f"Missing checkpoint keys on resume: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected checkpoint keys on resume: {unexpected_keys}")

    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as exc:
            print(f"Could not load scheduler state on resume: {exc}")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0)))
    best_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
    print(
        f"Resuming from {args.resume_checkpoint}: "
        f"start_epoch={start_epoch}, best_epoch={best_epoch}, best_val_loss={best_loss:.5f}"
    )
    return start_epoch, best_epoch, best_loss


def add_run_metadata(
    results: Dict,
    args,
    model: SpatialChordReg,
    data_info: Dict,
    missing: Sequence[str],
) -> Dict:
    results["experiment"] = args.experiment
    results["rna_feature"] = getattr(args, "rna_feature", DEFAULT_RNA_FEATURE)
    results["rna_var_feature"] = resolve_rna_var_feature(args)
    results["rna_feature_subdir"] = getattr(
        args, "rna_feature_subdir", DEFAULT_RNA_FEATURE_SUBDIR
    )
    results["knn_k"] = args.knn_k if args.experiment == "spatial_pc" else None
    results["he_scales"] = resolve_he_scales(args)
    results["fusion_type"] = args.fusion_type
    fusion_name = str(args.fusion_type)
    results["protein_spatial_readout"] = bool(
        getattr(model, "use_protein_spatial_readout", fusion_name.endswith("_psr"))
    )
    results["pcif_rank"] = (
        getattr(model, "pcif_rank", getattr(args, "pcif_rank", 0))
        if "pcif" in fusion_name or "moe" in fusion_name
        else getattr(args, "pcif_rank", 0)
    )
    results["spatial_control"] = args.spatial_control if args.experiment == "spatial_pc" else None
    results["split_strategy"] = args.split_strategy
    results["modality_mask_train"] = args.modality_mask_train
    results["mask_prob"] = args.mask_prob
    for key in (
        "lr",
        "weight_decay",
        "lr_schedule",
        "warmup_epochs",
        "warmup_start_factor",
        "min_lr_ratio",
        "plateau_patience",
        "plateau_factor",
        "plateau_threshold",
        "early_stop_patience",
        "early_stop_min_delta",
        "checkpoint_metric",
        "best_checkpoint_value",
        "best_checkpoint_val_loss",
        "min_val_loss",
        "min_val_loss_epoch",
        "no_decay_norm_bias",
        "stop_epoch",
        "stop_reason",
    ):
        if hasattr(args, key):
            results[key] = getattr(args, key)
    results["missing_modalities"] = list(missing)
    results["scale_names"] = model.token_names()
    results["n_train_cells"] = int(data_info.get("n_train", 0))
    results["n_val_cells"] = int(data_info.get("n_val", 0))
    results["n_test_cells"] = int(data_info.get("n_test", 0))
    results["n_buffer_cells"] = int(data_info.get("n_buffer", 0))
    results["n_val_buffer_cells"] = int(data_info.get("n_val_buffer", 0))
    results["n_proteins"] = int(data_info.get("n_proteins", 0))
    if hasattr(args, "best_epoch"):
        results["best_epoch"] = args.best_epoch
    if hasattr(args, "best_val_loss"):
        results["best_val_loss"] = args.best_val_loss
    for key in (
        "n_train",
        "n_val",
        "n_test",
        "n_buffer",
        "n_val_buffer",
        "spatial_block_grid_size",
        "spatial_block_buffer_knn",
        "spatial_block_buffer_mult",
        "spatial_block_buffer_radius",
        "spatial_block_buffer_excluded",
        "spatial_block_test_blocks",
        "region_tile_um",
        "region_window_shape",
        "test_region_bounds_um",
        "test_ratio_actual",
        "region_buffer_um",
        "region_val_window_shape",
        "region_val_buffer_um",
        "val_region_bounds_um",
        "val_ratio_actual",
    ):
        if key in data_info:
            results[key] = data_info[key]
    return results


def save_run_static_artifacts(output_dir: Path, args, data_info: Dict) -> None:
    np.savez(
        output_dir / "split_indices.npz",
        train_idx=np.asarray(data_info["train_idx"], dtype=np.int64),
        val_idx=np.asarray(data_info["val_idx"], dtype=np.int64),
        test_idx=np.asarray(data_info["test_idx"], dtype=np.int64),
        buffer_idx=np.asarray(data_info.get("buffer_idx", []), dtype=np.int64),
        val_buffer_idx=np.asarray(data_info.get("val_buffer_idx", []), dtype=np.int64),
        train_pool_idx=np.asarray(data_info.get("train_pool_idx", data_info["train_idx"]), dtype=np.int64),
    )
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(clean_json(vars(args)), f, indent=2)
    with open(output_dir / "protein_names.json", "w") as f:
        json.dump(list(data_info["protein_names"]), f, indent=2)
    with open(output_dir / "label_scaler.json", "w") as f:
        json.dump(
            {
                "target_transform": "arcsinh",
                "label_space": "arcsinh_zscore",
                "mean": np.asarray(data_info["prot_mean"]).tolist(),
                "std": np.asarray(data_info["prot_std"]).tolist(),
            },
            f,
            indent=2,
        )


@torch.no_grad()
def run_model_evaluation(
    model: SpatialChordReg,
    val_loader,
    device: torch.device,
    feature_keys: List[str],
    zero_std: torch.Tensor,
    data_info: Dict,
    args,
    output_dir: Path,
    missing: Optional[Sequence[str]] = None,
    json_name: str = "results.json",
    save_arrays: bool = True,
) -> Dict:
    missing = list(missing or [])
    is_graph = args.experiment == "spatial_pc"
    n_proteins = data_info["n_proteins"]

    if is_graph:
        pred_std, fg_score, attn_mean = infer_graph(
            model,
            val_loader,
            device,
            feature_keys,
            zero_std,
            n_eval_nodes=len(data_info["test_idx"]),
            n_proteins=n_proteins,
            test_idx=data_info["test_idx"],
            index_mode=data_info.get("graph_eval_index_mode", "global"),
            args=args,
            missing_modalities=missing,
        )
        true_log = data_info["prot_log_test"][: pred_std.shape[0]]
    else:
        pred_std, fg_score, attn_mean = infer_dense(
            model,
            val_loader,
            device,
            feature_keys,
            zero_std,
            args,
            missing_modalities=missing,
        )
        true_log = data_info["prot_log_test"][: pred_std.shape[0]]

    pred_log = destandardize(pred_std, data_info["prot_mean"], data_info["prot_std"])
    fg_eval_score = pred_log if fg_score is None else fg_score
    results = evaluate_predictions(pred_log, true_log, fg_eval_score, data_info["protein_names"])
    if fg_score is None:
        results["foreground_score"] = "predicted_arcsinh_abundance"
    add_run_metadata(results, args, model, data_info, missing)

    if save_arrays:
        suffix = missing_suffix(missing)
        if suffix == "none":
            np.save(output_dir / "pred_protein_log.npy", pred_log)
            np.save(output_dir / "true_protein_log.npy", true_log)
            np.save(output_dir / "pred_protein_eval_space.npy", pred_log)
            np.save(output_dir / "true_protein_eval_space.npy", true_log)
            np.save(output_dir / "scale_attention_mean.npy", attn_mean)
        else:
            np.save(output_dir / f"pred_protein_log_missing_{suffix}.npy", pred_log)
            np.save(output_dir / f"true_protein_log_missing_{suffix}.npy", true_log)
            np.save(output_dir / f"pred_protein_eval_space_missing_{suffix}.npy", pred_log)
            np.save(output_dir / f"true_protein_eval_space_missing_{suffix}.npy", true_log)
            np.save(output_dir / f"scale_attention_mean_missing_{suffix}.npy", attn_mean)
        write_metric_tables(
            output_dir=output_dir,
            results=results,
            pred_log=pred_log,
            true_log=true_log,
            protein_names=data_info["protein_names"],
            test_idx=data_info["test_idx"][: pred_log.shape[0]],
            suffix=suffix,
        )

    with open(output_dir / json_name, "w") as f:
        json.dump(clean_json(results), f, indent=2)
    return results
