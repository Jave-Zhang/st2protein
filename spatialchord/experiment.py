# -*- coding: utf-8 -*-
"""Experiment construction helpers for SpatialChord-Reg."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from spatialchord.config import ROOT  # noqa: F401 - importing sets repository import path

from dataset import (
    get_dataloader,
    get_neighbor_loader,
    load_spatial_data,
    load_spatial_graph_data,
    load_spatial_graph_split_data,
)
from spatialchord.reg_models import SpatialChordReg


class WarmupPlateauScheduler:
    """Linear warmup followed by validation-loss ReduceLROnPlateau."""

    requires_metric = True

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_epochs: int,
        warmup_start_factor: float,
        min_lr: float,
        plateau_factor: float,
        plateau_patience: int,
        plateau_threshold: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.last_epoch = 0
        self._last_lr = [group["lr"] for group in optimizer.param_groups]
        if self.warmup_epochs > 0:
            self.warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            self._last_lr = self.warmup.get_last_lr()
        else:
            self.warmup = None
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode="abs",
            min_lr=min_lr,
        )

    def step(self, metric: float | None = None) -> None:
        self.last_epoch += 1
        if self.warmup is not None and self.last_epoch <= self.warmup_epochs:
            self.warmup.step()
        else:
            if metric is None:
                raise ValueError("WarmupPlateauScheduler.step requires a validation metric")
            self.plateau.step(metric)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self) -> List[float]:
        return list(self._last_lr)

    def state_dict(self) -> Dict:
        return {
            "warmup_epochs": self.warmup_epochs,
            "last_epoch": self.last_epoch,
            "last_lr": list(self._last_lr),
            "warmup": self.warmup.state_dict() if self.warmup is not None else None,
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.last_epoch = int(state_dict.get("last_epoch", 0))
        self._last_lr = list(state_dict.get("last_lr", self._last_lr))
        if self.warmup is not None and state_dict.get("warmup") is not None:
            self.warmup.load_state_dict(state_dict["warmup"])
        if state_dict.get("plateau") is not None:
            self.plateau.load_state_dict(state_dict["plateau"])


def _optimizer_params(args, model: SpatialChordReg):
    if not getattr(args, "no_decay_norm_bias", False):
        return model.parameters()
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_loaders(args, modalities: List[str]):
    if args.experiment == "spatial_pc":
        if args.split_strategy == "region_holdout":
            split_data, data_info = load_spatial_graph_split_data(
                h5ad_path=args.h5ad_path,
                feature_dir=args.feature_dir,
                modalities=modalities,
                rna_feature=args.rna_feature,
                rna_var_feature=args.rna_var_feature,
                rna_feature_subdir=args.rna_feature_subdir,
                train_ratio=args.train_ratio,
                seed=args.seed,
                knn_k=args.knn_k,
                split_strategy=args.split_strategy,
                spatial_block_grid_size=args.spatial_block_grid_size,
                spatial_block_buffer_k=args.spatial_block_buffer_k,
                spatial_block_buffer_mult=args.spatial_block_buffer_mult,
                region_tile_um=args.region_tile_um,
                region_test_window_w=args.region_test_window_w,
                region_test_window_h=args.region_test_window_h,
                region_buffer_um=args.region_buffer_um,
                region_val_window_w=args.region_val_window_w,
                region_val_window_h=args.region_val_window_h,
                region_val_buffer_um=args.region_val_buffer_um,
                spatial_control=args.spatial_control,
            )
            train_loader = get_neighbor_loader(
                split_data["train"],
                input_nodes=torch.arange(split_data["train"].num_nodes, dtype=torch.long),
                batch_size=args.batch_size,
                num_neighbors=[args.knn_k],
                shuffle=True,
                num_workers=args.num_workers,
                subgraph_type="induced",
            )
            val_loader = get_neighbor_loader(
                split_data["val"],
                input_nodes=torch.arange(split_data["val"].num_nodes, dtype=torch.long),
                batch_size=args.batch_size,
                num_neighbors=[args.knn_k],
                shuffle=False,
                num_workers=args.num_workers,
                subgraph_type="induced",
            )
            test_loader = get_neighbor_loader(
                split_data["test"],
                input_nodes=torch.arange(split_data["test"].num_nodes, dtype=torch.long),
                batch_size=args.batch_size,
                num_neighbors=[args.knn_k],
                shuffle=False,
                num_workers=args.num_workers,
                subgraph_type="induced",
            )
            return train_loader, val_loader, test_loader, data_info

        data, data_info = load_spatial_graph_data(
            h5ad_path=args.h5ad_path,
            feature_dir=args.feature_dir,
            modalities=modalities,
            rna_feature=args.rna_feature,
            rna_var_feature=args.rna_var_feature,
            rna_feature_subdir=args.rna_feature_subdir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            knn_k=args.knn_k,
            split_strategy=args.split_strategy,
            spatial_block_grid_size=args.spatial_block_grid_size,
            spatial_block_buffer_k=args.spatial_block_buffer_k,
            spatial_block_buffer_mult=args.spatial_block_buffer_mult,
            region_tile_um=args.region_tile_um,
            region_test_window_w=args.region_test_window_w,
            region_test_window_h=args.region_test_window_h,
            region_buffer_um=args.region_buffer_um,
            region_val_window_w=args.region_val_window_w,
            region_val_window_h=args.region_val_window_h,
            region_val_buffer_um=args.region_val_buffer_um,
            spatial_control=args.spatial_control,
        )
        train_idx_t = torch.tensor(data_info["train_idx"], dtype=torch.long)
        test_idx_t = torch.tensor(data_info["test_idx"], dtype=torch.long)
        train_loader = get_neighbor_loader(
            data,
            input_nodes=train_idx_t,
            batch_size=args.batch_size,
            num_neighbors=[args.knn_k],
            shuffle=True,
            num_workers=args.num_workers,
            subgraph_type="induced",
        )
        val_loader = get_neighbor_loader(
            data,
            input_nodes=test_idx_t,
            batch_size=args.batch_size,
            num_neighbors=[args.knn_k],
            shuffle=False,
            num_workers=args.num_workers,
            subgraph_type="induced",
        )
        return train_loader, val_loader, val_loader, data_info

    train_ds, val_ds, test_ds, data_info = load_spatial_data(
        h5ad_path=args.h5ad_path,
        feature_dir=args.feature_dir,
        modalities=modalities,
        rna_feature=args.rna_feature,
        rna_var_feature=args.rna_var_feature,
        rna_feature_subdir=args.rna_feature_subdir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        split_strategy=args.split_strategy,
        spatial_block_grid_size=args.spatial_block_grid_size,
        spatial_block_buffer_k=args.spatial_block_buffer_k,
        spatial_block_buffer_mult=args.spatial_block_buffer_mult,
        region_tile_um=args.region_tile_um,
        region_test_window_w=args.region_test_window_w,
        region_test_window_h=args.region_test_window_h,
        region_buffer_um=args.region_buffer_um,
        region_val_window_w=args.region_val_window_w,
        region_val_window_h=args.region_val_window_h,
        region_val_buffer_um=args.region_val_buffer_um,
    )
    train_loader = get_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = get_dataloader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    return train_loader, val_loader, test_loader, data_info


def make_zero_std(data_info: Dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        -np.asarray(data_info["prot_mean"]) / np.asarray(data_info["prot_std"]),
        dtype=torch.float32,
        device=device,
    )


def build_model(args, feature_dims: Dict[str, int], n_proteins: int) -> SpatialChordReg:
    return SpatialChordReg(
        feature_dims=feature_dims,
        n_proteins=n_proteins,
        dim=args.dim,
        dropout=args.dropout,
        use_unc_gate=not args.no_unc_gate,
        direct_head=args.direct_head,
        use_spatial=args.experiment == "spatial_pc",
        spatial_k=args.knn_k,
        fusion_type=args.fusion_type,
        pcif_rank=getattr(args, "pcif_rank", 0),
    )


def build_optimizer_scheduler(args, model: SpatialChordReg) -> Tuple[torch.optim.Optimizer, object]:
    optimizer = torch.optim.AdamW(
        _optimizer_params(args, model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    min_lr = args.lr * getattr(args, "min_lr_ratio", 0.05)
    schedule = getattr(args, "lr_schedule", "cosine")
    warmup_epochs = max(int(getattr(args, "warmup_epochs", 0)), 0)
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs, 1), eta_min=min_lr
        )
    elif schedule == "warmup_cosine":
        main_epochs = max(args.epochs - warmup_epochs, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=min_lr
        )
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=getattr(args, "warmup_start_factor", 0.1),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            scheduler = cosine
    elif schedule == "plateau":
        scheduler = WarmupPlateauScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=getattr(args, "warmup_start_factor", 0.1),
            min_lr=min_lr,
            plateau_factor=getattr(args, "plateau_factor", 0.5),
            plateau_patience=getattr(args, "plateau_patience", 2),
            plateau_threshold=getattr(args, "plateau_threshold", 1e-4),
        )
    else:
        raise ValueError(f"Unsupported lr_schedule: {schedule}")
    return optimizer, scheduler
