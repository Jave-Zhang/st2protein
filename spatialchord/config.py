# -*- coding: utf-8 -*-
"""Configuration and modality helpers for SpatialChord-Reg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RNA_FEATURE = "scvi128_h512"
DEFAULT_RNA_FEATURE_SUBDIR = "scvi_variants"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolve_context_modality(feature_dir: str) -> str:
    feature_dir_p = Path(feature_dir)
    if (feature_dir_p / "feats_he_context_rawctx-409.npy").exists():
        return "he_context"
    if (feature_dir_p / "feats_he_context_rawctx-224.npy").exists():
        return "he_context_224"
    return "he_micro"


def resolve_he_scales(args) -> str:
    if args.he_scales is not None:
        return args.he_scales
    return "none" if args.experiment == "rna_hurdle" else "both"


def resolve_rna_var_feature(args) -> str:
    value = getattr(args, "rna_var_feature", "auto")
    if value != "auto":
        return value
    return "matched"


def selected_modalities(args) -> List[str]:
    modalities = ["rna_latent"]
    if resolve_rna_var_feature(args) != "none":
        modalities.append("rna_var")
    he_scales = resolve_he_scales(args)
    if args.experiment in ("fusion_pc", "spatial_pc"):
        if he_scales in ("cell", "both"):
            modalities.append("he_cell")
        if he_scales in ("context", "both"):
            modalities.append(resolve_context_modality(args.feature_dir))
    return modalities


def parse_missing_modalities(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    out = []
    for item in value.split(","):
        item = item.strip()
        if item and item != "none":
            out.append(item)
    return out


def missing_suffix(missing_modalities: Sequence[str]) -> str:
    if not missing_modalities:
        return "none"
    return "_".join(missing_modalities)


def parse_args():
    parser = argparse.ArgumentParser(description="SpatialChord-Reg training")
    parser.add_argument(
        "--experiment",
        choices=["rna_hurdle", "fusion_pc", "spatial_pc"],
        required=True,
    )
    parser.add_argument(
        "--he_scales",
        choices=["none", "cell", "context", "both"],
        default=None,
        help="H&E scales to use. Defaults to none for rna_hurdle and both otherwise.",
    )
    parser.add_argument(
        "--fusion_type",
        choices=[
            "protein_conditioned",
            "shared_attention",
            "concat",
            "pcif_pairwise",
            "pcif_hier",
            "hybrid_pcif",
            "hybrid_pcif_hier",
            "pc_moe",
            "pc_moe_hier",
            "pc_moe_static",
            "pc_moe_hier_static",
            "concat_psr",
            "pcif_hier_psr",
            "hybrid_pcif_psr",
            "hybrid_pcif_hier_psr",
            "pc_moe_psr",
            "pc_moe_static_psr",
        ],
        default="protein_conditioned",
    )
    parser.add_argument(
        "--pcif_rank",
        type=int,
        default=0,
        help="Low-rank PCIF interaction size. 0 uses dim // 2.",
    )
    parser.add_argument(
        "--spatial_control",
        choices=["true_knn", "random_neighbors", "permuted_coords"],
        default="true_knn",
    )
    parser.add_argument(
        "--split_strategy",
        choices=["random", "spatial_block", "region_holdout"],
        default="random",
    )
    parser.add_argument("--spatial_block_grid_size", type=int, default=5)
    parser.add_argument("--spatial_block_buffer_k", type=int, default=8)
    parser.add_argument("--spatial_block_buffer_mult", type=float, default=2.0)
    parser.add_argument("--region_tile_um", type=float, default=800.0)
    parser.add_argument("--region_test_window_w", type=int, default=4)
    parser.add_argument("--region_test_window_h", type=int, default=3)
    parser.add_argument("--region_buffer_um", type=float, default=80.0)
    parser.add_argument("--region_val_window_w", type=int, default=2)
    parser.add_argument("--region_val_window_h", type=int, default=2)
    parser.add_argument("--region_val_buffer_um", type=float, default=40.0)
    parser.add_argument(
        "--modality_mask_train",
        choices=["none", "single_drop"],
        default="none",
    )
    parser.add_argument("--mask_prob", type=float, default=0.3)
    parser.add_argument(
        "--eval_missing_modalities",
        type=str,
        default="",
        help="Comma-separated missing modality names to evaluate after training.",
    )
    parser.add_argument("--h5ad_path", type=str, default="../../processed/xenium_rna_prot.h5ad")
    parser.add_argument("--feature_dir", type=str, default="../../processed")
    parser.add_argument(
        "--rna_feature",
        choices=[DEFAULT_RNA_FEATURE],
        default=DEFAULT_RNA_FEATURE,
        help="RNA feature matrix to use as the rna_latent token.",
    )
    parser.add_argument(
        "--rna_var_feature",
        choices=["auto", "matched", "zeros", "none"],
        default="auto",
        help="Variance feature for RNA uncertainty gating. auto uses the matched scVI variance.",
    )
    parser.add_argument(
        "--rna_feature_subdir",
        type=str,
        default=DEFAULT_RNA_FEATURE_SUBDIR,
        help="Subdirectory under feature_dir containing the default scVI RNA feature files.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr_schedule",
        choices=["cosine", "warmup_cosine", "plateau"],
        default="cosine",
        help="Learning-rate schedule. cosine preserves the legacy behavior.",
    )
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--plateau_patience", type=int, default=2)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_threshold", type=float, default=1e-4)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation improvement. 0 disables it.",
    )
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint_metric",
        choices=["val_loss", "val_pcc", "val_spearman", "val_pcc_spearman"],
        default="val_loss",
        help="Validation metric used for best.pt selection and early stopping.",
    )
    parser.add_argument(
        "--no_decay_norm_bias",
        action="store_true",
        help="Exclude bias and 1D parameters such as LayerNorm weights from weight decay.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--knn_k", type=int, default=16)
    parser.add_argument("--lambda_abund", type=float, default=0.5)
    parser.add_argument("--lambda_corr", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--direct_head", action="store_true")
    parser.add_argument("--no_unc_gate", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    return parser.parse_args()
