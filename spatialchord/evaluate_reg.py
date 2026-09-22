# -*- coding: utf-8 -*-
"""Evaluate a SpatialChord-Reg checkpoint under full or missing-modality inputs."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatialchord.artifacts import load_checkpoint, run_model_evaluation
from spatialchord.config import (
    DEFAULT_RNA_FEATURE,
    DEFAULT_RNA_FEATURE_SUBDIR,
    missing_suffix,
    parse_missing_modalities,
    selected_modalities,
    set_seed,
)
from spatialchord.experiment import build_loaders, build_model, make_zero_std


DEFAULT_CONFIG = {
    "experiment": "spatial_pc",
    "h5ad_path": "../../processed/xenium_rna_prot.h5ad",
    "feature_dir": "../../processed",
    "rna_feature": DEFAULT_RNA_FEATURE,
    "rna_var_feature": "auto",
    "rna_feature_subdir": DEFAULT_RNA_FEATURE_SUBDIR,
    "device": "cuda:0",
    "batch_size": 2048,
    "train_ratio": 0.9,
    "num_workers": 8,
    "seed": 42,
    "dim": 256,
    "dropout": 0.1,
    "knn_k": 16,
    "direct_head": False,
    "no_unc_gate": False,
    "max_train_batches": 0,
    "max_eval_batches": 0,
    "he_scales": None,
    "fusion_type": "protein_conditioned",
    "pcif_rank": 0,
    "spatial_control": "true_knn",
    "split_strategy": "random",
    "spatial_block_grid_size": 5,
    "spatial_block_buffer_k": 8,
    "spatial_block_buffer_mult": 2.0,
    "region_tile_um": 800.0,
    "region_test_window_w": 4,
    "region_test_window_h": 3,
    "region_buffer_um": 80.0,
    "region_val_window_w": 2,
    "region_val_window_h": 2,
    "region_val_buffer_um": 40.0,
    "modality_mask_train": "none",
    "mask_prob": 0.3,
    "eval_missing_modalities": "",
}


def build_args(config: dict, cli_args) -> Namespace:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    for key in ("h5ad_path", "feature_dir", "device", "batch_size", "num_workers", "max_eval_batches"):
        value = getattr(cli_args, key)
        if value is not None:
            merged[key] = value
    merged["output_dir"] = str(cli_args.output_dir)
    merged["eval_missing_modalities"] = cli_args.eval_missing_modalities
    return Namespace(**merged)


def parse_eval_specs(value: str) -> list[list[str]]:
    specs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "none":
            specs.append([])
        else:
            specs.append(parse_missing_modalities(item))
    return specs or [[]]


def main(cli_args) -> None:
    run_dir = Path(cli_args.run_dir)
    checkpoint_path = Path(cli_args.checkpoint) if cli_args.checkpoint else run_dir / "best.pt"
    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    eval_cli = Namespace(**vars(cli_args))
    eval_cli.output_dir = str(output_dir)
    args = build_args(ckpt.get("config", {}), eval_cli)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    modalities = selected_modalities(args)
    _, _, test_loader, data_info = build_loaders(args, modalities)
    feature_keys = list(data_info["modalities"])
    feature_dims = data_info["modality_dims"]
    zero_std = make_zero_std(data_info, device)

    model = build_model(args, feature_dims, data_info["n_proteins"]).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing_keys:
        print(f"Missing checkpoint keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected checkpoint keys: {unexpected_keys}")

    args.best_epoch = ckpt.get("epoch")
    args.best_val_loss = ckpt.get("val_loss")

    for missing in parse_eval_specs(args.eval_missing_modalities):
        suffix = missing_suffix(missing)
        run_model_evaluation(
            model,
            test_loader,
            device,
            feature_keys,
            zero_std,
            data_info,
            args,
            output_dir,
            missing=missing,
            json_name=f"results_missing_{suffix}.json",
            save_arrays=not cli_args.no_save_arrays,
        )
        print(f"Saved missing-modality evaluation: {output_dir / f'results_missing_{suffix}.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SpatialChord-Reg checkpoint")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--h5ad_path", type=str, default=None)
    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument(
        "--eval_missing_modalities",
        type=str,
        default="none,rna,he_cell,he_context,spatial",
    )
    parser.add_argument("--no_save_arrays", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
