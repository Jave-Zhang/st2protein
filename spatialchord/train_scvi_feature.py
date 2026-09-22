# -*- coding: utf-8 -*-
"""Train the default scVI RNA feature extractor and export posterior arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd


def configure_device(device: str) -> Tuple[str, int | str]:
    if device.startswith("cuda"):
        gpu_id = device.split(":", 1)[1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return "gpu", 1
    return "cpu", "auto"


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_history(history: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    frames = []
    summary = {}
    for name, df in history.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        renamed = df.copy()
        renamed.columns = [name if len(df.columns) == 1 else f"{name}_{c}" for c in df.columns]
        frames.append(renamed)
        series = renamed.iloc[:, 0].astype(float)
        summary[name] = {
            "first": float(series.iloc[0]),
            "last": float(series.iloc[-1]),
            "min": float(series.min()),
            "min_epoch": int(series.idxmin()),
        }
    if frames:
        pd.concat(frames, axis=1).to_csv(output_dir / "history.csv")
    with open(output_dir / "history_summary.json", "w") as f:
        json.dump(clean_json(summary), f, indent=2)


def train_and_extract(args) -> None:
    accelerator, devices = configure_device(args.device)

    import scvi
    import torch

    if accelerator == "gpu" and not torch.cuda.is_available():
        print("CUDA is not available in this process; falling back to CPU")
        accelerator, devices = "cpu", "auto"
    elif accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    scvi.settings.seed = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.h5ad_path} ...")
    adata = ad.read_h5ad(args.h5ad_path)
    print(f"  cells={adata.n_obs}, genes={adata.n_vars}")

    adata.layers["counts"] = adata.X.copy()

    batch_key = "batch_id" if (args.use_batch_key and "batch_id" in adata.obs.columns) else None
    if batch_key:
        n_unique = int(adata.obs[batch_key].nunique())
        print(f"  Using batch_key='{batch_key}' with {n_unique} batches")

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)

    model = scvi.model.SCVI(
        adata,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        gene_likelihood=args.gene_likelihood,
        dispersion=args.dispersion,
        latent_distribution=args.latent_distribution,
        dropout_rate=args.dropout_rate,
        use_layer_norm=args.use_layer_norm,
        use_batch_norm=args.use_batch_norm,
    )

    config = {
        "h5ad_path": args.h5ad_path,
        "output_dir": str(output_dir),
        "n_latent": args.n_latent,
        "n_hidden": args.n_hidden,
        "n_layers": args.n_layers,
        "dropout_rate": args.dropout_rate,
        "gene_likelihood": args.gene_likelihood,
        "dispersion": args.dispersion,
        "latent_distribution": args.latent_distribution,
        "use_layer_norm": args.use_layer_norm,
        "use_batch_norm": args.use_batch_norm,
        "use_batch_key": args.use_batch_key,
        "batch_key": batch_key,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "validation_size": args.validation_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "device": args.device,
        "accelerator": accelerator,
        "devices": devices,
    }
    with open(output_dir / "scvi_config.json", "w") as f:
        json.dump(clean_json(config), f, indent=2)

    print(
        "Training scVI "
        f"(latent={args.n_latent}, hidden={args.n_hidden}, layers={args.n_layers}, "
        f"likelihood={args.gene_likelihood}, dispersion={args.dispersion}, "
        f"latent_distribution={args.latent_distribution}) ..."
    )
    model.train(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        train_size=args.train_size,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        plan_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
    )

    save_history(model.history, output_dir)
    if "elbo_validation" in model.history:
        val = model.history["elbo_validation"].iloc[:, 0].astype(float)
        print(f"  Best valid ELBO: {val.min():.4f} @ epoch {int(val.idxmin())}")
        print(f"  Final valid ELBO: {val.iloc[-1]:.4f}")

    print("Extracting posterior mean ...")
    mu = model.get_latent_representation(adata, give_mean=True).astype(np.float32)
    np.save(output_dir / "rna_mu.npy", mu)
    print(f"  Saved {output_dir / 'rna_mu.npy'} shape={mu.shape}")

    print("Extracting posterior variance ...")
    model.module.eval()
    var_list = []
    dl = model._make_data_loader(adata, batch_size=args.batch_size)
    with torch.inference_mode():
        for tensors in dl:
            inference_input = model.module._get_inference_input(tensors)
            outputs = model.module.inference(**inference_input)
            var_list.append(outputs["qz"].variance.cpu().numpy())
    var = np.concatenate(var_list, axis=0).astype(np.float32)
    np.save(output_dir / "rna_var.npy", var)
    print(f"  Saved {output_dir / 'rna_var.npy'} shape={var.shape}")

    model_path = output_dir / "scvi_model"
    model.save(str(model_path), overwrite=True)
    print(f"  scVI model saved to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and export the default scVI-128 hidden512 RNA feature")
    parser.add_argument("--h5ad_path", type=str, default="../../processed/xenium_rna_prot.h5ad")
    parser.add_argument("--output_dir", type=str, default="../../processed/scvi_variants/scvi128_h512")
    parser.add_argument("--n_latent", type=int, default=128)
    parser.add_argument("--n_hidden", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--gene_likelihood", choices=["zinb", "nb", "poisson", "normal"], default="nb")
    parser.add_argument(
        "--dispersion",
        choices=["gene", "gene-batch", "gene-label", "gene-cell"],
        default="gene",
    )
    parser.add_argument("--latent_distribution", choices=["normal", "ln"], default="normal")
    parser.add_argument("--use_layer_norm", choices=["both", "encoder", "decoder", "none"], default="both")
    parser.add_argument("--use_batch_norm", choices=["both", "encoder", "decoder", "none"], default="none")
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--train_size", type=float, default=0.9)
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--use_batch_key", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    train_and_extract(parse_args())
