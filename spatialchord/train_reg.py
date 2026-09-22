# -*- coding: utf-8 -*-
"""Train SpatialChord-Reg experiments.

Run from the scChord repository root, for example:
    python spatialchord/train_reg.py --experiment fusion_pc
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatialchord.artifacts import (
    add_run_metadata,
    build_checkpoint_payload,
    load_checkpoint,
    maybe_resume_training,
    run_model_evaluation,
    save_run_static_artifacts,
    write_metric_tables,
)
from spatialchord.config import (
    missing_suffix,
    parse_args,
    parse_missing_modalities,
    selected_modalities,
    set_seed,
)
from spatialchord.engine import (
    batch_to_device,
    infer_dense,
    infer_graph,
    pyg_batch_to_dict,
    train_dense_epoch,
    train_graph_epoch,
    validate_dense,
    validate_graph,
)
from spatialchord.evaluation import (
    clean_json,
    destandardize,
    evaluate_predictions,
    foreground_metrics,
    safe_row_corr,
)
from spatialchord.experiment import (
    build_loaders,
    build_model,
    build_optimizer_scheduler,
    make_zero_std,
)
from spatialchord.losses import corr_loss, regression_loss


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_loss_base",
    "train_loss_bce",
    "train_loss_abund",
    "train_loss_corr",
    "val_loss",
    "val_loss_base",
    "val_loss_bce",
    "val_loss_abund",
    "val_loss_corr",
    "val_pcc_protein_mean",
    "val_spearman_protein_mean",
    "val_rmse_global",
    "val_fg_auprc_mean",
    "checkpoint_score",
    "best_checkpoint_value",
    "best_checkpoint_val_loss",
    "min_val_loss",
    "min_val_loss_epoch",
    "lr",
    "best_epoch",
    "best_val_loss",
    "improved",
    "epochs_since_improve",
]


def scheduler_step(scheduler, val_loss: float) -> None:
    if getattr(scheduler, "requires_metric", False):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def append_train_history(output_dir: Path, row: dict) -> None:
    path = output_dir / "train_history.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def checkpoint_mode(metric: str) -> str:
    return "min" if metric == "val_loss" else "max"


def initial_checkpoint_score(metric: str) -> float:
    return float("inf") if checkpoint_mode(metric) == "min" else -float("inf")


def finite_or_default(value: float, default: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else default


def checkpoint_score(metric: str, val_m: dict) -> float:
    if metric == "val_loss":
        return finite_or_default(val_m["loss"], float("inf"))
    if metric == "val_pcc":
        return finite_or_default(val_m["val_pcc_protein_mean"], -float("inf"))
    if metric == "val_spearman":
        return finite_or_default(val_m["val_spearman_protein_mean"], -float("inf"))
    if metric == "val_pcc_spearman":
        pcc = finite_or_default(val_m["val_pcc_protein_mean"], -float("inf"))
        spearman = finite_or_default(val_m["val_spearman_protein_mean"], -float("inf"))
        if not math.isfinite(pcc) or not math.isfinite(spearman):
            return -float("inf")
        return 0.7 * pcc + 0.3 * spearman
    raise ValueError(f"Unsupported checkpoint_metric: {metric}")


def is_better(score: float, best_score: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return score < best_score - min_delta
    if mode == "max":
        return score > best_score + min_delta
    raise ValueError(f"Unsupported checkpoint mode: {mode}")


def main(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = selected_modalities(args)
    train_loader, val_loader, test_loader, data_info = build_loaders(args, modalities)
    save_run_static_artifacts(output_dir, args, data_info)

    feature_keys = list(data_info["modalities"])
    feature_dims = data_info["modality_dims"]
    n_proteins = data_info["n_proteins"]
    zero_std = make_zero_std(data_info, device)

    model = build_model(args, feature_dims, n_proteins).to(device)
    optimizer, scheduler = build_optimizer_scheduler(args, model)
    start_epoch, best_epoch, best_loss = maybe_resume_training(
        args, model, optimizer, scheduler, device
    )
    is_graph = args.experiment == "spatial_pc"
    ckpt_metric = getattr(args, "checkpoint_metric", "val_loss")
    ckpt_mode = checkpoint_mode(ckpt_metric)
    best_checkpoint_value = initial_checkpoint_score(ckpt_metric)
    best_checkpoint_val_loss = float("inf")
    min_val_loss = float("inf")
    min_val_loss_epoch = 0
    if start_epoch > 1 and ckpt_metric == "val_loss":
        best_checkpoint_value = best_loss
        best_checkpoint_val_loss = best_loss
        min_val_loss = best_loss
        min_val_loss_epoch = best_epoch
    epochs_since_improve = 0
    stop_reason = "max_epochs"
    last_epoch = start_epoch - 1
    if start_epoch <= 1:
        history_path = output_dir / "train_history.csv"
        if history_path.exists():
            history_path.unlink()

    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        if is_graph:
            train_m = train_graph_epoch(
                model, train_loader, optimizer, device, feature_keys, zero_std, args
            )
            val_m = validate_graph(
                model, val_loader, device, feature_keys, zero_std, args, data_info
            )
        else:
            train_m = train_dense_epoch(
                model, train_loader, optimizer, device, feature_keys, zero_std, args
            )
            val_m = validate_dense(
                model, val_loader, device, feature_keys, zero_std, args, data_info
            )
        score = checkpoint_score(ckpt_metric, val_m)
        improved = is_better(score, best_checkpoint_value, ckpt_mode, 0.0)
        meaningful_improvement = is_better(
            score,
            best_checkpoint_value,
            ckpt_mode,
            getattr(args, "early_stop_min_delta", 0.0),
        )
        if val_m["loss"] < min_val_loss:
            min_val_loss = val_m["loss"]
            min_val_loss_epoch = epoch
            if ckpt_metric != "val_loss":
                torch.save(
                    build_checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        args,
                        feature_keys,
                        data_info,
                        min_val_loss_epoch,
                        min_val_loss,
                    ),
                    output_dir / "best_val_loss.pt",
                )
        if improved:
            best_checkpoint_value = score
            best_checkpoint_val_loss = val_m["loss"]
            best_loss = best_checkpoint_val_loss
            best_epoch = epoch
            torch.save(
                build_checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                    feature_keys,
                    data_info,
                    best_epoch,
                    best_checkpoint_val_loss,
                ),
                output_dir / "best.pt",
            )
        if meaningful_improvement:
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        scheduler_step(scheduler, val_m["loss"])
        lr = scheduler.get_last_lr()[0]

        append_train_history(
            output_dir,
            {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_loss_base": train_m["loss_base"],
                "train_loss_bce": train_m["loss_bce"],
                "train_loss_abund": train_m["loss_abund"],
                "train_loss_corr": train_m["loss_corr"],
                "val_loss": val_m["loss"],
                "val_loss_base": val_m["loss_base"],
                "val_loss_bce": val_m["loss_bce"],
                "val_loss_abund": val_m["loss_abund"],
                "val_loss_corr": val_m["loss_corr"],
                "val_pcc_protein_mean": val_m["val_pcc_protein_mean"],
                "val_spearman_protein_mean": val_m["val_spearman_protein_mean"],
                "val_rmse_global": val_m["val_rmse_global"],
                "val_fg_auprc_mean": val_m["val_fg_auprc_mean"],
                "checkpoint_score": score,
                "best_checkpoint_value": best_checkpoint_value,
                "best_checkpoint_val_loss": best_checkpoint_val_loss,
                "min_val_loss": min_val_loss,
                "min_val_loss_epoch": min_val_loss_epoch,
                "lr": lr,
                "best_epoch": best_epoch,
                "best_val_loss": best_checkpoint_val_loss,
                "improved": int(improved),
                "epochs_since_improve": epochs_since_improve,
            },
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_m['loss']:.5f} val_loss={val_m['loss']:.5f} "
            f"{ckpt_metric}={score:.5f} "
            f"lr={lr:.2e} best_epoch={best_epoch} "
            f"epochs_since_improve={epochs_since_improve}"
        )

        torch.save(
            build_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                feature_keys,
                data_info,
                best_epoch,
                best_checkpoint_val_loss,
            ),
            output_dir / "last.pt",
        )

        patience = int(getattr(args, "early_stop_patience", 0))
        if patience > 0 and epochs_since_improve >= patience:
            stop_reason = f"early_stop_patience_{patience}"
            print(
                f"Early stopping at epoch {epoch}: "
                f"best_epoch={best_epoch}, best_{ckpt_metric}={best_checkpoint_value:.5f}, "
                f"best_checkpoint_val_loss={best_checkpoint_val_loss:.5f}"
            )
            break

    args.stop_epoch = last_epoch
    args.stop_reason = stop_reason
    args.best_epoch = best_epoch
    args.best_val_loss = best_checkpoint_val_loss
    args.best_checkpoint_value = best_checkpoint_value
    args.best_checkpoint_val_loss = best_checkpoint_val_loss
    args.min_val_loss = min_val_loss
    args.min_val_loss_epoch = min_val_loss_epoch
    torch.save(
        build_checkpoint_payload(
            model,
            optimizer,
            scheduler,
            last_epoch,
            args,
            feature_keys,
            data_info,
            best_epoch,
            best_checkpoint_val_loss,
        ),
        output_dir / "final.pt",
    )
    print(
        f"Best epoch: {best_epoch}, best {ckpt_metric}: {best_checkpoint_value:.5f}, "
        f"best checkpoint val loss: {best_checkpoint_val_loss:.5f}, "
        f"min val loss: {min_val_loss:.5f} at epoch {min_val_loss_epoch}"
    )

    ckpt = load_checkpoint(output_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    run_model_evaluation(
        model,
        test_loader,
        device,
        feature_keys,
        zero_std,
        data_info,
        args,
        output_dir,
        missing=[],
        json_name="results.json",
    )

    for missing_name in parse_missing_modalities(args.eval_missing_modalities):
        missing = [missing_name]
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
        )
    print(f"Saved regression artifacts to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
