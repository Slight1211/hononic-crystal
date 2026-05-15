from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_first_band_gap_multitask import (
    Scaler,
    build_cosine_warmup_scheduler,
    build_scaler,
    get_graph_feature_indices,
    move_batch_tensor,
    resolve_amp_config,
    set_seed,
)
from train_gap_index_then_regress import (
    CachedGapGroupDataset,
    GapBoundsRegressor,
    build_index_with_gap_groups,
    collate_gap_groups,
    evaluate_regressor,
)


@dataclass(frozen=True)
class RegressorCvConfig:
    name: str
    hidden_dim: int
    layers: int
    dropout: float
    lr: float
    weight_decay: float
    smooth_l1_beta: float


DEFAULT_CONFIGS = [
    RegressorCvConfig("baseline_h224_l8", 224, 8, 0.10, 3e-4, 1e-5, 1.0),
    RegressorCvConfig("dropout005", 224, 8, 0.05, 3e-4, 1e-5, 1.0),
    RegressorCvConfig("dropout015", 224, 8, 0.15, 3e-4, 1e-5, 1.0),
    RegressorCvConfig("dropout020", 224, 8, 0.20, 3e-4, 1e-5, 1.0),
    RegressorCvConfig("lr0002", 224, 8, 0.10, 2e-4, 1e-5, 1.0),
    RegressorCvConfig("lr0004", 224, 8, 0.10, 4e-4, 1e-5, 1.0),
    RegressorCvConfig("lr0005", 224, 8, 0.10, 5e-4, 1e-5, 1.0),
    RegressorCvConfig("width192", 192, 8, 0.10, 3e-4, 1e-5, 1.0),
    RegressorCvConfig("width256", 256, 8, 0.10, 3e-4, 1e-5, 1.0),
    RegressorCvConfig("layers6", 224, 6, 0.10, 3e-4, 1e-5, 1.0),
    RegressorCvConfig("layers10", 224, 10, 0.10, 3e-4, 1e-5, 1.0),
    RegressorCvConfig("huber_beta05", 224, 8, 0.10, 3e-4, 1e-5, 0.5),
]


def load_config_grid(path: Path | None) -> list[RegressorCvConfig]:
    if path is None:
        return DEFAULT_CONFIGS
    raw = json.loads(path.read_text(encoding="utf-8"))
    configs = []
    for item in raw:
        configs.append(
            RegressorCvConfig(
                name=str(item["name"]),
                hidden_dim=int(item.get("hidden_dim", 224)),
                layers=int(item.get("layers", 8)),
                dropout=float(item.get("dropout", 0.10)),
                lr=float(item.get("lr", 3e-4)),
                weight_decay=float(item.get("weight_decay", 1e-5)),
                smooth_l1_beta=float(item.get("smooth_l1_beta", 1.0)),
            )
        )
    return configs


def stratified_like_subset(indices: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    n_select = max(1, int(round(shuffled.shape[0] * fraction)))
    selected = shuffled[:n_select]
    selected.sort()
    return selected.astype(np.int64)


def make_fold_ids(size: int, selected_indices: np.ndarray, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_ids = np.full(size, -1, dtype=np.int16)
    shuffled = np.array(selected_indices, copy=True)
    rng.shuffle(shuffled)
    for fold_id, part in enumerate(np.array_split(shuffled, folds)):
        fold_ids[part] = fold_id
    return fold_ids


def metrics_to_dict(metrics) -> dict[str, float]:
    return {
        "loss": metrics.loss,
        "lower_mae_khz": metrics.lower_mae_khz,
        "upper_mae_khz": metrics.upper_mae_khz,
        "mean_bound_mae_khz": 0.5 * (metrics.lower_mae_khz + metrics.upper_mae_khz),
        "width_mae_khz": metrics.width_mae_khz,
        "lower_rmse_khz": metrics.lower_rmse_khz,
        "upper_rmse_khz": metrics.upper_rmse_khz,
        "width_rmse_khz": metrics.width_rmse_khz,
        "lower_bias_khz": metrics.lower_bias_khz,
        "upper_bias_khz": metrics.upper_bias_khz,
        "width_bias_khz": metrics.width_bias_khz,
        "lower_r2": metrics.lower_r2,
        "upper_r2": metrics.upper_r2,
        "width_r2": metrics.width_r2,
    }


def train_one_fold(
    *,
    indexed,
    feature_indices: np.ndarray,
    config: RegressorCvConfig,
    fold_id: int,
    fold_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    use_grad_scaler: bool,
    run_dir: Path,
) -> dict[str, object]:
    train_mask = (fold_ids >= 0) & (fold_ids != fold_id)
    val_mask = fold_ids == fold_id
    train_dataset = CachedGapGroupDataset(indexed, train_mask, feature_indices=feature_indices)
    val_dataset = CachedGapGroupDataset(indexed, val_mask, feature_indices=feature_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_gap_groups,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_gap_groups,
        persistent_workers=args.num_workers > 0,
    )

    sample_batch = next(iter(train_loader))
    node_dim = int(sample_batch["node_x"].shape[1])
    edge_dim = int(sample_batch["edge_attr"].shape[1])
    graph_dim = int(sample_batch["graph_x"].shape[1])
    scaler: Scaler = build_scaler(indexed.gap_targets[train_mask, :2])
    model = GapBoundsRegressor(
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_dim=graph_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
    ).to(device)
    scaler = scaler.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_cosine_warmup_scheduler(
        optimizer=optimizer,
        total_steps=max(len(train_loader) * args.epochs, 1),
        warmup_fraction=args.warmup_fraction,
        min_lr_scale=args.min_lr_scale,
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    config_dir = run_dir / config.name
    config_dir.mkdir(parents=True, exist_ok=True)
    history_path = config_dir / f"fold_{fold_id:02d}_history.jsonl"

    best_state = None
    best_score = float("inf")
    best_epoch = 0
    best_row: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
            gap_target = move_batch_tensor(batch, "gap_target", device, non_blocking=amp_enabled)
            target_lu = gap_target[:, :2]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred_norm = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                target_norm = scaler.encode(target_lu)
                loss = F.smooth_l1_loss(pred_norm, target_norm, beta=config.smooth_l1_beta)
            if use_grad_scaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            train_loss_sum += float(loss.item()) * int(target_lu.shape[0])
            train_count += int(target_lu.shape[0])

        val_metrics = evaluate_regressor(model, val_loader, scaler, device, amp_enabled, amp_dtype)
        metric_dict = metrics_to_dict(val_metrics)
        row: dict[str, object] = {
            "config": config.name,
            "fold": fold_id,
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"val_{key}": value for key, value in metric_dict.items()},
        }
        history.append(row)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        score = float(row[f"val_{args.best_metric}"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_row = row
            if args.save_checkpoints:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[gap{args.target_gap_index}_cv] config={config.name} fold={fold_id + 1}/{args.folds} "
            f"epoch={epoch}/{args.epochs} train_loss={row['train_loss']:.6f} "
            f"val_mean_bound={metric_dict['mean_bound_mae_khz']:.6f} "
            f"val_lower={metric_dict['lower_mae_khz']:.6f} "
            f"val_upper={metric_dict['upper_mae_khz']:.6f} "
            f"val_width={metric_dict['width_mae_khz']:.6f} "
            f"r2_lower={metric_dict['lower_r2']:.6f} "
            f"r2_upper={metric_dict['upper_r2']:.6f} "
            f"best_{args.best_metric}={best_score:.6f}@{best_epoch}",
            flush=True,
        )

    if args.save_checkpoints and best_state is not None:
        torch.save(best_state, config_dir / f"fold_{fold_id:02d}_best.pt")

    summary = {
        "config": config.__dict__,
        "fold": fold_id,
        "elapsed_seconds": time.perf_counter() - start,
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metric": args.best_metric,
        "best_row": best_row,
        "history": history,
    }
    (config_dir / f"fold_{fold_id:02d}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del train_loader, val_loader, train_dataset, val_dataset, model, optimizer, scheduler, grad_scaler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def aggregate_results(fold_summaries: list[dict[str, object]], best_metric: str) -> dict[str, object]:
    by_config: dict[str, list[dict[str, object]]] = {}
    for summary in fold_summaries:
        by_config.setdefault(str(summary["config"]["name"]), []).append(summary)

    aggregate: dict[str, object] = {}
    metric_names = [
        "mean_bound_mae_khz",
        "lower_mae_khz",
        "upper_mae_khz",
        "width_mae_khz",
        "lower_rmse_khz",
        "upper_rmse_khz",
        "width_rmse_khz",
        "lower_r2",
        "upper_r2",
        "width_r2",
    ]
    for name, summaries in by_config.items():
        best_rows = [s["best_row"] for s in summaries if s.get("best_row") is not None]
        values = {
            metric: [float(row[f"val_{metric}"]) for row in best_rows]
            for metric in metric_names
        }
        aggregate[name] = {
            "folds": len(best_rows),
            "config": summaries[0]["config"],
            "best_epochs": [int(s["best_epoch"]) for s in summaries],
            "metric_mean": {key: float(np.mean(vals)) for key, vals in values.items()},
            "metric_std": {key: float(np.std(vals, ddof=0)) for key, vals in values.items()},
            "selection_score": float(np.mean(values[best_metric])),
        }
    ranked = sorted(aggregate.items(), key=lambda item: item[1]["selection_score"])
    return {
        "aggregate": aggregate,
        "ranked_configs": [{"name": name, **payload} for name, payload in ranked],
        "best_config_name": ranked[0][0] if ranked else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold CV for a gap-index-specific lower/upper regressor.")
    parser.add_argument("--preprocessed-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-gap-index", type=int, default=3, choices=[2, 3])
    parser.add_argument("--sample-fraction", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp-mode", type=str, default="bf16", choices=["auto", "bf16", "fp16", "off"])
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--min-lr-scale", type=float, default=0.10)
    parser.add_argument("--graph-feature-mode", type=str, default="novel105_phase_edge_shell")
    parser.add_argument("--major-gap-indices", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-sample-index", type=int, default=0)
    parser.add_argument("--max-sample-index", type=int, default=0)
    parser.add_argument("--configs-json", type=Path, default=None)
    parser.add_argument("--best-metric", choices=["mean_bound_mae_khz", "width_mae_khz"], default="mean_bound_mae_khz")
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.sample_fraction <= 1.0):
        raise ValueError("--sample-fraction must be in (0, 1].")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    set_seed(args.seed)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    configs = load_config_grid(args.configs_json)
    major_gap_indices = (int(args.major_gap_indices[0]), int(args.major_gap_indices[1]))
    try:
        target_group_id = major_gap_indices.index(int(args.target_gap_index))
    except ValueError as exc:
        raise ValueError("--target-gap-index must be one of --major-gap-indices.") from exc

    print(f"[gap{args.target_gap_index}_cv] building positive-gap index...", flush=True)
    indexed = build_index_with_gap_groups(
        cache_roots=args.preprocessed_root,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        major_gap_indices=major_gap_indices,
        min_sample_index=args.min_sample_index,
        max_sample_index=args.max_sample_index,
    )
    target_indices = np.flatnonzero(indexed.group_ids == target_group_id)
    selected_indices = stratified_like_subset(target_indices, args.sample_fraction, args.seed)
    fold_ids = make_fold_ids(indexed.size, selected_indices, args.folds, args.seed + 19)
    feature_indices = get_graph_feature_indices(args.graph_feature_mode)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled, amp_dtype, use_grad_scaler, resolved_amp_mode = resolve_amp_config(device, args.amp_mode)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "task": f"gap{args.target_gap_index}_lower_upper_regression_cv",
        "target_gap_index": int(args.target_gap_index),
        "target_group_id": int(target_group_id),
        "preprocessed_roots": [str(root) for root in args.preprocessed_root],
        "major_gap_indices": list(major_gap_indices),
        "target_available_samples": int(target_indices.shape[0]),
        "selected_target_samples": int(selected_indices.shape[0]),
        "sample_fraction": args.sample_fraction,
        "folds": args.folds,
        "fold_counts": [int(np.sum(fold_ids == fold_id)) for fold_id in range(args.folds)],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "graph_feature_mode": args.graph_feature_mode,
        "device": str(device),
        "amp_mode": resolved_amp_mode,
        "best_metric": args.best_metric,
        "seed": args.seed,
        "configs": [config.__dict__ for config in configs],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)

    fold_summaries: list[dict[str, object]] = []
    start = time.perf_counter()
    for config in configs:
        print(f"[gap{args.target_gap_index}_cv] starting config={config.name}", flush=True)
        for fold_id in range(args.folds):
            summary = train_one_fold(
                indexed=indexed,
                feature_indices=feature_indices,
                config=config,
                fold_id=fold_id,
                fold_ids=fold_ids,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                use_grad_scaler=use_grad_scaler,
                run_dir=run_dir,
            )
            fold_summaries.append(summary)
            partial = aggregate_results(fold_summaries, args.best_metric)
            partial["elapsed_seconds"] = time.perf_counter() - start
            (run_dir / "partial_summary.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")

    final = aggregate_results(fold_summaries, args.best_metric)
    final["metadata"] = metadata
    final["elapsed_seconds"] = time.perf_counter() - start
    final["fold_summaries"] = fold_summaries
    (run_dir / "cv_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
