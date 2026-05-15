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
from torch.utils.data import DataLoader

from train_first_band_gap_multitask import (
    build_cosine_warmup_scheduler,
    get_graph_feature_indices,
    move_batch_tensor,
    resolve_amp_config,
    set_seed,
)
from train_gap_index_classifier_improved import (
    build_index_with_four_classes,
    compute_class_weights,
    evaluate_classifier,
    focal_cross_entropy,
    make_sampler,
)
from train_gap_index_then_regress import (
    CachedGapGroupDataset,
    GapGroupClassifier,
    collate_gap_groups,
)


@dataclass(frozen=True)
class CvConfig:
    name: str
    hidden_dim: int
    layers: int
    dropout: float
    lr: float
    weight_decay: float
    class_weight_scheme: str
    sampler: str
    sampler_power: float
    focal_gamma: float
    label_smoothing: float


DEFAULT_CONFIGS = [
    CvConfig(
        name="balanced_focal15",
        hidden_dim=224,
        layers=8,
        dropout=0.10,
        lr=3e-4,
        weight_decay=1e-5,
        class_weight_scheme="sqrt_inverse",
        sampler="weighted",
        sampler_power=0.50,
        focal_gamma=1.5,
        label_smoothing=0.02,
    ),
    CvConfig(
        name="stronger_minority",
        hidden_dim=224,
        layers=8,
        dropout=0.12,
        lr=3e-4,
        weight_decay=1e-5,
        class_weight_scheme="sqrt_inverse",
        sampler="weighted",
        sampler_power=0.70,
        focal_gamma=2.0,
        label_smoothing=0.02,
    ),
    CvConfig(
        name="softer_focal",
        hidden_dim=224,
        layers=8,
        dropout=0.08,
        lr=3e-4,
        weight_decay=1e-5,
        class_weight_scheme="sqrt_inverse",
        sampler="weighted",
        sampler_power=0.35,
        focal_gamma=1.0,
        label_smoothing=0.01,
    ),
]


def stratified_fraction_subset(
    labels: np.ndarray,
    fraction: float,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    per_class: dict[str, dict[str, int]] = {}
    for class_id in range(num_classes):
        idx = np.flatnonzero(labels == class_id)
        rng.shuffle(idx)
        target = max(1, int(round(idx.shape[0] * fraction)))
        chosen = np.sort(idx[:target])
        selected_parts.append(chosen)
        per_class[str(class_id)] = {
            "available": int(idx.shape[0]),
            "selected": int(chosen.shape[0]),
        }
    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    metadata = {
        "requested_fraction": float(fraction),
        "selected_samples": int(selected.shape[0]),
        "per_class": per_class,
    }
    return selected.astype(np.int64), metadata


def make_stratified_fold_ids(
    labels: np.ndarray,
    selected_indices: np.ndarray,
    folds: int,
    num_classes: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_ids = np.full(labels.shape[0], -1, dtype=np.int16)
    selected_mask = np.zeros(labels.shape[0], dtype=bool)
    selected_mask[selected_indices] = True
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(selected_mask & (labels == class_id))
        rng.shuffle(class_indices)
        for fold_id, part in enumerate(np.array_split(class_indices, folds)):
            fold_ids[part] = fold_id
    return fold_ids


def counts_for_mask(labels: np.ndarray, mask: np.ndarray, num_classes: int) -> list[int]:
    return np.bincount(labels[mask].astype(np.int64), minlength=num_classes).astype(int).tolist()


def load_config_grid(path: Path | None) -> list[CvConfig]:
    if path is None:
        return DEFAULT_CONFIGS
    raw = json.loads(path.read_text(encoding="utf-8"))
    configs = []
    for item in raw:
        configs.append(
            CvConfig(
                name=str(item["name"]),
                hidden_dim=int(item.get("hidden_dim", 224)),
                layers=int(item.get("layers", 8)),
                dropout=float(item.get("dropout", 0.10)),
                lr=float(item.get("lr", 3e-4)),
                weight_decay=float(item.get("weight_decay", 1e-5)),
                class_weight_scheme=str(item.get("class_weight_scheme", "sqrt_inverse")),
                sampler=str(item.get("sampler", "weighted")),
                sampler_power=float(item.get("sampler_power", 0.50)),
                focal_gamma=float(item.get("focal_gamma", 1.5)),
                label_smoothing=float(item.get("label_smoothing", 0.02)),
            )
        )
    return configs


def train_one_fold(
    *,
    indexed,
    feature_indices: np.ndarray,
    config: CvConfig,
    fold_id: int,
    fold_ids: np.ndarray,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    use_grad_scaler: bool,
    run_dir: Path,
) -> dict[str, object]:
    num_classes = len(class_names)
    train_mask = (fold_ids >= 0) & (fold_ids != fold_id)
    val_mask = fold_ids == fold_id
    train_dataset = CachedGapGroupDataset(indexed, train_mask, feature_indices=feature_indices)
    val_dataset = CachedGapGroupDataset(indexed, val_mask, feature_indices=feature_indices)

    train_labels = train_dataset.group_ids.astype(np.int64)
    class_weights = compute_class_weights(train_labels, num_classes, config.class_weight_scheme)
    sampler = make_sampler(train_labels, config.sampler_power) if config.sampler == "weighted" else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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
    model = GapGroupClassifier(
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_dim=graph_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_cosine_warmup_scheduler(
        optimizer=optimizer,
        total_steps=max(len(train_loader) * args.epochs, 1),
        warmup_fraction=args.warmup_fraction,
        min_lr_scale=args.min_lr_scale,
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    class_weights_device = class_weights.to(device)

    config_dir = run_dir / config.name
    config_dir.mkdir(parents=True, exist_ok=True)
    history_path = config_dir / f"fold_{fold_id:02d}_history.jsonl"
    best_score = -1.0
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
            group_id = move_batch_tensor(batch, "group_id", device, non_blocking=amp_enabled)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                loss = focal_cross_entropy(
                    logits=logits,
                    target=group_id,
                    class_weights=class_weights_device,
                    gamma=config.focal_gamma,
                    label_smoothing=config.label_smoothing,
                )
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
            train_loss_sum += float(loss.item()) * int(group_id.shape[0])
            train_count += int(group_id.shape[0])

        val_metrics = evaluate_classifier(
            model=model,
            loader=val_loader,
            class_weights=class_weights,
            class_names=class_names,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
        row: dict[str, object] = {
            "config": config.name,
            "fold": fold_id,
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "val_balanced_accuracy": val_metrics.balanced_accuracy,
            "val_macro_f1": val_metrics.macro_f1,
            "val_weighted_f1": val_metrics.weighted_f1,
            "val_per_class": val_metrics.per_class,
            "val_confusion_matrix": val_metrics.confusion_matrix,
        }
        history.append(row)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        score = float(row[args.best_metric])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_row = row

        no_gap_recall = val_metrics.per_class["no_gap"]["recall"]
        other_recall = val_metrics.per_class["other"]["recall"]
        print(
            f"[cv] config={config.name} fold={fold_id + 1}/{args.folds} "
            f"epoch={epoch}/{args.epochs} train_loss={row['train_loss']:.6f} "
            f"val_macro_f1={val_metrics.macro_f1:.6f} "
            f"val_bal_acc={val_metrics.balanced_accuracy:.6f} "
            f"no_gap_rec={no_gap_recall:.4f} other_rec={other_recall:.4f} "
            f"best_{args.best_metric}={best_score:.6f}@{best_epoch}",
            flush=True,
        )

    summary = {
        "config": config.__dict__,
        "fold": fold_id,
        "elapsed_seconds": time.perf_counter() - start,
        "train_counts": counts_for_mask(indexed.group_ids, train_mask, num_classes),
        "val_counts": counts_for_mask(indexed.group_ids, val_mask, num_classes),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_row": best_row,
        "history": history,
    }
    (config_dir / f"fold_{fold_id:02d}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    del train_loader, val_loader, train_dataset, val_dataset, model, optimizer, scheduler, grad_scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def aggregate_results(fold_summaries: list[dict[str, object]], class_names: list[str], best_metric: str) -> dict[str, object]:
    by_config: dict[str, list[dict[str, object]]] = {}
    for summary in fold_summaries:
        by_config.setdefault(str(summary["config"]["name"]), []).append(summary)

    aggregate: dict[str, object] = {}
    for name, summaries in by_config.items():
        best_rows = [s["best_row"] for s in summaries if s.get("best_row") is not None]
        metrics = {
            "val_macro_f1": [float(r["val_macro_f1"]) for r in best_rows],
            "val_balanced_accuracy": [float(r["val_balanced_accuracy"]) for r in best_rows],
            "val_accuracy": [float(r["val_accuracy"]) for r in best_rows],
            "val_weighted_f1": [float(r["val_weighted_f1"]) for r in best_rows],
        }
        per_class = {}
        for class_name in class_names:
            per_class[class_name] = {
                key: [float(r["val_per_class"][class_name][key]) for r in best_rows]
                for key in ["precision", "recall", "f1"]
            }
        aggregate[name] = {
            "folds": len(best_rows),
            "config": summaries[0]["config"],
            "best_epochs": [int(s["best_epoch"]) for s in summaries],
            "metric_mean": {key: float(np.mean(values)) for key, values in metrics.items()},
            "metric_std": {key: float(np.std(values, ddof=0)) for key, values in metrics.items()},
            "per_class_mean": {
                class_name: {key: float(np.mean(values)) for key, values in cls.items()}
                for class_name, cls in per_class.items()
            },
            "per_class_std": {
                class_name: {key: float(np.std(values, ddof=0)) for key, values in cls.items()}
                for class_name, cls in per_class.items()
            },
            "selection_score": float(np.mean(metrics[best_metric])),
        }
    ranked = sorted(aggregate.items(), key=lambda item: item[1]["selection_score"], reverse=True)
    return {
        "aggregate": aggregate,
        "ranked_configs": [{"name": name, **payload} for name, payload in ranked],
        "best_config_name": ranked[0][0] if ranked else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified k-fold CV for the four-class gap-index classifier.")
    parser.add_argument("--preprocessed-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fraction", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260514)
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
    parser.add_argument("--best-metric", choices=["val_macro_f1", "val_balanced_accuracy"], default="val_macro_f1")
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
    class_names = ["no_gap", f"gap{major_gap_indices[0]}", f"gap{major_gap_indices[1]}", "other"]
    num_classes = len(class_names)

    print(f"[cv] building full four-class index from {len(args.preprocessed_root)} cache root(s)...", flush=True)
    indexed = build_index_with_four_classes(
        cache_roots=args.preprocessed_root,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        major_gap_indices=major_gap_indices,
        min_sample_index=args.min_sample_index,
        max_sample_index=args.max_sample_index,
    )
    selected_indices, subset_metadata = stratified_fraction_subset(
        labels=indexed.group_ids,
        fraction=args.sample_fraction,
        num_classes=num_classes,
        seed=args.seed,
    )
    fold_ids = make_stratified_fold_ids(
        labels=indexed.group_ids,
        selected_indices=selected_indices,
        folds=args.folds,
        num_classes=num_classes,
        seed=args.seed + 17,
    )
    feature_indices = get_graph_feature_indices(args.graph_feature_mode)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled, amp_dtype, use_grad_scaler, resolved_amp_mode = resolve_amp_config(device, args.amp_mode)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "preprocessed_roots": [str(root) for root in args.preprocessed_root],
        "class_mode": "four-class",
        "class_names": class_names,
        "full_dataset_samples": int(indexed.size),
        "full_class_counts": np.bincount(indexed.group_ids, minlength=num_classes).astype(int).tolist(),
        "subset": subset_metadata,
        "folds": args.folds,
        "fold_counts": [
            counts_for_mask(indexed.group_ids, fold_ids == fold_id, num_classes)
            for fold_id in range(args.folds)
        ],
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
        print(f"[cv] starting config={config.name}", flush=True)
        for fold_id in range(args.folds):
            summary = train_one_fold(
                indexed=indexed,
                feature_indices=feature_indices,
                config=config,
                fold_id=fold_id,
                fold_ids=fold_ids,
                class_names=class_names,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                use_grad_scaler=use_grad_scaler,
                run_dir=run_dir,
            )
            fold_summaries.append(summary)
            partial = aggregate_results(fold_summaries, class_names, args.best_metric)
            partial["elapsed_seconds"] = time.perf_counter() - start
            (run_dir / "partial_summary.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")

    final = aggregate_results(fold_summaries, class_names, args.best_metric)
    final["metadata"] = metadata
    final["elapsed_seconds"] = time.perf_counter() - start
    final["fold_summaries"] = fold_summaries
    (run_dir / "cv_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
