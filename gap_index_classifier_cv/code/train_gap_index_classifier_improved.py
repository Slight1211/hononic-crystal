import argparse
import json
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_first_band_gap_multitask import (
    build_cosine_warmup_scheduler,
    discover_preprocessed_chunk_paths,
    get_graph_feature_indices,
    move_batch_tensor,
    resolve_amp_config,
    set_seed,
)
from train_gap_index_then_regress import (
    CachedGapGroupDataset,
    GapGroupClassifier,
    IndexedGapGroupSamples,
    collate_gap_groups,
    load_cache_source_infos,
    map_preprocessed_chunk_to_original,
)


@dataclass
class ClassifierEval:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    y_true: np.ndarray
    y_pred: np.ndarray


def map_gap_index_to_four_class(has_gap: bool, gap_index: int, major_gap_indices: tuple[int, int]) -> int:
    if not has_gap or gap_index < 0:
        return 0
    if gap_index == major_gap_indices[0]:
        return 1
    if gap_index == major_gap_indices[1]:
        return 2
    return 3


def build_index_with_four_classes(
    cache_roots: list[Path],
    max_batches: int,
    max_samples: int,
    major_gap_indices: tuple[int, int],
    min_sample_index: int = 0,
    max_sample_index: int = 0,
) -> IndexedGapGroupSamples:
    chunk_paths = discover_preprocessed_chunk_paths(cache_roots, max_batches=max_batches)
    infos = load_cache_source_infos(cache_roots)

    chunk_ids: list[int] = []
    local_ids: list[int] = []
    sample_ids: list[int] = []
    gap_targets: list[list[float]] = []
    gap_index_values: list[int] = []
    group_ids: list[int] = []
    seen_sample_ids: set[int] = set()
    duplicate_samples_skipped = 0
    no_gap_samples = 0
    sample_index_filtered_skipped = 0

    for chunk_id, chunk_path in enumerate(chunk_paths):
        chunk = torch.load(chunk_path, map_location="cpu")
        original_path = map_preprocessed_chunk_to_original(chunk_path, infos)
        with original_path.open("rb") as f:
            records = pickle.load(f)
        sample_index_chunk = chunk["sample_index"].numpy()
        gap_target_chunk = chunk["gap_target"].numpy()
        has_gap_chunk = chunk["has_gap"].numpy()
        if len(records) != int(sample_index_chunk.shape[0]):
            raise RuntimeError(f"Chunk/sample count mismatch: {chunk_path} vs {original_path}")

        for local_id, record in enumerate(records):
            if max_samples > 0 and len(sample_ids) >= max_samples:
                break
            record_sample_index = int(record["sample_index"])
            cache_sample_index = int(sample_index_chunk[local_id])
            if record_sample_index != cache_sample_index:
                raise RuntimeError(
                    f"Sample index mismatch at {original_path} local_id={local_id}: "
                    f"record={record_sample_index}, cache={cache_sample_index}"
                )
            if cache_sample_index < min_sample_index or (max_sample_index > 0 and cache_sample_index > max_sample_index):
                sample_index_filtered_skipped += 1
                continue
            if cache_sample_index in seen_sample_ids:
                duplicate_samples_skipped += 1
                continue
            seen_sample_ids.add(cache_sample_index)

            has_gap = bool(float(has_gap_chunk[local_id]) > 0.5)
            gap_index = int(record["gap_index"]) if has_gap and record["gap_index"] is not None else -1
            class_id = map_gap_index_to_four_class(has_gap, gap_index, major_gap_indices)
            if class_id == 0:
                no_gap_samples += 1

            chunk_ids.append(chunk_id)
            local_ids.append(local_id)
            sample_ids.append(cache_sample_index)
            gap_targets.append(gap_target_chunk[local_id].astype(np.float32).tolist())
            gap_index_values.append(gap_index)
            group_ids.append(class_id)
        if max_samples > 0 and len(sample_ids) >= max_samples:
            break

    if not sample_ids:
        raise RuntimeError("No samples were found while building the four-class index.")

    return IndexedGapGroupSamples(
        chunk_paths=[str(p) for p in chunk_paths],
        chunk_ids=np.asarray(chunk_ids, dtype=np.int32),
        local_ids=np.asarray(local_ids, dtype=np.int32),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        gap_targets=np.asarray(gap_targets, dtype=np.float32),
        gap_index_values=np.asarray(gap_index_values, dtype=np.int64),
        group_ids=np.asarray(group_ids, dtype=np.int64),
        duplicate_samples_skipped=duplicate_samples_skipped,
        no_gap_samples_skipped=no_gap_samples,
        sample_index_filtered_skipped=sample_index_filtered_skipped,
    )


def make_stratified_split_masks(
    labels: np.ndarray,
    eligible_mask: np.ndarray,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(labels.shape[0], dtype=bool)
    val_mask = np.zeros(labels.shape[0], dtype=bool)
    test_mask = np.zeros(labels.shape[0], dtype=bool)

    for class_id in range(num_classes):
        idx = np.flatnonzero(eligible_mask & (labels == class_id))
        rng.shuffle(idx)
        n = int(idx.shape[0])
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]
        test_mask[test_idx] = True
        val_mask[val_idx] = True
        train_mask[train_idx] = True

    return train_mask, val_mask, test_mask


def compute_class_weights(labels: np.ndarray, num_classes: int, scheme: str) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    if scheme == "none":
        weights = np.ones(num_classes, dtype=np.float64)
    elif scheme == "inverse":
        weights = counts.sum() / np.maximum(counts, 1.0)
    elif scheme == "sqrt_inverse":
        weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    elif scheme == "effective":
        beta = 0.9999
        weights = (1.0 - beta) / np.maximum(1.0 - np.power(beta, counts), 1e-12)
    else:
        raise ValueError(f"Unknown class weight scheme: {scheme}")
    weights = weights / max(float(weights.mean()), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    probs = torch.softmax(logits.float(), dim=1).gather(1, target.view(-1, 1)).squeeze(1).clamp(1e-6, 1.0)
    focal = torch.pow(1.0 - probs, gamma)
    return torch.mean(focal.to(ce.dtype) * ce)


def evaluate_classifier(
    model: GapGroupClassifier,
    loader: DataLoader,
    class_weights: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    gamma: float,
    label_smoothing: float,
) -> ClassifierEval:
    model.eval()
    class_weights = class_weights.to(device)
    loss_sum = 0.0
    sample_count = 0
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
            group_id = move_batch_tensor(batch, "group_id", device, non_blocking=amp_enabled)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                loss = focal_cross_entropy(logits, group_id, class_weights, gamma, label_smoothing)

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_list.append(pred)
            y_true_list.append(group_id.cpu().numpy())
            loss_sum += float(loss.item()) * int(group_id.shape[0])
            sample_count += int(group_id.shape[0])

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, name in enumerate(class_names)
    }
    return ClassifierEval(
        loss=loss_sum / max(sample_count, 1),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        per_class=per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))).astype(int).tolist(),
        y_true=y_true,
        y_pred=y_pred,
    )


def make_sampler(labels: np.ndarray, power: float) -> WeightedRandomSampler:
    counts = np.bincount(labels.astype(np.int64)).astype(np.float64)
    weights_by_class = np.power(1.0 / np.maximum(counts, 1.0), power)
    sample_weights = weights_by_class[labels.astype(np.int64)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(labels.shape[0]),
        replacement=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved standalone gap-index classifier training.")
    parser.add_argument("--preprocessed-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-mode", choices=["four-class"], default="four-class")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=448)
    parser.add_argument("--hidden-dim", type=int, default=224)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp-mode", type=str, default="bf16", choices=["auto", "bf16", "fp16", "off"])
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--min-lr-scale", type=float, default=0.10)
    parser.add_argument("--graph-feature-mode", type=str, default="novel105_phase_edge_shell")
    parser.add_argument("--major-gap-indices", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--min-sample-index", type=int, default=0)
    parser.add_argument("--max-sample-index", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--class-weight-scheme", choices=["none", "inverse", "sqrt_inverse", "effective"], default="sqrt_inverse")
    parser.add_argument("--sampler", choices=["none", "weighted"], default="weighted")
    parser.add_argument("--sampler-power", type=float, default=0.50)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--best-metric", choices=["macro_f1", "balanced_accuracy"], default="macro_f1")
    args = parser.parse_args()

    set_seed(args.seed)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    major_gap_indices = (int(args.major_gap_indices[0]), int(args.major_gap_indices[1]))
    indexed = build_index_with_four_classes(
        cache_roots=args.preprocessed_root,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        major_gap_indices=major_gap_indices,
        min_sample_index=args.min_sample_index,
        max_sample_index=args.max_sample_index,
    )

    eligible = np.ones(indexed.size, dtype=bool)
    num_classes = 4
    class_names = ["no_gap", f"gap{major_gap_indices[0]}", f"gap{major_gap_indices[1]}", "other"]

    train_mask, val_mask, test_mask = make_stratified_split_masks(
        labels=indexed.group_ids,
        eligible_mask=eligible,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        num_classes=num_classes,
    )
    feature_indices = get_graph_feature_indices(args.graph_feature_mode)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled, amp_dtype, use_grad_scaler, resolved_amp_mode = resolve_amp_config(device, args.amp_mode)

    train_dataset = CachedGapGroupDataset(indexed, train_mask, feature_indices=feature_indices)
    val_dataset = CachedGapGroupDataset(indexed, val_mask, feature_indices=feature_indices)
    test_dataset = CachedGapGroupDataset(indexed, test_mask, feature_indices=feature_indices)

    train_labels = train_dataset.group_ids.astype(np.int64)
    class_weights = compute_class_weights(train_labels, num_classes, args.class_weight_scheme)
    sampler = make_sampler(train_labels, args.sampler_power) if args.sampler == "weighted" else None
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
    test_loader = DataLoader(
        test_dataset,
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
    model = GapGroupClassifier(node_dim, edge_dim, graph_dim, args.hidden_dim, args.layers, args.dropout, num_classes=num_classes).to(device)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "preprocessed_roots": [str(root) for root in args.preprocessed_root],
        "class_mode": args.class_mode,
        "class_names": class_names,
        "major_gap_indices": list(major_gap_indices),
        "graph_feature_mode": args.graph_feature_mode,
        "graph_feature_dim": graph_dim,
        "train_group_counts": np.bincount(indexed.group_ids[train_mask], minlength=num_classes).astype(int).tolist(),
        "val_group_counts": np.bincount(indexed.group_ids[val_mask], minlength=num_classes).astype(int).tolist(),
        "test_group_counts": np.bincount(indexed.group_ids[test_mask], minlength=num_classes).astype(int).tolist(),
        "eligible_samples": int(eligible.sum()),
        "excluded_samples": int((~eligible).sum()),
        "device": str(device),
        "amp_mode": resolved_amp_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "class_weight_scheme": args.class_weight_scheme,
        "class_weights": class_weights.tolist(),
        "sampler": args.sampler,
        "sampler_power": args.sampler_power,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "best_metric": args.best_metric,
        "seed": args.seed,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_cosine_warmup_scheduler(
        optimizer=optimizer,
        total_steps=max(len(train_loader) * args.epochs, 1),
        warmup_fraction=args.warmup_fraction,
        min_lr_scale=args.min_lr_scale,
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    class_weights_device = class_weights.to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    history: list[dict[str, object]] = []
    history_path = run_dir / "history.jsonl"
    start_time = time.perf_counter()

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
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
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
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
        row: dict[str, object] = {
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

        score = float(getattr(val_metrics, args.best_metric))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "gap_index_classifier_improved_best.pt")

        print(
            f"[improved_classifier] epoch {epoch}/{args.epochs} "
            f"train_loss={row['train_loss']:.6f} "
            f"val_bal_acc={val_metrics.balanced_accuracy:.6f} "
            f"val_macro_f1={val_metrics.macro_f1:.6f} "
            f"best_{args.best_metric}={best_score:.6f}@{best_epoch}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("No best classifier checkpoint was selected.")
    model.load_state_dict(best_state)
    test_metrics = evaluate_classifier(
        model=model,
        loader=test_loader,
        class_weights=class_weights,
        class_names=class_names,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    summary = {
        "finished_at": datetime.now().isoformat(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metric": args.best_metric,
        "best_model_path": str(run_dir / "gap_index_classifier_improved_best.pt"),
        "history": history,
        "test_metrics": {
            "loss": test_metrics.loss,
            "accuracy": test_metrics.accuracy,
            "balanced_accuracy": test_metrics.balanced_accuracy,
            "macro_f1": test_metrics.macro_f1,
            "weighted_f1": test_metrics.weighted_f1,
            "per_class": test_metrics.per_class,
            "confusion_matrix": test_metrics.confusion_matrix,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
