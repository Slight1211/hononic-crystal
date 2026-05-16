from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_first_band_gap_multitask import (
    build_scaler,
    get_graph_feature_indices,
    make_split_masks,
    resolve_amp_config,
    set_seed,
)
from train_gap_index_then_regress import (
    CachedGapGroupDataset,
    collate_gap_groups,
    build_index_with_gap_groups,
    evaluate_regressor,
    train_regressor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gap-bound regressors without training the gap-index classifier.")
    parser.add_argument("--preprocessed-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--regressor-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=224)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp-mode", type=str, default="bf16", choices=["auto", "bf16", "fp16", "off"])
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--min-lr-scale", type=float, default=0.10)
    parser.add_argument("--graph-feature-mode", type=str, default="novel105_phase_edge_shell")
    parser.add_argument("--major-gap-indices", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--min-group-train-samples", type=int, default=200)
    parser.add_argument("--min-sample-index", type=int, default=0)
    parser.add_argument("--max-sample-index", type=int, default=0)
    parser.add_argument(
        "--regressor-filter",
        nargs="+",
        default=None,
        choices=["gap2_expert", "gap3_expert", "other_fallback"],
        help="Optional subset of regressors to train.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    major_gap_indices = (int(args.major_gap_indices[0]), int(args.major_gap_indices[1]))
    indexed = build_index_with_gap_groups(
        cache_roots=args.preprocessed_root,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        major_gap_indices=major_gap_indices,
        min_sample_index=args.min_sample_index,
        max_sample_index=args.max_sample_index,
    )

    train_indices, val_indices, test_indices = make_split_masks(indexed.sample_ids)
    feature_indices = get_graph_feature_indices(args.graph_feature_mode)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled, amp_dtype, use_grad_scaler, resolved_amp_mode = resolve_amp_config(device, args.amp_mode)

    sample_dataset = CachedGapGroupDataset(indexed, train_indices, feature_indices=feature_indices)
    sample_loader = DataLoader(
        sample_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_gap_groups,
        persistent_workers=args.num_workers > 0,
    )
    sample_batch = next(iter(sample_loader))
    node_dim = int(sample_batch["node_x"].shape[1])
    edge_dim = int(sample_batch["edge_attr"].shape[1])
    graph_dim = int(sample_batch["graph_x"].shape[1])

    group_names = [f"gap{major_gap_indices[0]}", f"gap{major_gap_indices[1]}", "other"]
    train_group_counts = np.bincount(indexed.group_ids[train_indices], minlength=3).astype(int)
    val_group_counts = np.bincount(indexed.group_ids[val_indices], minlength=3).astype(int)
    test_group_counts = np.bincount(indexed.group_ids[test_indices], minlength=3).astype(int)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "preprocessed_roots": [str(root) for root in args.preprocessed_root],
        "major_gap_indices": list(major_gap_indices),
        "group_names": group_names,
        "graph_feature_mode": args.graph_feature_mode,
        "graph_feature_dim": int(graph_dim),
        "train_samples": int(train_indices.sum()),
        "val_samples": int(val_indices.sum()),
        "test_samples": int(test_indices.sum()),
        "train_group_counts": train_group_counts.tolist(),
        "val_group_counts": val_group_counts.tolist(),
        "test_group_counts": test_group_counts.tolist(),
        "unique_positive_gap_samples": int(indexed.size),
        "duplicate_samples_skipped": int(indexed.duplicate_samples_skipped),
        "no_gap_samples_skipped": int(indexed.no_gap_samples_skipped),
        "sample_index_filtered_skipped": int(indexed.sample_index_filtered_skipped),
        "device": str(device),
        "amp_mode": resolved_amp_mode,
        "regressor_epochs": args.regressor_epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "regressor_filter": args.regressor_filter,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    gap2_train = train_indices & (indexed.group_ids == 0)
    gap2_val = val_indices & (indexed.group_ids == 0)
    gap2_test = test_indices & (indexed.group_ids == 0)
    gap3_train = train_indices & (indexed.group_ids == 1)
    gap3_val = val_indices & (indexed.group_ids == 1)
    gap3_test = test_indices & (indexed.group_ids == 1)
    other_train = train_indices & (indexed.group_ids == 2)
    other_val = val_indices & (indexed.group_ids == 2)
    other_test = test_indices & (indexed.group_ids == 2)

    regressor_specs = [
        ("gap2_expert", gap2_train, gap2_val, gap2_test),
        ("gap3_expert", gap3_train, gap3_val, gap3_test),
    ]
    if int(other_train.sum()) < args.min_group_train_samples:
        regressor_specs.append(("other_fallback", train_indices, val_indices, test_indices))
    else:
        regressor_specs.append(("other_fallback", other_train, other_val, other_test))
    if args.regressor_filter is not None:
        selected_names = set(args.regressor_filter)
        regressor_specs = [spec for spec in regressor_specs if spec[0] in selected_names]
        if not regressor_specs:
            raise ValueError(f"No regressors selected by --regressor-filter={args.regressor_filter}")

    start_time = time.perf_counter()
    regressor_infos: dict[str, dict[str, object]] = {}

    for name, train_sel, val_sel, test_sel in regressor_specs:
        train_dataset = CachedGapGroupDataset(indexed, train_sel, feature_indices=feature_indices)
        val_dataset = CachedGapGroupDataset(indexed, val_sel, feature_indices=feature_indices)
        test_dataset = CachedGapGroupDataset(indexed, test_sel, feature_indices=feature_indices)
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
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_gap_groups,
            persistent_workers=args.num_workers > 0,
        )
        scaler = build_scaler(indexed.gap_targets[train_sel, :2])
        model, info = train_regressor(
            name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            target_scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            use_grad_scaler=use_grad_scaler,
            node_dim=node_dim,
            edge_dim=edge_dim,
            graph_dim=graph_dim,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.regressor_epochs,
            warmup_fraction=args.warmup_fraction,
            min_lr_scale=args.min_lr_scale,
            output_dir=run_dir,
        )
        test_metrics = evaluate_regressor(model, test_loader, scaler, device, amp_enabled, amp_dtype)
        regressor_infos[name] = {
            **info,
            "train_samples": int(train_sel.sum()),
            "val_samples": int(val_sel.sum()),
            "test_samples": int(test_sel.sum()),
            "test_metrics": {
                "loss": test_metrics.loss,
                "lower_mae_khz": test_metrics.lower_mae_khz,
                "upper_mae_khz": test_metrics.upper_mae_khz,
                "width_mae_khz": test_metrics.width_mae_khz,
                "lower_rmse_khz": test_metrics.lower_rmse_khz,
                "upper_rmse_khz": test_metrics.upper_rmse_khz,
                "width_rmse_khz": test_metrics.width_rmse_khz,
                "lower_bias_khz": test_metrics.lower_bias_khz,
                "upper_bias_khz": test_metrics.upper_bias_khz,
                "width_bias_khz": test_metrics.width_bias_khz,
                "lower_r2": test_metrics.lower_r2,
                "upper_r2": test_metrics.upper_r2,
                "width_r2": test_metrics.width_r2,
            },
        }

    summary = {
        "finished_at": datetime.now().isoformat(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "regressors": regressor_infos,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
