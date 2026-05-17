from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    GapBoundsRegressor,
    build_index_with_gap_groups,
    collate_gap_groups,
    predict_regressor,
)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    ss_res = float(np.sum(err * err))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered * centered))
    return float("nan") if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def save_prediction_csv(path: Path, pred: dict[str, np.ndarray]) -> None:
    pred_width = pred["pred_upper_khz"] - pred["pred_lower_khz"]
    data = np.column_stack(
        [
            pred["sample_index"],
            pred["gap_index_value"],
            pred["true_lower_khz"],
            pred["pred_lower_khz"],
            pred["true_upper_khz"],
            pred["pred_upper_khz"],
            pred["true_width_khz"],
            pred_width,
        ]
    )
    header = (
        "sample_index,gap_index_value,true_lower_khz,pred_lower_khz,"
        "true_upper_khz,pred_upper_khz,true_width_khz,pred_width_khz"
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10g")


def load_metadata(run_dir: Path) -> dict[str, object]:
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_one(
    *,
    name: str,
    run_dir: Path,
    checkpoint_path: Path,
    indexed,
    train_sel: np.ndarray,
    test_sel: np.ndarray,
    feature_indices: np.ndarray,
    node_dim: int,
    edge_dim: int,
    graph_dim: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
) -> dict[str, np.ndarray]:
    scaler = build_scaler(indexed.gap_targets[train_sel, :2])
    model = GapBoundsRegressor(
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_dim=graph_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    test_dataset = CachedGapGroupDataset(indexed, test_sel, feature_indices=feature_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_gap_groups,
        persistent_workers=num_workers > 0,
    )
    pred = predict_regressor(model, test_loader, scaler, device, amp_enabled, amp_dtype)
    save_prediction_csv(run_dir / "test_predictions.csv", pred)
    print(
        f"{name}: n={len(pred['sample_index'])} "
        f"lower_mae={mae(pred['true_lower_khz'], pred['pred_lower_khz']):.6f} "
        f"upper_mae={mae(pred['true_upper_khz'], pred['pred_upper_khz']):.6f} "
        f"lower_r2={r2_score(pred['true_lower_khz'], pred['pred_lower_khz']):.6f} "
        f"upper_r2={r2_score(pred['true_upper_khz'], pred['pred_upper_khz']):.6f}",
        flush=True,
    )
    return pred


def plot_panel(ax, true: np.ndarray, pred: np.ndarray, label: str, letter: str, color: str) -> None:
    if true.size > 25000:
        rng = np.random.default_rng(20260517)
        idx = rng.choice(true.size, size=25000, replace=False)
        true_plot = true[idx]
        pred_plot = pred[idx]
    else:
        true_plot = true
        pred_plot = pred

    ax.scatter(true_plot, pred_plot, s=5, alpha=0.18, color=color, linewidths=0)
    lo = float(min(np.min(true), np.min(pred)))
    hi = float(max(np.max(true), np.max(pred)))
    pad = max((hi - lo) * 0.04, 0.1)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="#202832", lw=1.2, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dfe6ee", linewidth=0.7, alpha=0.85)
    ax.set_xlabel("True frequency (kHz)")
    ax.set_ylabel("Predicted frequency (kHz)")
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, va="top", ha="left", fontsize=16, fontweight="bold")
    ax.text(
        0.05,
        0.12,
        f"{label}\nMAE={mae(true, pred):.3f} kHz\n$R^2$={r2_score(true, pred):.4f}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#d9e0e8", "alpha": 0.9},
    )


def make_plot(gap2: dict[str, np.ndarray], gap3: dict[str, np.ndarray], output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
            "font.size": 11,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 7.4), constrained_layout=True)
    plot_panel(axes[0, 0], gap2["true_lower_khz"], gap2["pred_lower_khz"], "gap index = 2, lower edge", "a", "#4C78A8")
    plot_panel(axes[0, 1], gap2["true_upper_khz"], gap2["pred_upper_khz"], "gap index = 2, upper edge", "b", "#4C78A8")
    plot_panel(axes[1, 0], gap3["true_lower_khz"], gap3["pred_lower_khz"], "gap index = 3, lower edge", "c", "#4F8F62")
    plot_panel(axes[1, 1], gap3["true_upper_khz"], gap3["pred_upper_khz"], "gap index = 3, upper edge", "d", "#4F8F62")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gap2/gap3 test predictions and predicted-vs-true plots.")
    parser.add_argument("--gap2-run-dir", type=Path, required=True)
    parser.add_argument("--gap3-run-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    gap2_meta = load_metadata(args.gap2_run_dir)
    gap3_meta = load_metadata(args.gap3_run_dir)
    if gap2_meta["preprocessed_roots"] != gap3_meta["preprocessed_roots"]:
        raise ValueError("gap2 and gap3 runs used different preprocessed roots.")
    if gap2_meta["graph_feature_mode"] != gap3_meta["graph_feature_mode"]:
        raise ValueError("gap2 and gap3 runs used different graph feature modes.")

    set_seed(int(gap2_meta["seed"]))
    cache_roots = [Path(p) for p in gap2_meta["preprocessed_roots"]]
    major_gap_indices = tuple(int(v) for v in gap2_meta["major_gap_indices"])
    indexed = build_index_with_gap_groups(cache_roots=cache_roots, max_batches=0, max_samples=0, major_gap_indices=major_gap_indices)
    train_indices, _, test_indices = make_split_masks(indexed.sample_ids)
    feature_indices = get_graph_feature_indices(str(gap2_meta["graph_feature_mode"]))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled, amp_dtype, _, _ = resolve_amp_config(device, str(gap2_meta.get("amp_mode", "bf16")))

    sample_dataset = CachedGapGroupDataset(indexed, train_indices, feature_indices=feature_indices)
    sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_gap_groups)
    sample_batch = next(iter(sample_loader))
    node_dim = int(sample_batch["node_x"].shape[1])
    edge_dim = int(sample_batch["edge_attr"].shape[1])
    graph_dim = int(sample_batch["graph_x"].shape[1])

    gap2_train = train_indices & (indexed.group_ids == 0)
    gap2_test = test_indices & (indexed.group_ids == 0)
    gap3_train = train_indices & (indexed.group_ids == 1)
    gap3_test = test_indices & (indexed.group_ids == 1)

    gap2 = infer_one(
        name="gap2_expert",
        run_dir=args.gap2_run_dir,
        checkpoint_path=args.gap2_run_dir / "gap2_expert_best.pt",
        indexed=indexed,
        train_sel=gap2_train,
        test_sel=gap2_test,
        feature_indices=feature_indices,
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_dim=graph_dim,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        batch_size=int(gap2_meta["batch_size"]),
        num_workers=int(gap2_meta["num_workers"]),
        hidden_dim=int(gap2_meta["hidden_dim"]),
        layers=int(gap2_meta["layers"]),
        dropout=float(gap2_meta["dropout"]),
    )
    gap3 = infer_one(
        name="gap3_expert",
        run_dir=args.gap3_run_dir,
        checkpoint_path=args.gap3_run_dir / "gap3_expert_best.pt",
        indexed=indexed,
        train_sel=gap3_train,
        test_sel=gap3_test,
        feature_indices=feature_indices,
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_dim=graph_dim,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        batch_size=int(gap3_meta["batch_size"]),
        num_workers=int(gap3_meta["num_workers"]),
        hidden_dim=int(gap3_meta["hidden_dim"]),
        layers=int(gap3_meta["layers"]),
        dropout=float(gap3_meta["dropout"]),
    )
    make_plot(gap2, gap3, args.output_pdf, args.output_png)
    print(f"wrote {args.output_pdf}", flush=True)
    print(f"wrote {args.output_png}", flush=True)


if __name__ == "__main__":
    main()
