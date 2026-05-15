from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_first_band_gap_multitask import (
    CachedMultiTaskDataset,
    GatedEdgeMessageLayer,
    AttentiveGraphPool,
    GRAPH_FEATURE_NAMES,
    MultiTaskDataset,
    ResidualMLPBlock,
    Scaler,
    TaskHead,
    build_cosine_warmup_scheduler,
    build_index,
    build_index_from_preprocessed,
    build_scaler,
    collate_graphs,
    compute_cls_metrics,
    get_graph_feature_indices,
    make_split_masks,
    move_batch_tensor,
    resolve_amp_config,
    scatter_max,
    scatter_mean,
    set_seed,
)


class GraphBackbone(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [GatedEdgeMessageLayer(hidden_dim=hidden_dim, edge_dim=hidden_dim, dropout=dropout) for _ in range(layers)]
        )
        self.layer_mix_logits = nn.Parameter(torch.zeros(layers + 1, dtype=torch.float32))
        self.attn_pool = AttentiveGraphPool(hidden_dim=hidden_dim)
        fusion_dim = hidden_dim * 9
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.trunk_refine = ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout, expansion=2)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_x: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        node_h = self.node_encoder(node_x)
        edge_h = self.edge_encoder(edge_attr)
        states = [node_h]
        for layer in self.layers:
            node_h = layer(node_h, edge_index, edge_h)
            states.append(node_h)
        batch_size = graph_x.shape[0]

        weights = torch.softmax(self.layer_mix_logits, dim=0)
        mixed_node = torch.zeros_like(states[0])
        for idx, state in enumerate(states):
            mixed_node = mixed_node + weights[idx] * state

        node_h_fp32 = node_h.float()
        mixed_node_fp32 = mixed_node.float()

        final_mean = scatter_mean(node_h_fp32, node_batch, batch_size)
        final_var = scatter_mean(node_h_fp32 * node_h_fp32, node_batch, batch_size) - final_mean * final_mean
        final_std = torch.sqrt(final_var.clamp_min(1e-8))
        final_max = scatter_max(node_h_fp32, node_batch, batch_size)
        final_attn = self.attn_pool(node_h, node_batch, batch_size)

        mixed_mean = scatter_mean(mixed_node_fp32, node_batch, batch_size)
        mixed_var = scatter_mean(mixed_node_fp32 * mixed_node_fp32, node_batch, batch_size) - mixed_mean * mixed_mean
        mixed_std = torch.sqrt(mixed_var.clamp_min(1e-8))
        mixed_max = scatter_max(mixed_node_fp32, node_batch, batch_size)
        mixed_attn = self.attn_pool(mixed_node, node_batch, batch_size)

        graph_feat = self.graph_encoder(graph_x)
        graph_embedding = torch.cat(
            [final_mean, final_std, final_max, final_attn, mixed_mean, mixed_std, mixed_max, mixed_attn, graph_feat],
            dim=1,
        ).to(node_h.dtype)
        return self.trunk_refine(self.trunk(graph_embedding))


class GapClassifierGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = GraphBackbone(
            node_dim=node_dim,
            edge_dim=edge_dim,
            graph_dim=graph_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        self.cls_head = TaskHead(hidden_dim=hidden_dim, output_dim=1, dropout=dropout)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_x: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        trunk = self.backbone(node_x, edge_index, edge_attr, graph_x, node_batch)
        return self.cls_head(trunk).squeeze(1).clamp(-30.0, 30.0)


class GapRegressorGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = GraphBackbone(
            node_dim=node_dim,
            edge_dim=edge_dim,
            graph_dim=graph_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        self.reg_head = TaskHead(hidden_dim=hidden_dim, output_dim=2, dropout=dropout)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_x: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        trunk = self.backbone(node_x, edge_index, edge_attr, graph_x, node_batch)
        return self.reg_head(trunk)


@dataclass
class ThresholdSelection:
    threshold: float
    balanced_accuracy: float
    specificity: float
    recall: float
    false_positive: int
    false_negative: int


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> ThresholdSelection:
    thresholds = np.linspace(0.5, 0.995, 200, dtype=np.float64)
    best: ThresholdSelection | None = None
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        recall = tp / max(int(np.sum(y_true == 1)), 1)
        specificity = tn / max(int(np.sum(y_true == 0)), 1)
        balanced_accuracy = 0.5 * (recall + specificity)
        item = ThresholdSelection(
            threshold=float(threshold),
            balanced_accuracy=float(balanced_accuracy),
            specificity=float(specificity),
            recall=float(recall),
            false_positive=fp,
            false_negative=fn,
        )
        if best is None:
            best = item
            continue
        candidate_key = (item.balanced_accuracy, item.specificity, item.recall, item.threshold)
        best_key = (best.balanced_accuracy, best.specificity, best.recall, best.threshold)
        if candidate_key > best_key:
            best = item
    if best is None:
        raise RuntimeError("Failed to choose classification threshold.")
    return best


def compute_classifier_loss(logits: torch.Tensor, has_gap: torch.Tensor, cls_weight: tuple[float, float], device: torch.device) -> torch.Tensor:
    neg_weight, pos_weight = cls_weight
    sample_weights = torch.where(
        has_gap > 0.5,
        torch.tensor(pos_weight, device=device),
        torch.tensor(neg_weight, device=device),
    )
    return F.binary_cross_entropy_with_logits(logits, has_gap, weight=sample_weights)


def evaluate_classifier(
    model: GapClassifierGNN,
    loader: DataLoader,
    cls_weight: tuple[float, float],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []
    sample_id_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
            has_gap = move_batch_tensor(batch, "has_gap", device, non_blocking=amp_enabled)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                loss = compute_classifier_loss(logits, has_gap, cls_weight, device)

            probs = torch.sigmoid(logits).float().cpu().numpy()
            y_true_list.append(has_gap.float().cpu().numpy())
            y_prob_list.append(probs)
            sample_id_list.append(batch["sample_index"].numpy())
            loss_sum += float(loss.item()) * has_gap.shape[0]
            sample_count += int(has_gap.shape[0])

    y_true = np.concatenate(y_true_list).astype(np.int64)
    y_prob = np.concatenate(y_prob_list)
    sample_ids = np.concatenate(sample_id_list).astype(np.int64)
    cls_metrics = compute_cls_metrics(y_true, y_prob)
    metrics = {
        "loss": loss_sum / max(sample_count, 1),
        **cls_metrics,
    }
    return metrics, y_true, y_prob, sample_ids


def evaluate_regressor(
    model: GapRegressorGNN,
    loader: DataLoader,
    scaler: Scaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    scaler = scaler.to(device)
    loss_sum = 0.0
    sample_count = 0
    lower_abs = 0.0
    width_abs = 0.0
    upper_abs = 0.0

    with torch.no_grad():
        for batch in loader:
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
            gap_target = move_batch_tensor(batch, "gap_target", device, non_blocking=amp_enabled)
            target_lw = torch.stack([gap_target[:, 0], gap_target[:, 2]], dim=1)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred_norm = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                target_norm = scaler.encode(target_lw)
                loss = F.smooth_l1_loss(pred_norm, target_norm)

            pred_lw = scaler.decode(pred_norm).float()
            lower_pred = pred_lw[:, 0]
            width_pred = pred_lw[:, 1].clamp_min(0.0)
            upper_pred = lower_pred + width_pred
            lower_true = gap_target[:, 0].float()
            width_true = gap_target[:, 2].float()
            upper_true = gap_target[:, 1].float()

            lower_abs += float(torch.abs(lower_pred - lower_true).sum().item())
            width_abs += float(torch.abs(width_pred - width_true).sum().item())
            upper_abs += float(torch.abs(upper_pred - upper_true).sum().item())
            loss_sum += float(loss.item()) * target_lw.shape[0]
            sample_count += int(target_lw.shape[0])

    return {
        "loss": loss_sum / max(sample_count, 1),
        "lower_mae_khz": lower_abs / max(sample_count, 1),
        "width_mae_khz": width_abs / max(sample_count, 1),
        "upper_from_lw_mae_khz": upper_abs / max(sample_count, 1),
    }


def predict_regressor(
    model: GapRegressorGNN,
    loader: DataLoader,
    scaler: Scaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    model.eval()
    scaler = scaler.to(device)
    sample_ids: list[np.ndarray] = []
    lower_pred: list[np.ndarray] = []
    width_pred: list[np.ndarray] = []
    upper_pred: list[np.ndarray] = []
    lower_true: list[np.ndarray] = []
    width_true: list[np.ndarray] = []
    upper_true: list[np.ndarray] = []
    has_gap_true: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
            gap_target = batch["gap_target"].numpy().astype(np.float64)
            has_gap = batch["has_gap"].numpy().astype(np.int64)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred_norm = model(node_x, edge_index, edge_attr, graph_x, node_batch)

            pred_lw = scaler.decode(pred_norm).float().cpu().numpy().astype(np.float64)
            lower_raw = pred_lw[:, 0]
            width_raw = np.maximum(pred_lw[:, 1], 0.0)
            upper_raw = lower_raw + width_raw

            sample_ids.append(batch["sample_index"].numpy().astype(np.int64))
            lower_pred.append(lower_raw)
            width_pred.append(width_raw)
            upper_pred.append(upper_raw)
            lower_true.append(gap_target[:, 0])
            upper_true.append(gap_target[:, 1])
            width_true.append(gap_target[:, 2])
            has_gap_true.append(has_gap)

    return {
        "sample_index": np.concatenate(sample_ids),
        "lower_raw_khz": np.concatenate(lower_pred),
        "upper_raw_khz": np.concatenate(upper_pred),
        "width_raw_khz": np.concatenate(width_pred),
        "lower_true_khz": np.concatenate(lower_true),
        "upper_true_khz": np.concatenate(upper_true),
        "width_true_khz": np.concatenate(width_true),
        "has_gap_true": np.concatenate(has_gap_true),
    }


def merge_predictions(
    classifier_probs: np.ndarray,
    reg_predictions: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, np.ndarray]:
    predicted_has_gap = (classifier_probs >= threshold).astype(np.int64)
    lower_final = np.where(predicted_has_gap == 1, reg_predictions["lower_raw_khz"], 0.0)
    width_final = np.where(predicted_has_gap == 1, np.maximum(reg_predictions["width_raw_khz"], 0.0), 0.0)
    upper_final = np.where(predicted_has_gap == 1, lower_final + width_final, 0.0)
    return {
        **reg_predictions,
        "has_gap_prob": classifier_probs,
        "predicted_has_gap": predicted_has_gap,
        "threshold": np.full_like(classifier_probs, threshold, dtype=np.float64),
        "lower_pred_khz": lower_final,
        "upper_pred_khz": upper_final,
        "width_pred_khz": width_final,
    }


def compute_final_metrics(predictions: dict[str, np.ndarray]) -> dict[str, float]:
    y_true = predictions["has_gap_true"].astype(np.int64)
    y_prob = predictions["has_gap_prob"].astype(np.float64)
    y_pred = predictions["predicted_has_gap"].astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(len(y_true), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    prob_metrics = compute_cls_metrics(y_true, y_prob)

    cls_metrics = {
        "cls_accuracy_at_threshold": float(accuracy),
        "cls_balanced_accuracy_at_threshold": float(0.5 * (recall + specificity)),
        "cls_precision_at_threshold": float(precision),
        "cls_recall_at_threshold": float(recall),
        "cls_specificity_at_threshold": float(specificity),
        "cls_f1_at_threshold": float(f1),
        "cls_roc_auc": float(prob_metrics["roc_auc"]),
        "false_positive": fp,
        "false_negative": fn,
    }

    metrics: dict[str, float] = cls_metrics
    for key in ["lower", "upper", "width"]:
        truth = predictions[f"{key}_true_khz"]
        pred = predictions[f"{key}_pred_khz"]
        err = pred - truth
        metrics[f"{key}_mae_all_khz"] = float(np.mean(np.abs(err)))
        metrics[f"{key}_rmse_all_khz"] = float(np.sqrt(np.mean(err**2)))
        metrics[f"{key}_bias_all_khz"] = float(np.mean(err))

    positive_mask = y_true == 1
    for key in ["lower", "upper", "width"]:
        truth = predictions[f"{key}_true_khz"][positive_mask]
        pred = predictions[f"{key}_pred_khz"][positive_mask]
        err = pred - truth
        metrics[f"{key}_mae_positive_khz"] = float(np.mean(np.abs(err)))
        metrics[f"{key}_rmse_positive_khz"] = float(np.sqrt(np.mean(err**2)))
        metrics[f"{key}_bias_positive_khz"] = float(np.mean(err))

    raw_upper_err = predictions["upper_raw_khz"][positive_mask] - predictions["upper_true_khz"][positive_mask]
    metrics["upper_raw_mae_positive_khz"] = float(np.mean(np.abs(raw_upper_err)))
    return metrics


def write_final_predictions(predictions: dict[str, np.ndarray], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "has_gap_true",
        "has_gap_prob",
        "predicted_has_gap",
        "threshold",
        "lower_true_khz",
        "upper_true_khz",
        "width_true_khz",
        "lower_raw_khz",
        "upper_raw_khz",
        "width_raw_khz",
        "lower_pred_khz",
        "upper_pred_khz",
        "width_pred_khz",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        size = predictions["sample_index"].shape[0]
        for idx in range(size):
            writer.writerow([predictions[key][idx] for key in fieldnames])


def build_datasets(
    indexed: Any,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    cache_size: int,
    feature_indices: np.ndarray,
    use_preprocessed: bool,
) -> dict[str, Dataset]:
    dataset_cls = CachedMultiTaskDataset if use_preprocessed else MultiTaskDataset
    positive_train = train_indices[indexed.has_gap[train_indices] > 0.5]
    positive_val = val_indices[indexed.has_gap[val_indices] > 0.5]
    positive_test = test_indices[indexed.has_gap[test_indices] > 0.5]
    return {
        "train_all": dataset_cls(indexed, train_indices, cache_size=cache_size, feature_indices=feature_indices),
        "val_all": dataset_cls(indexed, val_indices, cache_size=cache_size, feature_indices=feature_indices),
        "test_all": dataset_cls(indexed, test_indices, cache_size=cache_size, feature_indices=feature_indices),
        "train_pos": dataset_cls(indexed, positive_train, cache_size=cache_size, feature_indices=feature_indices),
        "val_pos": dataset_cls(indexed, positive_val, cache_size=cache_size, feature_indices=feature_indices),
        "test_pos": dataset_cls(indexed, positive_test, cache_size=cache_size, feature_indices=feature_indices),
        "positive_train_indices": positive_train,
        "positive_val_indices": positive_val,
        "positive_test_indices": positive_test,
    }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": collate_graphs,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)


def train_classifier(
    model: GapClassifierGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cls_weight: tuple[float, float],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    use_grad_scaler: bool,
    grad_clip: float,
    epochs: int,
    run_dir: Path,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], dict[str, float], np.ndarray, np.ndarray]:
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_y_true: np.ndarray | None = None
    best_y_prob: np.ndarray | None = None
    best_key: tuple[float, float, float] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        last_lr = optimizer.param_groups[0]["lr"]

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
            has_gap = move_batch_tensor(batch, "has_gap", device, non_blocking=amp_enabled)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                loss = compute_classifier_loss(logits, has_gap, cls_weight, device)

            if use_grad_scaler:
                previous_scale = grad_scaler.get_scale()
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                if grad_scaler.get_scale() >= previous_scale:
                    scheduler.step()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

            running_loss += float(loss.item()) * has_gap.shape[0]
            seen += int(has_gap.shape[0])
            last_lr = optimizer.param_groups[0]["lr"]

        train_loss = running_loss / max(seen, 1)
        val_metrics, y_true, y_prob, _ = evaluate_classifier(model, val_loader, cls_weight, device, amp_enabled, amp_dtype)
        epoch_summary = {"epoch": epoch, "train_loss": train_loss, "lr": last_lr, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_summary)
        print(json.dumps({"stage": "classifier", **epoch_summary}, ensure_ascii=False), flush=True)

        candidate_key = (
            float(val_metrics["balanced_accuracy"]),
            float(val_metrics["roc_auc"]),
            -float(val_metrics["loss"]),
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_metrics = val_metrics
            best_y_true = y_true.copy()
            best_y_prob = y_prob.copy()
            torch.save(
                {
                    "model_state": best_state,
                    "val_metrics": best_metrics,
                    "epoch": epoch,
                },
                run_dir / "classifier_best.pt",
            )

    if best_state is None or best_metrics is None or best_y_true is None or best_y_prob is None:
        raise RuntimeError("Classifier training did not produce a checkpoint.")
    return history, best_state, best_metrics, best_y_true, best_y_prob


def train_regressor(
    model: GapRegressorGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: Scaler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    use_grad_scaler: bool,
    grad_clip: float,
    epochs: int,
    run_dir: Path,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], dict[str, float]]:
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    scaler_device = scaler.to(device)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        last_lr = optimizer.param_groups[0]["lr"]

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
            gap_target = move_batch_tensor(batch, "gap_target", device, non_blocking=amp_enabled)
            target_lw = torch.stack([gap_target[:, 0], gap_target[:, 2]], dim=1)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred_norm = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                target_norm = scaler_device.encode(target_lw)
                loss = F.smooth_l1_loss(pred_norm, target_norm)

            if use_grad_scaler:
                previous_scale = grad_scaler.get_scale()
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                if grad_scaler.get_scale() >= previous_scale:
                    scheduler.step()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

            running_loss += float(loss.item()) * target_lw.shape[0]
            seen += int(target_lw.shape[0])
            last_lr = optimizer.param_groups[0]["lr"]

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_regressor(model, val_loader, scaler, device, amp_enabled, amp_dtype)
        epoch_summary = {"epoch": epoch, "train_loss": train_loss, "lr": last_lr, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_summary)
        print(json.dumps({"stage": "regressor", **epoch_summary}, ensure_ascii=False), flush=True)

        if val_metrics["loss"] < best_loss:
            best_loss = float(val_metrics["loss"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state": best_state,
                    "scaler_mean": scaler.mean,
                    "scaler_std": scaler.std,
                    "val_metrics": best_metrics,
                    "epoch": epoch,
                },
                run_dir / "regressor_best.pt",
            )

    if best_state is None or best_metrics is None:
        raise RuntimeError("Regressor training did not produce a checkpoint.")
    return history, best_state, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage GNN for band-gap classification and gated regression.")
    parser.add_argument("--data-root", type=Path, action="append", dest="data_roots")
    parser.add_argument("--preprocessed-root", type=Path, action="append", dest="preprocessed_roots")
    parser.add_argument("--output-dir", type=Path, default=Path("gnn_two_stage_runs"))
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--classifier-epochs", type=int, default=4)
    parser.add_argument("--regressor-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--hidden-dim", type=int, default=224)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--graph-feature-mode", type=str, default="full_novel")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--min-lr-scale", type=float, default=0.10)
    parser.add_argument("--amp-mode", type=str, default="auto", choices=["auto", "fp16", "bf16", "off"])
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    data_roots = args.data_roots or [Path(r"E:\datasets"), Path(r"E:\datasets\NWW2")]
    preprocessed_roots = args.preprocessed_roots or [
        Path(r"E:\datasets_graph_cache_novel109_v1\source_000"),
        Path(r"E:\datasets_graph_cache_novel109_v1\source_001"),
    ]

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_enabled, amp_dtype, use_grad_scaler, resolved_amp_mode = resolve_amp_config(device, args.amp_mode)
    overall_start = time.time()

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_indices = get_graph_feature_indices(args.graph_feature_mode)
    selected_feature_names = [GRAPH_FEATURE_NAMES[int(i)] for i in feature_indices.tolist()]

    print("Building sample index...", flush=True)
    use_preprocessed = len(preprocessed_roots) > 0
    if use_preprocessed:
        indexed = build_index_from_preprocessed(preprocessed_roots, max_batches=args.max_batches, max_samples=args.max_samples)
    else:
        indexed = build_index(data_roots, max_batches=args.max_batches, max_samples=args.max_samples)
    train_mask, val_mask, test_mask = make_split_masks(indexed.sample_ids)
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)
    test_indices = np.flatnonzero(test_mask)

    datasets = build_datasets(
        indexed=indexed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        cache_size=args.cache_size,
        feature_indices=feature_indices,
        use_preprocessed=use_preprocessed,
    )

    train_all_loader = make_loader(datasets["train_all"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_all_loader = make_loader(datasets["val_all"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)
    test_all_loader = make_loader(datasets["test_all"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)
    train_pos_loader = make_loader(datasets["train_pos"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, device=device)
    val_pos_loader = make_loader(datasets["val_pos"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)
    test_pos_loader = make_loader(datasets["test_pos"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device)

    train_has_gap = indexed.has_gap[train_indices]
    pos_count = float(train_has_gap.sum())
    neg_count = float(len(train_has_gap) - pos_count)
    cls_weight = (
        min(math.sqrt(0.5 * len(train_has_gap) / max(neg_count, 1.0)), 16.0),
        min(math.sqrt(0.5 * len(train_has_gap) / max(pos_count, 1.0)), 2.0),
    )

    reg_targets_train = indexed.gap_targets[datasets["positive_train_indices"]][:, [0, 2]]
    reg_scaler = build_scaler(reg_targets_train)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_roots": [str(path) for path in data_roots],
        "preprocessed_roots": [str(path) for path in preprocessed_roots],
        "use_preprocessed": use_preprocessed,
        "device": str(device),
        "amp_mode": resolved_amp_mode,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
        "graph_feature_mode": args.graph_feature_mode,
        "graph_feature_dim": int(len(feature_indices)),
        "graph_feature_names": selected_feature_names,
        "train_samples": int(len(datasets["train_all"])),
        "val_samples": int(len(datasets["val_all"])),
        "test_samples": int(len(datasets["test_all"])),
        "train_positive_samples": int(len(datasets["train_pos"])),
        "val_positive_samples": int(len(datasets["val_pos"])),
        "test_positive_samples": int(len(datasets["test_pos"])),
        "train_gap_ratio": float(indexed.has_gap[train_indices].mean()),
        "val_gap_ratio": float(indexed.has_gap[val_indices].mean()),
        "test_gap_ratio": float(indexed.has_gap[test_indices].mean()),
        "cls_weight": cls_weight,
        "reg_scaler_mean": reg_scaler.mean.tolist(),
        "reg_scaler_std": reg_scaler.std.tolist(),
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "dropout": args.dropout,
        "num_workers": args.num_workers,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    classifier = GapClassifierGNN(
        node_dim=11,
        edge_dim=7,
        graph_dim=len(feature_indices),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    cls_optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cls_scheduler = build_cosine_warmup_scheduler(
        cls_optimizer,
        total_steps=max(args.classifier_epochs * len(train_all_loader), 1),
        warmup_fraction=args.warmup_fraction,
        min_lr_scale=args.min_lr_scale,
    )
    cls_history, cls_best_state, cls_best_metrics, cls_val_true, cls_val_prob = train_classifier(
        model=classifier,
        train_loader=train_all_loader,
        val_loader=val_all_loader,
        cls_weight=cls_weight,
        optimizer=cls_optimizer,
        scheduler=cls_scheduler,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        use_grad_scaler=use_grad_scaler,
        grad_clip=args.grad_clip,
        epochs=args.classifier_epochs,
        run_dir=run_dir,
    )
    classifier.load_state_dict(cls_best_state)
    threshold = choose_threshold(cls_val_true, cls_val_prob)
    cls_test_metrics, cls_test_true, cls_test_prob, cls_test_ids = evaluate_classifier(
        classifier, test_all_loader, cls_weight, device, amp_enabled, amp_dtype
    )

    regressor = GapRegressorGNN(
        node_dim=11,
        edge_dim=7,
        graph_dim=len(feature_indices),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    reg_optimizer = torch.optim.AdamW(regressor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    reg_scheduler = build_cosine_warmup_scheduler(
        reg_optimizer,
        total_steps=max(args.regressor_epochs * len(train_pos_loader), 1),
        warmup_fraction=args.warmup_fraction,
        min_lr_scale=args.min_lr_scale,
    )
    reg_history, reg_best_state, reg_best_metrics = train_regressor(
        model=regressor,
        train_loader=train_pos_loader,
        val_loader=val_pos_loader,
        scaler=reg_scaler,
        optimizer=reg_optimizer,
        scheduler=reg_scheduler,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        use_grad_scaler=use_grad_scaler,
        grad_clip=args.grad_clip,
        epochs=args.regressor_epochs,
        run_dir=run_dir,
    )
    regressor.load_state_dict(reg_best_state)

    test_reg_predictions = predict_regressor(regressor, test_all_loader, reg_scaler, device, amp_enabled, amp_dtype)
    if not np.array_equal(test_reg_predictions["sample_index"], cls_test_ids):
        raise RuntimeError("Sample index mismatch between classifier and regressor predictions.")
    final_predictions = merge_predictions(cls_test_prob, test_reg_predictions, threshold.threshold)
    final_metrics = compute_final_metrics(final_predictions)
    write_final_predictions(final_predictions, run_dir / "test_predictions.csv")

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - overall_start,
        "classifier": {
            "best_val_metrics": cls_best_metrics,
            "test_metrics": cls_test_metrics,
            "threshold_selection": threshold.__dict__,
            "history": cls_history,
        },
        "regressor": {
            "best_val_metrics": reg_best_metrics,
            "test_positive_metrics": evaluate_regressor(regressor, test_pos_loader, reg_scaler, device, amp_enabled, amp_dtype),
            "history": reg_history,
        },
        "final_test_metrics": final_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
