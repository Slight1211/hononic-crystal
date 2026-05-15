from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from train_first_band_gap_multitask import (
    GRAPH_FEATURE_NAMES,
    Scaler,
    TaskHead,
    TensorChunkCache,
    build_cosine_warmup_scheduler,
    build_scaler,
    discover_preprocessed_chunk_paths,
    get_graph_feature_indices,
    make_split_masks,
    move_batch_tensor,
    resolve_amp_config,
    set_seed,
)
from train_first_band_gap_multitask import collate_graphs as _unused_collate_graphs
from train_two_stage_gap_gnn import GraphBackbone


@dataclass
class CacheSourceInfo:
    scan_root: Path
    cache_root: Path
    data_roots: list[Path]


@dataclass
class IndexedGapGroupSamples:
    chunk_paths: list[str]
    chunk_ids: np.ndarray
    local_ids: np.ndarray
    sample_ids: np.ndarray
    gap_targets: np.ndarray
    gap_index_values: np.ndarray
    group_ids: np.ndarray
    duplicate_samples_skipped: int = 0
    no_gap_samples_skipped: int = 0
    sample_index_filtered_skipped: int = 0

    @property
    def size(self) -> int:
        return int(self.sample_ids.shape[0])


@dataclass
class ClassifierMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float


@dataclass
class RegressorMetrics:
    loss: float
    lower_mae_khz: float
    upper_mae_khz: float
    width_mae_khz: float
    lower_rmse_khz: float
    upper_rmse_khz: float
    width_rmse_khz: float
    lower_bias_khz: float
    upper_bias_khz: float
    width_bias_khz: float
    lower_r2: float
    upper_r2: float
    width_r2: float


class CachedGapGroupDataset(Dataset):
    def __init__(
        self,
        indexed: IndexedGapGroupSamples,
        selection: np.ndarray,
        cache_size: int = 8,
        feature_indices: np.ndarray | None = None,
    ) -> None:
        self.chunk_paths = indexed.chunk_paths
        self.chunk_ids = indexed.chunk_ids[selection]
        self.local_ids = indexed.local_ids[selection]
        self.sample_ids = indexed.sample_ids[selection]
        self.gap_targets = indexed.gap_targets[selection]
        self.gap_index_values = indexed.gap_index_values[selection]
        self.group_ids = indexed.group_ids[selection]
        self.feature_indices = feature_indices
        self.cache = TensorChunkCache(max_items=cache_size)

    def __len__(self) -> int:
        return int(self.sample_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        chunk_path = self.chunk_paths[int(self.chunk_ids[index])]
        chunk = self.cache.get(chunk_path)
        local_id = int(self.local_ids[index])
        node_start = int(chunk["node_ptr"][local_id].item())
        node_end = int(chunk["node_ptr"][local_id + 1].item())
        edge_start = int(chunk["edge_ptr"][local_id].item())
        edge_end = int(chunk["edge_ptr"][local_id + 1].item())
        graph_x = chunk["graph_x"][local_id]
        if self.feature_indices is not None:
            graph_x = graph_x[self.feature_indices]
        return {
            "node_x": chunk["node_x"][node_start:node_end],
            "edge_index": chunk["edge_index"][:, edge_start:edge_end],
            "edge_attr": chunk["edge_attr"][edge_start:edge_end],
            "graph_x": graph_x,
            "gap_target": self.gap_targets[index],
            "gap_index_value": int(self.gap_index_values[index]),
            "group_id": int(self.group_ids[index]),
            "sample_index": int(self.sample_ids[index]),
        }


def collate_gap_groups(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    node_x_list = []
    edge_index_list = []
    edge_attr_list = []
    graph_x_list = []
    gap_target_list = []
    group_id_list = []
    gap_index_value_list = []
    sample_ids = []
    batch_index_list = []
    node_offset = 0

    for graph_id, item in enumerate(batch):
        node_x = torch.as_tensor(item["node_x"])
        edge_index = torch.as_tensor(item["edge_index"]).long() + node_offset
        edge_attr = torch.as_tensor(item["edge_attr"])
        graph_x = torch.as_tensor(item["graph_x"])
        gap_target = torch.as_tensor(item["gap_target"])

        node_x_list.append(node_x)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        graph_x_list.append(graph_x)
        gap_target_list.append(gap_target)
        group_id_list.append(int(item["group_id"]))
        gap_index_value_list.append(int(item["gap_index_value"]))
        sample_ids.append(int(item["sample_index"]))
        batch_index_list.append(torch.full((node_x.shape[0],), graph_id, dtype=torch.long))
        node_offset += node_x.shape[0]

    return {
        "node_x": torch.cat(node_x_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "graph_x": torch.stack(graph_x_list, dim=0),
        "gap_target": torch.stack(gap_target_list, dim=0),
        "group_id": torch.tensor(group_id_list, dtype=torch.long),
        "gap_index_value": torch.tensor(gap_index_value_list, dtype=torch.long),
        "node_batch": torch.cat(batch_index_list, dim=0),
        "sample_index": torch.tensor(sample_ids, dtype=torch.long),
    }


class GapGroupClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        num_classes: int,
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
        self.head = TaskHead(hidden_dim=hidden_dim, output_dim=num_classes, dropout=dropout)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_x: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        trunk = self.backbone(node_x, edge_index, edge_attr, graph_x, node_batch)
        return self.head(trunk)


class GapBoundsRegressor(nn.Module):
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
        self.head = TaskHead(hidden_dim=hidden_dim, output_dim=2, dropout=dropout)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_x: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        trunk = self.backbone(node_x, edge_index, edge_attr, graph_x, node_batch)
        return self.head(trunk)


def load_cache_source_infos(cache_roots: list[Path]) -> list[CacheSourceInfo]:
    infos: list[CacheSourceInfo] = []
    for cache_root in cache_roots:
        scan_root = cache_root.expanduser().resolve()
        manifest_path = scan_root / "cache_manifest.json"
        resolved_cache_root = scan_root
        if not manifest_path.exists():
            for parent in scan_root.parents:
                candidate = parent / "cache_manifest.json"
                if candidate.exists():
                    manifest_path = candidate
                    resolved_cache_root = parent.resolve()
                    break
            else:
                raise FileNotFoundError(
                    f"Missing cache manifest for preprocessed root: {scan_root}"
                )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_roots = [Path(p) for p in manifest["data_roots"]]
        infos.append(
            CacheSourceInfo(
                scan_root=scan_root,
                cache_root=resolved_cache_root,
                data_roots=data_roots,
            )
        )
    return infos


def map_gap_index_to_group(gap_index: int, major_gap_indices: tuple[int, int]) -> int:
    if gap_index == major_gap_indices[0]:
        return 0
    if gap_index == major_gap_indices[1]:
        return 1
    return 2


def map_preprocessed_chunk_to_original(chunk_path: Path, infos: list[CacheSourceInfo]) -> Path:
    resolved = chunk_path.resolve()
    for info in infos:
        try:
            rel = resolved.relative_to(info.cache_root)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            break
        source_token = rel.parts[0]
        if not source_token.startswith("source_"):
            break
        source_idx = int(source_token.split("_")[1])
        if source_idx >= len(info.data_roots):
            raise IndexError(f"Cache source index {source_idx} out of range for {chunk_path}")
        target_root = info.data_roots[source_idx]
        orig_tail = list(rel.parts[1:])
        orig_tail[-1] = orig_tail[-1].replace("graph_chunk_", "pilot_chunk_").replace(".pt", ".pkl")
        return target_root.joinpath(*orig_tail)
    raise FileNotFoundError(f"Could not map cached chunk to original pkl path: {chunk_path}")


def build_index_with_gap_groups(
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
    no_gap_samples_skipped = 0
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
            raise RuntimeError(
                f"Chunk/sample count mismatch between cache and original data: {chunk_path} vs {original_path}"
            )
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
            if float(has_gap_chunk[local_id]) <= 0.5:
                no_gap_samples_skipped += 1
                continue
            gap_index = int(record["gap_index"]) if record["gap_index"] is not None else -1
            if gap_index < 0:
                no_gap_samples_skipped += 1
                continue
            chunk_ids.append(chunk_id)
            local_ids.append(local_id)
            sample_ids.append(cache_sample_index)
            gap_targets.append(gap_target_chunk[local_id].astype(np.float32).tolist())
            gap_index_values.append(gap_index)
            group_ids.append(map_gap_index_to_group(gap_index, major_gap_indices))
        if max_samples > 0 and len(sample_ids) >= max_samples:
            break

    if not sample_ids:
        raise RuntimeError("No positive gap samples were found while building index.")

    return IndexedGapGroupSamples(
        chunk_paths=[str(p) for p in chunk_paths],
        chunk_ids=np.asarray(chunk_ids, dtype=np.int32),
        local_ids=np.asarray(local_ids, dtype=np.int32),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        gap_targets=np.asarray(gap_targets, dtype=np.float32),
        gap_index_values=np.asarray(gap_index_values, dtype=np.int64),
        group_ids=np.asarray(group_ids, dtype=np.int64),
        duplicate_samples_skipped=duplicate_samples_skipped,
        no_gap_samples_skipped=no_gap_samples_skipped,
        sample_index_filtered_skipped=sample_index_filtered_skipped,
    )


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_classifier(
    model: GapGroupClassifier,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[ClassifierMetrics, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    sample_id_list: list[np.ndarray] = []

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
                loss = F.cross_entropy(logits, group_id, weight=class_weights)

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_list.append(pred)
            y_true_list.append(group_id.cpu().numpy())
            sample_id_list.append(batch["sample_index"].numpy())
            loss_sum += float(loss.item()) * int(group_id.shape[0])
            sample_count += int(group_id.shape[0])

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    sample_ids = np.concatenate(sample_id_list).astype(np.int64)
    metrics = compute_multiclass_metrics(y_true, y_pred)
    return (
        ClassifierMetrics(
            loss=loss_sum / max(sample_count, 1),
            accuracy=metrics["accuracy"],
            balanced_accuracy=metrics["balanced_accuracy"],
            macro_f1=metrics["macro_f1"],
        ),
        y_true,
        y_pred,
        sample_ids,
    )


def evaluate_regressor(
    model: GapBoundsRegressor,
    loader: DataLoader,
    scaler: Scaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> RegressorMetrics:
    model.eval()
    scaler = scaler.to(device)
    loss_sum = 0.0
    sample_count = 0
    lower_errs: list[np.ndarray] = []
    upper_errs: list[np.ndarray] = []
    width_errs: list[np.ndarray] = []
    lower_true_values: list[np.ndarray] = []
    upper_true_values: list[np.ndarray] = []
    width_true_values: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
            gap_target = move_batch_tensor(batch, "gap_target", device, non_blocking=amp_enabled)
            target_lu = gap_target[:, :2]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred_norm = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                target_norm = scaler.encode(target_lu)
                loss = F.smooth_l1_loss(pred_norm, target_norm)

            pred_lu = scaler.decode(pred_norm).float().cpu().numpy().astype(np.float64)
            true_lu = target_lu.float().cpu().numpy().astype(np.float64)
            lower_err = pred_lu[:, 0] - true_lu[:, 0]
            upper_err = pred_lu[:, 1] - true_lu[:, 1]
            true_width = true_lu[:, 1] - true_lu[:, 0]
            width_err = (pred_lu[:, 1] - pred_lu[:, 0]) - true_width
            lower_errs.append(lower_err)
            upper_errs.append(upper_err)
            width_errs.append(width_err)
            lower_true_values.append(true_lu[:, 0])
            upper_true_values.append(true_lu[:, 1])
            width_true_values.append(true_width)
            loss_sum += float(loss.item()) * int(target_lu.shape[0])
            sample_count += int(target_lu.shape[0])

    lower = np.concatenate(lower_errs)
    upper = np.concatenate(upper_errs)
    width = np.concatenate(width_errs)
    lower_true = np.concatenate(lower_true_values)
    upper_true = np.concatenate(upper_true_values)
    width_true = np.concatenate(width_true_values)

    def r2_score_from_error(error: np.ndarray, true: np.ndarray) -> float:
        ss_res = float(np.sum(error ** 2))
        centered = true - float(np.mean(true))
        ss_tot = float(np.sum(centered ** 2))
        if ss_tot <= 0.0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    return RegressorMetrics(
        loss=loss_sum / max(sample_count, 1),
        lower_mae_khz=float(np.mean(np.abs(lower))),
        upper_mae_khz=float(np.mean(np.abs(upper))),
        width_mae_khz=float(np.mean(np.abs(width))),
        lower_rmse_khz=float(np.sqrt(np.mean(lower ** 2))),
        upper_rmse_khz=float(np.sqrt(np.mean(upper ** 2))),
        width_rmse_khz=float(np.sqrt(np.mean(width ** 2))),
        lower_bias_khz=float(np.mean(lower)),
        upper_bias_khz=float(np.mean(upper)),
        width_bias_khz=float(np.mean(width)),
        lower_r2=r2_score_from_error(lower, lower_true),
        upper_r2=r2_score_from_error(upper, upper_true),
        width_r2=r2_score_from_error(width, width_true),
    )


def predict_regressor(
    model: GapBoundsRegressor,
    loader: DataLoader,
    scaler: Scaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    model.eval()
    scaler = scaler.to(device)
    sample_ids: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    true_list: list[np.ndarray] = []
    group_list: list[np.ndarray] = []
    gap_index_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred_norm = model(node_x, edge_index, edge_attr, graph_x, node_batch)

            pred_lu = scaler.decode(pred_norm).float().cpu().numpy().astype(np.float64)
            sample_ids.append(batch["sample_index"].numpy().astype(np.int64))
            pred_list.append(pred_lu)
            true_list.append(batch["gap_target"].numpy().astype(np.float64))
            group_list.append(batch["group_id"].numpy().astype(np.int64))
            gap_index_list.append(batch["gap_index_value"].numpy().astype(np.int64))

    pred_all = np.concatenate(pred_list, axis=0)
    true_all = np.concatenate(true_list, axis=0)
    return {
        "sample_index": np.concatenate(sample_ids),
        "pred_lower_khz": pred_all[:, 0],
        "pred_upper_khz": pred_all[:, 1],
        "true_lower_khz": true_all[:, 0],
        "true_upper_khz": true_all[:, 1],
        "true_width_khz": true_all[:, 2],
        "group_id": np.concatenate(group_list),
        "gap_index_value": np.concatenate(gap_index_list),
    }


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    use_grad_scaler: bool,
    class_weights: torch.Tensor,
    node_dim: int,
    edge_dim: int,
    graph_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    warmup_fraction: float,
    min_lr_scale: float,
    output_dir: Path,
) -> tuple[GapGroupClassifier, dict[str, object]]:
    model = GapGroupClassifier(node_dim, edge_dim, graph_dim, hidden_dim, layers, dropout, num_classes=3).to(device)
    class_weights = class_weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_cosine_warmup_scheduler(
        optimizer=optimizer,
        total_steps=max(len(train_loader) * epochs, 1),
        warmup_fraction=warmup_fraction,
        min_lr_scale=min_lr_scale,
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    best_state = None
    best_val_bal_acc = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
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
                loss = F.cross_entropy(logits, group_id, weight=class_weights)
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

        val_metrics, _, _, _ = evaluate_classifier(model, val_loader, class_weights, device, amp_enabled, amp_dtype)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_count, 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_balanced_accuracy": val_metrics.balanced_accuracy,
                "val_macro_f1": val_metrics.macro_f1,
            }
        )
        print(
            f"[classifier] epoch {epoch}/{epochs} "
            f"train_loss={history[-1]['train_loss']:.6f} "
            f"val_bal_acc={val_metrics.balanced_accuracy:.6f} "
            f"val_macro_f1={val_metrics.macro_f1:.6f}",
            flush=True,
        )
        if val_metrics.balanced_accuracy > best_val_bal_acc:
            best_val_bal_acc = val_metrics.balanced_accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Classifier best state missing.")
    model.load_state_dict(best_state)
    model_path = output_dir / "gap_index_classifier_best.pt"
    torch.save(best_state, model_path)
    final_val_metrics, _, _, _ = evaluate_classifier(model, val_loader, class_weights, device, amp_enabled, amp_dtype)
    return model, {
        "history": history,
        "best_val_metrics": {
            "loss": final_val_metrics.loss,
            "accuracy": final_val_metrics.accuracy,
            "balanced_accuracy": final_val_metrics.balanced_accuracy,
            "macro_f1": final_val_metrics.macro_f1,
        },
        "model_path": str(model_path),
    }


def train_regressor(
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_scaler: Scaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    use_grad_scaler: bool,
    node_dim: int,
    edge_dim: int,
    graph_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    warmup_fraction: float,
    min_lr_scale: float,
    output_dir: Path,
) -> tuple[GapBoundsRegressor, dict[str, object]]:
    model = GapBoundsRegressor(node_dim, edge_dim, graph_dim, hidden_dim, layers, dropout).to(device)
    scaler = target_scaler.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_cosine_warmup_scheduler(
        optimizer=optimizer,
        total_steps=max(len(train_loader) * epochs, 1),
        warmup_fraction=warmup_fraction,
        min_lr_scale=min_lr_scale,
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    best_state = None
    best_val_mae = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
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
                loss = F.smooth_l1_loss(pred_norm, target_norm)
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

        val_metrics = evaluate_regressor(model, val_loader, target_scaler, device, amp_enabled, amp_dtype)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_count, 1),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_loss": val_metrics.loss,
                "val_lower_mae_khz": val_metrics.lower_mae_khz,
                "val_upper_mae_khz": val_metrics.upper_mae_khz,
                "val_width_mae_khz": val_metrics.width_mae_khz,
                "val_lower_r2": val_metrics.lower_r2,
                "val_upper_r2": val_metrics.upper_r2,
                "val_width_r2": val_metrics.width_r2,
            }
        )
        print(
            f"[{name}] epoch {epoch}/{epochs} "
            f"train_loss={history[-1]['train_loss']:.6f} "
            f"val_lower={val_metrics.lower_mae_khz:.6f} "
            f"val_upper={val_metrics.upper_mae_khz:.6f} "
            f"val_width={val_metrics.width_mae_khz:.6f} "
            f"r2_lower={val_metrics.lower_r2:.6f} "
            f"r2_upper={val_metrics.upper_r2:.6f} "
            f"r2_width={val_metrics.width_r2:.6f}",
            flush=True,
        )
        if val_metrics.width_mae_khz < best_val_mae:
            best_val_mae = val_metrics.width_mae_khz
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"Regressor best state missing for {name}.")
    model.load_state_dict(best_state)
    model_path = output_dir / f"{name}_best.pt"
    torch.save(best_state, model_path)
    final_val_metrics = evaluate_regressor(model, val_loader, target_scaler, device, amp_enabled, amp_dtype)
    return model, {
        "history": history,
        "best_val_metrics": {
            "loss": final_val_metrics.loss,
            "lower_mae_khz": final_val_metrics.lower_mae_khz,
            "upper_mae_khz": final_val_metrics.upper_mae_khz,
            "width_mae_khz": final_val_metrics.width_mae_khz,
            "lower_r2": final_val_metrics.lower_r2,
            "upper_r2": final_val_metrics.upper_r2,
            "width_r2": final_val_metrics.width_r2,
        },
        "model_path": str(model_path),
    }


def compute_class_weights(group_ids: np.ndarray) -> torch.Tensor:
    counts = np.bincount(group_ids.astype(np.int64), minlength=3).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_final_metrics(
    true_lower: np.ndarray,
    true_upper: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
) -> dict[str, float]:
    true_width = true_upper - true_lower
    pred_width = pred_upper - pred_lower
    lower_err = pred_lower - true_lower
    upper_err = pred_upper - true_upper
    width_err = pred_width - true_width

    def r2_score_from_error(error: np.ndarray, true: np.ndarray) -> float:
        ss_res = float(np.sum(error ** 2))
        centered = true - float(np.mean(true))
        ss_tot = float(np.sum(centered ** 2))
        if ss_tot <= 0.0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    return {
        "lower_mae_khz": float(np.mean(np.abs(lower_err))),
        "upper_mae_khz": float(np.mean(np.abs(upper_err))),
        "width_mae_khz": float(np.mean(np.abs(width_err))),
        "lower_rmse_khz": float(np.sqrt(np.mean(lower_err ** 2))),
        "upper_rmse_khz": float(np.sqrt(np.mean(upper_err ** 2))),
        "width_rmse_khz": float(np.sqrt(np.mean(width_err ** 2))),
        "lower_bias_khz": float(np.mean(lower_err)),
        "upper_bias_khz": float(np.mean(upper_err)),
        "width_bias_khz": float(np.mean(width_err)),
        "lower_r2": r2_score_from_error(lower_err, true_lower),
        "upper_r2": r2_score_from_error(upper_err, true_upper),
        "width_r2": r2_score_from_error(width_err, true_width),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gap_index classifier first, then route to expert regressors.")
    parser.add_argument("--preprocessed-root", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--classifier-epochs", type=int, default=4)
    parser.add_argument("--regressor-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=224)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp-mode", type=str, default="auto", choices=["auto", "bf16", "fp16", "off"])
    parser.add_argument("--warmup-fraction", type=float, default=0.08)
    parser.add_argument("--min-lr-scale", type=float, default=0.10)
    parser.add_argument("--graph-feature-mode", type=str, default="novel105_phase_edge_shell")
    parser.add_argument("--major-gap-indices", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--min-group-train-samples", type=int, default=200)
    parser.add_argument("--min-sample-index", type=int, default=0)
    parser.add_argument("--max-sample-index", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_roots = [Path(root) for root in args.preprocessed_root]
    major_gap_indices = (int(args.major_gap_indices[0]), int(args.major_gap_indices[1]))
    indexed = build_index_with_gap_groups(
        cache_roots=cache_roots,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        major_gap_indices=major_gap_indices,
        min_sample_index=args.min_sample_index,
        max_sample_index=args.max_sample_index,
    )

    train_mask, val_mask, test_mask = make_split_masks(indexed.sample_ids)
    train_indices = train_mask
    val_indices = val_mask
    test_indices = test_mask

    feature_indices = get_graph_feature_indices(args.graph_feature_mode)
    train_dataset = CachedGapGroupDataset(indexed, train_indices, feature_indices=feature_indices)
    val_dataset = CachedGapGroupDataset(indexed, val_indices, feature_indices=feature_indices)
    test_dataset = CachedGapGroupDataset(indexed, test_indices, feature_indices=feature_indices)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled, amp_dtype, use_grad_scaler, resolved_amp_mode = resolve_amp_config(device, args.amp_mode)

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

    sample_batch = next(iter(train_loader))
    node_dim = int(sample_batch["node_x"].shape[1])
    edge_dim = int(sample_batch["edge_attr"].shape[1])
    graph_dim = int(sample_batch["graph_x"].shape[1])

    train_group_counts = np.bincount(indexed.group_ids[train_indices], minlength=3).astype(int)
    val_group_counts = np.bincount(indexed.group_ids[val_indices], minlength=3).astype(int)
    test_group_counts = np.bincount(indexed.group_ids[test_indices], minlength=3).astype(int)
    class_weights = compute_class_weights(indexed.group_ids[train_indices])

    group_names = [f"gap{major_gap_indices[0]}", f"gap{major_gap_indices[1]}", "other"]
    metadata = {
        "created_at": datetime.now().isoformat(),
        "preprocessed_roots": [str(root) for root in cache_roots],
        "major_gap_indices": list(major_gap_indices),
        "group_names": group_names,
        "graph_feature_mode": args.graph_feature_mode,
        "graph_feature_dim": int(graph_dim),
        "graph_feature_names": [GRAPH_FEATURE_NAMES[i] for i in feature_indices.tolist()],
        "train_samples": int(train_indices.sum()),
        "val_samples": int(val_indices.sum()),
        "test_samples": int(test_indices.sum()),
        "unique_positive_gap_samples": int(indexed.size),
        "duplicate_samples_skipped": int(indexed.duplicate_samples_skipped),
        "no_gap_samples_skipped": int(indexed.no_gap_samples_skipped),
        "sample_index_filtered_skipped": int(indexed.sample_index_filtered_skipped),
        "min_sample_index": int(args.min_sample_index),
        "max_sample_index": int(args.max_sample_index),
        "train_group_counts": train_group_counts.tolist(),
        "val_group_counts": val_group_counts.tolist(),
        "test_group_counts": test_group_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "device": str(device),
        "amp_mode": resolved_amp_mode,
        "classifier_epochs": args.classifier_epochs,
        "regressor_epochs": args.regressor_epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "min_group_train_samples": args.min_group_train_samples,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    start_time = time.perf_counter()

    classifier_model, classifier_info = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        use_grad_scaler=use_grad_scaler,
        class_weights=class_weights,
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_dim=graph_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.classifier_epochs,
        warmup_fraction=args.warmup_fraction,
        min_lr_scale=args.min_lr_scale,
        output_dir=run_dir,
    )
    classifier_val_metrics, _, _, _ = evaluate_classifier(
        classifier_model, val_loader, class_weights.to(device), device, amp_enabled, amp_dtype
    )
    classifier_test_metrics, y_true_test, y_pred_test, sample_ids_test = evaluate_classifier(
        classifier_model, test_loader, class_weights.to(device), device, amp_enabled, amp_dtype
    )

    regressor_specs: list[tuple[str, np.ndarray, np.ndarray]] = []
    gap2_train = train_indices & (indexed.group_ids == 0)
    gap2_val = val_indices & (indexed.group_ids == 0)
    gap3_train = train_indices & (indexed.group_ids == 1)
    gap3_val = val_indices & (indexed.group_ids == 1)
    other_train = train_indices & (indexed.group_ids == 2)
    other_val = val_indices & (indexed.group_ids == 2)

    regressor_specs.append(("gap2_expert", gap2_train, gap2_val))
    regressor_specs.append(("gap3_expert", gap3_train, gap3_val))
    if int(other_train.sum()) < args.min_group_train_samples:
        regressor_specs.append(("other_fallback", train_indices, val_indices))
    else:
        regressor_specs.append(("other_fallback", other_train, other_val))

    regressor_models: dict[str, GapBoundsRegressor] = {}
    regressor_infos: dict[str, dict[str, object]] = {}
    regressor_predictions: dict[str, dict[str, np.ndarray]] = {}

    for name, train_sel, val_sel in regressor_specs:
        reg_train_dataset = CachedGapGroupDataset(indexed, train_sel, feature_indices=feature_indices)
        reg_val_dataset = CachedGapGroupDataset(indexed, val_sel, feature_indices=feature_indices)
        reg_train_loader = DataLoader(
            reg_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_gap_groups,
            persistent_workers=args.num_workers > 0,
        )
        reg_val_loader = DataLoader(
            reg_val_dataset,
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
            train_loader=reg_train_loader,
            val_loader=reg_val_loader,
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
        regressor_models[name] = model
        regressor_infos[name] = {
            **info,
            "train_samples": int(train_sel.sum()),
            "val_samples": int(val_sel.sum()),
        }
        regressor_predictions[name] = predict_regressor(
            model=model,
            loader=test_loader,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

    base_prediction = regressor_predictions["other_fallback"]
    sample_order = base_prediction["sample_index"]
    group_true = base_prediction["group_id"]
    gap_index_true = base_prediction["gap_index_value"]
    true_lower = base_prediction["true_lower_khz"]
    true_upper = base_prediction["true_upper_khz"]
    classifier_pred = y_pred_test

    if not np.array_equal(sample_order, sample_ids_test):
        raise RuntimeError("Classifier/regressor sample orders do not match on the test set.")

    pred_lower = np.asarray(regressor_predictions["other_fallback"]["pred_lower_khz"], dtype=np.float64).copy()
    pred_upper = np.asarray(regressor_predictions["other_fallback"]["pred_upper_khz"], dtype=np.float64).copy()

    gap2_mask = classifier_pred == 0
    gap3_mask = classifier_pred == 1
    pred_lower[gap2_mask] = regressor_predictions["gap2_expert"]["pred_lower_khz"][gap2_mask]
    pred_upper[gap2_mask] = regressor_predictions["gap2_expert"]["pred_upper_khz"][gap2_mask]
    pred_lower[gap3_mask] = regressor_predictions["gap3_expert"]["pred_lower_khz"][gap3_mask]
    pred_upper[gap3_mask] = regressor_predictions["gap3_expert"]["pred_upper_khz"][gap3_mask]

    final_metrics = compute_final_metrics(true_lower, true_upper, pred_lower, pred_upper)
    per_group_metrics = {}
    for group_id, name in enumerate(group_names):
        mask = group_true == group_id
        if int(mask.sum()) == 0:
            continue
        per_group_metrics[name] = {
            "count": int(mask.sum()),
            **compute_final_metrics(true_lower[mask], true_upper[mask], pred_lower[mask], pred_upper[mask]),
        }

    pred_csv = run_dir / "test_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_index",
                "gap_index_true",
                "group_true",
                "group_pred",
                "lower_true_khz",
                "upper_true_khz",
                "width_true_khz",
                "lower_pred_khz",
                "upper_pred_khz",
                "width_pred_khz",
            ]
        )
        for row in zip(
            sample_order,
            gap_index_true,
            group_true,
            classifier_pred,
            true_lower,
            true_upper,
            true_upper - true_lower,
            pred_lower,
            pred_upper,
            pred_upper - pred_lower,
            strict=True,
        ):
            writer.writerow(row)

    elapsed = time.perf_counter() - start_time
    summary = {
        "finished_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "classifier": {
            **classifier_info,
            "val_metrics": {
                "loss": classifier_val_metrics.loss,
                "accuracy": classifier_val_metrics.accuracy,
                "balanced_accuracy": classifier_val_metrics.balanced_accuracy,
                "macro_f1": classifier_val_metrics.macro_f1,
            },
            "test_metrics": {
                "loss": classifier_test_metrics.loss,
                "accuracy": classifier_test_metrics.accuracy,
                "balanced_accuracy": classifier_test_metrics.balanced_accuracy,
                "macro_f1": classifier_test_metrics.macro_f1,
            },
        },
        "regressors": regressor_infos,
        "final_test_metrics": final_metrics,
        "per_group_test_metrics": per_group_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
