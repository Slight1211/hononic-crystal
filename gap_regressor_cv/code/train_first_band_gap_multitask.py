from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from node_pwe_repro import CELL_SIZE_M, EPOXY, STEEL, polygon_structure_factor


SUPPORT_HARMONICS = 8
SUPPORT_SAMPLES = 32
BRAGG_FEATURE_DIM = 8
BOUNDARY_SPECTRAL_DIM = 6
PHYSICS_PROXY_DIM = 10
DUAL_SPACE_DIM = 4
VERTEX_SCATTER_DIM = 6
RADIAL_COMPETITION_DIM = 6
CURVATURE_BLOCH_DIM = 4
DIRECTIONAL_PHASE_DIM = 8
EDGE_ORIENTATION_DIM = 8
CORNER_COUPLING_DIM = 4
SHELL_INTERFERENCE_DIM = 6
STRUCTURE_HKS: list[tuple[int, int]] = [
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 0),
    (4, 2),
    (4, 4),
]
BASE_GRAPH_FEATURE_NAMES = [
    "area_fraction",
    "sqrt_area_fraction",
    "node_count_norm",
    "perimeter",
    "isoperimetric_quotient",
    "edge_len_mean",
    "edge_len_std",
    "edge_len_min",
    "edge_len_max",
    "radius_mean",
    "radius_std",
    "radius_min",
    "radius_max",
    "bbox_width",
    "bbox_height",
]
SUPPORT_FEATURE_NAMES = [
    "support_mean",
    "support_std",
    "support_min",
    "support_max",
    *[f"support_harmonic_{i}" for i in range(1, SUPPORT_HARMONICS + 1)],
]
STRUCTURE_FEATURE_NAMES = [f"sf_{h}_{k}" for (h, k) in STRUCTURE_HKS]
BRAGG_FEATURE_NAMES = [
    "bragg_shell1_energy",
    "bragg_shell2_energy",
    "bragg_shell3_energy",
    "bragg_shell4_energy",
    "bragg_m_over_x",
    "bragg_shell2_over_shell1",
    "bragg_shell3_over_shell2",
    "bragg_x_minus_m",
]
BOUNDARY_SPECTRAL_NAMES = [
    "boundary_m4_real",
    "boundary_m4_imag",
    "boundary_m8_abs",
    "boundary_m8_real",
    "corner_turn_mean",
    "corner_turn_std",
]
PHYSICS_PROXY_NAMES = [
    "bloch_fx_transverse",
    "bloch_fm_transverse",
    "bloch_fx_longitudinal",
    "bloch_fm_longitudinal",
    "radius_of_gyration",
    "gap_x_proxy",
    "gap_m_proxy",
    "gap_complete_proxy",
    "gap_balance_proxy",
    "gap_inertia_proxy",
]
DUAL_SPACE_FEATURE_NAMES = [
    "dualspace_x_edge_proxy",
    "dualspace_m_edge_proxy",
    "dualspace_balance_proxy",
    "dualspace_inertia_proxy",
]
VERTEX_SCATTER_NAMES = [
    "vertex_q4_abs",
    "vertex_q8_abs",
    "corner_q4_abs",
    "corner_q8_abs",
    "radial_step_mean",
    "radial_step_std",
]
RADIAL_COMPETITION_NAMES = [
    "support_even_odd_ratio",
    "support_low_high_ratio",
    "support_h4_over_h2",
    "support_h8_over_h4",
    "support_peak_trough_ratio",
    "support_entropy",
]
CURVATURE_BLOCH_NAMES = [
    "curv_gap_x_proxy",
    "curv_gap_m_proxy",
    "corner_bragg_proxy",
    "anisotropic_transition_proxy",
]
DIRECTIONAL_PHASE_NAMES = [
    "support_phase_h2_real",
    "support_phase_h2_imag",
    "support_phase_h4_real",
    "support_phase_h4_imag",
    "support_phase_h6_real",
    "support_phase_h6_imag",
    "support_phase_h8_real",
    "support_phase_h8_imag",
]
EDGE_ORIENTATION_NAMES = [
    "edge_orient_q2_real",
    "edge_orient_q2_imag",
    "edge_orient_q4_real",
    "edge_orient_q4_imag",
    "edge_orient_q6_real",
    "edge_orient_q6_imag",
    "edge_orient_q8_real",
    "edge_orient_q8_imag",
]
CORNER_COUPLING_NAMES = [
    "corner_mode_q4_real",
    "corner_mode_q4_imag",
    "corner_mode_q8_real",
    "corner_mode_q8_imag",
]
SHELL_INTERFERENCE_NAMES = [
    "shell12_coupling",
    "shell13_coupling",
    "shell24_coupling",
    "x_minus_diag_shell2",
    "shell12_balance",
    "shell34_balance",
]
GRAPH_FEATURE_NAMES = (
    BASE_GRAPH_FEATURE_NAMES
    + SUPPORT_FEATURE_NAMES
    + STRUCTURE_FEATURE_NAMES
    + BRAGG_FEATURE_NAMES
    + BOUNDARY_SPECTRAL_NAMES
    + PHYSICS_PROXY_NAMES
    + DUAL_SPACE_FEATURE_NAMES
    + VERTEX_SCATTER_NAMES
    + RADIAL_COMPETITION_NAMES
    + CURVATURE_BLOCH_NAMES
    + DIRECTIONAL_PHASE_NAMES
    + EDGE_ORIENTATION_NAMES
    + CORNER_COUPLING_NAMES
    + SHELL_INTERFERENCE_NAMES
)
GRAPH_FEATURE_DIM = (
    15
    + 4
    + SUPPORT_HARMONICS
    + len(STRUCTURE_HKS)
    + BRAGG_FEATURE_DIM
    + BOUNDARY_SPECTRAL_DIM
    + PHYSICS_PROXY_DIM
    + DUAL_SPACE_DIM
    + VERTEX_SCATTER_DIM
    + RADIAL_COMPETITION_DIM
    + CURVATURE_BLOCH_DIM
    + DIRECTIONAL_PHASE_DIM
    + EDGE_ORIENTATION_DIM
    + CORNER_COUPLING_DIM
    + SHELL_INTERFERENCE_DIM
)
assert len(GRAPH_FEATURE_NAMES) == GRAPH_FEATURE_DIM
_baseline_stop = len(BASE_GRAPH_FEATURE_NAMES) + len(SUPPORT_FEATURE_NAMES) + len(STRUCTURE_FEATURE_NAMES)
_bragg_stop = _baseline_stop + len(BRAGG_FEATURE_NAMES)
_boundary_stop = _bragg_stop + len(BOUNDARY_SPECTRAL_NAMES)
_physics_stop = _boundary_stop + len(PHYSICS_PROXY_NAMES)
_dual_space_stop = _physics_stop + len(DUAL_SPACE_FEATURE_NAMES)
_vertex_stop = _dual_space_stop + len(VERTEX_SCATTER_NAMES)
_radial_stop = _vertex_stop + len(RADIAL_COMPETITION_NAMES)
_curvature_stop = _radial_stop + len(CURVATURE_BLOCH_NAMES)
_phase_stop = _curvature_stop + len(DIRECTIONAL_PHASE_NAMES)
_edge_orient_stop = _phase_stop + len(EDGE_ORIENTATION_NAMES)
_corner_stop = _edge_orient_stop + len(CORNER_COUPLING_NAMES)
FEATURE_GROUP_SLICES = {
    "baseline": slice(0, _baseline_stop),
    "bragg": slice(_baseline_stop, _bragg_stop),
    "boundary": slice(_bragg_stop, _boundary_stop),
    "physics": slice(_boundary_stop, _physics_stop),
    "dual_space": slice(_physics_stop, _dual_space_stop),
    "vertex_scatter": slice(_dual_space_stop, _vertex_stop),
    "radial_competition": slice(_vertex_stop, _radial_stop),
    "curvature_bloch": slice(_radial_stop, _curvature_stop),
    "directional_phase": slice(_curvature_stop, _phase_stop),
    "edge_orientation": slice(_phase_stop, _edge_orient_stop),
    "corner_coupling": slice(_edge_orient_stop, _corner_stop),
    "shell_interference": slice(_corner_stop, GRAPH_FEATURE_DIM),
}


def get_graph_feature_indices(mode: str) -> np.ndarray:
    if mode == "full_novel":
        return np.arange(GRAPH_FEATURE_DIM, dtype=np.int64)
    if mode == "full_dual_space":
        return np.arange(FEATURE_GROUP_SLICES["dual_space"].stop, dtype=np.int64)
    if mode == "baseline_plus_physics":
        return np.arange(FEATURE_GROUP_SLICES["physics"].stop, dtype=np.int64)
    if mode == "novel105_phase_edge_shell":
        first = np.arange(FEATURE_GROUP_SLICES["edge_orientation"].stop, dtype=np.int64)
        second = np.arange(FEATURE_GROUP_SLICES["shell_interference"].start, GRAPH_FEATURE_DIM, dtype=np.int64)
        return np.concatenate([first, second], axis=0)
    if mode == "novel101_edge_corner_shell":
        first = np.arange(FEATURE_GROUP_SLICES["directional_phase"].start, dtype=np.int64)
        second = np.arange(FEATURE_GROUP_SLICES["edge_orientation"].start, GRAPH_FEATURE_DIM, dtype=np.int64)
        return np.concatenate([first, second], axis=0)
    raise ValueError(f"Unknown graph feature mode: {mode}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(vertices: np.ndarray) -> float:
    edges = np.roll(vertices, -1, axis=0) - vertices
    return float(np.linalg.norm(edges, axis=1).sum())


def polygon_second_moments(vertices: np.ndarray) -> tuple[float, float]:
    current = np.asarray(vertices, dtype=np.float64)
    nxt = np.roll(current, -1, axis=0)
    x0 = current[:, 0]
    y0 = current[:, 1]
    x1 = nxt[:, 0]
    y1 = nxt[:, 1]
    cross = x0 * y1 - x1 * y0
    area_signed = 0.5 * np.sum(cross)
    ix = np.sum((y0 * y0 + y0 * y1 + y1 * y1) * cross) / 12.0
    iy = np.sum((x0 * x0 + x0 * x1 + x1 * x1) * cross) / 12.0
    if area_signed < 0.0:
        ix = -ix
        iy = -iy
    return float(ix), float(iy)


class ChunkCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def get(self, path: str) -> list[dict[str, Any]]:
        if path in self._cache:
            records = self._cache.pop(path)
            self._cache[path] = records
            return records
        with Path(path).open("rb") as f:
            records = pickle.load(f)
        self._cache[path] = records
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return records


@dataclass
class IndexedSamples:
    chunk_paths: list[str]
    chunk_ids: np.ndarray
    local_ids: np.ndarray
    sample_ids: np.ndarray
    has_gap: np.ndarray
    gap_targets: np.ndarray
    band1_targets: np.ndarray

    @property
    def size(self) -> int:
        return int(self.sample_ids.shape[0])


def discover_batch_dirs(data_roots: list[Path], max_batches: int = 0) -> list[Path]:
    batch_dirs: list[Path] = []
    seen: set[str] = set()
    for data_root in data_roots:
        root = data_root.expanduser()
        candidates = sorted(p for p in root.rglob("batch_*") if p.is_dir())
        kept_for_root = 0
        for batch_dir in candidates:
            if not any(batch_dir.glob("pilot_chunk_*.pkl")):
                continue
            key = str(batch_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            batch_dirs.append(batch_dir)
            kept_for_root += 1
            if max_batches > 0 and kept_for_root >= max_batches:
                break
    batch_dirs.sort()
    if not batch_dirs:
        roots_text = ", ".join(str(root) for root in data_roots)
        raise FileNotFoundError(f"No batch_* directories with pilot_chunk_*.pkl found under: {roots_text}")
    return batch_dirs


def discover_chunk_paths(data_roots: list[Path], max_batches: int = 0) -> list[Path]:
    batch_dirs = discover_batch_dirs(data_roots, max_batches=max_batches)
    chunk_paths: list[Path] = []
    for batch_dir in batch_dirs:
        chunk_paths.extend(sorted(batch_dir.glob("pilot_chunk_*.pkl")))
    if not chunk_paths:
        roots_text = ", ".join(str(root) for root in data_roots)
        raise FileNotFoundError(f"No pilot_chunk_*.pkl files found under: {roots_text}")
    return chunk_paths


def build_index(data_roots: list[Path], max_batches: int = 0, max_samples: int = 0) -> IndexedSamples:
    chunk_paths = discover_chunk_paths(data_roots, max_batches=max_batches)
    chunk_ids: list[int] = []
    local_ids: list[int] = []
    sample_ids: list[int] = []
    has_gap: list[float] = []
    gap_targets: list[list[float]] = []
    band1_targets: list[np.ndarray] = []

    for chunk_id, chunk_path in enumerate(chunk_paths):
        with chunk_path.open("rb") as f:
            records = pickle.load(f)
        for local_id, record in enumerate(records):
            bands = np.asarray(record["bands_khz"], dtype=np.float32)
            lower = float(record["lower_khz"]) if record["lower_khz"] is not None else 0.0
            upper = float(record["upper_khz"]) if record["upper_khz"] is not None else 0.0
            width = float(record["gap_width_khz"]) if record["gap_width_khz"] is not None else 0.0

            chunk_ids.append(chunk_id)
            local_ids.append(local_id)
            sample_ids.append(int(record["sample_index"]))
            has_gap.append(1.0 if record["has_gap"] else 0.0)
            gap_targets.append([lower, upper, width])
            band1_targets.append(bands[:, 0].astype(np.float32))
            if max_samples > 0 and len(sample_ids) >= max_samples:
                break
        if max_samples > 0 and len(sample_ids) >= max_samples:
            break

    return IndexedSamples(
        chunk_paths=[str(p) for p in chunk_paths],
        chunk_ids=np.asarray(chunk_ids, dtype=np.int32),
        local_ids=np.asarray(local_ids, dtype=np.int32),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        has_gap=np.asarray(has_gap, dtype=np.float32),
        gap_targets=np.asarray(gap_targets, dtype=np.float32),
        band1_targets=np.asarray(band1_targets, dtype=np.float32),
    )


def discover_preprocessed_batch_dirs(cache_roots: list[Path], max_batches: int = 0) -> list[Path]:
    batch_dirs: list[Path] = []
    seen: set[str] = set()
    for cache_root in cache_roots:
        root = cache_root.expanduser()
        candidates = sorted(p for p in root.rglob("batch_*") if p.is_dir())
        kept_for_root = 0
        for batch_dir in candidates:
            if not any(batch_dir.glob("graph_chunk_*.pt")):
                continue
            key = str(batch_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            batch_dirs.append(batch_dir)
            kept_for_root += 1
            if max_batches > 0 and kept_for_root >= max_batches:
                break
    batch_dirs.sort()
    if not batch_dirs:
        roots_text = ", ".join(str(root) for root in cache_roots)
        raise FileNotFoundError(f"No batch_* directories with graph_chunk_*.pt found under: {roots_text}")
    return batch_dirs


def discover_preprocessed_chunk_paths(cache_roots: list[Path], max_batches: int = 0) -> list[Path]:
    batch_dirs = discover_preprocessed_batch_dirs(cache_roots, max_batches=max_batches)
    chunk_paths: list[Path] = []
    for batch_dir in batch_dirs:
        chunk_paths.extend(sorted(batch_dir.glob("graph_chunk_*.pt")))
    if not chunk_paths:
        roots_text = ", ".join(str(root) for root in cache_roots)
        raise FileNotFoundError(f"No graph_chunk_*.pt files found under: {roots_text}")
    return chunk_paths


def build_index_from_preprocessed(cache_roots: list[Path], max_batches: int = 0, max_samples: int = 0) -> IndexedSamples:
    chunk_paths = discover_preprocessed_chunk_paths(cache_roots, max_batches=max_batches)
    chunk_ids: list[int] = []
    local_ids: list[int] = []
    sample_ids: list[int] = []
    has_gap: list[float] = []
    gap_targets: list[list[float]] = []
    band1_targets: list[np.ndarray] = []

    for chunk_id, chunk_path in enumerate(chunk_paths):
        chunk = torch.load(chunk_path, map_location="cpu")
        chunk_dim = int(chunk["graph_x"].shape[1])
        if chunk_dim != GRAPH_FEATURE_DIM:
            raise RuntimeError(
                f"Cached graph feature dimension mismatch for {chunk_path}: "
                f"found {chunk_dim}, expected {GRAPH_FEATURE_DIM}. "
                "Please regenerate the preprocessed cache with the current feature set."
            )
        sample_index = chunk["sample_index"].numpy()
        has_gap_chunk = chunk["has_gap"].numpy()
        gap_target_chunk = chunk["gap_target"].numpy()
        band1_chunk = chunk["band1"].numpy()
        for local_id in range(int(sample_index.shape[0])):
            chunk_ids.append(chunk_id)
            local_ids.append(local_id)
            sample_ids.append(int(sample_index[local_id]))
            has_gap.append(float(has_gap_chunk[local_id]))
            gap_targets.append(gap_target_chunk[local_id].astype(np.float32).tolist())
            band1_targets.append(band1_chunk[local_id].astype(np.float32))
            if max_samples > 0 and len(sample_ids) >= max_samples:
                break
        if max_samples > 0 and len(sample_ids) >= max_samples:
            break

    return IndexedSamples(
        chunk_paths=[str(p) for p in chunk_paths],
        chunk_ids=np.asarray(chunk_ids, dtype=np.int32),
        local_ids=np.asarray(local_ids, dtype=np.int32),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        has_gap=np.asarray(has_gap, dtype=np.float32),
        gap_targets=np.asarray(gap_targets, dtype=np.float32),
        band1_targets=np.asarray(band1_targets, dtype=np.float32),
    )


def make_split_masks(sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    buckets = sample_ids % 10
    test_mask = buckets == 0
    val_mask = buckets == 1
    train_mask = ~(test_mask | val_mask)
    return train_mask, val_mask, test_mask


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, eps)


def compute_support_signature(vertices: np.ndarray, num_angles: int = SUPPORT_SAMPLES) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False, dtype=np.float32)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return np.max(vertices @ directions.T, axis=0).astype(np.float32)


def compute_support_harmonics(vertices: np.ndarray, num_angles: int = SUPPORT_SAMPLES, keep: int = SUPPORT_HARMONICS) -> np.ndarray:
    support = compute_support_signature(vertices, num_angles=num_angles)
    support_mean = float(np.mean(support))
    normalized = support / max(support_mean, 1e-8)
    coeffs = np.fft.rfft(normalized)
    magnitudes = np.abs(coeffs[1 : keep + 1]).astype(np.float32)
    if magnitudes.shape[0] < keep:
        magnitudes = np.pad(magnitudes, (0, keep - magnitudes.shape[0]))
    return np.concatenate(
        [
            np.asarray(
                [
                    support_mean,
                    float(np.std(support)),
                    float(np.min(support)),
                    float(np.max(support)),
                ],
                dtype=np.float32,
            ),
            magnitudes,
        ]
    ).astype(np.float32)


def compute_structure_features(vertices: np.ndarray) -> np.ndarray:
    values = [abs(polygon_structure_factor(hk, vertices.astype(float))) for hk in STRUCTURE_HKS]
    return np.asarray(values, dtype=np.float32)


def compute_bragg_proxy_features(structure_features: np.ndarray) -> np.ndarray:
    eps = 1e-8
    amp = {hk: float(structure_features[i]) for i, hk in enumerate(STRUCTURE_HKS)}
    a10 = amp[(1, 0)]
    a11 = amp[(1, 1)]
    a20 = amp[(2, 0)]
    a21 = amp[(2, 1)]
    a22 = amp[(2, 2)]
    a30 = amp[(3, 0)]
    a31 = amp[(3, 1)]
    a32 = amp[(3, 2)]
    a33 = amp[(3, 3)]
    a40 = amp[(4, 0)]
    a42 = amp[(4, 2)]
    a44 = amp[(4, 4)]
    shell1 = math.sqrt(a10 * a10 + a11 * a11)
    shell2 = math.sqrt(a20 * a20 + a21 * a21 + a22 * a22)
    shell3 = math.sqrt(a30 * a30 + a31 * a31 + a32 * a32 + a33 * a33)
    shell4 = math.sqrt(a40 * a40 + a42 * a42 + a44 * a44)
    return np.asarray(
        [
            shell1,
            shell2,
            shell3,
            shell4,
            a11 / (a10 + eps),
            shell2 / (shell1 + eps),
            shell3 / (shell2 + eps),
            a10 - a11,
        ],
        dtype=np.float32,
    )


def compute_boundary_spectral_features(vertices: np.ndarray) -> np.ndarray:
    current = np.asarray(vertices, dtype=np.float64)
    nxt = np.roll(current, -1, axis=0)
    edges = nxt - current
    lengths = np.linalg.norm(edges, axis=1)
    perimeter = max(float(lengths.sum()), 1e-8)
    weights = lengths / perimeter
    tangent_angles = np.arctan2(edges[:, 1], edges[:, 0])
    m4 = np.sum(weights * np.exp(1j * 4.0 * tangent_angles))
    m8 = np.sum(weights * np.exp(1j * 8.0 * tangent_angles))
    backward = current - np.roll(current, 1, axis=0)
    backward_unit = normalize_vectors(backward)
    forward_unit = normalize_vectors(edges)
    turn_cos = np.sum(backward_unit * forward_unit, axis=1)
    turn_sin = backward_unit[:, 0] * forward_unit[:, 1] - backward_unit[:, 1] * forward_unit[:, 0]
    turn_angles = np.abs(np.arctan2(turn_sin, turn_cos))
    return np.asarray(
        [
            float(np.real(m4)),
            float(np.imag(m4)),
            float(np.abs(m8)),
            float(np.real(m8)),
            float(np.mean(turn_angles)),
            float(np.std(turn_angles)),
        ],
        dtype=np.float32,
    )


def compute_physics_proxy_features(area_fraction: float, structure_features: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    eps = 1e-8
    rho_eff = (1.0 - area_fraction) * EPOXY.density + area_fraction * STEEL.density
    lam_eff = (1.0 - area_fraction) * EPOXY.lame_lambda + area_fraction * STEEL.lame_lambda
    mu_eff = (1.0 - area_fraction) * EPOXY.lame_mu + area_fraction * STEEL.lame_mu
    c_t = math.sqrt(max(mu_eff / rho_eff, eps))
    c_l = math.sqrt(max((lam_eff + 2.0 * mu_eff) / rho_eff, eps))
    f_x_t = c_t / (CELL_SIZE_M * 1e3)
    f_m_t = math.sqrt(2.0) * c_t / (CELL_SIZE_M * 1e3)
    f_x_l = c_l / (CELL_SIZE_M * 1e3)
    f_m_l = math.sqrt(2.0) * c_l / (CELL_SIZE_M * 1e3)
    amp = {hk: float(structure_features[i]) for i, hk in enumerate(STRUCTURE_HKS)}
    gap_x_proxy = f_x_t * amp[(1, 0)]
    gap_m_proxy = f_m_t * amp[(1, 1)]
    gap_complete_proxy = min(gap_x_proxy, gap_m_proxy)
    ix, iy = polygon_second_moments(vertices)
    area_exact = max(polygon_area(vertices), eps)
    radius_of_gyration = math.sqrt(max((ix + iy) / area_exact, eps))
    return np.asarray(
        [
            f_x_t,
            f_m_t,
            f_x_l,
            f_m_l,
            radius_of_gyration,
            gap_x_proxy,
            gap_m_proxy,
            gap_complete_proxy,
            gap_m_proxy / (gap_x_proxy + eps),
            gap_complete_proxy * radius_of_gyration,
        ],
        dtype=np.float32,
    )


def compute_dual_space_features(
    boundary_spectral_features: np.ndarray,
    physics_proxy_features: np.ndarray,
) -> np.ndarray:
    m4_real = float(boundary_spectral_features[0])
    m4_imag = float(boundary_spectral_features[1])
    m8_abs = float(boundary_spectral_features[2])
    turn_mean = float(boundary_spectral_features[4])
    gap_x_proxy = float(physics_proxy_features[5])
    gap_m_proxy = float(physics_proxy_features[6])
    gap_balance_proxy = float(physics_proxy_features[8])
    gap_inertia_proxy = float(physics_proxy_features[9])
    return np.asarray(
        [
            gap_x_proxy * (1.0 + abs(m4_real)),
            gap_m_proxy * (1.0 + m8_abs),
            gap_balance_proxy * (1.0 + abs(m4_imag)),
            gap_inertia_proxy * (1.0 + turn_mean),
        ],
        dtype=np.float32,
    )


def compute_vertex_scattering_features(vertices: np.ndarray) -> np.ndarray:
    current = np.asarray(vertices, dtype=np.float64)
    nxt = np.roll(current, -1, axis=0)
    prv = np.roll(current, 1, axis=0)
    radius = np.linalg.norm(current, axis=1)
    theta = np.arctan2(current[:, 1], current[:, 0])
    radius_weights = radius / max(float(radius.sum()), 1e-8)
    vertex_q4 = np.sum(radius_weights * np.exp(1j * 4.0 * theta))
    vertex_q8 = np.sum(radius_weights * np.exp(1j * 8.0 * theta))

    backward = current - prv
    forward = nxt - current
    backward_unit = normalize_vectors(backward)
    forward_unit = normalize_vectors(forward)
    turn_cos = np.sum(backward_unit * forward_unit, axis=1)
    turn_sin = backward_unit[:, 0] * forward_unit[:, 1] - backward_unit[:, 1] * forward_unit[:, 0]
    turn_angles = np.abs(np.arctan2(turn_sin, turn_cos))
    corner_weights = turn_angles / max(float(turn_angles.sum()), 1e-8)
    corner_q4 = np.sum(corner_weights * np.exp(1j * 4.0 * theta))
    corner_q8 = np.sum(corner_weights * np.exp(1j * 8.0 * theta))

    radial_steps = np.abs(np.roll(radius, -1) - radius)
    return np.asarray(
        [
            float(np.abs(vertex_q4)),
            float(np.abs(vertex_q8)),
            float(np.abs(corner_q4)),
            float(np.abs(corner_q8)),
            float(np.mean(radial_steps)),
            float(np.std(radial_steps)),
        ],
        dtype=np.float32,
    )


def compute_radial_competition_features(vertices: np.ndarray) -> np.ndarray:
    support = compute_support_signature(vertices, num_angles=SUPPORT_SAMPLES)
    support_mean = max(float(np.mean(support)), 1e-8)
    normalized = support / support_mean
    coeffs = np.abs(np.fft.rfft(normalized))[1 : SUPPORT_HARMONICS + 1]
    if coeffs.shape[0] < SUPPORT_HARMONICS:
        coeffs = np.pad(coeffs, (0, SUPPORT_HARMONICS - coeffs.shape[0]))
    even_energy = float(np.sum(coeffs[1::2]))
    odd_energy = float(np.sum(coeffs[0::2]))
    low_energy = float(np.sum(coeffs[:4]))
    high_energy = float(np.sum(coeffs[4:8]))
    p = coeffs / max(float(np.sum(coeffs)), 1e-8)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    return np.asarray(
        [
            even_energy / max(odd_energy, 1e-8),
            low_energy / max(high_energy, 1e-8),
            float(coeffs[3] / max(coeffs[1], 1e-8)),
            float(coeffs[7] / max(coeffs[3], 1e-8)),
            float(np.max(support) / max(np.min(support), 1e-8)),
            entropy,
        ],
        dtype=np.float32,
    )


def compute_curvature_bloch_features(
    vertices: np.ndarray,
    structure_features: np.ndarray,
    boundary_spectral_features: np.ndarray,
    physics_proxy_features: np.ndarray,
) -> np.ndarray:
    current = np.asarray(vertices, dtype=np.float64)
    nxt = np.roll(current, -1, axis=0)
    prv = np.roll(current, 1, axis=0)
    edges = nxt - current
    lengths = np.linalg.norm(edges, axis=1)
    backward = current - prv
    backward_unit = normalize_vectors(backward)
    forward_unit = normalize_vectors(edges)
    turn_cos = np.sum(backward_unit * forward_unit, axis=1)
    turn_sin = backward_unit[:, 0] * forward_unit[:, 1] - backward_unit[:, 1] * forward_unit[:, 0]
    turn_angles = np.abs(np.arctan2(turn_sin, turn_cos))
    curvature_strength = float(np.sum(turn_angles * lengths) / max(np.sum(lengths), 1e-8))
    shell1_energy = math.sqrt(float(structure_features[0]) ** 2 + float(structure_features[1]) ** 2)
    shell2_energy = math.sqrt(float(structure_features[2]) ** 2 + float(structure_features[3]) ** 2 + float(structure_features[4]) ** 2)
    gap_x_proxy = float(physics_proxy_features[5])
    gap_m_proxy = float(physics_proxy_features[6])
    boundary_m4_real = float(boundary_spectral_features[0])
    return np.asarray(
        [
            curvature_strength * gap_x_proxy,
            curvature_strength * gap_m_proxy,
            curvature_strength * shell1_energy,
            (shell2_energy - shell1_energy) * boundary_m4_real,
        ],
        dtype=np.float32,
    )


def compute_directional_phase_features(vertices: np.ndarray) -> np.ndarray:
    support = compute_support_signature(vertices, num_angles=SUPPORT_SAMPLES)
    normalized = support / max(float(np.mean(support)), 1e-8)
    coeffs = np.fft.rfft(normalized)
    features: list[float] = []
    for harmonic in [2, 4, 6, 8]:
        coeff = coeffs[harmonic] if harmonic < coeffs.shape[0] else 0.0 + 0.0j
        magnitude = abs(coeff)
        if magnitude < 1e-8:
            features.extend([0.0, 0.0])
        else:
            phasor = coeff / magnitude
            features.extend([float(np.real(phasor)), float(np.imag(phasor))])
    return np.asarray(features, dtype=np.float32)


def compute_edge_orientation_features(vertices: np.ndarray) -> np.ndarray:
    current = np.asarray(vertices, dtype=np.float64)
    nxt = np.roll(current, -1, axis=0)
    edges = nxt - current
    lengths = np.linalg.norm(edges, axis=1)
    weights = lengths / max(float(lengths.sum()), 1e-8)
    theta = np.arctan2(edges[:, 1], edges[:, 0])
    features: list[float] = []
    for harmonic in [2, 4, 6, 8]:
        q = np.sum(weights * np.exp(1j * harmonic * theta))
        features.extend([float(np.real(q)), float(np.imag(q))])
    return np.asarray(features, dtype=np.float32)


def compute_corner_coupling_features(vertices: np.ndarray) -> np.ndarray:
    current = np.asarray(vertices, dtype=np.float64)
    nxt = np.roll(current, -1, axis=0)
    prv = np.roll(current, 1, axis=0)
    backward = current - prv
    forward = nxt - current
    backward_unit = normalize_vectors(backward)
    forward_unit = normalize_vectors(forward)
    turn_cos = np.sum(backward_unit * forward_unit, axis=1)
    turn_sin = backward_unit[:, 0] * forward_unit[:, 1] - backward_unit[:, 1] * forward_unit[:, 0]
    turn_angles = np.abs(np.arctan2(turn_sin, turn_cos))
    theta = np.arctan2(current[:, 1], current[:, 0])
    weights = turn_angles / max(float(turn_angles.sum()), 1e-8)
    q4 = np.sum(weights * np.exp(1j * 4.0 * theta))
    q8 = np.sum(weights * np.exp(1j * 8.0 * theta))
    return np.asarray([float(np.real(q4)), float(np.imag(q4)), float(np.real(q8)), float(np.imag(q8))], dtype=np.float32)


def compute_shell_interference_features(structure_features: np.ndarray) -> np.ndarray:
    eps = 1e-8
    amp = {hk: float(structure_features[i]) for i, hk in enumerate(STRUCTURE_HKS)}
    shell1 = math.sqrt(amp[(1, 0)] ** 2 + amp[(1, 1)] ** 2)
    shell2 = math.sqrt(amp[(2, 0)] ** 2 + amp[(2, 1)] ** 2 + amp[(2, 2)] ** 2)
    shell3 = math.sqrt(amp[(3, 0)] ** 2 + amp[(3, 1)] ** 2 + amp[(3, 2)] ** 2 + amp[(3, 3)] ** 2)
    shell4 = math.sqrt(amp[(4, 0)] ** 2 + amp[(4, 2)] ** 2 + amp[(4, 4)] ** 2)
    x_minus_diag_shell2 = (amp[(2, 0)] + amp[(2, 1)]) - amp[(2, 2)]
    return np.asarray(
        [
            shell1 * shell2,
            shell1 * shell3,
            shell2 * shell4,
            x_minus_diag_shell2,
            (shell1 - shell2) / (shell1 + shell2 + eps),
            (shell3 - shell4) / (shell3 + shell4 + eps),
        ],
        dtype=np.float32,
    )


def record_to_graph(record: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(record["vertices"], dtype=np.float32)
    count = vertices.shape[0]
    nxt = np.roll(vertices, -1, axis=0)
    prv = np.roll(vertices, 1, axis=0)

    edge_forward = nxt - vertices
    edge_backward = vertices - prv
    forward_length = np.linalg.norm(edge_forward, axis=1, keepdims=True)
    backward_length = np.linalg.norm(edge_backward, axis=1, keepdims=True)

    centered = vertices
    radius = np.linalg.norm(centered, axis=1, keepdims=True)
    angle = np.arctan2(centered[:, 1:2], centered[:, 0:1])
    unit_forward = normalize_vectors(edge_forward)
    unit_backward = normalize_vectors(edge_backward)
    turn_cos = np.sum(unit_backward * unit_forward, axis=1, keepdims=True)
    turn_sin = unit_backward[:, :1] * unit_forward[:, 1:2] - unit_backward[:, 1:2] * unit_forward[:, :1]
    local_triangle = 0.5 * np.abs(
        edge_backward[:, :1] * edge_forward[:, 1:2] - edge_backward[:, 1:2] * edge_forward[:, :1]
    )
    neighbor_radius_mean = (np.roll(radius, 1, axis=0) + radius + np.roll(radius, -1, axis=0)) / 3.0

    node_features = np.concatenate(
        [
            centered,
            radius,
            np.cos(angle),
            np.sin(angle),
            backward_length,
            forward_length,
            turn_cos,
            turn_sin,
            local_triangle,
            neighbor_radius_mean,
        ],
        axis=1,
    ).astype(np.float32)

    src = np.arange(count, dtype=np.int64)
    dst = (src + 1) % count
    reverse_src = dst
    reverse_dst = src
    edge_index = np.stack(
        [
            np.concatenate([src, reverse_src]),
            np.concatenate([dst, reverse_dst]),
        ],
        axis=0,
    )

    reverse_edge = -edge_forward
    forward_unit = normalize_vectors(edge_forward)
    reverse_unit = normalize_vectors(reverse_edge)
    midpoint = 0.5 * (vertices + nxt)
    midpoint_radius = np.linalg.norm(midpoint, axis=1, keepdims=True)
    radius_delta = np.linalg.norm(nxt, axis=1, keepdims=True) - radius
    edge_attr = np.concatenate(
        [
            np.concatenate([edge_forward, reverse_edge], axis=0),
            np.concatenate([forward_unit, reverse_unit], axis=0),
            np.concatenate([forward_length, forward_length], axis=0),
            np.concatenate([radius_delta, -radius_delta], axis=0),
            np.concatenate([midpoint_radius, midpoint_radius], axis=0),
        ],
        axis=1,
    ).astype(np.float32)

    area_fraction = float(record["area_fraction"])
    perimeter = polygon_perimeter(vertices)
    radius_flat = radius[:, 0]
    lengths_flat = forward_length[:, 0]
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    iso = 4.0 * math.pi * area_fraction / max(perimeter * perimeter, 1e-8)
    base_graph_features = np.asarray(
        [
            area_fraction,
            math.sqrt(max(area_fraction, 1e-8)),
            float(count) / 64.0,
            perimeter,
            iso,
            float(lengths_flat.mean()),
            float(lengths_flat.std()),
            float(lengths_flat.min()),
            float(lengths_flat.max()),
            float(radius_flat.mean()),
            float(radius_flat.std()),
            float(radius_flat.min()),
            float(radius_flat.max()),
            float(bbox_size[0]),
            float(bbox_size[1]),
        ],
        dtype=np.float32,
    )
    support_features = compute_support_harmonics(vertices)
    structure_features = compute_structure_features(vertices)
    bragg_features = compute_bragg_proxy_features(structure_features)
    boundary_spectral_features = compute_boundary_spectral_features(vertices)
    physics_proxy_features = compute_physics_proxy_features(area_fraction, structure_features, vertices)
    dual_space_features = compute_dual_space_features(boundary_spectral_features, physics_proxy_features)
    vertex_scatter_features = compute_vertex_scattering_features(vertices)
    radial_competition_features = compute_radial_competition_features(vertices)
    curvature_bloch_features = compute_curvature_bloch_features(
        vertices,
        structure_features,
        boundary_spectral_features,
        physics_proxy_features,
    )
    directional_phase_features = compute_directional_phase_features(vertices)
    edge_orientation_features = compute_edge_orientation_features(vertices)
    corner_coupling_features = compute_corner_coupling_features(vertices)
    shell_interference_features = compute_shell_interference_features(structure_features)
    graph_features = np.concatenate(
        [
            base_graph_features,
            support_features,
            structure_features,
            bragg_features,
            boundary_spectral_features,
            physics_proxy_features,
            dual_space_features,
            vertex_scatter_features,
            radial_competition_features,
            curvature_bloch_features,
            directional_phase_features,
            edge_orientation_features,
            corner_coupling_features,
            shell_interference_features,
        ],
        axis=0,
    ).astype(np.float32)

    bands = np.asarray(record["bands_khz"], dtype=np.float32)
    band1 = bands[:, 0].astype(np.float32)
    has_gap = 1.0 if record["has_gap"] else 0.0
    lower = float(record["lower_khz"]) if record["lower_khz"] is not None else 0.0
    upper = float(record["upper_khz"]) if record["upper_khz"] is not None else 0.0
    width = float(record["gap_width_khz"]) if record["gap_width_khz"] is not None else 0.0

    return {
        "node_x": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "graph_x": graph_features,
        "band1": band1,
        "has_gap": np.asarray([has_gap], dtype=np.float32),
        "gap_target": np.asarray([lower, upper, width], dtype=np.float32),
        "sample_index": int(record["sample_index"]),
    }


class MultiTaskDataset(Dataset):
    def __init__(self, indexed: IndexedSamples, selection: np.ndarray, cache_size: int = 8, feature_indices: np.ndarray | None = None):
        self.chunk_paths = indexed.chunk_paths
        self.chunk_ids = indexed.chunk_ids[selection]
        self.local_ids = indexed.local_ids[selection]
        self.sample_ids = indexed.sample_ids[selection]
        self.has_gap = indexed.has_gap[selection]
        self.cache = ChunkCache(max_items=cache_size)
        self.feature_indices = feature_indices

    def __len__(self) -> int:
        return int(self.sample_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        chunk_path = self.chunk_paths[int(self.chunk_ids[index])]
        records = self.cache.get(chunk_path)
        record = records[int(self.local_ids[index])]
        item = record_to_graph(record)
        if self.feature_indices is not None:
            item["graph_x"] = item["graph_x"][self.feature_indices]
        return item


class TensorChunkCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, path: str) -> dict[str, Any]:
        if path in self._cache:
            item = self._cache.pop(path)
            self._cache[path] = item
            return item
        item = torch.load(path, map_location="cpu")
        self._cache[path] = item
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return item


class CachedMultiTaskDataset(Dataset):
    def __init__(self, indexed: IndexedSamples, selection: np.ndarray, cache_size: int = 8, feature_indices: np.ndarray | None = None):
        self.chunk_paths = indexed.chunk_paths
        self.chunk_ids = indexed.chunk_ids[selection]
        self.local_ids = indexed.local_ids[selection]
        self.sample_ids = indexed.sample_ids[selection]
        self.has_gap = indexed.has_gap[selection]
        self.cache = TensorChunkCache(max_items=cache_size)
        self.feature_indices = feature_indices

    def __len__(self) -> int:
        return int(self.sample_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
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
            "band1": chunk["band1"][local_id],
            "has_gap": chunk["has_gap"][local_id : local_id + 1],
            "gap_target": chunk["gap_target"][local_id],
            "sample_index": int(chunk["sample_index"][local_id].item()),
        }


def collate_graphs(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    node_x_list = []
    edge_index_list = []
    edge_attr_list = []
    graph_x_list = []
    band1_list = []
    has_gap_list = []
    gap_target_list = []
    sample_ids = []
    batch_index_list = []
    node_offset = 0

    for graph_id, item in enumerate(batch):
        node_x = torch.as_tensor(item["node_x"])
        edge_index = torch.as_tensor(item["edge_index"]).long() + node_offset
        edge_attr = torch.as_tensor(item["edge_attr"])
        graph_x = torch.as_tensor(item["graph_x"])
        band1 = torch.as_tensor(item["band1"])
        has_gap = torch.as_tensor(item["has_gap"])
        gap_target = torch.as_tensor(item["gap_target"])

        node_x_list.append(node_x)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        graph_x_list.append(graph_x)
        band1_list.append(band1)
        has_gap_list.append(has_gap)
        gap_target_list.append(gap_target)
        sample_ids.append(item["sample_index"])
        batch_index_list.append(torch.full((node_x.shape[0],), graph_id, dtype=torch.long))
        node_offset += node_x.shape[0]

    return {
        "node_x": torch.cat(node_x_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "graph_x": torch.stack(graph_x_list, dim=0),
        "band1": torch.stack(band1_list, dim=0),
        "has_gap": torch.cat(has_gap_list, dim=0),
        "gap_target": torch.stack(gap_target_list, dim=0),
        "node_batch": torch.cat(batch_index_list, dim=0),
        "sample_index": torch.tensor(sample_ids, dtype=torch.long),
    }


def scatter_mean(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, values.shape[1], dtype=values.dtype, device=values.device)
    out.index_add_(0, index, values)
    counts = torch.zeros(dim_size, 1, dtype=values.dtype, device=values.device)
    ones = torch.ones(values.shape[0], 1, dtype=values.dtype, device=values.device)
    counts.index_add_(0, index, ones)
    return out / counts.clamp_min(1.0)


def scatter_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, values.shape[1], dtype=values.dtype, device=values.device)
    out.index_add_(0, index, values)
    return out


def scatter_max(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.full((dim_size, values.shape[1]), -torch.inf, dtype=values.dtype, device=values.device)
    expanded_index = index.unsqueeze(1).expand(-1, values.shape[1])
    out.scatter_reduce_(0, expanded_index, values, reduce="amax", include_self=True)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    scores_fp32 = scores.float()
    scores_2d = scores_fp32.unsqueeze(1)
    max_scores = scatter_max(scores_2d, index, dim_size).squeeze(1)
    stabilized = (scores_fp32 - max_scores[index]).clamp(-60.0, 60.0)
    exp_scores = torch.exp(stabilized)
    denom = torch.zeros(dim_size, dtype=exp_scores.dtype, device=exp_scores.device)
    denom.index_add_(0, index, exp_scores)
    weights = exp_scores / denom[index].clamp_min(1e-8)
    return weights.to(scores.dtype)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, expansion: int = 2):
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class GatedEdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.ffn = ResidualMLPBlock(hidden_dim=hidden_dim, dropout=dropout, expansion=2)

    def forward(self, node_x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        node_h = self.node_norm(node_x)
        edge_h = self.edge_norm(edge_attr)
        src, dst = edge_index
        edge_inputs = torch.cat([node_h[src], node_h[dst], edge_h], dim=1)
        gates = torch.sigmoid(self.gate(edge_inputs))
        messages = gates * self.message(edge_inputs)
        aggregated_mean = scatter_mean(messages, dst, node_x.shape[0])
        aggregated_max = scatter_max(messages, dst, node_x.shape[0])
        updated = self.update(torch.cat([node_h, aggregated_mean, aggregated_max], dim=1))
        out = self.out_norm(node_x + updated)
        return self.ffn(out)


class AttentiveGraphPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_h: torch.Tensor, node_batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        with torch.autocast(device_type=node_h.device.type, enabled=False):
            node_fp32 = node_h.float()
            scores = self.score(node_fp32).squeeze(1)
            weights = scatter_softmax(scores, node_batch, batch_size).unsqueeze(1).float()
            pooled = scatter_sum(node_fp32 * weights, node_batch, batch_size)
        return pooled.to(node_h.dtype)


class TaskHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskGapGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        band_points: int,
    ):
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
        self.band_head = TaskHead(hidden_dim=hidden_dim, output_dim=band_points, dropout=dropout)
        self.cls_head = TaskHead(hidden_dim=hidden_dim, output_dim=1, dropout=dropout)
        self.gap_head = TaskHead(hidden_dim=hidden_dim, output_dim=3, dropout=dropout)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_x: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        trunk = self.trunk_refine(self.trunk(graph_embedding))
        cls_logit = self.cls_head(trunk).squeeze(1).clamp(-30.0, 30.0)
        return {
            "band1": self.band_head(trunk),
            "cls_logit": cls_logit,
            "gap": self.gap_head(trunk),
        }


@dataclass
class Scaler:
    mean: torch.Tensor
    std: torch.Tensor

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def to(self, device: torch.device) -> "Scaler":
        return Scaler(mean=self.mean.to(device), std=self.std.to(device))


def build_scaler(values: np.ndarray) -> Scaler:
    tensor = torch.from_numpy(values.astype(np.float32))
    std = tensor.std(dim=0).clamp_min(1e-6)
    return Scaler(mean=tensor.mean(dim=0), std=std)


def move_batch_tensor(batch: dict[str, torch.Tensor], key: str, device: torch.device, non_blocking: bool) -> torch.Tensor:
    return batch[key].to(device, non_blocking=non_blocking)


def resolve_amp_config(device: torch.device, amp_mode: str) -> tuple[bool, torch.dtype, bool, str]:
    if device.type != "cuda" or amp_mode == "off":
        return False, torch.float32, False, "off"
    if amp_mode == "bf16":
        return True, torch.bfloat16, False, "bf16"
    if amp_mode == "fp16":
        return True, torch.float16, True, "fp16"
    if amp_mode == "auto":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False, "bf16"
        return True, torch.float16, True, "fp16"
    raise ValueError(f"Unknown amp_mode: {amp_mode}")


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
    min_lr_scale: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    min_lr_scale = float(max(0.0, min(min_lr_scale, 1.0)))

    def lr_lambda(step: int) -> float:
        step = step + 1
        if step <= warmup_steps:
            return step / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_cls_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def evaluate(
    model: MultiTaskGapGNN,
    loader: DataLoader,
    band_scaler: Scaler,
    gap_scaler: Scaler,
    cls_weight: tuple[float, float] | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    positive_only: bool = False,
) -> dict[str, float]:
    model.eval()
    band_scaler = band_scaler.to(device)
    gap_scaler = gap_scaler.to(device)
    loss_sum = 0.0
    sample_count = 0
    band_mae_sum = 0.0
    gap_mae_sum = torch.zeros(3, dtype=torch.float64)
    gap_positive_count = 0
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
            edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
            edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
            graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
            node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
            band1 = move_batch_tensor(batch, "band1", device, non_blocking=amp_enabled)
            has_gap = move_batch_tensor(batch, "has_gap", device, non_blocking=amp_enabled)
            gap_target = move_batch_tensor(batch, "gap_target", device, non_blocking=amp_enabled)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                band_norm = band_scaler.encode(band1)
                gap_norm = gap_scaler.encode(gap_target)
                band_loss = F.mse_loss(outputs["band1"], band_norm)

            if positive_only:
                cls_loss = outputs["cls_logit"].sum() * 0.0
                positive_mask = torch.ones_like(has_gap, dtype=torch.bool)
            else:
                if cls_weight is None:
                    raise ValueError("cls_weight must not be None when positive_only is False")
                neg_weight, pos_weight = cls_weight
                sample_weights = torch.where(
                    has_gap > 0.5,
                    torch.tensor(pos_weight, device=device),
                    torch.tensor(neg_weight, device=device),
                )
                cls_loss = F.binary_cross_entropy_with_logits(outputs["cls_logit"], has_gap, weight=sample_weights)
                positive_mask = has_gap > 0.5

            if positive_mask.any():
                gap_loss = F.smooth_l1_loss(outputs["gap"][positive_mask], gap_norm[positive_mask])
                gap_pred = gap_scaler.decode(outputs["gap"][positive_mask]).float().cpu()
                gap_true = gap_target[positive_mask].float().cpu()
                gap_mae_sum += torch.abs(gap_pred - gap_true).sum(dim=0).double()
                gap_positive_count += int(positive_mask.sum().item())
            else:
                gap_loss = outputs["gap"].sum() * 0.0

            if positive_only:
                loss = band_loss + gap_loss
            else:
                loss = band_loss + 0.15 * cls_loss + gap_loss
            loss_sum += float(loss.item()) * band1.shape[0]
            sample_count += int(band1.shape[0])

            band_pred = band_scaler.decode(outputs["band1"]).float().cpu()
            band_mae_sum += float(torch.abs(band_pred - band1.float().cpu()).sum().item())
            if not positive_only:
                y_true_list.append(has_gap.float().cpu().numpy())
                y_prob_list.append(torch.sigmoid(outputs["cls_logit"]).float().cpu().numpy())

    metrics = {
        "loss": loss_sum / max(sample_count, 1),
        "band1_mae_khz": band_mae_sum / max(sample_count * band_scaler.mean.numel(), 1),
        "gap_mae_lower_khz": float(gap_mae_sum[0] / max(gap_positive_count, 1)),
        "gap_mae_upper_khz": float(gap_mae_sum[1] / max(gap_positive_count, 1)),
        "gap_mae_width_khz": float(gap_mae_sum[2] / max(gap_positive_count, 1)),
    }
    if not positive_only:
        y_true = np.concatenate(y_true_list).astype(np.int64)
        y_prob = np.concatenate(y_prob_list)
        cls_metrics = compute_cls_metrics(y_true, y_prob)
        metrics.update({f"cls_{key}": value for key, value in cls_metrics.items()})
    return metrics


def write_predictions(
    model: MultiTaskGapGNN,
    loader: DataLoader,
    band_scaler: Scaler,
    gap_scaler: Scaler,
    device: torch.device,
    output_csv: Path,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    positive_only: bool = False,
) -> None:
    model.eval()
    gap_scaler = gap_scaler.to(device)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_index",
                "has_gap_true",
                "has_gap_prob",
                "lower_true_khz",
                "upper_true_khz",
                "width_true_khz",
                "lower_pred_khz",
                "upper_pred_khz",
                "width_pred_khz",
            ]
        )
        with torch.no_grad():
            for batch in loader:
                node_x = move_batch_tensor(batch, "node_x", device, non_blocking=amp_enabled)
                edge_index = move_batch_tensor(batch, "edge_index", device, non_blocking=amp_enabled)
                edge_attr = move_batch_tensor(batch, "edge_attr", device, non_blocking=amp_enabled)
                graph_x = move_batch_tensor(batch, "graph_x", device, non_blocking=amp_enabled)
                node_batch = move_batch_tensor(batch, "node_batch", device, non_blocking=amp_enabled)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                if positive_only:
                    probs = np.ones(batch["sample_index"].shape[0], dtype=np.float32)
                else:
                    probs = torch.sigmoid(outputs["cls_logit"]).float().cpu().numpy()
                gap_pred = gap_scaler.decode(outputs["gap"]).float().cpu().numpy()
                gap_true = batch["gap_target"].numpy()
                has_gap_true = batch["has_gap"].numpy()
                sample_ids = batch["sample_index"].numpy()
                for sid, y, p, truth, pred in zip(sample_ids, has_gap_true, probs, gap_true, gap_pred, strict=True):
                    writer.writerow([int(sid), int(y), float(p), *truth.tolist(), *pred.tolist()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multitask GNN for first-band curve, gap existence, and gap bounds.")
    parser.add_argument("--data-root", type=Path, action="append", dest="data_roots")
    parser.add_argument("--preprocessed-root", type=Path, action="append", dest="preprocessed_roots")
    parser.add_argument("--output-dir", type=Path, default=Path("gnn_multitask_runs"))
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--graph-feature-mode", type=str, default="novel105_phase_edge_shell")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--min-lr-scale", type=float, default=0.10)
    parser.add_argument("--amp-mode", type=str, default="auto", choices=["auto", "fp16", "bf16", "off"])
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--positive-only", action="store_true")
    args = parser.parse_args()
    data_roots = args.data_roots or [Path(r"E:\datasets"), Path(r"E:\datasets\NWW2")]
    preprocessed_roots = args.preprocessed_roots or []

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

    if args.positive_only:
        positive_selector = indexed.has_gap > 0.5
        train_indices = train_indices[positive_selector[train_indices]]
        val_indices = val_indices[positive_selector[val_indices]]
        test_indices = test_indices[positive_selector[test_indices]]

    dataset_cls = CachedMultiTaskDataset if use_preprocessed else MultiTaskDataset
    train_dataset = dataset_cls(indexed, train_indices, cache_size=args.cache_size, feature_indices=feature_indices)
    val_dataset = dataset_cls(indexed, val_indices, cache_size=args.cache_size, feature_indices=feature_indices)
    test_dataset = dataset_cls(indexed, test_indices, cache_size=args.cache_size, feature_indices=feature_indices)

    loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "collate_fn": collate_graphs,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    band_scaler = build_scaler(indexed.band1_targets[train_indices])
    if args.positive_only:
        gap_scaler = build_scaler(indexed.gap_targets[train_indices])
        cls_weight = None
    else:
        positive_train = train_indices[indexed.has_gap[train_indices] > 0.5]
        gap_scaler = build_scaler(indexed.gap_targets[positive_train])

        train_has_gap = indexed.has_gap[train_indices]
        pos_count = float(train_has_gap.sum())
        neg_count = float(len(train_has_gap) - pos_count)
        cls_weight = (
            min(math.sqrt(0.5 * len(train_has_gap) / max(neg_count, 1.0)), 16.0),
            min(math.sqrt(0.5 * len(train_has_gap) / max(pos_count, 1.0)), 2.0),
        )

    model = MultiTaskGapGNN(
        node_dim=11,
        edge_dim=7,
        graph_dim=len(feature_indices),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        band_points=indexed.band1_targets.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    total_steps = max(args.epochs * len(train_loader), 1)
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=args.warmup_fraction,
        min_lr_scale=args.min_lr_scale,
    )
    band_scaler_device = band_scaler.to(device)
    gap_scaler_device = gap_scaler.to(device)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_roots": [str(path) for path in data_roots],
        "preprocessed_roots": [str(path) for path in preprocessed_roots],
        "use_preprocessed": use_preprocessed,
        "device": str(device),
        "graph_feature_mode": args.graph_feature_mode,
        "graph_feature_dim": int(len(feature_indices)),
        "graph_feature_names": selected_feature_names,
        "model_name": "multiscale_gated_gap_gnn_v2_positive_only" if args.positive_only else "multiscale_gated_gap_gnn_v2",
        "positive_only": bool(args.positive_only),
        "amp_mode": resolved_amp_mode,
        "max_batches": args.max_batches,
        "max_samples": args.max_samples,
        "train_samples": int(len(train_dataset)),
        "val_samples": int(len(val_dataset)),
        "test_samples": int(len(test_dataset)),
        "train_gap_ratio": float(indexed.has_gap[train_indices].mean()),
        "val_gap_ratio": float(indexed.has_gap[val_indices].mean()),
        "test_gap_ratio": float(indexed.has_gap[test_indices].mean()),
        "cls_weight": cls_weight,
        "band_scaler_mean_shape": list(band_scaler.mean.shape),
        "gap_scaler_mean": gap_scaler.mean.tolist(),
        "gap_scaler_std": gap_scaler.std.tolist(),
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype),
        "pin_memory": loader_kwargs["pin_memory"],
        "num_workers": args.num_workers,
        "grad_clip": args.grad_clip,
        "warmup_fraction": args.warmup_fraction,
        "min_lr_scale": args.min_lr_scale,
        "total_steps": total_steps,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    best_val = float("inf")
    best_state = None
    history: list[dict[str, float]] = []
    start_time = time.time()
    print("Training...", flush=True)
    for epoch in range(1, args.epochs + 1):
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
            band1 = move_batch_tensor(batch, "band1", device, non_blocking=amp_enabled)
            has_gap = move_batch_tensor(batch, "has_gap", device, non_blocking=amp_enabled)
            gap_target = move_batch_tensor(batch, "gap_target", device, non_blocking=amp_enabled)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                band_norm = band_scaler_device.encode(band1)
                gap_norm = gap_scaler_device.encode(gap_target)

                outputs = model(node_x, edge_index, edge_attr, graph_x, node_batch)
                band_loss = F.mse_loss(outputs["band1"], band_norm)

                if args.positive_only:
                    cls_loss = outputs["cls_logit"].sum() * 0.0
                    gap_loss = F.smooth_l1_loss(outputs["gap"], gap_norm)
                else:
                    if cls_weight is None:
                        raise ValueError("cls_weight must not be None when positive_only is False")
                    neg_weight, pos_weight = cls_weight
                    sample_weights = torch.where(
                        has_gap > 0.5,
                        torch.tensor(pos_weight, device=device),
                        torch.tensor(neg_weight, device=device),
                    )
                    cls_loss = F.binary_cross_entropy_with_logits(outputs["cls_logit"], has_gap, weight=sample_weights)

                    positive_mask = has_gap > 0.5
                    if positive_mask.any():
                        gap_loss = F.smooth_l1_loss(outputs["gap"][positive_mask], gap_norm[positive_mask])
                    else:
                        gap_loss = outputs["gap"].sum() * 0.0

                if args.positive_only:
                    loss = band_loss + gap_loss
                else:
                    loss = band_loss + 0.15 * cls_loss + gap_loss

            if use_grad_scaler:
                previous_scale = grad_scaler.get_scale()
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                if grad_scaler.get_scale() >= previous_scale:
                    scheduler.step()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()

            running_loss += float(loss.item()) * band1.shape[0]
            seen += int(band1.shape[0])
            last_lr = optimizer.param_groups[0]["lr"]

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(
            model,
            val_loader,
            band_scaler,
            gap_scaler,
            cls_weight,
            device,
            amp_enabled,
            amp_dtype,
            positive_only=args.positive_only,
        )
        epoch_summary = {"epoch": epoch, "train_loss": train_loss, "lr": last_lr, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False), flush=True)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "band_scaler_mean": band_scaler.mean,
                    "band_scaler_std": band_scaler.std,
                    "gap_scaler_mean": gap_scaler.mean,
                    "gap_scaler_std": gap_scaler.std,
                    "metadata": metadata,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                run_dir / "best_model.pt",
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        test_loader,
        band_scaler,
        gap_scaler,
        cls_weight,
        device,
        amp_enabled,
        amp_dtype,
        positive_only=args.positive_only,
    )
    write_predictions(
        model,
        test_loader,
        band_scaler,
        gap_scaler,
        device,
        run_dir / "test_predictions.csv",
        amp_enabled,
        amp_dtype,
        positive_only=args.positive_only,
    )

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - start_time,
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
