from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from node_pwe_repro import (
    EPOXY,
    STEEL,
    build_pwe_matrices,
    build_k_path,
    solve_in_plane_bands,
    first_complete_gap,
    polygon_structure_factor,
)


@dataclass(frozen=True)
class GenerationConfig:
    total_samples: int = 5000
    chunk_size: int = 250
    order: int = 7
    points_per_segment: int = 21
    bands_to_keep: int = 12
    min_sector_vertices: int = 1
    max_sector_vertices: int = 8
    min_area_fraction: float = 0.04
    max_area_fraction: float = 0.72
    sampling_mode: str = "radial_sector"
    base_seed: int = 20260402
    cpu_fraction: float = 0.30
    max_workers: int = 0
    start_index: int = 0
    batch_id: str = ""
    output_dir: str = ""
    resume: bool = False


def polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def unique_points(points: list[tuple[float, float]], tol: float = 1e-10) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for point in points:
        if not any(abs(point[0] - q[0]) < tol and abs(point[1] - q[1]) < tol for q in unique):
            unique.append(point)
    return unique


def build_polygon_from_sector(sector: list[tuple[float, float]]) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for x, y in sector:
        points.extend(
            [
                (x, y),
                (y, x),
                (-y, x),
                (-x, y),
                (-x, -y),
                (-y, -x),
                (y, -x),
                (x, -y),
            ]
        )

    deduped = unique_points(points)
    deduped.sort(key=lambda p: math.atan2(p[1], p[0]))
    return np.asarray(deduped, dtype=float)


def sample_sector_points_radial(rng: random.Random, count: int) -> list[tuple[float, float]]:
    angles = sorted(rng.uniform(0.03, math.pi / 4.0 - 0.03) for _ in range(count))
    radii = [rng.uniform(0.10, 0.42) for _ in range(count)]
    return [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]


def sample_sector_points_square_bounded(rng: random.Random, count: int) -> list[tuple[float, float]]:
    # Sample inside the first 45-degree wedge of the square cell:
    # 0 < y < x < 0.5. After symmetry expansion, all vertices remain in the square,
    # so the geometry is no longer implicitly capped by a radius-0.42 circle.
    coord_min = 0.01
    coord_max = 0.49
    diag_margin = 0.01
    sector: list[tuple[float, float]] = []
    attempts = 0
    while len(sector) < count and attempts < 5000:
        attempts += 1
        x = rng.uniform(coord_min, coord_max)
        y = rng.uniform(coord_min, coord_max)
        if y > x - diag_margin:
            continue
        sector.append((x, y))
    if len(sector) != count:
        raise RuntimeError("Failed to sample square-bounded sector points.")
    sector.sort(key=lambda p: math.atan2(p[1], p[0]))
    return sector


def sample_random_polygon(seed: int, cfg: GenerationConfig) -> np.ndarray:
    rng = random.Random(seed)
    mode = cfg.sampling_mode.strip().lower()
    for _ in range(2000):
        count = rng.randint(cfg.min_sector_vertices, cfg.max_sector_vertices)
        if mode == "square_bounded":
            sector = sample_sector_points_square_bounded(rng, count)
        else:
            sector = sample_sector_points_radial(rng, count)
        vertices = build_polygon_from_sector(sector)
        area_fraction = polygon_area(vertices)
        if cfg.min_area_fraction <= area_fraction <= cfg.max_area_fraction:
            return vertices
    raise RuntimeError("Failed to sample a valid polygon.")


def solve_sample(sample_index: int, cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = GenerationConfig(**cfg_dict)
    sample_seed = cfg.base_seed + sample_index
    vertices = sample_random_polygon(sample_seed, cfg)
    reciprocal_vectors, matrices = build_pwe_matrices(
        cfg.order,
        lambda hk: polygon_structure_factor(hk, vertices),
        EPOXY,
        STEEL,
    )
    bands_khz = solve_in_plane_bands(
        reciprocal_vectors,
        matrices,
        points_per_segment=cfg.points_per_segment,
        bands_to_keep=cfg.bands_to_keep,
    )
    gap = first_complete_gap(bands_khz)
    return {
        "sample_index": sample_index,
        "seed": sample_seed,
        "vertices": vertices,
        "bands_khz": bands_khz,
        "gap_index": gap.gap_index,
        "lower_khz": gap.lower_khz,
        "upper_khz": gap.upper_khz,
        "gap_width_khz": None if gap.lower_khz is None or gap.upper_khz is None else gap.upper_khz - gap.lower_khz,
        "has_gap": gap.gap_index is not None,
        "area_fraction": polygon_area(vertices),
        "node_count": int(vertices.shape[0]),
    }


def resolve_workers(cfg: GenerationConfig) -> int:
    if cfg.max_workers > 0:
        return cfg.max_workers
    logical = os.cpu_count() or 1
    return max(1, math.floor(logical * cfg.cpu_fraction))


def save_chunk(output_dir: Path, chunk_index: int, records: list[dict[str, Any]]) -> Path:
    records = sorted(records, key=lambda item: int(item["sample_index"]))
    path = output_dir / f"pilot_chunk_{chunk_index:04d}.pkl"
    with path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_existing_chunk_info(output_dir: Path) -> tuple[int, int]:
    chunk_paths = sorted(output_dir.glob("pilot_chunk_*.pkl"))
    if not chunk_paths:
        return 0, 1

    completed = 0
    max_chunk_index = 0
    for path in chunk_paths:
        with path.open("rb") as f:
            records = pickle.load(f)
        completed += len(records)
        try:
            idx = int(path.stem.split("_")[-1])
            max_chunk_index = max(max_chunk_index, idx)
        except Exception:
            pass
    return completed, max_chunk_index + 1


def write_progress(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_k_path_rows(points_per_segment: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    k_path = build_k_path(points_per_segment)
    corners = ["M", "Gamma", "X", "M"]
    segment_labels = ["M-Gamma", "Gamma-X", "X-M"]
    rows: list[dict[str, Any]] = []

    cumulative = 0.0
    previous = None
    path_index = 0
    for segment_index, segment_label in enumerate(segment_labels):
        for step in range(points_per_segment):
            if segment_index > 0 and step == 0:
                continue

            kx, ky = k_path[path_index]
            if previous is not None:
                cumulative += float(np.linalg.norm(k_path[path_index] - previous))

            special_point = ""
            if segment_index == 0 and step == 0:
                special_point = corners[0]
            elif segment_index == 0 and step == points_per_segment - 1:
                special_point = corners[1]
            elif segment_index == 1 and step == points_per_segment - 1:
                special_point = corners[2]
            elif segment_index == 2 and step == points_per_segment - 1:
                special_point = corners[3]

            rows.append(
                {
                    "path_index": path_index,
                    "segment": segment_label,
                    "segment_step": step,
                    "kx_rad_per_m": float(kx),
                    "ky_rad_per_m": float(ky),
                    "cumulative_distance": cumulative,
                    "special_point": special_point,
                }
            )
            previous = k_path[path_index]
            path_index += 1

    return k_path, rows


def write_k_path_files(output_dir: Path, points_per_segment: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    k_path, rows = build_k_path_rows(points_per_segment)
    csv_path = output_dir / "k_path.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path_index",
                "segment",
                "segment_step",
                "kx_rad_per_m",
                "ky_rad_per_m",
                "cumulative_distance",
                "special_point",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return k_path, rows


def load_config_defaults(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config JSON must contain an object: {path}")

    known = set(GenerationConfig.__dataclass_fields__.keys())
    return {key: value for key, value in payload.items() if key in known}


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pilot node-PWE dataset with Python multiprocessing.")
    parser.add_argument("--config-json", type=str, default="", help="Optional JSON file providing default arguments.")
    parser.add_argument("--total-samples", type=int, default=defaults.get("total_samples", 5000))
    parser.add_argument("--chunk-size", type=int, default=defaults.get("chunk_size", 250))
    parser.add_argument("--order", type=int, default=defaults.get("order", 7))
    parser.add_argument("--points-per-segment", type=int, default=defaults.get("points_per_segment", 21))
    parser.add_argument("--bands-to-keep", type=int, default=defaults.get("bands_to_keep", 12))
    parser.add_argument("--min-sector-vertices", type=int, default=defaults.get("min_sector_vertices", 1))
    parser.add_argument("--max-sector-vertices", type=int, default=defaults.get("max_sector_vertices", 8))
    parser.add_argument("--min-area-fraction", type=float, default=defaults.get("min_area_fraction", 0.04))
    parser.add_argument("--max-area-fraction", type=float, default=defaults.get("max_area_fraction", 0.72))
    parser.add_argument("--sampling-mode", type=str, default=defaults.get("sampling_mode", "radial_sector"))
    parser.add_argument("--cpu-fraction", type=float, default=defaults.get("cpu_fraction", 0.30))
    parser.add_argument("--max-workers", type=int, default=defaults.get("max_workers", 0))
    parser.add_argument("--base-seed", type=int, default=defaults.get("base_seed", 20260402))
    parser.add_argument("--start-index", type=int, default=defaults.get("start_index", 0))
    parser.add_argument("--batch-id", type=str, default=defaults.get("batch_id", ""))
    parser.add_argument("--output-dir", type=str, default=defaults.get("output_dir", ""))
    parser.add_argument("--resume", action="store_true", default=bool(defaults.get("resume", False)))
    return parser


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config-json", type=str, default="")
    bootstrap_args, _ = bootstrap.parse_known_args()

    defaults: dict[str, Any] = {}
    if bootstrap_args.config_json:
        defaults = load_config_defaults(bootstrap_args.config_json)

    parser = build_parser(defaults)
    args = parser.parse_args()

    cfg = GenerationConfig(
        total_samples=args.total_samples,
        chunk_size=args.chunk_size,
        order=args.order,
        points_per_segment=args.points_per_segment,
        bands_to_keep=args.bands_to_keep,
        min_sector_vertices=args.min_sector_vertices,
        max_sector_vertices=args.max_sector_vertices,
        min_area_fraction=args.min_area_fraction,
        max_area_fraction=args.max_area_fraction,
        sampling_mode=args.sampling_mode,
        cpu_fraction=args.cpu_fraction,
        max_workers=args.max_workers,
        base_seed=args.base_seed,
        start_index=args.start_index,
        batch_id=args.batch_id,
        output_dir=args.output_dir,
        resume=args.resume,
    )

    root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        batch_tag = f"_{cfg.batch_id}" if cfg.batch_id else ""
        index_tag = f"_idx{cfg.start_index}to{cfg.start_index + cfg.total_samples - 1}"
        output_dir = root / (
            f"pilot_dataset_py_{cfg.total_samples}_nodes{cfg.min_sector_vertices}to{cfg.max_sector_vertices}"
            f"_order{cfg.order}_k{cfg.points_per_segment}{batch_tag}{index_tag}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_count, next_chunk_index = load_existing_chunk_info(output_dir)
    if existing_count > 0 and not cfg.resume:
        raise RuntimeError(
            f"Output directory already has {existing_count} samples. Use --resume to continue: {output_dir}"
        )

    workers = resolve_workers(cfg)
    k_path, k_path_rows = write_k_path_files(output_dir, cfg.points_per_segment)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "resolved_workers": workers,
        "python_pid": os.getpid(),
        "resume": cfg.resume,
        "existing_samples": existing_count,
        "k_path_file": "k_path.csv",
        "k_path_labels": ["M", "Gamma", "X", "M"],
        "k_path": k_path.tolist(),
        "k_path_rows": k_path_rows,
        "global_start_index": cfg.start_index,
        "global_end_index": cfg.start_index + cfg.total_samples - 1,
        "batch_id": cfg.batch_id,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OutputDir: {output_dir}", flush=True)
    if cfg.batch_id:
        print(f"BatchId: {cfg.batch_id}", flush=True)
    print(f"Workers: {workers}", flush=True)
    print(f"Total samples: {cfg.total_samples}", flush=True)
    print(f"Chunk size: {cfg.chunk_size}", flush=True)
    print(f"Global sample range: {cfg.start_index}..{cfg.start_index + cfg.total_samples - 1}", flush=True)
    print(f"Resume: {cfg.resume}", flush=True)
    print(f"Existing samples: {existing_count}", flush=True)

    cfg_dict = asdict(cfg)
    buffer: list[dict[str, Any]] = []
    completed = existing_count
    chunk_index = next_chunk_index
    start = time.time()

    if completed >= cfg.total_samples:
        summary = {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "total_samples": cfg.total_samples,
            "resolved_workers": workers,
            "elapsed_seconds": 0.0,
            "chunks_written": chunk_index - 1,
            "output_dir": str(output_dir),
            "completed_samples": completed,
            "global_start_index": cfg.start_index,
            "global_end_index": cfg.start_index + cfg.total_samples - 1,
            "batch_id": cfg.batch_id,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("dataset already complete", flush=True)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        sample_range_start = cfg.start_index + existing_count
        sample_range_end = cfg.start_index + cfg.total_samples
        futures = [executor.submit(solve_sample, i, cfg_dict) for i in range(sample_range_start, sample_range_end)]
        for future in as_completed(futures):
            record = future.result()
            buffer.append(record)
            completed += 1

            if len(buffer) >= cfg.chunk_size:
                path = save_chunk(output_dir, chunk_index, buffer)
                elapsed = time.time() - start
                generated_now = completed - existing_count
                avg = elapsed / max(generated_now, 1)
                remain = cfg.total_samples - completed
                eta = remain * avg
                write_progress(
                    output_dir,
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "completed_samples": completed,
                        "total_samples": cfg.total_samples,
                        "generated_this_run": generated_now,
                        "chunk_index_written": chunk_index,
                        "last_chunk": path.name,
                        "avg_seconds_per_sample_this_run": avg,
                        "eta_minutes": eta / 60.0,
                        "global_start_index": cfg.start_index,
                        "global_end_index": cfg.start_index + cfg.total_samples - 1,
                        "batch_id": cfg.batch_id,
                    },
                )
                print(
                    f"saved {path.name} | completed {completed}/{cfg.total_samples} | "
                    f"avg {avg:.3f}s/sample | ETA {eta/60:.1f} min"
                , flush=True)
                buffer = []
                chunk_index += 1

    if buffer:
        path = save_chunk(output_dir, chunk_index, buffer)
        write_progress(
            output_dir,
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "completed_samples": completed,
                "total_samples": cfg.total_samples,
                "generated_this_run": completed - existing_count,
                "chunk_index_written": chunk_index,
                "last_chunk": path.name,
                "avg_seconds_per_sample_this_run": (time.time() - start) / max(completed - existing_count, 1),
                "eta_minutes": 0.0,
                "global_start_index": cfg.start_index,
                "global_end_index": cfg.start_index + cfg.total_samples - 1,
                "batch_id": cfg.batch_id,
            },
        )
        print(f"saved {path.name} | completed {completed}/{cfg.total_samples}", flush=True)

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "total_samples": cfg.total_samples,
        "resolved_workers": workers,
        "elapsed_seconds": time.time() - start,
        "chunks_written": chunk_index,
        "output_dir": str(output_dir),
        "completed_samples": completed,
        "resume": cfg.resume,
        "existing_samples_at_start": existing_count,
        "global_start_index": cfg.start_index,
        "global_end_index": cfg.start_index + cfg.total_samples - 1,
        "batch_id": cfg.batch_id,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("dataset generation finished", flush=True)


if __name__ == "__main__":
    main()
