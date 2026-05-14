from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

from node_pwe_repro import (
    EPOXY,
    STEEL,
    build_pwe_matrices,
    first_complete_gap,
    polygon_area,
    polygon_structure_factor,
    regular_polygon,
    solve_in_plane_bands,
)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def radial_perturbation_polygon(
    rng: np.random.Generator,
    nodes: int,
    radius_ratio: float,
    modes: int,
    coefficient_sigma: float,
    rel_bound: float,
    max_attempts: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, nodes, endpoint=False)
    base_area = np.pi * radius_ratio**2
    mode_numbers = np.arange(2, modes + 1, dtype=float)

    for _ in range(max_attempts):
        # Smooth manufacturing-like boundary perturbations; higher modes decay
        # quickly to avoid jagged or self-intersecting polygons.
        sigma = coefficient_sigma / (mode_numbers**2)
        cos_coeff = rng.normal(0.0, sigma)
        sin_coeff = rng.normal(0.0, sigma)

        perturbation = np.zeros_like(theta)
        for mode, a_m, b_m in zip(mode_numbers, cos_coeff, sin_coeff):
            perturbation += a_m * np.cos(mode * theta) + b_m * np.sin(mode * theta)

        radial_factor = 1.0 + perturbation
        vertices = np.column_stack(
            [
                radius_ratio * radial_factor * np.cos(theta),
                radius_ratio * radial_factor * np.sin(theta),
            ]
        )
        scale = np.sqrt(base_area / polygon_area(vertices))
        vertices *= scale
        radii = np.linalg.norm(vertices, axis=1)
        rel_after_scale = radii / radius_ratio - 1.0

        if np.max(np.abs(rel_after_scale)) <= rel_bound:
            coeffs = np.column_stack([mode_numbers, cos_coeff, sin_coeff]).reshape(-1)
            return vertices, rel_after_scale, coeffs

    raise RuntimeError("Failed to generate a valid perturbed circle.")


def solve_target_gap_for_polygon(
    vertices: np.ndarray,
    order: int,
    points_per_segment: int,
    bands_to_keep: int,
    target_gap_index: int,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    reciprocal_vectors, matrices = build_pwe_matrices(
        order,
        lambda hk: polygon_structure_factor(hk, vertices),
        background=EPOXY,
        inclusion=STEEL,
    )
    bands = solve_in_plane_bands(
        reciprocal_vectors,
        matrices,
        points_per_segment=points_per_segment,
        bands_to_keep=bands_to_keep,
    )
    first_gap = first_complete_gap(bands)
    band_index = target_gap_index - 1
    lower = float(np.max(bands[:, band_index]))
    upper = float(np.min(bands[:, band_index + 1]))
    return bands, {
        "gap_index": target_gap_index,
        "first_complete_gap_index": first_gap.gap_index,
        "lower_khz": lower,
        "upper_khz": upper,
        "width_khz": upper - lower,
    }


def solve_sample(task: dict[str, int | float]) -> dict[str, float | int | None]:
    sample_id = int(task["sample_id"])
    seed = int(task["seed"])
    rng = np.random.default_rng(seed)
    vertices, rel_perturbation, coeffs = radial_perturbation_polygon(
        rng,
        nodes=int(task["nodes"]),
        radius_ratio=float(task["radius_ratio"]),
        modes=int(task["modes"]),
        coefficient_sigma=float(task["coefficient_sigma"]),
        rel_bound=float(task["rel_bound"]),
    )
    _, gap = solve_target_gap_for_polygon(
        vertices,
        order=int(task["order"]),
        points_per_segment=int(task["points_per_segment"]),
        bands_to_keep=int(task["bands"]),
        target_gap_index=int(task["target_gap_index"]),
    )
    row: dict[str, float | int | None] = {
        "sample_id": sample_id,
        "seed": seed,
        "area_fraction": float(polygon_area(vertices)),
        "max_abs_radial_perturbation": float(np.max(np.abs(rel_perturbation))),
        "rms_radial_perturbation": float(np.sqrt(np.mean(rel_perturbation**2))),
        **gap,
    }
    for index, value in enumerate(coeffs):
        row[f"coeff_{index:02d}"] = float(value)
    return row


def save_histograms(rows: list[dict[str, float | int | None]], output_path: Path) -> None:
    lower = np.array([float(row["lower_khz"]) for row in rows])
    upper = np.array([float(row["upper_khz"]) for row in rows])
    width = np.array([float(row["width_khz"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), dpi=180)
    for ax, data, title, color in [
        (axes[0], lower, "Lower edge", "#2f6f9f"),
        (axes[1], upper, "Upper edge", "#5b8c3a"),
        (axes[2], width, "Band-gap width", "#b0612c"),
    ]:
        ax.hist(data, bins=min(28, max(10, int(np.sqrt(data.size)))), color=color, alpha=0.82)
        ax.axvline(np.mean(data), color="black", linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_example_shapes(
    rows: list[dict[str, float | int | None]],
    output_path: Path,
    *,
    nodes: int,
    radius_ratio: float,
    modes: int,
    coefficient_sigma: float,
    rel_bound: float,
) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row["width_khz"]))
    examples = [
        ("min width", sorted_rows[0]),
        ("median width", sorted_rows[len(sorted_rows) // 2]),
        ("max width", sorted_rows[-1]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0), dpi=180)
    for ax, (label, row) in zip(axes, examples):
        rng = np.random.default_rng(int(row["seed"]))
        vertices, _, _ = radial_perturbation_polygon(
            rng,
            nodes=nodes,
            radius_ratio=radius_ratio,
            modes=modes,
            coefficient_sigma=coefficient_sigma,
            rel_bound=rel_bound,
        )
        closed = np.vstack([vertices, vertices[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#1f4f7a", linewidth=1.5)
        ax.fill(closed[:, 0], closed[:, 1], color="#7aa6c7", alpha=0.45)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{label}\n{float(row['width_khz']):.3f} kHz")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.31, 0.31)
        ax.set_ylim(-0.31, 0.31)
        ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geometry uncertainty analysis for a circular scatterer with smooth random boundary perturbations."
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--points-per-segment", type=int, default=15)
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--target-gap-index", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--radius-ratio", type=float, default=13.0 / 50.0)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--coefficient-sigma", type=float, default=0.08)
    parser.add_argument("--rel-bound", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("uncertainty_geometry_circle_order10_n1000"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    regular_vertices = regular_polygon(args.radius_ratio, args.nodes)
    _, baseline = solve_target_gap_for_polygon(
        regular_vertices,
        order=args.order,
        points_per_segment=args.points_per_segment,
        bands_to_keep=args.bands,
        target_gap_index=args.target_gap_index,
    )

    tasks = [
        {
            "sample_id": sample_id,
            "seed": args.seed + sample_id * 7919,
            "nodes": args.nodes,
            "radius_ratio": args.radius_ratio,
            "modes": args.modes,
            "coefficient_sigma": args.coefficient_sigma,
            "rel_bound": args.rel_bound,
            "order": args.order,
            "points_per_segment": args.points_per_segment,
            "bands": args.bands,
            "target_gap_index": args.target_gap_index,
        }
        for sample_id in range(1, args.samples + 1)
    ]

    rows: list[dict[str, float | int | None]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {executor.submit(solve_sample, task): int(task["sample_id"]) for task in tasks}
        for done_count, future in enumerate(as_completed(future_to_id), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"{done_count:4d}/{args.samples} "
                f"(sample {int(row['sample_id']):4d}): "
                f"{float(row['lower_khz']):.4f}-{float(row['upper_khz']):.4f} kHz, "
                f"width={float(row['width_khz']):.4f} kHz",
                flush=True,
            )

    rows.sort(key=lambda row: int(row["sample_id"]))
    lower = np.array([float(row["lower_khz"]) for row in rows])
    upper = np.array([float(row["upper_khz"]) for row in rows])
    width = np.array([float(row["width_khz"]) for row in rows])
    max_perturb = np.array([float(row["max_abs_radial_perturbation"]) for row in rows])
    rms_perturb = np.array([float(row["rms_radial_perturbation"]) for row in rows])

    first_gap_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["first_complete_gap_index"])
        first_gap_counts[key] = first_gap_counts.get(key, 0) + 1

    summary = {
        "settings": {
            "samples": args.samples,
            "order": args.order,
            "nodes": args.nodes,
            "points_per_segment": args.points_per_segment,
            "bands": args.bands,
            "target_gap_index": args.target_gap_index,
            "radius_ratio": args.radius_ratio,
            "modes": args.modes,
            "coefficient_sigma": args.coefficient_sigma,
            "relative_radial_bound": args.rel_bound,
            "area_normalized": True,
            "workers": args.workers,
            "elapsed_seconds": time.time() - start,
            "first_complete_gap_counts": first_gap_counts,
        },
        "baseline": baseline,
        "statistics": {
            "lower_khz": summarize(lower),
            "upper_khz": summarize(upper),
            "width_khz": summarize(width),
            "max_abs_radial_perturbation": summarize(max_perturb),
            "rms_radial_perturbation": summarize(rms_perturb),
        },
    }

    fieldnames = list(rows[0].keys())
    csv_path = args.out_dir / "circle_geometry_uncertainty_order10_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / "circle_geometry_uncertainty_order10_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    histogram_path = args.out_dir / "circle_geometry_uncertainty_order10_histograms.png"
    save_histograms(rows, histogram_path)

    examples_path = args.out_dir / "circle_geometry_uncertainty_order10_examples.png"
    save_example_shapes(
        rows,
        examples_path,
        nodes=args.nodes,
        radius_ratio=args.radius_ratio,
        modes=args.modes,
        coefficient_sigma=args.coefficient_sigma,
        rel_bound=args.rel_bound,
    )

    print("\nBaseline:", baseline)
    print(json.dumps(summary["statistics"], indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {histogram_path}")
    print(f"wrote {examples_path}")


if __name__ == "__main__":
    main()
