from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from circle_material_oat_uncertainty_order10 import (
    CASES,
    material_with_young_modulus,
    precompute_indicator_matrix,
    solve_target_gap_pair,
)
from circle_uncertainty_order10 import solve_one
from node_pwe_repro import EPOXY, STEEL, Material, regular_polygon


_RECIPROCAL_VECTORS: np.ndarray | None = None
_INDICATOR: np.ndarray | None = None
_POINTS_PER_SEGMENT = 15
_TARGET_GAP_INDEX = 3
_K_INDICES: list[int] | None = None


FACTOR_NAMES = ["steel_E", "steel_rho", "epoxy_E", "epoxy_rho"]
FACTOR_LABELS = {
    "steel_E": r"Steel $E$",
    "steel_rho": r"Steel $\rho$",
    "epoxy_E": r"Epoxy $E$",
    "epoxy_rho": r"Epoxy $\rho$",
}
OUTPUTS = [
    ("lower_khz", "Lower edge"),
    ("upper_khz", "Upper edge"),
    ("width_khz", "Width"),
]


def init_worker(
    order: int,
    nodes: int,
    points_per_segment: int,
    target_gap_index: int,
    k_indices: list[int] | None,
) -> None:
    global _RECIPROCAL_VECTORS, _INDICATOR, _POINTS_PER_SEGMENT, _TARGET_GAP_INDEX, _K_INDICES
    vertices = regular_polygon(radius_ratio=13.0 / 50.0, nodes=nodes)
    _RECIPROCAL_VECTORS, _INDICATOR = precompute_indicator_matrix(order, vertices)
    _POINTS_PER_SEGMENT = points_per_segment
    _TARGET_GAP_INDEX = target_gap_index
    _K_INDICES = k_indices


def make_materials_from_factors(factors: dict[str, float]) -> tuple[Material, Material]:
    steel = material_with_young_modulus(STEEL, STEEL.young_modulus * factors["steel_E"])
    steel = Material(
        name=steel.name,
        density=STEEL.density * factors["steel_rho"],
        young_modulus=steel.young_modulus,
        shear_modulus=steel.shear_modulus,
    )
    epoxy = material_with_young_modulus(EPOXY, EPOXY.young_modulus * factors["epoxy_E"])
    epoxy = Material(
        name=epoxy.name,
        density=EPOXY.density * factors["epoxy_rho"],
        young_modulus=epoxy.young_modulus,
        shear_modulus=epoxy.shear_modulus,
    )
    return steel, epoxy


def solve_sobol_sample(task: dict[str, float | int | str]) -> dict[str, float | int | str]:
    if _RECIPROCAL_VECTORS is None or _INDICATOR is None:
        raise RuntimeError("Worker has not been initialized.")

    factors = {name: float(task[f"{name}_factor"]) for name in FACTOR_NAMES}
    steel, epoxy = make_materials_from_factors(factors)
    gap = solve_target_gap_pair(
        _RECIPROCAL_VECTORS,
        _INDICATOR,
        epoxy=epoxy,
        steel=steel,
        points_per_segment=_POINTS_PER_SEGMENT,
        target_gap_index=_TARGET_GAP_INDEX,
        k_indices=_K_INDICES,
    )
    return {
        **task,
        "steel_E": steel.young_modulus,
        "steel_rho": steel.density,
        "steel_mu": steel.shear_modulus,
        "epoxy_E": epoxy.young_modulus,
        "epoxy_rho": epoxy.density,
        "epoxy_mu": epoxy.shear_modulus,
        **gap,
    }


def make_k_indices(points_per_segment: int, start: int | None, stop: int | None) -> list[int] | None:
    if start is None and stop is None:
        return None
    full_count = 3 * points_per_segment - 2
    first = 0 if start is None else start
    last = full_count - 1 if stop is None else stop
    return list(range(first, last + 1))


def build_sobol_design(base_samples: int, rel_bound: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    dimension = len(FACTOR_NAMES)
    sampler = qmc.Sobol(d=2 * dimension, scramble=True, seed=seed)
    if base_samples > 0 and (base_samples & (base_samples - 1)) == 0:
        unit = sampler.random_base2(m=int(np.log2(base_samples)))
    else:
        unit = sampler.random(base_samples)
    lower = 1.0 - rel_bound
    upper = 1.0 + rel_bound
    scaled = lower + (upper - lower) * unit
    return scaled[:, :dimension], scaled[:, dimension:]


def build_tasks(a_matrix: np.ndarray, b_matrix: np.ndarray) -> list[dict[str, float | int | str]]:
    tasks: list[dict[str, float | int | str]] = []
    for matrix_name, matrix in [("A", a_matrix), ("B", b_matrix)]:
        for row_index, factors in enumerate(matrix, start=1):
            task: dict[str, float | int | str] = {
                "sample_type": matrix_name,
                "sample_id": row_index,
                "replaced_factor": "",
            }
            for name, value in zip(FACTOR_NAMES, factors):
                task[f"{name}_factor"] = float(value)
            tasks.append(task)

    for factor_index, factor_name in enumerate(FACTOR_NAMES):
        ab_matrix = np.array(a_matrix, copy=True)
        ab_matrix[:, factor_index] = b_matrix[:, factor_index]
        for row_index, factors in enumerate(ab_matrix, start=1):
            task = {
                "sample_type": "AB",
                "sample_id": row_index,
                "replaced_factor": factor_name,
            }
            for name, value in zip(FACTOR_NAMES, factors):
                task[f"{name}_factor"] = float(value)
            tasks.append(task)
    return tasks


def values_for(rows: list[dict[str, float | int | str]], output_key: str) -> np.ndarray:
    return np.array([float(row[output_key]) for row in rows], dtype=float)


def summarize_sobol(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    by_type: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_type.setdefault((str(row["sample_type"]), str(row["replaced_factor"])), []).append(row)
    for value in by_type.values():
        value.sort(key=lambda row: int(row["sample_id"]))

    a_rows = by_type[("A", "")]
    b_rows = by_type[("B", "")]
    output_summary: dict[str, object] = {}
    for output_key, output_label in OUTPUTS:
        y_a = values_for(a_rows, output_key)
        y_b = values_for(b_rows, output_key)
        variance = float(np.var(np.concatenate([y_a, y_b]), ddof=1))
        factor_results = {}
        for factor_name in FACTOR_NAMES:
            y_ab = values_for(by_type[("AB", factor_name)], output_key)
            first_order = float(np.mean(y_b * (y_ab - y_a)) / variance)
            total_order = float(0.5 * np.mean((y_a - y_ab) ** 2) / variance)
            factor_results[factor_name] = {
                "label": FACTOR_LABELS[factor_name],
                "first_order": first_order,
                "total_order": total_order,
            }
        output_summary[output_key] = {
            "label": output_label,
            "mean": float(np.mean(np.concatenate([y_a, y_b]))),
            "std": float(np.std(np.concatenate([y_a, y_b]), ddof=1)),
            "variance": variance,
            "factors": factor_results,
        }
    return output_summary


def save_sobol_figure(summary: dict[str, object], output_pdf: Path, output_png: Path) -> None:
    labels = [FACTOR_LABELS[name].replace("$", "") for name in FACTOR_NAMES]
    x = np.arange(len(FACTOR_NAMES), dtype=float)
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), dpi=220, sharey=True)
    for ax, (output_key, output_label) in zip(axes, OUTPUTS):
        output = summary[output_key]
        factors = output["factors"]  # type: ignore[index]
        s1 = np.array([factors[name]["first_order"] for name in FACTOR_NAMES], dtype=float)  # type: ignore[index]
        st = np.array([factors[name]["total_order"] for name in FACTOR_NAMES], dtype=float)  # type: ignore[index]
        ax.bar(x - width / 2.0, s1, width=width, color="white", edgecolor="black", linewidth=0.8, label=r"$S_i$")
        ax.bar(x + width / 2.0, st, width=width, color="0.55", edgecolor="black", linewidth=0.8, label=r"$T_i$")
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_title(output_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.24, linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Sobol index", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.27, wspace=0.14)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sobol global sensitivity analysis for circular-scatterer material uncertainty."
    )
    parser.add_argument("--base-samples", type=int, default=256)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--points-per-segment", type=int, default=15)
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--target-gap-index", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--rel-bound", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k-index-min", type=int, default=8)
    parser.add_argument("--k-index-max", type=int, default=16)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("uncertainty_material_sobol_order10_n256"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(r"C:\Users\Sligh\Downloads\Pic\uncertainty"),
    )
    args = parser.parse_args()

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_indices = make_k_indices(args.points_per_segment, args.k_index_min, args.k_index_max)
    a_matrix, b_matrix = build_sobol_design(args.base_samples, args.rel_bound, args.seed)
    tasks = build_tasks(a_matrix, b_matrix)

    vertices = regular_polygon(radius_ratio=13.0 / 50.0, nodes=args.nodes)
    reciprocal_vectors, indicator = precompute_indicator_matrix(args.order, vertices)
    baseline = solve_one(
        reciprocal_vectors,
        indicator,
        epoxy=EPOXY,
        steel=STEEL,
        points_per_segment=args.points_per_segment,
        bands_to_keep=args.bands,
        target_gap_index=args.target_gap_index,
    )

    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.order, args.nodes, args.points_per_segment, args.target_gap_index, k_indices),
    ) as executor:
        futures = [executor.submit(solve_sobol_sample, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if completed == 1 or completed % 50 == 0 or completed == len(tasks):
                elapsed = time.time() - started
                print(
                    f"{completed:4d}/{len(tasks)} Sobol samples completed "
                    f"({row['sample_type']} {row['replaced_factor']}, "
                    f"width={float(row['width_khz']):.4f} kHz), elapsed={elapsed:.1f}s",
                    flush=True,
                )

    rows.sort(
        key=lambda row: (
            str(row["sample_type"]),
            str(row["replaced_factor"]),
            int(row["sample_id"]),
        )
    )
    summary = summarize_sobol(rows)
    settings = {
        "base_samples": args.base_samples,
        "evaluations": len(rows),
        "inputs": FACTOR_NAMES,
        "distribution": "independent uniform material factors",
        "relative_bound": args.rel_bound,
        "order": args.order,
        "nodes": args.nodes,
        "points_per_segment": args.points_per_segment,
        "bands": args.bands,
        "target_gap_index": args.target_gap_index,
        "seed": args.seed,
        "k_indices": k_indices,
        "workers": args.workers,
        "elapsed_seconds": time.time() - started,
        "note": "All Sobol design points are directly evaluated by node-based PWE; no interpolation or surrogate model is used.",
        "nominal_materials": {
            "steel": asdict(STEEL),
            "steel_poisson_ratio": STEEL.poisson_ratio,
            "epoxy": asdict(EPOXY),
            "epoxy_poisson_ratio": EPOXY.poisson_ratio,
        },
    }

    csv_path = args.out_dir / "material_sobol_uncertainty_order10_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / "material_sobol_uncertainty_order10_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "settings": settings,
                "baseline": baseline,
                "sobol": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure_pdf = args.figure_dir / f"material_sobol_uncertainty_order10_n{args.base_samples}.pdf"
    figure_png = args.figure_dir / f"material_sobol_uncertainty_order10_n{args.base_samples}.png"
    save_sobol_figure(summary, figure_pdf, figure_png)

    print("\nBaseline:", baseline)
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_pdf}")
    print(f"wrote {figure_png}")


if __name__ == "__main__":
    main()
