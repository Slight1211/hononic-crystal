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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from circle_material_oat_uncertainty_order10 import (
    CASES,
    materials_for_case,
    precompute_indicator_matrix,
    solve_target_gap_pair,
)
from circle_uncertainty_order10 import solve_one, summarize, truncated_normal_factors
from node_pwe_repro import EPOXY, STEEL, regular_polygon


_RECIPROCAL_VECTORS: np.ndarray | None = None
_INDICATOR: np.ndarray | None = None
_POINTS_PER_SEGMENT = 15
_TARGET_GAP_INDEX = 3
_K_INDICES: list[int] | None = None


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


def solve_direct_sample(task: dict[str, float | int | str]) -> dict[str, float | int | str]:
    if _RECIPROCAL_VECTORS is None or _INDICATOR is None:
        raise RuntimeError("Worker has not been initialized.")

    case_name = str(task["case"])
    case_label = str(task["case_label"])
    factor = float(task["factor"])
    sample_id = int(task["sample_id"])
    steel, epoxy = materials_for_case(case_name, factor)
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
        "case": case_name,
        "case_label": case_label,
        "sample_id": sample_id,
        "source": "direct_pwe",
        "factor": factor,
        "steel_E": steel.young_modulus,
        "steel_rho": steel.density,
        "steel_mu": steel.shear_modulus,
        "epoxy_E": epoxy.young_modulus,
        "epoxy_rho": epoxy.density,
        "epoxy_mu": epoxy.shear_modulus,
        **gap,
    }


def save_histograms(
    results_by_case: dict[str, list[dict[str, float | int | str]]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(9.2, 8.45), dpi=220)
    metrics = [
        ("lower_khz", "Lower edge", "#2f6f9f"),
        ("upper_khz", "Upper edge", "#5b8c3a"),
        ("width_khz", "Band-gap width", "#b0612c"),
    ]

    for row_index, (case_name, case_label) in enumerate(CASES):
        case_results = results_by_case[case_name]
        for col_index, (key, column_title, color) in enumerate(metrics):
            ax = axes[row_index, col_index]
            values = np.array([float(row[key]) for row in case_results], dtype=float)
            bins = min(18, max(7, int(np.sqrt(values.size))))
            ax.hist(values, bins=bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.35)
            ax.axvline(values.mean(), color="black", linewidth=1.0)
            ax.grid(True, alpha=0.22, linewidth=0.5)
            ax.tick_params(labelsize=8)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            if row_index == 0:
                ax.set_title(column_title, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{case_label}\nCount", fontsize=9)
                ax.text(
                    -0.28,
                    1.06,
                    chr(97 + row_index),
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    clip_on=False,
                )
            if row_index == len(CASES) - 1:
                ax.set_xlabel("Frequency (kHz)", fontsize=9)

    fig.subplots_adjust(left=0.095, right=0.992, top=0.955, bottom=0.065, hspace=0.44, wspace=0.25)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_k_indices(points_per_segment: int, start: int | None, stop: int | None) -> list[int] | None:
    if start is None and stop is None:
        return None
    full_count = 3 * points_per_segment - 2
    first = 0 if start is None else start
    last = full_count - 1 if stop is None else stop
    return list(range(first, last + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct one-at-a-time material uncertainty for the circular scatterer."
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--points-per-segment", type=int, default=15)
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--target-gap-index", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--rel-bound", type=float, default=0.05)
    parser.add_argument("--rel-sigma", type=float, default=0.05 / 3.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--k-index-min", type=int, default=8)
    parser.add_argument("--k-index-max", type=int, default=16)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("uncertainty_material_oat_direct_order10_n300"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(r"C:\Users\Sligh\Downloads\Pic\uncertainty"),
    )
    args = parser.parse_args()

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    factors = truncated_normal_factors(rng, args.samples, args.rel_bound, args.rel_sigma)
    k_indices = make_k_indices(args.points_per_segment, args.k_index_min, args.k_index_max)

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

    tasks: list[dict[str, float | int | str]] = []
    for case_name, case_label in CASES:
        for sample_id, factor in enumerate(factors, start=1):
            tasks.append(
                {
                    "case": case_name,
                    "case_label": case_label,
                    "sample_id": sample_id,
                    "factor": float(factor),
                }
            )

    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.order, args.nodes, args.points_per_segment, args.target_gap_index, k_indices),
    ) as executor:
        futures = [executor.submit(solve_direct_sample, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                elapsed = time.time() - started
                print(
                    f"{completed:4d}/{len(tasks)} direct samples completed "
                    f"({row['case']}, width={float(row['width_khz']):.4f} kHz), "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    rows.sort(key=lambda row: (str(row["case"]), int(row["sample_id"])))
    results_by_case = {case_name: [] for case_name, _ in CASES}
    for row in rows:
        results_by_case[str(row["case"])].append(row)

    statistics = {}
    for case_name, case_label in CASES:
        case_results = results_by_case[case_name]
        statistics[case_name] = {
            "label": case_label,
            "lower_khz": summarize(np.array([float(row["lower_khz"]) for row in case_results])),
            "upper_khz": summarize(np.array([float(row["upper_khz"]) for row in case_results])),
            "width_khz": summarize(np.array([float(row["width_khz"]) for row in case_results])),
        }

    csv_path = args.out_dir / "material_oat_direct_uncertainty_order10_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "settings": {
            "samples_per_case": args.samples,
            "order": args.order,
            "nodes": args.nodes,
            "points_per_segment": args.points_per_segment,
            "bands": args.bands,
            "target_gap_index": args.target_gap_index,
            "seed": args.seed,
            "distribution": "one-at-a-time direct truncated normal sampling",
            "relative_bound": args.rel_bound,
            "relative_sigma": args.rel_sigma,
            "k_indices": k_indices,
            "workers": args.workers,
            "elapsed_seconds": time.time() - started,
            "note": "Only one material property is varied in each case. Each random sample is directly evaluated by node-based PWE; no interpolation is used.",
            "nominal_materials": {
                "steel": asdict(STEEL),
                "steel_poisson_ratio": STEEL.poisson_ratio,
                "epoxy": asdict(EPOXY),
                "epoxy_poisson_ratio": EPOXY.poisson_ratio,
            },
        },
        "baseline": baseline,
        "statistics": statistics,
    }
    summary_path = args.out_dir / "material_oat_direct_uncertainty_order10_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure_pdf = args.figure_dir / "material_oat_uncertainty_order10_n300.pdf"
    figure_png = args.figure_dir / "material_oat_uncertainty_order10_n300.png"
    save_histograms(results_by_case, figure_pdf, figure_png)

    print("\nBaseline:", baseline)
    print(json.dumps(statistics, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_pdf}")
    print(f"wrote {figure_png}")


if __name__ == "__main__":
    main()
