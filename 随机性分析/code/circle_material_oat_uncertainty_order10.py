from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.linalg import eigh

from circle_uncertainty_order10 import (
    build_matrices_from_indicator,
    precompute_indicator_matrix,
    solve_one,
    summarize,
    truncated_normal_factors,
)
from node_pwe_repro import EPOXY, STEEL, Material, build_k_path, regular_polygon


CASES = [
    ("steel_E", "Steel $E$"),
    ("steel_rho", r"Steel $\rho$"),
    ("epoxy_E", "Epoxy $E$"),
    ("epoxy_rho", r"Epoxy $\rho$"),
]


def material_with_young_modulus(material: Material, young_modulus: float) -> Material:
    poisson_ratio = material.poisson_ratio
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    return Material(
        name=material.name,
        density=material.density,
        young_modulus=young_modulus,
        shear_modulus=shear_modulus,
    )


def materials_for_case(case_name: str, factor: float) -> tuple[Material, Material]:
    steel = STEEL
    epoxy = EPOXY
    if case_name == "steel_E":
        steel = material_with_young_modulus(STEEL, STEEL.young_modulus * factor)
    elif case_name == "steel_rho":
        steel = Material(
            name=STEEL.name,
            density=STEEL.density * factor,
            young_modulus=STEEL.young_modulus,
            shear_modulus=STEEL.shear_modulus,
        )
    elif case_name == "epoxy_E":
        epoxy = material_with_young_modulus(EPOXY, EPOXY.young_modulus * factor)
    elif case_name == "epoxy_rho":
        epoxy = Material(
            name=EPOXY.name,
            density=EPOXY.density * factor,
            young_modulus=EPOXY.young_modulus,
            shear_modulus=EPOXY.shear_modulus,
        )
    else:
        raise ValueError(f"Unknown case: {case_name}")
    return steel, epoxy


def solve_target_gap_pair(
    reciprocal_vectors: np.ndarray,
    indicator: np.ndarray,
    epoxy: Material,
    steel: Material,
    points_per_segment: int,
    target_gap_index: int,
    k_indices: list[int] | None = None,
) -> dict[str, float | int]:
    matrices = build_matrices_from_indicator(indicator, background=epoxy, inclusion=steel)
    k_path = build_k_path(points_per_segment)
    if k_indices is not None:
        k_path = k_path[k_indices]
    mass = matrices["mass"]
    mu = matrices["mu"]
    lam = matrices["lambda"]
    lam2mu = matrices["lambda_plus_2mu"]
    zeros = np.zeros_like(mass)
    generalized_mass = np.block([[mass, zeros], [zeros, mass]])

    lower_band: list[float] = []
    upper_band: list[float] = []
    pair = [target_gap_index - 1, target_gap_index]
    for k_vector in k_path:
        shifted = reciprocal_vectors + k_vector
        kx = shifted[:, 0]
        ky = shifted[:, 1]
        p11 = (kx[:, None] * kx[None, :]) * lam2mu + (ky[:, None] * ky[None, :]) * mu
        p22 = (ky[:, None] * ky[None, :]) * lam2mu + (kx[:, None] * kx[None, :]) * mu
        p12 = (kx[:, None] * ky[None, :]) * lam + (ky[:, None] * kx[None, :]) * mu
        stiffness = np.block([[p11, p12], [p12.conj().T, p22]])
        omega_squared = eigh(
            stiffness,
            generalized_mass,
            eigvals_only=True,
            subset_by_index=pair,
        )
        frequencies_khz = np.sqrt(np.maximum(omega_squared, 0.0)) / (2.0 * np.pi * 1e3)
        lower_band.append(float(frequencies_khz[0]))
        upper_band.append(float(frequencies_khz[1]))

    lower = max(lower_band)
    upper = min(upper_band)
    return {
        "gap_index": target_gap_index,
        "lower_khz": lower,
        "upper_khz": upper,
        "width_khz": upper - lower,
    }


def save_case_histograms(
    results_by_case: dict[str, list[dict[str, float | int | str | None]]],
    labels_by_case: dict[str, str],
    output_pdf: Path,
    output_png: Path,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(9.2, 8.4), dpi=220)
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
            ax.axvline(np.mean(values), color="black", linewidth=1.0)
            ax.grid(True, alpha=0.22, linewidth=0.5)
            ax.tick_params(labelsize=8)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            if row_index == 0:
                ax.set_title(column_title, fontsize=10)
            if col_index == 0:
                ax.set_ylabel(f"{chr(97 + row_index)} {case_label}\nCount", fontsize=9)
            if row_index == len(CASES) - 1:
                ax.set_xlabel("Frequency (kHz)", fontsize=9)

    fig.tight_layout(pad=0.9, h_pad=1.1, w_pad=1.0)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-at-a-time material uncertainty for the circular scatterer."
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--points-per-segment", type=int, default=15)
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--target-gap-index", type=int, default=3)
    parser.add_argument("--k-index-min", type=int, default=None)
    parser.add_argument("--k-index-max", type=int, default=None)
    parser.add_argument("--rel-bound", type=float, default=0.05)
    parser.add_argument("--rel-sigma", type=float, default=0.05 / 3.0)
    parser.add_argument(
        "--response-levels",
        type=int,
        default=0,
        help="If positive, solve this many one-at-a-time PWE control levels and interpolate the Monte Carlo samples.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("uncertainty_material_oat_order10_n300"))
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(r"C:\Users\Sligh\Downloads\Pic\uncertainty"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    factors = truncated_normal_factors(rng, args.samples, args.rel_bound, args.rel_sigma)

    vertices = regular_polygon(radius_ratio=13.0 / 50.0, nodes=args.nodes)
    reciprocal_vectors, indicator = precompute_indicator_matrix(args.order, vertices)
    k_indices = None
    if args.k_index_min is not None or args.k_index_max is not None:
        full_count = len(range(0, 3 * args.points_per_segment - 2))
        start = 0 if args.k_index_min is None else args.k_index_min
        stop = full_count - 1 if args.k_index_max is None else args.k_index_max
        k_indices = list(range(start, stop + 1))

    baseline = solve_one(
        reciprocal_vectors,
        indicator,
        epoxy=EPOXY,
        steel=STEEL,
        points_per_segment=args.points_per_segment,
        bands_to_keep=args.bands,
        target_gap_index=args.target_gap_index,
    )

    all_rows: list[dict[str, float | int | str | None]] = []
    response_rows: list[dict[str, float | int | str | None]] = []
    results_by_case: dict[str, list[dict[str, float | int | str | None]]] = {}
    response_factors = None
    if args.response_levels > 0:
        response_factors = np.linspace(1.0 - args.rel_bound, 1.0 + args.rel_bound, args.response_levels)

    for case_index, (case_name, case_label) in enumerate(CASES, start=1):
        case_results: list[dict[str, float | int | str | None]] = []
        if response_factors is not None:
            response_lower: list[float] = []
            response_upper: list[float] = []
            for level_id, response_factor in enumerate(response_factors, start=1):
                steel, epoxy = materials_for_case(case_name, float(response_factor))
                gap = solve_target_gap_pair(
                    reciprocal_vectors,
                    indicator,
                    epoxy=epoxy,
                    steel=steel,
                    points_per_segment=args.points_per_segment,
                    target_gap_index=args.target_gap_index,
                    k_indices=k_indices,
                )
                response_lower.append(float(gap["lower_khz"]))
                response_upper.append(float(gap["upper_khz"]))
                response_rows.append(
                    {
                        "case": case_name,
                        "case_label": case_label,
                        "level_id": level_id,
                        "factor": float(response_factor),
                        **gap,
                    }
                )
            lower_samples = np.interp(factors, response_factors, np.array(response_lower))
            upper_samples = np.interp(factors, response_factors, np.array(response_upper))
            for sample_id, factor in enumerate(factors, start=1):
                steel, epoxy = materials_for_case(case_name, float(factor))
                gap = {
                    "gap_index": args.target_gap_index,
                    "lower_khz": float(lower_samples[sample_id - 1]),
                    "upper_khz": float(upper_samples[sample_id - 1]),
                    "width_khz": float(upper_samples[sample_id - 1] - lower_samples[sample_id - 1]),
                }
                row = {
                    "case": case_name,
                    "case_label": case_label,
                    "sample_id": sample_id,
                    "source": "interpolated_response",
                    "factor": float(factor),
                    "steel_E": steel.young_modulus,
                    "steel_rho": steel.density,
                    "steel_mu": steel.shear_modulus,
                    "epoxy_E": epoxy.young_modulus,
                    "epoxy_rho": epoxy.density,
                    "epoxy_mu": epoxy.shear_modulus,
                    **gap,
                }
                case_results.append(row)
                all_rows.append(row)
        else:
            for sample_id, factor in enumerate(factors, start=1):
                steel, epoxy = materials_for_case(case_name, float(factor))
                gap = solve_target_gap_pair(
                    reciprocal_vectors,
                    indicator,
                    epoxy=epoxy,
                    steel=steel,
                    points_per_segment=args.points_per_segment,
                    target_gap_index=args.target_gap_index,
                    k_indices=k_indices,
                )
                row = {
                    "case": case_name,
                    "case_label": case_label,
                    "sample_id": sample_id,
                    "source": "direct_pwe",
                    "factor": float(factor),
                    "steel_E": steel.young_modulus,
                    "steel_rho": steel.density,
                    "steel_mu": steel.shear_modulus,
                    "epoxy_E": epoxy.young_modulus,
                    "epoxy_rho": epoxy.density,
                    "epoxy_mu": epoxy.shear_modulus,
                    **gap,
                }
                case_results.append(row)
                all_rows.append(row)
        results_by_case[case_name] = case_results
        widths = np.array([float(row["width_khz"]) for row in case_results], dtype=float)
        print(
            f"{case_index}/4 {case_name}: width mean={np.mean(widths):.4f} kHz, "
            f"std={np.std(widths, ddof=1):.4f} kHz",
            flush=True,
        )

    statistics = {}
    for case_name, case_label in CASES:
        case_results = results_by_case[case_name]
        statistics[case_name] = {
            "label": case_label,
            "lower_khz": summarize(np.array([float(row["lower_khz"]) for row in case_results])),
            "upper_khz": summarize(np.array([float(row["upper_khz"]) for row in case_results])),
            "width_khz": summarize(np.array([float(row["width_khz"]) for row in case_results])),
        }

    summary = {
        "settings": {
            "samples_per_case": args.samples,
            "order": args.order,
            "nodes": args.nodes,
            "points_per_segment": args.points_per_segment,
            "bands": args.bands,
            "seed": args.seed,
            "distribution": "one-at-a-time truncated normal factor",
            "relative_bound": args.rel_bound,
            "relative_sigma": args.rel_sigma,
            "k_indices": k_indices,
            "response_levels": args.response_levels,
            "note": "Only one material property is varied in each case. Poisson's ratio is kept fixed when Young's modulus is varied.",
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

    csv_path = args.out_dir / "material_oat_uncertainty_order10_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = args.out_dir / "material_oat_uncertainty_order10_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if response_rows:
        response_path = args.out_dir / "material_oat_uncertainty_order10_response_levels.csv"
        with response_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(response_rows[0].keys()))
            writer.writeheader()
            writer.writerows(response_rows)

    figure_pdf = args.figure_dir / f"material_oat_uncertainty_order10_n{args.samples}.pdf"
    figure_png = args.figure_dir / f"material_oat_uncertainty_order10_n{args.samples}.png"
    save_case_histograms(results_by_case, dict(CASES), figure_pdf, figure_png)

    print("\nBaseline:", baseline)
    print(json.dumps(statistics, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_pdf}")
    print(f"wrote {figure_png}")


if __name__ == "__main__":
    main()
