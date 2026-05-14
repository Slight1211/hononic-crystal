from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from node_pwe_repro import (
    CELL_SIZE_M,
    EPOXY,
    STEEL,
    Material,
    build_k_path,
    build_reciprocal_indices,
    first_complete_gap,
    polygon_structure_factor,
    regular_polygon,
    solve_in_plane_bands,
)


def truncated_normal_factors(
    rng: np.random.Generator,
    count: int,
    rel_bound: float,
    rel_sigma: float,
) -> np.ndarray:
    factors = np.empty(count, dtype=float)
    filled = 0
    while filled < count:
        draw = rng.normal(loc=1.0, scale=rel_sigma, size=count - filled)
        draw = draw[(draw >= 1.0 - rel_bound) & (draw <= 1.0 + rel_bound)]
        take = min(draw.size, count - filled)
        if take > 0:
            factors[filled : filled + take] = draw[:take]
            filled += take
    return factors


def precompute_indicator_matrix(order: int, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hk_list = build_reciprocal_indices(order)
    size = len(hk_list)
    indicator_cache = {
        (dh, dk): polygon_structure_factor((dh, dk), vertices)
        for dh in range(-2 * order, 2 * order + 1)
        for dk in range(-2 * order, 2 * order + 1)
    }

    indicator = np.empty((size, size), dtype=np.complex128)
    for row, (h_row, k_row) in enumerate(hk_list):
        for col, (h_col, k_col) in enumerate(hk_list):
            indicator[row, col] = indicator_cache[(h_row - h_col, k_row - k_col)]
    reciprocal_vectors = 2.0 * np.pi * np.array(hk_list, dtype=float) / CELL_SIZE_M
    return reciprocal_vectors, indicator


def coefficient_matrix(
    indicator: np.ndarray,
    background_value: float,
    inclusion_value: float,
) -> np.ndarray:
    return (
        background_value * np.eye(indicator.shape[0], dtype=np.complex128)
        + (inclusion_value - background_value) * indicator
    )


def build_matrices_from_indicator(
    indicator: np.ndarray,
    background: Material,
    inclusion: Material,
) -> dict[str, np.ndarray]:
    rho_matrix = coefficient_matrix(indicator, background.density, inclusion.density)
    mu_inv = coefficient_matrix(indicator, 1.0 / background.lame_mu, 1.0 / inclusion.lame_mu)
    lam_inv = coefficient_matrix(
        indicator,
        1.0 / background.lame_lambda,
        1.0 / inclusion.lame_lambda,
    )
    lam2mu_inv = coefficient_matrix(
        indicator,
        1.0 / (background.lame_lambda + 2.0 * background.lame_mu),
        1.0 / (inclusion.lame_lambda + 2.0 * inclusion.lame_mu),
    )
    return {
        "mass": rho_matrix,
        "mu": np.linalg.inv(mu_inv),
        "lambda": np.linalg.inv(lam_inv),
        "lambda_plus_2mu": np.linalg.inv(lam2mu_inv),
    }


def solve_one(
    reciprocal_vectors: np.ndarray,
    indicator: np.ndarray,
    epoxy: Material,
    steel: Material,
    points_per_segment: int,
    bands_to_keep: int,
    target_gap_index: int | None = None,
) -> dict[str, float | int | None]:
    matrices = build_matrices_from_indicator(indicator, background=epoxy, inclusion=steel)
    bands = solve_in_plane_bands(
        reciprocal_vectors,
        matrices,
        points_per_segment=points_per_segment,
        bands_to_keep=bands_to_keep,
    )
    if target_gap_index is not None:
        band_index = target_gap_index - 1
        lower = float(np.max(bands[:, band_index]))
        upper = float(np.min(bands[:, band_index + 1]))
        width = upper - lower
        return {
            "gap_index": target_gap_index,
            "lower_khz": lower,
            "upper_khz": upper,
            "width_khz": width,
        }
    gap = first_complete_gap(bands)
    width = None
    if gap.lower_khz is not None and gap.upper_khz is not None:
        width = gap.upper_khz - gap.lower_khz
    return {
        "gap_index": gap.gap_index,
        "lower_khz": gap.lower_khz,
        "upper_khz": gap.upper_khz,
        "width_khz": width,
    }


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


def save_histograms(results: list[dict[str, float | int | None]], output_path: Path) -> None:
    lower = np.array([float(row["lower_khz"]) for row in results], dtype=float)
    upper = np.array([float(row["upper_khz"]) for row in results], dtype=float)
    width = np.array([float(row["width_khz"]) for row in results], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), dpi=180)
    for ax, data, title, color in [
        (axes[0], lower, "Lower edge", "#2f6f9f"),
        (axes[1], upper, "Upper edge", "#5b8c3a"),
        (axes[2], width, "Band-gap width", "#b0612c"),
    ]:
        ax.hist(data, bins=min(16, max(6, int(np.sqrt(data.size)))), color=color, alpha=0.82)
        ax.axvline(np.mean(data), color="black", linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo uncertainty check for the circular scatterer using node-based PWE."
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--points-per-segment", type=int, default=15)
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--target-gap-index", type=int, default=3)
    parser.add_argument("--rel-bound", type=float, default=0.05)
    parser.add_argument("--rel-sigma", type=float, default=0.05 / 3.0)
    parser.add_argument("--out-dir", type=Path, default=Path("uncertainty_circle_order10"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
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

    steel_e = truncated_normal_factors(rng, args.samples, args.rel_bound, args.rel_sigma)
    steel_rho = truncated_normal_factors(rng, args.samples, args.rel_bound, args.rel_sigma)
    epoxy_e = truncated_normal_factors(rng, args.samples, args.rel_bound, args.rel_sigma)
    epoxy_rho = truncated_normal_factors(rng, args.samples, args.rel_bound, args.rel_sigma)

    results: list[dict[str, float | int | None]] = []
    for sample_id in range(args.samples):
        steel = Material(
            name="steel",
            density=STEEL.density * steel_rho[sample_id],
            young_modulus=STEEL.young_modulus * steel_e[sample_id],
            shear_modulus=STEEL.shear_modulus,
        )
        epoxy = Material(
            name="epoxy",
            density=EPOXY.density * epoxy_rho[sample_id],
            young_modulus=EPOXY.young_modulus * epoxy_e[sample_id],
            shear_modulus=EPOXY.shear_modulus,
        )
        gap = solve_one(
            reciprocal_vectors,
            indicator,
            epoxy=epoxy,
            steel=steel,
            points_per_segment=args.points_per_segment,
            bands_to_keep=args.bands,
            target_gap_index=args.target_gap_index,
        )
        row = {
            "sample_id": sample_id + 1,
            "steel_E_factor": float(steel_e[sample_id]),
            "steel_rho_factor": float(steel_rho[sample_id]),
            "epoxy_E_factor": float(epoxy_e[sample_id]),
            "epoxy_rho_factor": float(epoxy_rho[sample_id]),
            "steel_E": steel.young_modulus,
            "steel_rho": steel.density,
            "epoxy_E": epoxy.young_modulus,
            "epoxy_rho": epoxy.density,
            **gap,
        }
        results.append(row)
        print(
            f"{sample_id + 1:4d}/{args.samples}: "
            f"{gap['lower_khz']:.4f}-{gap['upper_khz']:.4f} kHz, "
            f"width={gap['width_khz']:.4f} kHz",
            flush=True,
        )

    valid = [row for row in results if row["lower_khz"] is not None and row["upper_khz"] is not None]
    lower = np.array([float(row["lower_khz"]) for row in valid], dtype=float)
    upper = np.array([float(row["upper_khz"]) for row in valid], dtype=float)
    width = np.array([float(row["width_khz"]) for row in valid], dtype=float)

    summary = {
        "settings": {
            "samples": args.samples,
            "valid_samples": len(valid),
            "order": args.order,
            "nodes": args.nodes,
            "points_per_segment": args.points_per_segment,
            "bands": args.bands,
            "seed": args.seed,
            "distribution": "independent truncated normal factors",
            "relative_bound": args.rel_bound,
            "relative_sigma": args.rel_sigma,
            "nominal_materials": {
                "steel": asdict(STEEL),
                "epoxy": asdict(EPOXY),
            },
        },
        "baseline": baseline,
        "statistics": {
            "lower_khz": summarize(lower),
            "upper_khz": summarize(upper),
            "width_khz": summarize(width),
        },
    }

    csv_path = args.out_dir / "circle_uncertainty_order10_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary_path = args.out_dir / "circle_uncertainty_order10_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure_path = args.out_dir / "circle_uncertainty_order10_histograms.png"
    save_histograms(valid, figure_path)

    print("\nBaseline:", baseline)
    print(json.dumps(summary["statistics"], indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
