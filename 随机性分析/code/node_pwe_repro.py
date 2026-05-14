from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class Material:
    name: str
    density: float
    young_modulus: float
    shear_modulus: float

    @property
    def poisson_ratio(self) -> float:
        return self.young_modulus / (2.0 * self.shear_modulus) - 1.0

    @property
    def lame_lambda(self) -> float:
        nu = self.poisson_ratio
        return 2.0 * self.shear_modulus * nu / (1.0 - 2.0 * nu)

    @property
    def lame_mu(self) -> float:
        return self.shear_modulus


@dataclass(frozen=True)
class GapResult:
    label: str
    gap_index: int | None
    lower_khz: float | None
    upper_khz: float | None


STEEL = Material(
    name="steel",
    density=7780.0,
    young_modulus=21.06e10,
    shear_modulus=8.10e10,
)

EPOXY = Material(
    name="epoxy",
    density=1180.0,
    young_modulus=4.35e9,
    shear_modulus=1.59e9,
)

CELL_SIZE_M = 0.05


def sinc_unscaled(x: np.ndarray) -> np.ndarray:
    values = np.ones_like(x, dtype=np.complex128)
    mask = np.abs(x) > 1e-12
    values[mask] = np.sin(x[mask]) / x[mask]
    return values


def polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def centered_square(side_ratio: float) -> np.ndarray:
    half = side_ratio / 2.0
    return np.array(
        [
            [-half, -half],
            [half, -half],
            [half, half],
            [-half, half],
        ],
        dtype=float,
    )


def regular_polygon(radius_ratio: float, nodes: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, nodes, endpoint=False)
    return np.column_stack(
        [radius_ratio * np.cos(angles), radius_ratio * np.sin(angles)]
    )


def circle_pixel_mask(radius_ratio: float, pixels: int) -> np.ndarray:
    xs = (np.arange(pixels) + 0.5) / pixels - 0.5
    ys = 0.5 - (np.arange(pixels) + 0.5) / pixels
    xx, yy = np.meshgrid(xs, ys)
    return xx**2 + yy**2 <= radius_ratio**2


def polygon_structure_factor(hk: tuple[int, int], vertices: np.ndarray) -> complex:
    h, k = hk
    if h == 0 and k == 0:
        return complex(polygon_area(vertices))

    current = np.asarray(vertices, dtype=float)
    nxt = np.roll(current, -1, axis=0)
    delta = nxt - current
    g_hat = 2.0 * np.pi * np.array([h, k], dtype=float)
    phase = np.exp(-0.5j * ((current + nxt) @ g_hat))
    arg = 0.5 * (delta @ g_hat)
    edge_factor = phase * sinc_unscaled(arg)

    # Use the better-conditioned branch to avoid division by values near zero.
    if abs(h) > abs(k):
        return np.sum(1j * delta[:, 1] * edge_factor / (2.0 * np.pi * h))
    return np.sum(-1j * delta[:, 0] * edge_factor / (2.0 * np.pi * k))


def pixel_structure_factor(hk: tuple[int, int], mask: np.ndarray) -> complex:
    occupied = np.argwhere(mask)
    if occupied.size == 0:
        return 0.0 + 0.0j

    pixels = mask.shape[0]
    xs = (occupied[:, 1] + 0.5) / pixels - 0.5
    ys = 0.5 - (occupied[:, 0] + 0.5) / pixels
    phase = np.exp(-2.0j * np.pi * (hk[0] * xs + hk[1] * ys))
    pixel_area_fraction = (1.0 / pixels) ** 2

    # Each occupied cell is treated as an exact square pixel, not a point sample.
    pixel_filter = np.sinc(hk[0] / pixels) * np.sinc(hk[1] / pixels)
    return pixel_area_fraction * pixel_filter * phase.sum()


def build_reciprocal_indices(order: int) -> list[tuple[int, int]]:
    return [(h, k) for h in range(-order, order + 1) for k in range(-order, order + 1)]


def coefficient_matrix_from_indicator(
    order: int,
    background_value: float,
    inclusion_value: float,
    indicator_lookup,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    hk_list = build_reciprocal_indices(order)
    indicator_cache = {
        (dh, dk): indicator_lookup((dh, dk))
        for dh in range(-2 * order, 2 * order + 1)
        for dk in range(-2 * order, 2 * order + 1)
    }

    size = len(hk_list)
    matrix = np.empty((size, size), dtype=np.complex128)
    for row, (h_row, k_row) in enumerate(hk_list):
        for col, (h_col, k_col) in enumerate(hk_list):
            diff = (h_row - h_col, k_row - k_col)
            delta = 1.0 if diff == (0, 0) else 0.0
            matrix[row, col] = (
                background_value * delta
                + (inclusion_value - background_value) * indicator_cache[diff]
            )
    return hk_list, matrix


def build_pwe_matrices(
    order: int,
    indicator_lookup,
    background: Material,
    inclusion: Material,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hk_list, rho_matrix = coefficient_matrix_from_indicator(
        order,
        background.density,
        inclusion.density,
        indicator_lookup,
    )
    _, mu_inv = coefficient_matrix_from_indicator(
        order,
        1.0 / background.lame_mu,
        1.0 / inclusion.lame_mu,
        indicator_lookup,
    )
    _, lam_inv = coefficient_matrix_from_indicator(
        order,
        1.0 / background.lame_lambda,
        1.0 / inclusion.lame_lambda,
        indicator_lookup,
    )
    _, lam2mu_inv = coefficient_matrix_from_indicator(
        order,
        1.0 / (background.lame_lambda + 2.0 * background.lame_mu),
        1.0 / (inclusion.lame_lambda + 2.0 * inclusion.lame_mu),
        indicator_lookup,
    )

    reciprocal_vectors = 2.0 * np.pi * np.array(hk_list, dtype=float) / CELL_SIZE_M
    return reciprocal_vectors, {
        "mass": rho_matrix,
        "mu": np.linalg.inv(mu_inv),
        "lambda": np.linalg.inv(lam_inv),
        "lambda_plus_2mu": np.linalg.inv(lam2mu_inv),
    }


def build_k_path(points_per_segment: int) -> np.ndarray:
    gamma = np.array([0.0, 0.0], dtype=float)
    x_point = np.array([np.pi / CELL_SIZE_M, 0.0], dtype=float)
    m_point = np.array([np.pi / CELL_SIZE_M, np.pi / CELL_SIZE_M], dtype=float)
    corners = [m_point, gamma, x_point, m_point]

    path: list[np.ndarray] = []
    for segment in range(len(corners) - 1):
        start = corners[segment]
        end = corners[segment + 1]
        for step in range(points_per_segment):
            alpha = step / (points_per_segment - 1)
            if segment > 0 and step == 0:
                continue
            path.append((1.0 - alpha) * start + alpha * end)
    return np.asarray(path)


def solve_in_plane_bands(
    reciprocal_vectors: np.ndarray,
    inverse_rule_matrices: dict[str, np.ndarray],
    points_per_segment: int,
    bands_to_keep: int,
) -> np.ndarray:
    k_path = build_k_path(points_per_segment)
    mass = inverse_rule_matrices["mass"]
    mu = inverse_rule_matrices["mu"]
    lam = inverse_rule_matrices["lambda"]
    lam2mu = inverse_rule_matrices["lambda_plus_2mu"]
    zeros = np.zeros_like(mass)

    band_values: list[np.ndarray] = []
    for k_vector in k_path:
        shifted = reciprocal_vectors + k_vector
        kx = shifted[:, 0]
        ky = shifted[:, 1]

        p11 = (kx[:, None] * kx[None, :]) * lam2mu + (ky[:, None] * ky[None, :]) * mu
        p22 = (ky[:, None] * ky[None, :]) * lam2mu + (kx[:, None] * kx[None, :]) * mu
        p12 = (kx[:, None] * ky[None, :]) * lam + (ky[:, None] * kx[None, :]) * mu

        stiffness = np.block([[p11, p12], [p12.conj().T, p22]])
        generalized_mass = np.block([[mass, zeros], [zeros, mass]])

        omega_squared = eigh(
            stiffness,
            generalized_mass,
            eigvals_only=True,
            subset_by_index=[0, bands_to_keep - 1],
        )
        frequencies_khz = np.sqrt(np.maximum(omega_squared, 0.0)) / (2.0 * np.pi * 1e3)
        band_values.append(frequencies_khz)

    return np.asarray(band_values)


def first_complete_gap(bands_khz: np.ndarray, tolerance: float = 1e-6) -> GapResult:
    band_count = bands_khz.shape[1]
    for index in range(band_count - 1):
        lower = float(np.max(bands_khz[:, index]))
        upper = float(np.min(bands_khz[:, index + 1]))
        if upper > lower + tolerance:
            return GapResult(
                label="gap",
                gap_index=index + 1,
                lower_khz=lower,
                upper_khz=upper,
            )
    return GapResult(label="gap", gap_index=None, lower_khz=None, upper_khz=None)


def compute_gap_for_polygon(
    label: str,
    vertices: np.ndarray,
    order: int,
    points_per_segment: int,
    bands_to_keep: int,
    background: Material,
    inclusion: Material,
) -> GapResult:
    reciprocal_vectors, matrices = build_pwe_matrices(
        order,
        lambda hk: polygon_structure_factor(hk, vertices),
        background,
        inclusion,
    )
    bands = solve_in_plane_bands(
        reciprocal_vectors,
        matrices,
        points_per_segment=points_per_segment,
        bands_to_keep=bands_to_keep,
    )
    gap = first_complete_gap(bands)
    return GapResult(label=label, gap_index=gap.gap_index, lower_khz=gap.lower_khz, upper_khz=gap.upper_khz)


def compute_gap_for_pixels(
    label: str,
    mask: np.ndarray,
    order: int,
    points_per_segment: int,
    bands_to_keep: int,
    background: Material,
    inclusion: Material,
) -> GapResult:
    reciprocal_vectors, matrices = build_pwe_matrices(
        order,
        lambda hk: pixel_structure_factor(hk, mask),
        background,
        inclusion,
    )
    bands = solve_in_plane_bands(
        reciprocal_vectors,
        matrices,
        points_per_segment=points_per_segment,
        bands_to_keep=bands_to_keep,
    )
    gap = first_complete_gap(bands)
    return GapResult(label=label, gap_index=gap.gap_index, lower_khz=gap.lower_khz, upper_khz=gap.upper_khz)


def run_benchmarks(
    order: int,
    points_per_segment: int,
    bands_to_keep: int,
    background: Material,
    inclusion: Material,
) -> dict[str, list[GapResult]]:
    results: dict[str, list[GapResult]] = {"square": [], "circle_nodes": [], "circle_pixels": []}

    square_vertices = centered_square(side_ratio=26.0 / 50.0)
    results["square"].append(
        compute_gap_for_polygon(
            label="square_4_nodes",
            vertices=square_vertices,
            order=order,
            points_per_segment=points_per_segment,
            bands_to_keep=bands_to_keep,
            background=background,
            inclusion=inclusion,
        )
    )

    radius_ratio = 13.0 / 50.0
    for node_count in [8, 24, 72, 96]:
        results["circle_nodes"].append(
            compute_gap_for_polygon(
                label=f"circle_nodes_{node_count}",
                vertices=regular_polygon(radius_ratio=radius_ratio, nodes=node_count),
                order=order,
                points_per_segment=points_per_segment,
                bands_to_keep=bands_to_keep,
                background=background,
                inclusion=inclusion,
            )
        )

    for pixels in [25, 50, 100, 200]:
        results["circle_pixels"].append(
            compute_gap_for_pixels(
                label=f"circle_pixels_{pixels}",
                mask=circle_pixel_mask(radius_ratio=radius_ratio, pixels=pixels),
                order=order,
                points_per_segment=points_per_segment,
                bands_to_keep=bands_to_keep,
                background=background,
                inclusion=inclusion,
            )
        )

    return results


def format_gap(result: GapResult) -> str:
    if result.lower_khz is None or result.upper_khz is None or result.gap_index is None:
        return f"{result.label:18s} no complete gap detected"
    return (
        f"{result.label:18s} gap#{result.gap_index:<2d} "
        f"{result.lower_khz:7.2f}-{result.upper_khz:7.2f} kHz"
    )


def dump_json(results: dict[str, list[GapResult]], output_path: Path) -> None:
    serializable = {
        group: [asdict(item) for item in group_results]
        for group, group_results in results.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python reproduction of the node-based in-plane PWE benchmark."
    )
    parser.add_argument("--order", type=int, default=7, help="Plane-wave truncation order.")
    parser.add_argument(
        "--points-per-segment",
        type=int,
        default=15,
        help="Bloch-path samples per segment.",
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=12,
        help="Number of in-plane bands to compute at each k-point.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for machine-readable benchmark output.",
    )
    args = parser.parse_args()

    results = run_benchmarks(
        order=args.order,
        points_per_segment=args.points_per_segment,
        bands_to_keep=args.bands,
        background=EPOXY,
        inclusion=STEEL,
    )

    print("Node-based PWE reproduction (in-plane modes only, inverse rule on stiffness blocks)")
    print(f"cell size = {CELL_SIZE_M * 1e3:.1f} mm, background = {EPOXY.name}, inclusion = {STEEL.name}")
    print(f"plane-wave order = {args.order}, k samples/segment = {args.points_per_segment}")
    print("")
    for section in ["square", "circle_nodes", "circle_pixels"]:
        print(section)
        for item in results[section]:
            print(f"  {format_gap(item)}")
        print("")

    if args.json_out is not None:
        dump_json(results, args.json_out)
        print(f"wrote json results to {args.json_out}")


if __name__ == "__main__":
    main()
