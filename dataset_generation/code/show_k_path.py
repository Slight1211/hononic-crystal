from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from node_pwe_repro import build_k_path


def build_rows(points_per_segment: int) -> list[dict[str, object]]:
    k_path = build_k_path(points_per_segment)
    segment_labels = ["M-Gamma", "Gamma-X", "X-M"]
    rows: list[dict[str, object]] = []

    cumulative = 0.0
    previous = None
    path_index = 0
    for segment_index, segment_label in enumerate(segment_labels):
        for step in range(points_per_segment):
            if segment_index > 0 and step == 0:
                continue

            kx, ky = k_path[path_index]
            if previous is not None:
                dx = float(kx - previous[0])
                dy = float(ky - previous[1])
                cumulative += (dx * dx + dy * dy) ** 0.5

            special = ""
            if segment_index == 0 and step == 0:
                special = "M"
            elif segment_index == 0 and step == points_per_segment - 1:
                special = "Gamma"
            elif segment_index == 1 and step == points_per_segment - 1:
                special = "X"
            elif segment_index == 2 and step == points_per_segment - 1:
                special = "M"

            rows.append(
                {
                    "path_index": path_index,
                    "segment": segment_label,
                    "segment_step": step,
                    "kx_rad_per_m": float(kx),
                    "ky_rad_per_m": float(ky),
                    "cumulative_distance": cumulative,
                    "special_point": special,
                }
            )
            previous = (float(kx), float(ky))
            path_index += 1

    return rows


def infer_points_per_segment(dataset_dir: Path) -> int:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Could not find manifest.json in {dataset_dir}. Please pass --points-per-segment explicitly."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    points_per_segment = config.get("points_per_segment")
    if not isinstance(points_per_segment, int):
        raise ValueError(
            f"manifest.json in {dataset_dir} does not contain an integer config.points_per_segment."
        )
    return points_per_segment


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Show or export the fixed PWE k-path used by the node-PWE dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset directory containing manifest.json.")
    parser.add_argument("--points-per-segment", type=int, default=0, help="Override path density directly.")
    parser.add_argument("--write-csv", type=Path, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    if args.points_per_segment > 0:
        points_per_segment = args.points_per_segment
    elif args.dataset_dir is not None:
        points_per_segment = infer_points_per_segment(args.dataset_dir)
    else:
        raise SystemExit("Provide either --dataset-dir or --points-per-segment.")

    rows = build_rows(points_per_segment)

    print(f"points_per_segment = {points_per_segment}")
    print("high-symmetry path = M -> Gamma -> X -> M")
    print(f"total_k_points = {len(rows)}")
    print("")
    print("path_index,segment,segment_step,kx_rad_per_m,ky_rad_per_m,cumulative_distance,special_point")
    for row in rows:
        print(
            f"{row['path_index']},{row['segment']},{row['segment_step']},"
            f"{row['kx_rad_per_m']:.8f},{row['ky_rad_per_m']:.8f},"
            f"{row['cumulative_distance']:.8f},{row['special_point']}"
        )

    if args.write_csv is not None:
        write_csv(args.write_csv, rows)
        print("")
        print(f"wrote csv to {args.write_csv}")


if __name__ == "__main__":
    main()
