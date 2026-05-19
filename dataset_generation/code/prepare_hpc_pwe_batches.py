from __future__ import annotations

import argparse
import csv
import json
import math
import stat
from datetime import datetime
from pathlib import Path


def write_text(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    if executable:
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_run_batch_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bash run_batch_from_json.sh <config-json> [extra generator args...]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="$1"
shift || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" generate_pilot_dataset_py.py --config-json "$CONFIG_PATH" --resume "$@"
"""


def build_slurm_script(plan_dir_name: str, batch_count: int) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=node-pwe-500k
#SBATCH --array=1-{batch_count}
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
CONFIG_LIST="$SCRIPT_DIR/{plan_dir_name}/config_paths.txt"
CONFIG_RELATIVE_PATH="$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$CONFIG_LIST")"

if [ -z "$CONFIG_RELATIVE_PATH" ]; then
  echo "No config found for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"
bash "$SCRIPT_DIR/run_batch_from_json.sh" "$SCRIPT_DIR/$CONFIG_RELATIVE_PATH"
"""


def build_pbs_script(plan_dir_name: str, batch_count: int) -> str:
    return f"""#!/usr/bin/env bash
#PBS -N node-pwe-500k
#PBS -J 1-{batch_count}
#PBS -l select=1:ncpus=12
#PBS -l walltime=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARRAY_ID="${{PBS_ARRAY_INDEX:-${{PBS_ARRAYID:-}}}}"
CONFIG_LIST="$SCRIPT_DIR/{plan_dir_name}/config_paths.txt"
CONFIG_RELATIVE_PATH="$(sed -n "${{ARRAY_ID}}p" "$CONFIG_LIST")"

if [ -z "$CONFIG_RELATIVE_PATH" ]; then
  echo "No config found for array index=${{ARRAY_ID}}"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"
bash "$SCRIPT_DIR/run_batch_from_json.sh" "$SCRIPT_DIR/$CONFIG_RELATIVE_PATH"
"""


def build_readme(plan_dir_name: str, batch_count: int) -> str:
    return f"""# HPC Node-PWE Bundle

This folder is ready to upload to an HPC cluster for large-scale node-PWE dataset generation.

## What is included

- `generate_pilot_dataset_py.py`: the Python dataset generator
- `node_pwe_repro.py`: the PWE solver
- `show_k_path.py`: exports or prints the fixed `M -> Gamma -> X -> M` k-path
- `prepare_hpc_pwe_batches.py`: regenerates batch plans if you want a different scale
- `run_batch_from_json.sh`: generic batch runner
- `submit_slurm_array.sh`: SLURM array template
- `submit_pbs_array.sh`: PBS array template
- `{plan_dir_name}/`: the current batch plan and per-batch JSON configs

## Current plan

- Total samples: `500000`
- Batch size: `5000`
- Batch count: `{batch_count}`
- Default `order`: `7`
- Default `points_per_segment`: `21`
- Default `bands_to_keep`: `12`
- Default node range: `1-8`

## Recommended usage

### Option 1: run one batch manually

```bash
bash run_batch_from_json.sh {plan_dir_name}/batch_configs/batch_0001.json
```

### Option 2: submit as a SLURM array

Before submitting, adjust the scheduler header in `submit_slurm_array.sh` if needed.

```bash
sbatch submit_slurm_array.sh
```

### Option 3: submit as a PBS array

Before submitting, adjust the scheduler header in `submit_pbs_array.sh` if needed.

```bash
qsub submit_pbs_array.sh
```

## Notes

- Each batch writes to its own output directory under `{plan_dir_name}/datasets/`.
- `sample_index` is globally unique across batches.
- Seeds are generated as `base_seed + sample_index`, so batches can be resumed independently.
- `k_path.csv` is written into each dataset directory.
- The runner script forces `OMP/MKL/OPENBLAS/NUMEXPR` threads to `1` by default, which is the safer choice on HPC.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a batch plan for large-scale node-PWE HPC generation.")
    parser.add_argument("--global-total-samples", type=int, default=500000)
    parser.add_argument("--batch-samples", type=int, default=5000)
    parser.add_argument("--order", type=int, default=7)
    parser.add_argument("--points-per-segment", type=int, default=21)
    parser.add_argument("--bands-to-keep", type=int, default=12)
    parser.add_argument("--min-sector-vertices", type=int, default=1)
    parser.add_argument("--max-sector-vertices", type=int, default=8)
    parser.add_argument("--sampling-mode", type=str, default="radial_sector")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--cpu-fraction", type=float, default=0.30)
    parser.add_argument("--base-seed", type=int, default=20260402)
    parser.add_argument("--plan-root", type=Path, default=Path("hpc_500k_plan"))
    parser.add_argument("--plan-dir-name", type=str, default="hpc_500k_plan")
    args = parser.parse_args()

    batch_count = math.ceil(args.global_total_samples / args.batch_samples)
    plan_root = args.plan_root
    configs_dir = plan_root / "batch_configs"
    datasets_dir = plan_root / "datasets"
    configs_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    config_paths: list[str] = []
    for batch_number in range(batch_count):
        start_index = batch_number * args.batch_samples
        remaining = args.global_total_samples - start_index
        total_samples = min(args.batch_samples, remaining)
        batch_id = f"batch_{batch_number + 1:04d}"
        config_rel = Path(args.plan_dir_name) / "batch_configs" / f"{batch_id}.json"
        output_rel = Path(args.plan_dir_name) / "datasets" / batch_id
        config = {
            "total_samples": total_samples,
            "chunk_size": args.chunk_size,
            "order": args.order,
            "points_per_segment": args.points_per_segment,
            "bands_to_keep": args.bands_to_keep,
            "min_sector_vertices": args.min_sector_vertices,
            "max_sector_vertices": args.max_sector_vertices,
            "sampling_mode": args.sampling_mode,
            "base_seed": args.base_seed,
            "cpu_fraction": args.cpu_fraction,
            "max_workers": args.max_workers,
            "start_index": start_index,
            "batch_id": batch_id,
            "output_dir": output_rel.as_posix(),
            "resume": True,
        }
        config_path = plan_root / "batch_configs" / f"{batch_id}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        config_paths.append(config_rel.as_posix())
        rows.append(
            {
                "batch_number": batch_number + 1,
                "batch_id": batch_id,
                "start_index": start_index,
                "end_index": start_index + total_samples - 1,
                "total_samples": total_samples,
                "config_path": config_rel.as_posix(),
                "output_dir": output_rel.as_posix(),
                "max_workers": args.max_workers,
            }
        )

    manifest_path = plan_root / "batch_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_number",
                "batch_id",
                "start_index",
                "end_index",
                "total_samples",
                "config_path",
                "output_dir",
                "max_workers",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    config_paths_path = plan_root / "config_paths.txt"
    config_paths_path.write_text("\n".join(config_paths) + "\n", encoding="utf-8")

    batch_plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "global_total_samples": args.global_total_samples,
        "batch_samples": args.batch_samples,
        "batch_count": batch_count,
        "order": args.order,
        "points_per_segment": args.points_per_segment,
        "bands_to_keep": args.bands_to_keep,
        "min_sector_vertices": args.min_sector_vertices,
        "max_sector_vertices": args.max_sector_vertices,
        "chunk_size": args.chunk_size,
        "max_workers": args.max_workers,
        "base_seed": args.base_seed,
        "plan_root": str(plan_root),
    }
    (plan_root / "batch_plan.json").write_text(json.dumps(batch_plan, indent=2), encoding="utf-8")

    script_root = plan_root.parent
    write_text(script_root / "run_batch_from_json.sh", build_run_batch_script(), executable=True)
    write_text(script_root / "submit_slurm_array.sh", build_slurm_script(args.plan_dir_name, batch_count), executable=True)
    write_text(script_root / "submit_pbs_array.sh", build_pbs_script(args.plan_dir_name, batch_count), executable=True)
    write_text(script_root / "README_HPC_500K.md", build_readme(args.plan_dir_name, batch_count))

    print(f"Prepared {batch_count} batches under {plan_root}")


if __name__ == "__main__":
    main()
