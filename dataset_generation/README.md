# Dataset Generation

This folder contains the code used to generate the polygonal phononic-crystal dataset used by the GNN classifier and branch-specific regressors.

## Included files

- `code/generate_pilot_dataset_py.py`: samples square-bounded symmetric polygonal inclusions, evaluates the node-based PWE solver, and writes chunked dataset records.
- `code/node_pwe_repro.py`: node-based in-plane PWE solver used by the generator.
- `code/show_k_path.py`: exports the fixed `M -> Gamma -> X -> M` wave-vector path.
- `code/prepare_hpc_pwe_batches.py`: creates batch JSON files for large HPC runs.
- `configs/smoke_square_bounded.json`: small local smoke-test configuration.
- `configs/full_dataset_square_bounded_example.json`: example configuration matching the square-bounded dataset construction used in the manuscript.

## Geometry rule

For the square-bounded dataset, each sample first draws `N_s` sector control points with `1 <= N_s <= 8`. The points are sampled in the first 45-degree sector using

```text
0.01 <= x <= 0.49,
0.01 <= y <= x - 0.01.
```

The sector points are sorted by `atan2(y, x)` and expanded with eightfold symmetry to obtain the full polygon in the square unit cell.

## Run a small local test

```bash
cd dataset_generation/code
python generate_pilot_dataset_py.py --config-json ../configs/smoke_square_bounded.json
```

The generated records contain polygon vertices, band frequencies, complete-gap index, lower and upper gap-edge frequencies, gap width, gap indicator, area fraction, and vertex count.

## Reproduce a large run

For a large run, edit `configs/full_dataset_square_bounded_example.json` or create multiple batch configs with `prepare_hpc_pwe_batches.py`. The raw generated chunks are intentionally not tracked in Git because they are large.
