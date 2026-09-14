# Dataset package

This package generates synthetic hex-grid sonar datasets for training and evaluation of planning and perception models. The generated samples encode a grid layout, obstacle configuration, sonar placement, and a scalar objective value derived from a navigation task under partial sonar observability.

The package is organized around a few core pieces:

- `cli.py`: command-line entry point for generating new datasets
- `generator.py`: dataset generation loop and worker orchestration
- `sample.py`: per-sample feature construction and evaluation
- `static_cache.py`: precomputed static geometry cache for faster repeated runs
- `merge_datasets.py`: merge streaming/partitioned dataset shards
- `convert_unknown_parts.py`: convert known-only samples to more-unknown variants
- `combine_local_features.py`: assemble a single local-feature tensor from split arrays

## Overview

Each sample is built from a hexagonal grid with:

- a fixed map height/width
- obstacle cells
- a goal cell
- a set of sonar cells
- a subset of sonars marked as known/unknown to the agent
- a start cell and a target output value for the planner objective

The resulting data is mostly NumPy arrays saved as `.npz` or memory-mapped `.npy` shards.

## Typical workflow

The main entry point is:

```bash
python -m dataset.cli --help
```

This exposes the main generation arguments such as grid size, number of obstacles, sonar counts, goal/start cells, forbidden radius, and placement policy.

Example:

```bash
python -m dataset.cli \
  --height 20 \
  --width 20 \
  --samples 100000 \
  --obstacle-number 60 \
  --sonar-number 8 \
  --unknown-sonar-number 2 \
  --goal-index 399 \
  --start-index 0 \
  --sonar-placement-policy random \
  --out data/training_dataset.npz
```

## Output format

The package exports feature dictionaries whose keys are produced by `Features.to_dict()` in `environment/features.py`.

The most important arrays in each sample are:

- `cost`: static cost map of the grid
- `outputs`: planner objective value(s) for each valid start node or a single focused value
- `reachable_mask`: finite-value mask for reachable states
- `goal_onehot`: one-hot goal indicator
- `obstacle_placement`: boolean map of obstacle cells
- `sonar_placement`: boolean map of sonar locations
- `axial_coordinates`: hex axial coordinates
- `cartesian_coordinates`: Euclidean coordinates
- `distance_to_goal`: distance from each cell to the goal
- `known_sonar_mask`: cells that are known sonar positions for the sample
- `unknown_sonar_mask`: cells that are treated as unknown sonar positions
- `neighbor_cost`, `neighbor_obstacle`, `neighbor_sonar`, `neighbor_mask`
- `ring_cost`, `ring_obstacle_count`, `ring_obstacle_density`, `ring_sonar_count`, `ring_sonar_density`

When saved as an `.npz`, the arrays are stacked as one dataset by sample.

For example, a generated dataset can look like:

```python
arr = np.load("data/training_dataset.npz")
print(arr.files)
```

and keys such as:

```python
['cost', 'outputs', 'obstacle_placement', 'sonar_placement', 'known_sonar_mask', ...]
```

## Generation modes and policies

The CLI supports a few important modes:

- `--single-value`: generate a single-value target for one start index instead of all valid starts
- `--start-index`: chosen starting node for single-value generation
- `--forbidden-radius`: excludes cells near start/goal from sonar placement
- `--sonar-placement-policy`: one of
  - `random`
  - `endpoint`
  - `mixture`
- `--starting_sonar_placement`: gives a fixed initial sonar placement to begin from
- `--best-sonar-split`: use the strongest known/unknown sonar split instead of random split

These are used in `dataset/sample.py` when constructing each feature sample.

## Streaming and shard-based generation

Large datasets can be generated incrementally without keeping all samples in memory using:

```bash
python -m dataset.cli --stream-output --samples 500000 --out data/my_dataset.npz
```

This writes temporary shard files under a sibling directory like:

```text
data/my_dataset_parts/
```

with metadata in:

```text
data/my_dataset_parts/meta.json
```

The package later merges shards via:

```bash
python dataset/merge_datasets.py
```

or directly by importing and using `merge_dataset(...)` from the module.

## Dataset conversion utilities

### Unknown-sonar conversions

`convert_unknown_parts.py` can transform an existing streaming dataset into a version with a different number of unknown sonars.

Examples:

```bash
python -m dataset.convert_unknown_parts \
  --input data/training_dataset_ALLKNOWN_parts \
  --output data/training_dataset_UNKNOWN1_parts \
  --unknown-count 1 \
  --mode random
```

or:

```bash
python -m dataset.convert_unknown_parts \
  --input data/training_dataset_UNKNOWN1_parts \
  --output data/training_dataset_UNKNOWN2_parts \
  --unknown-count 2 \
  --mode best \
  --start-index 0
```

The `--mode all` option expands each sample into all valid unknown-sonar splits.

### Local feature assembly

`combine_local_features.py` assembles a flattened local-feature tensor from a parts dataset. This is useful when a model expects a single 3D tensor of shape:

```text
(samples, cells, channels)
```

instead of separate arrays per feature.

## Precomputation and cache

`static_cache.py` caches expensive static geometry such as:

- `visibility_map.npy`
- `base_contributions.npy`
- `meta.json`

These are stored under a cache directory such as:

```text
data/precomp/
```

This avoids recomputing the same static features across many workers and repeated dataset generation jobs.

## Package layout

```text
dataset/
├── README.md
├── cli.py
├── generator.py
├── sample.py
├── static_cache.py
├── merge_datasets.py
├── convert_unknown_parts.py
├── combine_local_features.py
├── generate_high_output_samples.py
├── utils.py
└── test/
    ├── analyze_keys.py
    ├── check_dataset.py
    ├── check_dtypes.py
    ├── check_tmp_parts.py
    └── dataset_statistics.py
```

## Notes

- Many arrays are boolean masks or float32 arrays, but exact dtypes depend on the generated sample type and the output stage.
- This package is intended for research data generation and is optimized for large-scale synthetic sample construction.
- For large jobs, using `--stream-output` and sharded generation is usually preferable to keeping the full dataset in memory.

## Minimal import-based usage

You can also generate a single sample directly in Python:

```python
from dataset.sample import sample_one
from dataset.static_cache import build_static

static = build_static(height=20, width=20, obstacle_indices=[...])
row = sample_one(
    static=static,
    sonar_number=8,
    unknown_sonar_number=2,
    goal_index=399,
    start_index=0,
    single_value=True,
    forbidden_radius=0,
    sonar_placement_policy="random",
)

print(row.keys())
```

This is the underlying primitive used by the dataset generator.
