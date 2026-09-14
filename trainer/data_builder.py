from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader

INF_REPLACEMENT = 100.0

def sanitize_cost(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=True)
    return np.nan_to_num(arr, posinf=INF_REPLACEMENT, neginf=INF_REPLACEMENT)

class PartsDataset(Dataset):
    """Dataset that reads a parts directory (meta.json + per-key .npy).

    It returns per-sample numpy arrays (not tensors) so the collate function can convert them
    to tensors in a central place (zero-copy when possible).
    """

    local_features = [
        "cost",
        "obstacle_placement",
        "sonar_placement",
        "axial_coordinates",
        "cartesian_coordinates",
        "distance_to_goal",
        "known_sonar_mask",
        "unknown_sonar_mask",
        "neighbor_cost",
        "neighbor_obstacle",
        "neighbor_sonar",
        "neighbor_mask",
        "ring_cost",
        "ring_obstacle_count",
        "ring_obstacle_density",
        "ring_sonar_count",
        "ring_sonar_density",
    ]

    global_features = [
        "goal_onehot",
        "obstacle_number",
        "sonar_number",
        "start_as_known_number",
        "start_as_unknown_number",
    ]

    def __init__(self, root: str | Path, train_cells: list[int] | None = None):
        self.root = Path(root)
        self.train_cells = train_cells
        self._mmaps: dict[str, np.memmap] | None = None

        # assume directory with meta.json
        self.parts_dir = self.root
        meta_path = self.parts_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in parts dir {self.parts_dir}")
        with open(meta_path, "r") as f:
            m = json.load(f)
        self._meta = m
        self.processed = int(m.get("processed", m.get("total", 0)))
        self._keys = list(m.get("keys", {}).keys())
        self.total_cells: int | None = None
        self.local_input_dimension: int | None = None
        self.global_context_dimension: int | None = None
        self._infer_from_meta(m)

        self.target_dimension: int | None = None
        self._get_target_dimension()

        self._batch_count = 0

    def _infer_from_meta(self, meta: dict):
        keys = meta.get("keys", {})

        cost_shape = tuple(keys["cost"]["shape"])
        self.total_cells = int(cost_shape[1])

        local_input_dimension: int = 0
        for key in self.local_features:
            shape = keys[key]["shape"]
            local_input_dimension += shape[-1] if len(shape) > 2 else 1

        self.local_input_dimension = local_input_dimension

        global_context_dimension: int = 0
        for key in self.global_features:
            shape = keys[key]["shape"]
            global_context_dimension += shape[-1] if len(shape) > 1 else 1
        self.global_context_dimension = global_context_dimension

    def _get_target_dimension(self):
        # THIS IS A HACK: we assume that the target dimension is the same for all samples, and we just read the first sample to get it.

        if self.train_cells is not None:
            self.target_dimension = len(self.train_cells)
        else:
            self._open_mmaps()
            reachable_mask = self._mmaps["reachable_mask"][0]
            self.target_dimension = int(np.sum(reachable_mask))

    """ FOR MULTIPLE LOCAL FEATURES FILES
    def _open_mmaps(self):
        if self._mmaps is not None:
            return
        mm = {}
        for k in self._keys:
            p = self.parts_dir / f"{k}.npy"
            if p.exists():
                mm[k] = np.load(p, mmap_mode="r")
        self._mmaps = mm
    """

    def _open_mmaps(self):
        if self._mmaps is not None:
            return

        mm = {}

        local_path = self.parts_dir / "local_features.npy"

        # If combined file exists, don't open individual local features.
        if local_path.exists():
            mm["local_features"] = np.load(
                local_path,
                mmap_mode="r",
            )

        for k in self._keys:
            if local_path.exists() and k in self.local_features:
                continue

            p = self.parts_dir / f"{k}.npy"
            if p.exists():
                mm[k] = np.load(p, mmap_mode="r")

        self._mmaps = mm

    def __len__(self) -> int:
        return int(self.processed)

    def __getitem__(self, idx: int):
        self._open_mmaps()
        mm = self._mmaps

        local_features = []
        for key in self.local_features:
            arr = mm[key][idx]
            if key in {"cost", "neighbor_cost", "ring_cost"}:
                arr = sanitize_cost(arr)
            else:
                arr = arr.astype(np.float32, copy=False)
            if arr.ndim == 1:
                arr = arr[:, None]
            local_features.append(arr)
        local_input = np.concatenate(local_features, axis=1)

        global_features = []
        for key in self.global_features:
            arr = np.atleast_1d(mm[key][idx]).astype(np.float32, copy=False)
            global_features.append(arr)
        global_context = np.concatenate(global_features)

        outputs = mm["outputs"][idx].astype(np.float32, copy=False)
        reachable_mask = mm["reachable_mask"][idx].astype(np.bool_, copy=False)
        if self.train_cells is not None:
            training_cell_indices = self.train_cells
        else:
            training_cell_indices = np.where(reachable_mask)[0]
        targets = outputs[training_cell_indices]

        return local_input, global_context, targets

    """ FOR MULTIPLE LOCAL FEATURES FILES
    def __getitems__(self, indices):
        start = time.perf_counter()

        self._open_mmaps()
        mm = self._mmaps

        local_features = []
        for key in self.local_features:
            t0 = time.perf_counter()
            arr = mm[key][indices]
            print(f"{key}: {time.perf_counter() - t0:.3f}s", flush=True)
            if key in {"cost", "neighbor_cost", "ring_cost"}:
                arr = sanitize_cost(arr)
            else:
                arr = arr.astype(np.float32, copy=False)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            local_features.append(arr)

        local_input = np.concatenate(local_features, axis=2)

        local_time = time.perf_counter()

        global_features = []
        for key in self.global_features:
            arr = mm[key][indices].astype(np.float32, copy=False)

            if arr.ndim == 1:
                arr = arr[:, None]

            global_features.append(arr)

        global_context = np.concatenate(global_features, axis=1)

        global_time = time.perf_counter()

        outputs = mm["outputs"][indices].astype(np.float32, copy=False)
        reachable_mask = mm["reachable_mask"][indices].astype(np.bool_, copy=False)

        if self.train_cells is not None:
            targets = outputs[:, self.train_cells]
        else:
            targets = [
                output[np.where(mask)[0]]
                for output, mask in zip(outputs, reachable_mask)
            ]

        end = time.perf_counter()

        print(
            f"local={local_time-start:.3f}s "
            f"global={global_time-local_time:.3f}s "
            f"outputs={end-global_time:.3f}s",
            flush=True,
        )

        return [
            (local_input[i], global_context[i], targets[i])
            for i in range(len(indices))
        ]
    """

    def __getitems__(self, indices):
        start = time.perf_counter()

        self._open_mmaps()
        mm = self._mmaps

        # ---------------------------------------------------------
        # LOCAL FEATURES
        # ---------------------------------------------------------
        if "local_features" in mm:
            # FAST PATH:
            # One memmap read instead of 17 separate reads.
            # Shape: (batch, 400, 49)

            local_input = mm["local_features"][indices].astype(
                np.float32,
                copy=False,
            )

            # Preserve the old sanitize_cost() behavior.
            #
            # Channel layout:
            #
            # cost                    -> 0
            # obstacle_placement      -> 1
            # sonar_placement         -> 2
            # axial_coordinates       -> 3:5
            # cartesian_coordinates   -> 5:7
            # distance_to_goal        -> 7
            # known_sonar_mask        -> 8
            # unknown_sonar_mask      -> 9
            # neighbor_cost           -> 10:16
            # neighbor_obstacle       -> 16:22
            # neighbor_sonar          -> 22:28
            # neighbor_mask           -> 28:34
            # ring_cost               -> 34:37
            # ring_obstacle_count     -> 37:40
            # ring_obstacle_density   -> 40:43
            # ring_sonar_count        -> 43:46
            # ring_sonar_density      -> 46:49

            local_input[:, :, 0] = sanitize_cost(
                local_input[:, :, 0]
            )

            local_input[:, :, 10:16] = sanitize_cost(
                local_input[:, :, 10:16]
            )

            local_input[:, :, 34:37] = sanitize_cost(
                local_input[:, :, 34:37]
            )

        else:
            # FALLBACK:
            # Dataset does not have local_features.npy.
            # Reconstruct the local input from the original files.

            local_features = []

            for key in self.local_features:
                arr = mm[key][indices]

                if key in {"cost", "neighbor_cost", "ring_cost"}:
                    arr = sanitize_cost(arr)
                else:
                    arr = arr.astype(
                        np.float32,
                        copy=False,
                    )

                if arr.ndim == 2:
                    arr = arr[:, :, None]

                local_features.append(arr)

            local_input = np.concatenate(
                local_features,
                axis=2,
            )

        local_time = time.perf_counter()

        # ---------------------------------------------------------
        # GLOBAL FEATURES
        # ---------------------------------------------------------
        global_features = []

        for key in self.global_features:
            arr = mm[key][indices].astype(
                np.float32,
                copy=False,
            )

            if arr.ndim == 1:
                arr = arr[:, None]

            global_features.append(arr)

        global_context = np.concatenate(
            global_features,
            axis=1,
        )

        global_time = time.perf_counter()

        # ---------------------------------------------------------
        # OUTPUTS / TARGETS
        # ---------------------------------------------------------
        outputs = mm["outputs"][indices].astype(
            np.float32,
            copy=False,
        )

        reachable_mask = mm["reachable_mask"][indices].astype(
            np.bool_,
            copy=False,
        )

        if self.train_cells is not None:
            targets = outputs[:, self.train_cells]
        else:
            targets = [
                output[np.where(mask)[0]]
                for output, mask in zip(
                    outputs,
                    reachable_mask,
                )
            ]

        end = time.perf_counter()

        # ---------------------------------------------------------
        # TIMING
        # ---------------------------------------------------------
        if self._batch_count % 100 == 0:
            path = (
                "combined"
                if "local_features" in mm
                else "individual"
            )

            print(
                f"Batch {self._batch_count}: "
                f"path={path} "
                f"local={local_time-start:.3f}s "
                f"global={global_time-local_time:.3f}s "
                f"outputs={end-global_time:.3f}s",
                flush=True,
            )

        self._batch_count += 1

        # ---------------------------------------------------------
        # RETURN
        # ---------------------------------------------------------
        return [
            (
                local_input[i],
                global_context[i],
                targets[i],
            )
            for i in range(len(indices))
        ]

def default_collate_fn(batch):
    """Convert a batch (list of tuples from `PartsDataset`) into torch tensors.

    Returns: (local_batch, global_context_batch, target_batch)
    where `local_batch` is (B, N, K), `global_context_batch` is (B, G),
    `target_batch` is (B, N).
    """
    # batch items: (local_feats, global_context, targets)
    locals_ = [item[0] for item in batch]
    globals_ = [item[1] for item in batch]
    targets = [item[2] for item in batch]

    local_np = np.stack(locals_, axis=0)
    global_np = np.stack(globals_, axis=0)
    target_np = np.stack(targets, axis=0)

    # convert to torch tensors (zero-copy where possible)
    if torch is None:
        raise RuntimeError("PyTorch is required for collate function")

    local_t = torch.from_numpy(local_np)
    global_t = torch.from_numpy(global_np)
    target_t = torch.from_numpy(target_np)

    return local_t, global_t, target_t


def build_dataloaders(
    root: str | Path,
    train_cells: list[int] | None = None,
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    max_samples: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    collate_fn=default_collate_fn,
    seed: int = 42,
):
    """Build train and validation DataLoaders and return dims needed for model instantiation.

    Returns: (train_loader, val_loader, local_input_dimension, global_context_dimension, total_cells, target_dimension)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to build dataloaders")
    #if not 0.0 < validation_fraction < 1.0:
    #    raise ValueError(f"validation_fraction must be in (0,1), got {validation_fraction}")

    print(f"Building dataloaders from {root} with train_cells={train_cells}, batch_size={batch_size}, validation_fraction={validation_fraction}, max_samples={max_samples}, num_workers={num_workers}, pin_memory={pin_memory}")

    base_ds = PartsDataset(root, train_cells=train_cells)
    ds = base_ds
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        if max_samples < len(ds):
            #indices = torch.randperm(len(ds))[:max_samples]
            indices = torch.arange(max_samples)
            ds = torch.utils.data.Subset(ds, indices)

    S = len(ds)
    if validation_fraction <= 0.0:
        train_ds = ds
        val_ds = ds
    else:
        n_val = int(max(1, round(validation_fraction * S))) if S > 1 else 0
        if n_val >= S:
            n_val = max(1, S - 1)
        n_train = S - n_val

        generator = torch.Generator()
        generator.manual_seed(seed)
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers // 2), pin_memory=pin_memory, collate_fn=collate_fn)

    print(f"Train dataset: {len(train_ds)} samples, Validation dataset: {len(val_ds)} samples")

    return train_loader, val_loader, base_ds.local_input_dimension, base_ds.global_context_dimension, base_ds.total_cells, base_ds.target_dimension
