import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environment.cost import CostMap
from environment.features import Features
from environment.utils import detection_probability
import dataset.static_cache as sc
from trainer.data_builder import PartsDataset, sanitize_cost
from trainer.loss import MSELoss
from trainer.model import LocalGlobalNet
from trainer.plot import plot_predictions
from trainer.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset


def _normalize_state_dict(state_dict: dict) -> dict:
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value
    return normalized


def _build_model_from_checkpoint(
    model_path: str | Path,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[LocalGlobalNet, dict]:
    state_dict = torch.load(model_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint is not a state_dict dictionary.")

    state_dict = _normalize_state_dict(state_dict)

    required_keys = [
        "local_hidden_weight",
        "local_output_weight",
        "global_hidden.weight",
        "global_output.weight",
    ]
    missing = [key for key in required_keys if key not in state_dict]
    if missing:
        raise KeyError(f"Missing checkpoint keys: {missing}")

    local_hidden_weight = state_dict["local_hidden_weight"]
    local_output_weight = state_dict["local_output_weight"]
    global_hidden_weight = state_dict["global_hidden.weight"]
    global_output_weight = state_dict["global_output.weight"]

    total_cells = width * height
    if local_hidden_weight.shape[0] != total_cells:
        raise ValueError(
            f"Checkpoint expects {local_hidden_weight.shape[0]} cells, "
            f"but width*height={total_cells}."
        )

    local_hidden_dimension = local_hidden_weight.shape[1]
    local_input_dimension = local_hidden_weight.shape[2]
    local_output_dimension = local_output_weight.shape[1]
    global_hidden_dimension = global_hidden_weight.shape[0]
    global_output_dimension = global_output_weight.shape[0]
    global_input_dimension = global_hidden_weight.shape[1]
    global_context_dimension = global_input_dimension - total_cells * local_output_dimension

    if global_context_dimension < 0:
        raise ValueError("Inferred global_context_dimension is negative.")

    model = LocalGlobalNet(
        rows=height,
        columns=width,
        local_input_dimension=local_input_dimension,
        local_hidden_dimension=local_hidden_dimension,
        local_output_dimension=local_output_dimension,
        global_context_dimension=global_context_dimension,
        global_hidden_dimension=global_hidden_dimension,
        global_output_dimension=global_output_dimension,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {
        "total_cells": total_cells,
        "local_input_dimension": local_input_dimension,
        "global_context_dimension": global_context_dimension,
        "global_output_dimension": global_output_dimension,
    }
    return model, meta


def _as_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None
    
def _extract_local_features(
    features: Features,
    features_dict: dict,
    total_cells: int,
    local_input_dimension: int,
) -> np.ndarray:
    candidate_names = [
        "local_features",
        "inputs",
        "local_input",
        "x_local",
        "feature_matrix",
    ]

    for name in candidate_names:
        if name in features_dict:
            arr = _as_numpy(features_dict[name])
            if arr is not None:
                if arr.ndim == 2 and arr.shape == (total_cells, local_input_dimension):
                    return arr.astype(np.float32, copy=False)
                if arr.ndim == 1 and arr.size == total_cells * local_input_dimension:
                    return arr.reshape(total_cells, local_input_dimension).astype(np.float32, copy=False)

    for name in candidate_names:
        if hasattr(features, name):
            arr = _as_numpy(getattr(features, name))
            if arr is not None:
                if arr.ndim == 2 and arr.shape == (total_cells, local_input_dimension):
                    return arr.astype(np.float32, copy=False)
                if arr.ndim == 1 and arr.size == total_cells * local_input_dimension:
                    return arr.reshape(total_cells, local_input_dimension).astype(np.float32, copy=False)

    excluded_keys = {
        "outputs",
        "output",
        "target",
        "targets",
        "label",
        "labels",
        "reachable_mask",
        "global_context",
        "context",
        "x_global",
        "goal_context",
        "goal_onehot",
        "goal_index",
        "obstacle_number",
        "sonar_number",
        "start_as_unknown_number",
        "start_as_known_number",
        "best_sonar_split",
    }

    channels = []
    debug_shapes = {}

    # Keep feature order exactly aligned with training/data loader logic.
    local_feature_order = list(PartsDataset.local_features)

    augmented = dict(features_dict)
    augmented["known_sonar_mask"] = features.known_sonar_mask
    augmented["unknown_sonar_mask"] = features.unknown_sonar_mask

    for key, value in augmented.items():
        arr = _as_numpy(value)
        if arr is not None:
            debug_shapes[key] = tuple(arr.shape)

    cost_keys = {"cost", "neighbor_cost", "ring_cost"}

    for key in local_feature_order:
        if key not in augmented:
            continue

        arr = _as_numpy(augmented[key])
        if arr is None:
            continue

        if key in cost_keys:
            arr = sanitize_cost(arr)
        else:
            arr = arr.astype(np.float32, copy=False)

        if arr.ndim == 1 and arr.size == total_cells:
            channels.append(arr.reshape(total_cells, 1))
        elif arr.ndim == 2 and arr.shape[0] == total_cells:
            channels.append(arr.reshape(total_cells, -1))

    if not channels:
        for key, value in augmented.items():
            if key in excluded_keys:
                continue

            arr = _as_numpy(value)
            if arr is None:
                continue

            if key in cost_keys:
                arr = sanitize_cost(arr)
            else:
                arr = arr.astype(np.float32, copy=False)

            if arr.ndim == 1 and arr.size == total_cells:
                channels.append(arr.reshape(total_cells, 1))
            elif arr.ndim == 2 and arr.shape[0] == total_cells:
                channels.append(arr.reshape(total_cells, -1))

    # If loose discovery above did not recover the expected matrix,
    # force assembly using the exact training order from PartsDataset.
    if channels:
        local_matrix = np.concatenate(channels, axis=1).astype(np.float32, copy=False)
        if local_matrix.shape == (total_cells, local_input_dimension):
            return local_matrix

    strict_channels = []
    strict_missing = []
    for key in PartsDataset.local_features:
        arr = _as_numpy(augmented.get(key))
        if arr is None:
            strict_missing.append(key)
            continue
        if key in cost_keys:
            arr = sanitize_cost(arr)
        else:
            arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 1 and arr.size == total_cells:
            strict_channels.append(arr.reshape(total_cells, 1))
        elif arr.ndim == 2 and arr.shape[0] == total_cells:
            strict_channels.append(arr.reshape(total_cells, -1))
        else:
            raise ValueError(
                f"Feature '{key}' has unsupported shape {arr.shape}; expected leading dim {total_cells}."
            )

    if strict_missing:
        raise ValueError(
            f"Missing required local features for strict assembly: {strict_missing}. Available keys: {sorted(list(augmented.keys()))}"
        )

    local_matrix = np.concatenate(strict_channels, axis=1).astype(np.float32, copy=False)
    if local_matrix.shape == (total_cells, local_input_dimension):
        return local_matrix

    raise ValueError(
        f"Built local feature matrix with shape {local_matrix.shape}, "
        f"expected ({total_cells}, {local_input_dimension}). "
        f"Available shapes: {debug_shapes}"
    )


def _extract_global_context(
    features: Features,
    features_dict: dict,
    cost_map: CostMap,
    goal_index: int,
    total_cells: int,
    global_context_dimension: int,
) -> np.ndarray:
    candidate_names = [
        "global_context",
        "context",
        "x_global",
        "goal_context",
    ]

    for name in candidate_names:
        if name in features_dict:
            arr = _as_numpy(features_dict[name])
            if arr is not None:
                if arr.ndim == 1 and arr.size == global_context_dimension:
                    return arr.astype(np.float32, copy=False)
                if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == global_context_dimension:
                    return arr.reshape(-1).astype(np.float32, copy=False)

    goal_onehot = _as_numpy(features_dict.get("goal_onehot"))
    if goal_onehot is None:
        goal_onehot = np.zeros(total_cells, dtype=np.float32)
        goal_onehot[goal_index] = 1.0
    else:
        goal_onehot = goal_onehot.astype(np.float32, copy=False).reshape(-1)

    # Build global context strictly with the same order used by PartsDataset.
    strict_global = {
        "goal_onehot": goal_onehot,
        "obstacle_number": float(features_dict.get("obstacle_number", 0.0)),
        "sonar_number": float(features_dict.get("sonar_number", cost_map.sonar_number)),
        "start_as_known_number": float(features_dict.get("start_as_known_number", 0.0)),
        "start_as_unknown_number": float(features_dict.get("start_as_unknown_number", 0.0)),
    }
    global_features = []
    for key in PartsDataset.global_features:
        if key not in strict_global:
            raise ValueError(f"Missing required global feature '{key}'")
        arr = np.atleast_1d(strict_global[key]).astype(np.float32, copy=False)
        global_features.append(arr)
    candidate = np.concatenate(global_features)

    if candidate.size == global_context_dimension:
        return candidate.astype(np.float32, copy=False)

    raise ValueError(
        f"Could not find global context of size {global_context_dimension}. "
        f"Built candidate size: {candidate.size}"
    )


def validate_positioning_with_sonars(
    sonar_indices: list[int],
    unknown_indices: list[int],
    model="best_model_mse.pt",
    width: int = 20,
    height: int = 20,
    obstacle_indices: list[int] | None = [43, 44, 24, 183, 184, 203, 204, 205, 224, 225, 244, 245, 262, 263, 264, 265, 285, 286, 287, 267, 132, 133, 152, 190, 191, 211, 212, 213, 355, 376, 113, 112, 94, 92, 91, 63, 64, 19, 18, 17, 16, 39, 38, 37, 59, 231, 232, 233, 234, 251, 252, 253, 271, 290, 291, 292, 309, 310, 329, 351],
    start_index: int = 0,
    goal_index: int = 399,
    compare_with_trainer: bool = True,
    plot_with_trainer: bool = False,
) -> dict:
    # Build static exactly like dataset generation path (with cache reuse when params match).
    static = sc.build_static(
        height=height,
        width=width,
        obstacle_indices=obstacle_indices,
        detection_probability=detection_probability,
    )
    raw_map = static.raw_map
    obstacle_map = static.obstacle_map

    cost_map = CostMap(
        obstacle_map,
        detection_probability,
        sonar_number=len(sonar_indices),
        sonar_indices=sonar_indices,
        base_contributions=static.base_contributions,
    )

    features = Features(
        static=static,
        cost_map=cost_map,
        goal_index=goal_index,
        start_as_unknown_number=len(unknown_indices),
        single_value=True,
        start_index=start_index,
        best_sonar_split=True,
    )

    features.known_sonar_mask.fill(False)
    features.unknown_sonar_mask.fill(False)

    sonar_array = np.asarray(cost_map.sonar_indices, dtype=np.int32)

    known_indices = [i for i in range(len(sonar_indices)) if i not in unknown_indices]
    features.known_sonar_mask[sonar_array[known_indices]] = True
    features.unknown_sonar_mask[sonar_array[unknown_indices]] = True

    features_dict = features.to_dict()

    result = {
        "raw_map": raw_map,
        "obstacle_map": obstacle_map,
        "cost_map": cost_map,
        "static": static,
        "features": features,
        "features_dict": features_dict,
        "sonar_indices": sonar_indices,
        "unknown_indices": unknown_indices,
        "known_sonar_mask": features.known_sonar_mask.copy(),
        "unknown_sonar_mask": features.unknown_sonar_mask.copy(),
    }

    if model is not None:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_obj, meta = _build_model_from_checkpoint(model, width, height, device)
            result["model_meta"] = meta

            local_features_np = _extract_local_features(
                features=features,
                features_dict=features_dict,
                total_cells=meta["total_cells"],
                local_input_dimension=meta["local_input_dimension"],
            )
            global_context_np = _extract_global_context(
                features=features,
                features_dict=features_dict,
                cost_map=cost_map,
                goal_index=goal_index,
                total_cells=meta["total_cells"],
                global_context_dimension=meta["global_context_dimension"],
            )

            local_features = torch.tensor(local_features_np, dtype=torch.float32, device=device).unsqueeze(0)
            global_context = torch.tensor(global_context_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                predictions = model_obj(local_features, global_context)

            result["predictions"] = predictions
            result["predicted_output"] = predictions.detach().cpu().numpy()

            if compare_with_trainer:
                # Build a 1-sample loader matching trainer.validate signature:
                # (local_batch, global_context_batch, target_batch).
                if int(meta["global_output_dimension"]) == 1:
                    target_np = np.asarray([float(features.outputs[start_index])], dtype=np.float32)
                else:
                    target_np = np.asarray(features.outputs, dtype=np.float32).reshape(-1)
                    if target_np.size != int(meta["global_output_dimension"]):
                        raise ValueError(
                            f"Target size mismatch for trainer path: got {target_np.size}, "
                            f"expected {int(meta['global_output_dimension'])}."
                        )

                target_tensor = torch.tensor(target_np, dtype=torch.float32).unsqueeze(0)
                one_sample_ds = TensorDataset(
                    torch.tensor(local_features_np, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(global_context_np, dtype=torch.float32).unsqueeze(0),
                    target_tensor,
                )
                one_sample_loader = DataLoader(one_sample_ds, batch_size=1, shuffle=False)

                trainer = Trainer(
                    model=model_obj,
                    optimizer=None,
                    loss=MSELoss(),
                    device=device,
                )
                trainer_loss = trainer.validate(one_sample_loader, None)
                result["trainer_validate_loss"] = float(trainer_loss)

                with torch.no_grad():
                    trainer_pred = model_obj(
                        torch.tensor(local_features_np, dtype=torch.float32, device=device).unsqueeze(0),
                        torch.tensor(global_context_np, dtype=torch.float32, device=device).unsqueeze(0),
                    )
                result["trainer_predicted_output"] = trainer_pred.detach().cpu().numpy()

                if plot_with_trainer:
                    plot_predictions(model_obj, one_sample_loader, device)

        except Exception as e:
            result["model_error"] = str(e)

    return result


if __name__ == "__main__":
    obstacles = [43, 44, 24, 183, 184, 203, 204, 205, 224, 225, 244, 245, 262, 263, 264, 265, 285, 286, 287, 267, 132, 133, 152, 190, 191, 211, 212, 213, 355, 376, 113, 112, 94, 92, 91, 63, 64, 19, 18, 17, 16, 39, 38, 37, 59, 231, 232, 233, 234, 251, 252, 253, 271, 290, 291, 292, 309, 310, 329, 351]

    available_indices = np.setdiff1d(
        np.arange(400),
        obstacles,
        assume_unique=True
    )

    sonar_indices = np.random.choice(
        available_indices,
        size=8,
        replace=False
    ).tolist()

    sonar_indices = [235, 236, 238, 259, 314, 349, 373, 391] # EXACT
    #sonar_indices = [236, 238, 256, 257, 333, 368, 369, 392]
    #sonar_indices = [65, 195, 258, 259, 275, 333, 369, 372]
    #sonar_indices = [85, 195, 256, 258, 259, 333, 369, 392]
    #sonar_indices = [216, 256, 258, 259, 314, 349, 373, 391]
    sonar_indices = [218, 235, 236, 259, 314, 349, 373, 391] # EXTREME MODEL 10%
    #sonar_indices = [178, 215, 256, 259, 295, 349, 373, 389] # RANDOM MODEL WITH BEST DATASET 2%
    unknown_indices = []

    #sonar_indices = [235, 256, 258, 259, 312, 369, 371, 373]
    #unknown_indices = [7] # sonar indices with base zero

    result = validate_positioning_with_sonars(
        sonar_indices=sonar_indices,
        unknown_indices=unknown_indices,
        start_index=0,
        model="best_partial_model_BESTDATASET_1UNKNOWN_extreme.pt",
        #model="best_model_BESTDATASET_ALLKNOWN_extreme.pt",
    )

    print(f"Map size: {result['raw_map'].total_cells}")
    print(f"Sonar count: {result['cost_map'].sonar_number}")
    print(f"Known sonars: {result['known_sonar_mask'].sum()}")
    print(f"Unknown sonars: {result['unknown_sonar_mask'].sum()}")
    print(f"Output at start: {result['features'].outputs[0]}")
    print(f"Reachable: {result['features'].reachable_mask[0]}")
    print(f"Predictions: {result.get('predicted_output', 'No predictions')}")
    print(f"Trainer predictions: {result.get('trainer_predicted_output', 'No trainer prediction')}")
    print(f"Trainer validate loss: {result.get('trainer_validate_loss', 'No trainer loss')}")
    print(f"Model error: {result.get('model_error', 'None')}")
    print(f"Model meta: {result.get('model_meta', 'None')}")