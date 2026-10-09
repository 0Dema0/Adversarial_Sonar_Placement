import argparse
import torch
import numpy as np

from trainer.data_builder import build_dataloaders
from trainer.model import LocalGlobalNet
from trainer.loss import MSELoss
from trainer.plot import plot_prediction_arrays

def parse_indices(s):
    if s is None:
        return None
    return [int(x) for x in s.split(",")]

@torch.no_grad()
def evaluate_and_collect(model, loader, device, loss_fn):
    model.eval()

    total_loss = 0.0
    num_batches = 0
    predictions = []
    targets = []
    max_prediction = -float("inf")
    max_info = None
    sample_idx = 0

    for local, global_context, target in loader:
        local = local.to(device)
        global_context = global_context.to(device)
        target = target.to(device)

        prediction = model(local, global_context)

        loss_map = (prediction - target) ** 2
        per_sample = loss_map.mean(dim=tuple(range(1, loss_map.ndim)))
        total_loss += loss_fn(per_sample, target).item()
        num_batches += 1

        prediction_values = prediction.detach().cpu().numpy()
        target_values = target.detach().cpu().numpy()
        predictions.append(prediction_values)
        targets.append(target_values)

        flat_predictions = prediction_values.reshape(-1)
        batch_max_index = int(np.argmax(flat_predictions))
        batch_max = float(flat_predictions[batch_max_index])

        if batch_max > max_prediction:
            max_prediction = batch_max
            max_info = {
                "sample": sample_idx + batch_max_index,
                "prediction": batch_max,
                "target": float(target_values.reshape(-1)[batch_max_index]),
                "sonar_placement": local[batch_max_index, :, 2].cpu().numpy(),
                "unknown_sonar_mask": local[batch_max_index, :, 9].cpu().numpy(),
            }

        sample_idx += target.shape[0]

    return total_loss / num_batches, np.concatenate(predictions), np.concatenate(targets), max_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--train_cells", type=str, default="0")
    parser.add_argument("--save", default=None)
    parser.add_argument("--alpha-envelope", action="store_true", default=False, help="Plot the alpha envelope of the predictions")
    args = parser.parse_args()

    train_cells = parse_indices(args.train_cells)

    test_loader, _, local_dim, global_dim, _, target_dim = build_dataloaders(
        args.data,
        train_cells=train_cells,
        batch_size=args.batch,
        validation_fraction=0.0,   # use the whole dataset as the validation split
    )

    model = LocalGlobalNet(
        rows=20,
        columns=20,
        local_input_dimension=local_dim,
        global_context_dimension=global_dim,
        global_output_dimension=target_dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    loss, predictions, targets, max_info = evaluate_and_collect(
        model, test_loader, device, MSELoss()
    )
    print(f"Test loss: {loss:.6f}")

    plot_prediction_arrays(
        predictions,
        targets,
        title="Predictions vs Targets",
        save_path=args.save,
        alpha_envelope=args.alpha_envelope,
    )

    sonar_positions = np.where(
        max_info["sonar_placement"] > 0
    )[0]

    unknown_sonar_positions = np.where(
        max_info["unknown_sonar_mask"] > 0
    )[0]

    print("\nMaximum prediction:")
    print(f"Sample:            {max_info['sample']}")
    print(f"Prediction:        {max_info['prediction']:.6f}")
    print(f"Target:            {max_info['target']:.6f}")
    print(f"Sonar positions:   {sonar_positions.tolist()}")
    print(f"Unknown sonar positions:{unknown_sonar_positions.tolist()}")

if __name__ == "__main__":
    main()
