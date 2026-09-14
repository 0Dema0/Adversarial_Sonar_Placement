import argparse
import torch

from trainer.data_builder import build_dataloaders
from trainer.model import LocalGlobalNet
from trainer.loss import MSELoss
from trainer.trainer import Trainer
from trainer.plot import plot_predictions

def parse_indices(s):
    if s is None:
        return None
    return [int(x) for x in s.split(",")]

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

    model.load_state_dict(torch.load(args.model, map_location=device))

    trainer = Trainer(
        model=model,
        optimizer=None,          # not needed for evaluation
        loss=MSELoss(),
        device=device,
    )

    loss = trainer.validate(test_loader, None)  # epoch is not needed for evaluation
    print(f"Test loss: {loss:.6f}")

    plot_predictions(model, test_loader, device, title="Predictions vs Targets", save_path=args.save, alpha_envelope=args.alpha_envelope)

if __name__ == "__main__":
    main()