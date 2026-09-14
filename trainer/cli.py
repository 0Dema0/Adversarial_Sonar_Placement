import argparse
from os import name
import torch

from trainer.data_builder import build_dataloaders
from trainer.model import LocalGlobalNet
from trainer.loss import MSELoss, CVaRLoss, AnnealedLoss, AnnealedExtremeMSELoss, AnnealedRandomMSELoss
from trainer.trainer import Trainer


def build_loss(name: str):
    if name == "mse":
        return MSELoss()

    if name == "cvar":
        return CVaRLoss(alpha=0.1, learnable_t=True)

    if name == "annealed":
        return AnnealedLoss(
            mse_loss=MSELoss(),
            cvar_loss=CVaRLoss(alpha=0.2, learnable_t=True),
            lambda_start=1.0, # weight of MSE at the start of training
            lambda_final=0.0, # weight of MSE at the end of training
            alpha_start=0.2, # weight of CVaR at the start of training
            alpha_final=0.0005, # weight of CVaR at the end of training
            warmup_epochs=10, # number of epochs to train with only MSE before annealing
            anneal_epochs=80, # number of epochs to anneal from lambda_start to lambda_final and alpha_start to alpha_final
        )

    if name == "extreme":
        return AnnealedExtremeMSELoss(
            rho_start=1.0,
            rho_final=0.001,
            warmup_epochs=10,
            anneal_epochs=80,
        )

    if name == "random":
        return AnnealedRandomMSELoss(
            rho_start=1.0,
            rho_final=0.001,
            warmup_epochs=10,
            anneal_epochs=80,
        )

    if name == "middle":
            return AnnealedRandomMSELoss(
                rho_start=1.0,
                rho_final=0.001,
                warmup_epochs=10,
                anneal_epochs=80,
            )
    
    raise ValueError(f"Unknown loss: {name}")


def parse_indices(s):
    if s is None:
        return None
    return [int(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=250000, help="Use only the first N samples from the dataset before splitting into train/val.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of the selected dataset to reserve for validation.")

    parser.add_argument("--loss", type=str, default="extreme", choices=["mse", "cvar", "annealed", "extreme", "random", "middle"])
    parser.add_argument("--train_cells", type=str, default="0")

    parser.add_argument("--save-path", type=str, default="best_knownandunknownBIG_model_BESTDATASET_1UNKNOWN_extreme.pt")
    parser.add_argument("--last-save-path", type=str, default="last_knownandunknownBIG_model_BESTDATASET_1UNKNOWN_extreme.pt")  # default="last_model_HACK_ALLKNOWN_annealed.pt"

    args = parser.parse_args()
    train_cells = parse_indices(args.train_cells)

    train_loader, val_loader, local_dim, global_dim, _, target_dim = build_dataloaders(
        args.data,
        train_cells=train_cells,
        batch_size=args.batch,
        validation_fraction=args.val_fraction,
        max_samples=args.max_samples,
    )

    model = LocalGlobalNet(
        rows=20,
        columns=20,
        local_input_dimension=local_dim,
        global_context_dimension=global_dim,
        global_output_dimension=target_dim,
    )

    loss_fn = build_loss(args.loss)
    params = list(model.parameters()) + list(loss_fn.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
    )

    trainer.fit(
        train_loader,
        val_loader,
        args.epochs,
        save_path=args.save_path,
        last_save_path=args.last_save_path,
    )

if __name__ == "__main__":
    main()

    # Best model for 0 unknowns: best_model_BESTDATASET_ALLKNOWN_extreme.pt