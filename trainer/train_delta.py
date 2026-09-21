"""Train a LocalGlobalNet that predicts the extra path cost of one unknown sonar.

Uses the unchanged ``PartsDataset``, ``LocalGlobalNet`` and ``Trainer``. The only
difference from ordinary training is the train/validation split: a delta dataset
built with ``--mode all`` contains every placement once per sonar, so rows are split
by ``source_row`` (whole placements go to either train or validation) to avoid
optimistic validation scores.

    python -m trainer.train_delta --data data/delta_1unknown_all_parts \
        --epochs 100 --save models/delta_best.pt

    # optional warm start from the existing all-known cost network (same dims)
    python -m trainer.train_delta --data data/delta_1unknown_all_parts \
        --init-from models/cost_best.pt --save models/delta_best.pt

At planning time the cost with sonar s unknown is then estimated as
    J1(P, s) ~= cost_net(features, all known) + delta_net(features, s unknown).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from trainer.data_builder import PartsDataset, default_collate_fn
from trainer.loss import AnnealedExtremeMSELoss, MSELoss
from trainer.model import LocalGlobalNet
from trainer.trainer import Trainer


def parse_indices(s):
    return None if s is None else [int(x) for x in s.split(",")]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_model(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total ({trainable:,} trainable)")
    for name, module in (("local", ["local_hidden_weight", "local_hidden_bias", "local_output_weight", "local_output_bias"]),
                         ("global", ["global_hidden", "global_output"])):
        count = 0
        for attr in module:
            obj = getattr(model, attr)
            count += sum(p.numel() for p in obj.parameters()) if hasattr(obj, "parameters") else obj.numel()
        print(f"  {name:6s}: {count:,}")


def fit_with_history(trainer, train_loader, val_loader, epochs, save_path, last_save_path, history_path):
    """Same loop as Trainer.fit, but records the curves and writes them to CSV."""
    history, best = [], float("inf")
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        stats = trainer.validate(val_loader, epoch, return_stats=True)
        row = {"epoch": epoch + 1, "train_loss": train_loss,
               "val_select": float(stats["val_select"]), "val_mse": float(stats["val_mse"])}
        history.append(row)
        print(f"{row['epoch']:3d} train={train_loss:.6f} val={row['val_select']:.6f} val_mse={row['val_mse']:.6f}")

        if row["val_select"] < best:
            best = row["val_select"]
            torch.save(trainer.model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

        if history_path:
            with open(history_path, "w") as handle:
                handle.write("epoch,train_loss,val_select,val_mse\n")
                for r in history:
                    handle.write(f"{r['epoch']},{r['train_loss']:.8f},{r['val_select']:.8f},{r['val_mse']:.8f}\n")

    if last_save_path:
        torch.save(trainer.model.state_dict(), last_save_path)
        print(f"Saved last model to {last_save_path}")
    return history


def target_column(dataset: PartsDataset, data_dir: Path, train_cells):
    """The delta value of every row, i.e. outputs[:, start_index]."""
    cell = train_cells[0] if train_cells else 0
    return np.asarray(np.load(data_dir / "outputs.npy", mmap_mode="r")[:len(dataset), cell], dtype=np.float64)


def limit_zero_fraction(indices: list[int], values: np.ndarray, max_fraction: float, rng: np.random.Generator) -> list[int]:
    """Drop random zero-target rows until they are at most ``max_fraction`` of the rows."""
    idx = np.asarray(indices)
    is_zero = values[idx] <= 0.0
    zeros, nonzeros = idx[is_zero], idx[~is_zero]
    if max_fraction >= 1.0 or len(zeros) == 0:
        return indices
    if max_fraction <= 0.0:
        keep = 0
    else:
        keep = int(np.floor(max_fraction / (1.0 - max_fraction) * len(nonzeros)))
    keep = min(keep, len(zeros))
    if keep == len(zeros):
        return indices
    kept_zeros = rng.choice(zeros, size=keep, replace=False) if keep else np.array([], dtype=idx.dtype)
    out = np.sort(np.concatenate((nonzeros, kept_zeros)))
    print(f"Zero-target rows in train: {len(zeros)} -> {keep} "
          f"({len(zeros) / len(idx) * 100:.1f}% -> {keep / max(len(out), 1) * 100:.1f}% of {len(out)} rows)")
    return out.tolist()


def grouped_split(dataset: PartsDataset, data_dir: Path, validation_fraction: float, max_placements: int | None, seed: int):
    n = len(dataset)
    group_path = data_dir / "source_row.npy"
    groups = np.load(group_path, mmap_mode="r")[:n] if group_path.exists() else np.arange(n)
    if not group_path.exists():
        print("source_row.npy not found; falling back to a per-row split")

    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    if max_placements is not None and max_placements < len(unique):
        unique = np.sort(rng.choice(unique, size=max_placements, replace=False))
    n_val = int(round(validation_fraction * len(unique)))
    val_groups = rng.choice(unique, size=n_val, replace=False) if n_val > 0 else np.array([], dtype=unique.dtype)

    in_use = np.isin(groups, unique)
    is_val = np.isin(groups, val_groups)
    train_idx = np.flatnonzero(in_use & ~is_val).tolist()
    val_idx = np.flatnonzero(in_use & is_val).tolist()
    print(f"Placements: {len(unique) - n_val} train / {n_val} val -> rows: {len(train_idx)} train / {len(val_idx)} val")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Delta _parts directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_cells", type=str, default="0", help="Start index (target cell)")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-placements", type=int, default=None, help="Use a random subset of placements")
    parser.add_argument("--max-zero-fraction", type=float, default=1.0,
                        help="Cap the share of zero-delta rows in the TRAINING split (e.g. 0.25); "
                             "validation is left untouched so its metrics stay representative")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--columns", type=int, default=20)
    parser.add_argument("--loss", choices=("mse", "extreme"), default="mse")
    parser.add_argument("--init-from", default=None, help="Warm-start from a compatible state_dict (e.g. the cost network)")
    parser.add_argument("--save", default="models/delta_best.pt")
    parser.add_argument("--last-save", default="models/delta_last.pt")
    parser.add_argument("--history", default="models/delta_history.csv", help="CSV of the loss curves ('' to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho_start", type=float, default=1)
    parser.add_argument("--rho_final", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=float, default=10)
    parser.add_argument("--anneal_epochs", type=float, default=80)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data_dir = Path(args.data)
    dataset = PartsDataset(data_dir, train_cells=parse_indices(args.train_cells))
    train_ds, val_ds = grouped_split(dataset, data_dir, args.validation_fraction, args.max_placements, args.seed)

    if args.max_zero_fraction < 1.0:
        if not 0.0 <= args.max_zero_fraction <= 1.0:
            raise ValueError("--max-zero-fraction must be in [0, 1]")
        values = target_column(dataset, data_dir, parse_indices(args.train_cells))
        train_ds = Subset(dataset, limit_zero_fraction(
            train_ds.indices, values, args.max_zero_fraction, np.random.default_rng(args.seed)))
        if len(train_ds.indices) == 0:
            raise ValueError("No training rows left after --max-zero-fraction")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin, collate_fn=default_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=max(0, args.num_workers // 2),
                            pin_memory=pin, collate_fn=default_collate_fn)

    model = LocalGlobalNet(
        rows=args.rows,
        columns=args.columns,
        local_input_dimension=dataset.local_input_dimension,
        global_context_dimension=dataset.global_context_dimension,
        global_output_dimension=dataset.target_dimension,
    )
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print(f"Initialised weights from {args.init_from}")

    describe_model(model)

    loss = MSELoss() if args.loss == "mse" else AnnealedExtremeMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = pick_device()
    print(f"Using device: {device}")
    trainer = Trainer(model=model, optimizer=optimizer, loss=loss, device=device)

    for path in (args.save, args.last_save, args.history):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    fit_with_history(trainer, train_loader, val_loader, args.epochs,
                     args.save, args.last_save, args.history or None)


if __name__ == "__main__":
    main()