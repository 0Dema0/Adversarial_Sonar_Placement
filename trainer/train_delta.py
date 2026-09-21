"""Train a LocalGlobalNet that predicts the extra path cost of one unknown sonar.

Uses the unchanged ``PartsDataset`` and ``LocalGlobalNet``. Differences from ordinary
training:

* rows are split into train/validation by ``source_row`` (whole placements), so the
  variants of one placement never straddle the split;
* ``--loss top`` (AnnealedTopPredTopTrueMSELoss in loss.py) trains only on the top-k rows
  by prediction (what an optimizer picks) and the top-k rows by true delta (the real
  best), so both a small delta predicted high and a large delta predicted low are fixed;
* checkpoints are selected on a tail metric with a FIXED definition, so it is
  comparable across epochs while rho is being annealed.

    python -m trainer.train_delta --data data/best_dataset_delta --loss symmetric \
        --rho_final 0.05 --epochs 100 --batch 512 --save models/delta_best.pt

At planning time the cost with sonar s unknown is then estimated as
    J1(P, s) ~= cost_net(features, all known) + delta_net(features, s unknown).
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from trainer.data_builder import PartsDataset, default_collate_fn
from trainer.loss import AnnealedExtremeMSELoss, AnnealedSymmetricTailMSELoss, AnnealedTopPredTopTrueMSELoss, MSELoss
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


# ---------------------------------------------------------------------------
# Tail selection and metrics
# ---------------------------------------------------------------------------

def tail_metrics(p: np.ndarray, t: np.ndarray, fraction: float) -> dict:
    """Fixed-definition validation metrics on 1-D predictions/targets."""
    n = len(t)
    k = max(1, int(math.ceil(fraction * n)))
    err2 = (p - t) ** 2
    sym = np.argsort(-(t + np.abs(p - t)), kind="stable")[:k]   # same key as AnnealedSymmetricTailMSELoss
    by_true = np.argsort(-t, kind="stable")[:k]
    by_pred = np.argsort(-p, kind="stable")[:k]
    true_topk_sum = float(t[by_true].sum())
    return {
        "val_mse": float(err2.mean()),
        "val_tail_sym": float(err2[sym].mean()),        # both failure modes, used for checkpointing
        "val_tail_true": float(err2[by_true].mean()),   # large deltas predicted low
        "val_tail_pred": float(err2[by_pred].mean()),   # rows the model calls large (false highs live here)
        # what an optimizer would get: true value of the model's top-k picks relative to the best possible top-k
        "topk_capture": float(t[by_pred].sum() / true_topk_sum) if true_topk_sum > 0 else float("nan"),
        "topk_overlap": float(len(np.intersect1d(by_pred, by_true)) / k),
    }


class DeltaTrainer(Trainer):
    """Trainer using any loss from trainer/loss.py, with fixed-definition validation.

    Training calls the loss exactly like Trainer does (per-sample squared error + target),
    optionally adding negative_weight * mean(relu(-prediction)^2). Validation computes
    tail metrics with FIXED definitions so checkpoints are comparable across epochs.
    """

    SELECT = {"mse": "val_mse", "extreme": "val_tail_true", "symmetric": "val_tail_sym", "top": "val_top_both"}

    def __init__(self, *args, mode: str = "mse", tail_fraction: float = 0.05, negative_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.tail_fraction = tail_fraction
        self.negative_weight = negative_weight
        self.last_negative_pct = 0.0

    @property
    def rho(self) -> float:
        return float(getattr(self.loss, "current_rho", 1.0))

    def train_epoch(self, loader, epoch):
        self.model.train()
        self.loss.set_epoch(epoch)
        total, negatives, count = 0.0, 0, 0
        for local, global_context, target in loader:
            local = local.to(self.device)
            global_context = global_context.to(self.device)
            target = target.to(self.device)

            prediction = self.model(local, global_context)
            loss_map = (prediction - target) ** 2
            per_sample = loss_map.mean(dim=tuple(range(1, loss_map.ndim)))
            if getattr(self.loss, "needs_prediction", False):
                loss = self.loss(per_sample, target, prediction)
            else:
                loss = self.loss(per_sample, target)
            if self.negative_weight > 0.0:
                loss = loss + self.negative_weight * torch.relu(-prediction).pow(2).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            negatives += int((prediction.detach() < 0).sum().item())
            count += prediction.numel()
        self.last_negative_pct = 100.0 * negatives / max(count, 1)
        return total / len(loader)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        preds, targets = [], []
        for local, global_context, target in loader:
            preds.append(self.model(local.to(self.device), global_context.to(self.device)).cpu().numpy())
            targets.append(target.numpy())
        return np.concatenate(preds).ravel(), np.concatenate(targets).ravel()

    def validate(self, loader, epoch, return_stats=False):
        p, t = self.predict(loader)
        stats = tail_metrics(p, t, self.tail_fraction)
        w = float(getattr(self.loss, "pred_weight", 0.5))
        stats["val_top_both"] = w * stats["val_tail_pred"] + (1.0 - w) * stats["val_tail_true"]
        stats["rho"] = self.rho
        stats["val_select"] = stats[self.SELECT[self.mode]]
        return stats if return_stats else stats["val_select"]


# ---------------------------------------------------------------------------
# Training loop with history
# ---------------------------------------------------------------------------

HISTORY_COLUMNS = ["epoch", "rho", "train_loss", "val_select", "val_mse", "val_top_both", "val_tail_sym", "val_tail_true",
                   "val_tail_pred", "topk_capture", "topk_overlap", "neg_pred_pct"]


def fit_with_history(trainer, train_loader, val_loader, epochs, save_path, last_save_path, history_path):
    history, best = [], float("inf")
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        stats = trainer.validate(val_loader, epoch, return_stats=True)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "neg_pred_pct": trainer.last_negative_pct, **stats}
        history.append(row)
        print(f"{row['epoch']:3d} rho={row['rho']:.3f} train={train_loss:.5f} | val_mse={row['val_mse']:.5f} "
              f"top_both={row['val_top_both']:.5f} tail_true={row['val_tail_true']:.5f} "
              f"tail_pred={row['val_tail_pred']:.5f} | capture={row['topk_capture']:.3f} "
              f"overlap={row['topk_overlap']:.3f} neg={row['neg_pred_pct']:.1f}%")

        if row["val_select"] < best:
            best = row["val_select"]
            torch.save(trainer.model.state_dict(), save_path)
            print(f"    saved best model ({trainer.SELECT[trainer.mode]}={best:.5f})")

        if history_path:
            with open(history_path, "w") as handle:
                handle.write(",".join(HISTORY_COLUMNS) + "\n")
                for r in history:
                    handle.write(",".join(f"{r[c]:.8g}" for c in HISTORY_COLUMNS) + "\n")

    if last_save_path:
        torch.save(trainer.model.state_dict(), last_save_path)
        print(f"Saved last model to {last_save_path}")
    return history


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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
    keep = 0 if max_fraction <= 0.0 else int(np.floor(max_fraction / (1.0 - max_fraction) * len(nonzeros)))
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


# ---------------------------------------------------------------------------

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
                        help="Cap the share of zero-delta rows in the TRAINING split (e.g. 0.25)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--columns", type=int, default=20)
    parser.add_argument("--loss", choices=("mse", "extreme", "symmetric", "top"), default="top",
                        help="top: MSE on top-k by prediction + top-k by true delta; extreme: top-k by true delta; "
                             "symmetric: top-k by true delta + |error|")
    parser.add_argument("--pred-weight", type=float, default=0.5,
                        help="--loss top: share of the loss on the top-k-by-prediction rows (rest: top-k by true)")
    parser.add_argument("--rho_start", type=float, default=1.0)
    parser.add_argument("--rho_final", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--anneal_epochs", type=int, default=80)
    parser.add_argument("--tail-fraction", type=float, default=None,
                        help="Fraction used by the fixed validation tail metrics (default: rho_final)")
    parser.add_argument("--negative-weight", type=float, default=0.0,
                        help="Weight of mean(relu(-prediction)^2); 0 = off")
    parser.add_argument("--init-from", default=None, help="Warm-start from a compatible state_dict (e.g. the cost network)")
    parser.add_argument("--save", default="models/delta_best.pt")
    parser.add_argument("--last-save", default="models/delta_last.pt")
    parser.add_argument("--history", default="models/delta_history.csv", help="CSV of the loss curves ('' to disable)")
    parser.add_argument("--seed", type=int, default=42)
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

    if args.loss == "mse":
        loss = MSELoss()
    else:
        schedule = dict(rho_start=args.rho_start, rho_final=args.rho_final,
                        warmup_epochs=args.warmup_epochs, anneal_epochs=args.anneal_epochs)
        if args.loss == "top":
            loss = AnnealedTopPredTopTrueMSELoss(pred_weight=args.pred_weight, **schedule)
        elif args.loss == "symmetric":
            loss = AnnealedSymmetricTailMSELoss(**schedule)
        else:
            loss = AnnealedExtremeMSELoss(**schedule)
        k_final = math.ceil(args.rho_final * args.batch)
        print(f"Loss: {args.loss} tail, rho {args.rho_start} -> {args.rho_final} "
              f"(warmup {args.warmup_epochs}, anneal {args.anneal_epochs} epochs); "
              f"{k_final} of {args.batch} rows per batch at the end")
        if k_final < 16:
            print(f"    note: only {k_final} rows per batch drive the gradient at rho_final; consider a larger --batch")

    tail_fraction = args.tail_fraction if args.tail_fraction is not None else args.rho_final
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = pick_device()
    print(f"Using device: {device}")
    trainer = DeltaTrainer(model=model, optimizer=optimizer, loss=loss, device=device,
                           mode=args.loss, tail_fraction=tail_fraction, negative_weight=args.negative_weight)

    for path in (args.save, args.last_save, args.history):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    fit_with_history(trainer, train_loader, val_loader, args.epochs,
                     args.save, args.last_save, args.history or None)


if __name__ == "__main__":
    main()