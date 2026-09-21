import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from trainer.data_builder import PartsDataset, default_collate_fn
from trainer.model import LocalGlobalNet
from trainer.train_delta import grouped_split, parse_indices, pick_device


def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for local, glob, target in loader:
            preds.append(model(local.to(device), glob.to(device)).cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(preds).ravel(), np.concatenate(targets).ravel()


def plot(p: np.ndarray, t: np.ndarray, save_path: str | None, title: str) -> None:
    import matplotlib
    matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt

    top = max(float(t.max()), float(p.max())) * 1.05
    low = min(0.0, float(p.min()))
    zeros = t==0
    print(np.max(p[zeros]))
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(t, p, s=4)
    ax.plot([0, top], [low, top], "k--", lw=1)
    ax.set_xlim(0, top); ax.set_ylim(low, top)
    ax.set_xlabel("true delta")
    ax.set_ylabel("predicted delta")
    ax.set_title(title)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--train_cells", type=str, default="0")
    parser.add_argument("--split", choices=("val", "all"), default="val",
                        help="val: the held-out placements from training (same seed/fraction); all: every row")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="must match training")
    parser.add_argument("--max-placements", type=int, default=None, help="must match training")
    parser.add_argument("--seed", type=int, default=42, help="must match training")
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--columns", type=int, default=20)
    parser.add_argument("--save", default=None, help="path for the plot (png)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    dataset = PartsDataset(data_dir, train_cells=parse_indices(args.train_cells))
    if args.split == "val":
        _, subset = grouped_split(dataset, data_dir, args.validation_fraction, args.max_placements, args.seed)
    else:
        subset = Subset(dataset, list(range(len(dataset))))
    loader = DataLoader(subset, batch_size=args.batch, shuffle=False, collate_fn=default_collate_fn)

    device = pick_device()
    print(f"Using device: {device}")
    model = LocalGlobalNet(
        rows=args.rows,
        columns=args.columns,
        local_input_dimension=dataset.local_input_dimension,
        global_context_dimension=dataset.global_context_dimension,
        global_output_dimension=dataset.target_dimension,
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    p, t = predict(model, loader, device)
   
    plot(p, t, args.save, f"Delta predictions ({args.split})")


if __name__ == "__main__":
    main()