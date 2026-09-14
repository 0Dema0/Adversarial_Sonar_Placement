from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("pgf")

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",   # or "xelatex", "lualatex"
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
})


def _is_parts_dir(path: Path) -> bool:
    return path.is_dir() and (path / "meta.json").exists() and (path / "outputs.npy").exists()


def _load_outputs_from_parts(parts_dir: Path) -> np.ndarray:
    arr = np.load(parts_dir / "outputs.npy", mmap_mode="r")
    return np.asarray(arr)


def _load_outputs_from_npz(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        if "outputs" not in data:
            raise KeyError(f"Key 'outputs' not found in {npz_path}")
        return np.asarray(data["outputs"])


def _select_output(outputs: np.ndarray, output_index: int | None) -> np.ndarray:
    if outputs.ndim == 0:
        vals = outputs.reshape(1)
    elif outputs.ndim == 1:
        vals = outputs
    else:
        per_sample = outputs.reshape(outputs.shape[0], -1)
        if output_index is None:
            raise ValueError(
                "This dataset has multidimensional outputs. Provide --output-index to select which output cell to plot."
            )
        if output_index < 0 or output_index >= per_sample.shape[1]:
            raise ValueError(
                f"--output-index {output_index} out of range for flattened output size {per_sample.shape[1]}"
            )
        vals = per_sample[:, output_index]

    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return vals


def _load_one(path_str: str, output_index: int | None) -> tuple[str, np.ndarray]:
    p = Path(path_str)
    if _is_parts_dir(p):
        outputs = _load_outputs_from_parts(p)
    elif p.is_file() and p.suffix.lower() == ".npz":
        outputs = _load_outputs_from_npz(p)
    else:
        raise ValueError(
            f"Unsupported input '{p}'. Provide either a parts folder with meta.json+outputs.npy or a .npz file."
        )

    vals = _select_output(outputs, output_index)
    return p.name, vals


def _parse_labels(labels_arg: list[str] | None, n: int) -> list[str] | None:
    if labels_arg is None:
        return None
    if len(labels_arg) != n:
        raise ValueError(f"--labels expects {n} values, got {len(labels_arg)}")
    return labels_arg


def _sanitize_for_filename(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    safe = safe.strip("_")
    return safe or "series"


def _build_fixed_width_bins(vals: np.ndarray, width: float) -> np.ndarray:
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    start = np.floor(vmin / width) * width
    end = np.ceil(vmax / width) * width
    if end <= start:
        end = start + width
    bins = np.arange(start, end + width, width, dtype=np.float64)
    if bins.size < 2:
        bins = np.array([start, start + width], dtype=np.float64)
    return bins


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot output distributions from one or more dataset sources (.npz and/or parts folders)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input paths: each can be a .npz file or a parts folder containing meta.json and outputs.npy",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for legend (must match number of inputs)",
    )
    parser.add_argument(
        "--output-index",
        type=int,
        default=0,
        help="Flattened output cell index to plot for multidimensional outputs. Not required for 1D outputs.",
    )
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins")
    parser.add_argument(
        "--density",
        action="store_true",
        default=False,
        help="Normalize histograms to probability density",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Histogram transparency",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Optional x-axis maximum (useful for clipping extreme outliers)",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=1.0,
        help="Bin width for the second per-input histogram (default: 1.0)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Outputs Distribution",
        help="Plot title",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output image path. If omitted, the figure is shown interactively.",
    )

    args = parser.parse_args()

    if args.bin_width <= 0:
        print("Error: --bin-width must be > 0", file=sys.stderr)
        return 2

    try:
        labels = _parse_labels(args.labels, len(args.inputs))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    series: list[tuple[str, np.ndarray]] = []
    for idx, item in enumerate(args.inputs):
        try:
            default_label, vals = _load_one(item, args.output_index)
            label = labels[idx] if labels is not None else default_label
            if vals.size == 0:
                print(f"Warning: no finite values found for {item}; skipping", file=sys.stderr)
                continue
            series.append((label, vals))
            print(
                f"Loaded {item}: n={vals.size} min={vals.min():.6f} max={vals.max():.6f} mean={vals.mean():.6f}",
                flush=True,
            )
        except Exception as exc:
            print(f"Error loading {item}: {exc}", file=sys.stderr)
            return 2

    if not series:
        print("No valid inputs to plot.", file=sys.stderr)
        return 2

    plt.figure(figsize=(10, 6))

    for label, vals in series:
        use_vals = vals
        if args.xmax is not None:
            use_vals = use_vals[use_vals <= args.xmax]
        plt.hist(
            use_vals,
            bins=args.bins,
            alpha=args.alpha,
            density=args.density,
            label=label,
            histtype="stepfilled",
        )

    plt.title(args.title)
    if args.output_index is None:
        plt.xlabel("outputs")
    else:
        plt.xlabel(f"outputs[index={args.output_index}]")
    plt.ylabel("density" if args.density else "count")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02, format="pgf")
        print(f"Saved plot to: {out_path}")

    # Second plot: one figure per input, with fixed-width bins and per-bin count/percentage labels.
    save_base = Path(args.save) if args.save else None
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, vals) in enumerate(series):
        use_vals = vals
        if args.xmax is not None:
            use_vals = use_vals[use_vals <= args.xmax]

        if use_vals.size == 0:
            print(f"Warning: no values left after xmax filtering for {label}; skipping fixed-width plot", file=sys.stderr)
            continue

        bins = _build_fixed_width_bins(use_vals, args.bin_width)
        counts, edges = np.histogram(use_vals, bins=bins)
        total = int(np.sum(counts))

        color = colors[idx % len(colors)] if colors else None
        plt.figure(figsize=(11, 6))
        counts_plot, edges_plot, patches = plt.hist(
            use_vals,
            bins=bins,
            color=color,
            alpha=0.55,
            density=False,
            label=label,
            histtype="bar",
            edgecolor="black",
            linewidth=0.8,
        )

        ymax = float(np.max(counts_plot)) if counts_plot.size else 0.0
        plt.ylim(top=ymax * 1.15)
        text_offset = 0.01 * ymax if ymax > 0 else 0.1
        for c, left, right in zip(counts, edges[:-1], edges[1:]):
            if c <= 0:
                continue
            x = 0.5 * (left + right)
            pct = (100.0 * c / total) if total > 0 else 0.0
            plt.text(
                x,
                c + text_offset,
                f"{int(c)}\n{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

        subtitle = f"{args.title} | {label} | bin width={args.bin_width:g}"
        plt.title(subtitle)
        if args.output_index is None:
            plt.xlabel("outputs")
        else:
            plt.xlabel(f"outputs[index={args.output_index}]")
        plt.ylabel("count")
        plt.grid(True, axis="y", alpha=0.2)
        plt.legend()
        plt.tight_layout()

        if save_base is not None:
            stem = save_base.stem
            suffix = ".pgf"
            safe_label = _sanitize_for_filename(label)
            per_input_name = f"{stem}_fixed_width_{idx+1}_{safe_label}{suffix}"
            per_input_path = save_base.with_name(per_input_name)
            per_input_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(per_input_path, bbox_inches="tight", pad_inches=0.02, format="pgf")
            print(f"Saved fixed-width plot to: {per_input_path}")

    if args.save is None:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
