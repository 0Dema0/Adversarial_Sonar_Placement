import matplotlib as mpl
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
})
import matplotlib.pyplot as plt
import numpy as np
import torch
import alphashape
from shapely.geometry import Polygon, MultiPolygon


@torch.no_grad()
def plot_predictions(model, loader, device, title="Predictions vs Targets", save_path=None, alpha_envelope=False):
    model.eval()

    predictions = []
    targets = []

    for local, global_context, target in loader:
        local = local.to(device)
        global_context = global_context.to(device)

        prediction = model(local, global_context)

        predictions.append(prediction.cpu().numpy())
        targets.append(target.numpy())

    predictions = np.concatenate(predictions, axis=0).ravel()
    targets = np.concatenate(targets, axis=0).ravel()

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, predictions, s=5, alpha=0.3, label="Samples")

    points = np.column_stack((targets, predictions))

    if alpha_envelope:
        alpha = 0.05   # smaller = tighter boundary, larger = smoother

        shape = alphashape.alphashape(points, alpha)

        if isinstance(shape, Polygon):
            polygons = [shape]
        elif isinstance(shape, MultiPolygon):
            polygons = list(shape.geoms)
        else:
            polygons = []

        for poly in polygons:
            boundary = np.asarray(poly.exterior.coords)

            plt.fill(
                boundary[:, 0],
                boundary[:, 1],
                alpha=0.15,
                label="Alpha Envelope",
            )

            plt.plot(
                boundary[:, 0],
                boundary[:, 1],
                "k--",
                linewidth=1,
            )

    lim_min = min(targets.min(), predictions.min())
    lim_max = max(targets.max(), predictions.max())
    plt.plot([-1, 14.5], [-1, 14.5], "r--", linewidth=2)

    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.grid(True)
    plt.xlim(-1, 14.5)
    plt.axis("equal")
    plt.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            pad_inches=0.02,
            format="pgf"
        )
    else:
        plt.show()