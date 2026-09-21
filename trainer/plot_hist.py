"""Plot the loss curves written by trainer/train_delta.py.

    python plot_history.py models/delta_history.csv
"""
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False   # trainer/plot.py turns this on; not needed here
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "models/delta_history.csv"
d = np.genfromtxt(path, delimiter=",", names=True)

plt.figure(figsize=(7, 4.5))
plt.plot(d["epoch"], d["train_loss"], label="train")
plt.plot(d["epoch"], d["val_mse"], label="validation")
plt.plot(d["epoch"], d["val_select"], label="validation")
plt.yscale("log")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.grid(True, which="both", alpha=0.3); plt.legend()
best = int(np.argmin(d["val_mse"]))
plt.axvline(d["epoch"][best], color="k", ls="--", lw=1)
plt.title(f"best val MSE {d['val_mse'][best]:.5f} @ epoch {int(d['epoch'][best])}")
plt.tight_layout(); plt.savefig(path.replace(".csv", ".png"), dpi=150); plt.show()