from __future__ import annotations

import time
import numpy as np


def save_dataset(path: str, dataset: dict[str, np.ndarray]) -> None:
    """Save dataset as compressed npz."""
    np.savez_compressed(path, **dataset)


class ETA:
    def __init__(self):
        self.start = time.time()

    def update(self, done: int, total: int) -> str:
        elapsed = time.time() - self.start
        remaining = elapsed / max(done, 1) * (total - done)

        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)

        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"


def stack_samples(samples: list[dict]) -> dict[str, np.ndarray]:
    """Stack a list of feature dictionaries."""

    keys = samples[0].keys()

    return {
        key: np.stack([sample[key] for sample in samples])
        for key in keys
    }