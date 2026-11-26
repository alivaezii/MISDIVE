"""Feature views and training-only preprocessing."""
from __future__ import annotations

import numpy as np

from .paths import views


def check_extractor_layout() -> None:
    """Fail if the installed extractor's group order or dimensions differ from the frozen view file."""
    import thrember
    layout = [[fe.name, fe.dim] for fe in thrember.PEFeatureExtractor().features]
    if layout != views()["extractor_groups"]:
        raise RuntimeError(f"extractor layout {layout} differs from configs/feature_views.json")


def view_columns(view: str) -> np.ndarray:
    """Raw-column indices (0..2567) of a view, from explicit frozen ranges."""
    ranges = views()[view]["column_ranges"]
    return np.concatenate([np.arange(a, b + 1) for a, b in ranges])


class Preprocessor:
    """Signed log1p, training z-score, removal of training-constant columns, clipping to [-clip, clip]."""

    def __init__(self, clip: float = 20.0, std_floor: float = 1e-6, chunk: int = 5000):
        self.clip, self.std_floor, self.chunk = clip, std_floor, chunk

    @staticmethod
    def _signed_log(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, arrays: list, masks: list) -> "Preprocessor":
        s1 = np.zeros(2568)
        s2 = np.zeros(2568)
        n = 0
        for arr, m in zip(arrays, masks):
            idx = np.flatnonzero(m)
            for a in range(0, len(idx), self.chunk):
                t = self._signed_log(arr[idx[a:a + self.chunk]])
                s1 += t.sum(0)
                s2 += (t ** 2).sum(0)
                n += t.shape[0]
        mean = s1 / n
        self.mean_ = mean.astype(np.float32)
        self.std_ = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0)).astype(np.float32)
        self.keep_ = np.flatnonzero(self.std_ > self.std_floor)
        return self

    def transform(self, arrays: list, masks: list, dtype=np.float16) -> np.ndarray:
        n = int(sum(np.count_nonzero(m) for m in masks))
        out = np.empty((n, len(self.keep_)), dtype=dtype)
        mu, sd = self.mean_[self.keep_], self.std_[self.keep_]
        r = 0
        for arr, m in zip(arrays, masks):
            idx = np.flatnonzero(m)
            for a in range(0, len(idx), self.chunk):
                t = self._signed_log(arr[idx[a:a + self.chunk]][:, self.keep_])
                t = np.clip((t - mu) / sd, -self.clip, self.clip)
                out[r:r + len(t)] = t.astype(dtype)
                r += len(t)
        return out

    def view_index(self, view: str) -> np.ndarray:
        """Positions of a view's retained columns within the preprocessed matrix."""
        return np.flatnonzero(np.isin(self.keep_, view_columns(view)))

    def state(self) -> dict:
        return {"mean": self.mean_, "std": self.std_, "keep": self.keep_}

    @classmethod
    def from_state(cls, z) -> "Preprocessor":
        p = cls()
        p.mean_, p.std_, p.keep_ = z["mean"], z["std"], z["keep"]
        return p
