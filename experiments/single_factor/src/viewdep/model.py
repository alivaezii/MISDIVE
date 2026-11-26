"""Two-branch view-specific detector (the INDEP_semantic architecture of the SENTRA project)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mlp(d_in: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


class TwoBranch(nn.Module):
    """Branch S reads the structural columns and branch C the content columns; no shared parameters."""

    def __init__(self, idx_s: np.ndarray, idx_c: np.ndarray):
        super().__init__()
        self.register_buffer("idx_s", torch.as_tensor(np.asarray(idx_s), dtype=torch.long))
        self.register_buffer("idx_c", torch.as_tensor(np.asarray(idx_c), dtype=torch.long))
        self.s = mlp(len(idx_s))
        self.c = mlp(len(idx_c))

    def forward(self, x: torch.Tensor):
        return self.s(x.index_select(1, self.idx_s)).squeeze(-1), self.c(x.index_select(1, self.idx_c)).squeeze(-1)


@torch.no_grad()
def score(model: TwoBranch, x: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Logits of both branches, shape (n, 2): column 0 = S, column 1 = C."""
    model.eval()
    out = []
    for a in range(0, len(x), batch):
        s, c = model(torch.from_numpy(np.asarray(x[a:a + batch], dtype=np.float32)))
        out.append(torch.stack([s, c], 1))
    return torch.cat(out).numpy().astype(np.float64)
