"""Training with validation-only early stopping, and branch-threshold calibration."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import TwoBranch, score
from .thresholds import joint_or_tpr


def fit(x_tr, y_tr, x_va, y_va, idx_s, idx_c, seed: int, cfg: dict, log=print):
    """Train one two-branch model; the loss is separable, so the branches are fitted independently."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    model = TwoBranch(idx_s, idx_c)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_weight = torch.tensor(float((y_tr == 0).sum() / max(1, (y_tr == 1).sum())))
    x_t = torch.from_numpy(x_tr)                     # float16 storage; batches are cast to float32
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = len(y_t)
    best, best_state, best_epoch, bad, history = -1.0, None, -1, 0, []
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(1000 * seed + epoch))
        loss_sum = 0.0
        for a in range(0, n, 512):
            b = perm[a:a + 512]
            xb, yb = x_t[b].float(), y_t[b]
            zs, zc = model(xb)
            loss = F.binary_cross_entropy_with_logits(zs, yb, pos_weight=pos_weight) + \
                F.binary_cross_entropy_with_logits(zc, yb, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(b)
        v = score(model, x_va)
        tpr = joint_or_tpr(v[:, 0], v[:, 1], y_va, 0.01, 51)
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "val_or_tpr_at_1pct": tpr, "seconds": time.time() - t0})
        log(f"  seed {seed} epoch {epoch}: loss={loss_sum / n:.4f} val_or_tpr={tpr:.4f}")
        if tpr > best + 1e-6:
            best, best_state, best_epoch, bad = tpr, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_or_tpr_at_1pct": best, "epochs_run": len(history),
                   "train_seconds": time.time() - t0, "history": history}
