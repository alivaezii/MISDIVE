"""MISDIVE Phase 1, ablation AB2: independently selected (selection-matched) MLP detectors for Win64.

Replays the clean-run Win64 two-branch MLP training exactly (same code path as viewdep.train.fit:
same seeds, initial weights, batch order, optimiser, loss), and changes only checkpoint selection:
each branch keeps the epoch that maximises ITS OWN validation TPR at 0.5% validation FPR (its own
frozen-threshold rule), with patience 5 on that branch's own metric. Because the two branch losses are
separable (no shared parameters; summed loss), each branch's trajectory is independent of its partner,
so this is equivalent to training six single-branch MLPs from the same initial weights. Training
continues while either branch is still improving (max 25 epochs).

Reproduction check: the joint OR-TPR history of the replay must equal the clean-run training log, and at
the clean-run best epoch the replayed branch logits must equal the clean-run validation logits.

Reads the clean repository read-only (training weeks 1-40 and validation weeks 41-46 only) and writes
only to OUT. Requires explicit authorisation flag.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(r"F:\ELSIVIER\SENTRA_AEiC2027")
OUT = Path(r"F:\ELSIVIER\MISDIVE_Phase1_pool\win64")
SEEDS = (42, 123, 2026)
ALPHA, EPOCHS, PATIENCE = 0.005, 25, 5


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def own_tpr(v_ben, v_mal):
    s = np.sort(v_ben)[::-1]
    thr = s[int(np.floor(ALPHA * len(s)))]
    return float((v_mal > thr).mean()), float(thr)


def main():
    if "--authorised-T" not in sys.argv:
        sys.exit("Refusing to train: authorisation T has not been given.")
    import pandas as pd
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(REPO / "src"))
    from viewdep.features import Preprocessor
    from viewdep.model import TwoBranch, score
    from viewdep.thresholds import joint_or_tpr
    OUT.mkdir(parents=True, exist_ok=True)
    prep = REPO / "data" / "prep" / "win64"
    art = REPO / "artifacts" / "win64"
    pp = Preprocessor.from_state(np.load(art / "preprocessor.npz"))
    idx_s, idx_c = pp.view_index("structural"), pp.view_index("content")
    x_tr = np.load(prep / "train.npy")
    x_va = np.load(prep / "validation.npy")
    y_tr = pd.read_parquet(prep / "train_meta.parquet").label.values.astype(np.int64)
    y_va = pd.read_parquet(prep / "validation_meta.parquet").label.values.astype(np.int64)
    ben, mal = y_va == 0, y_va == 1
    record = {"design": "AB2 selection-matched MLP (replayed clean-run training; own-metric checkpoint selection)",
              "own_metric": "validation TPR at the floor(0.005*N0)-th largest validation benign score",
              "epochs_max": EPOCHS, "patience": PATIENCE, "torch": torch.__version__, "seeds": {}}
    for seed in SEEDS:
        clean_log = json.loads((art / f"train_log_s{seed}.json").read_text())
        clean_val = np.load(art / f"val_logits_s{seed}.npy")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(8)
        model = TwoBranch(idx_s, idx_c)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        pos_weight = torch.tensor(float((y_tr == 0).sum() / max(1, (y_tr == 1).sum())))
        x_t, y_t = torch.from_numpy(x_tr), torch.from_numpy(y_tr.astype(np.float32))
        n = len(y_t)
        br = {b: dict(best=-1.0, state=None, epoch=-1, bad=0, stopped=False) for b in ("S", "C")}
        hist, t0, repro = [], time.time(), {}
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(1000 * seed + epoch))
            for a in range(0, n, 512):
                b = perm[a:a + 512]
                xb, yb = x_t[b].float(), y_t[b]
                zs, zc = model(xb)
                loss = F.binary_cross_entropy_with_logits(zs, yb, pos_weight=pos_weight) + \
                    F.binary_cross_entropy_with_logits(zc, yb, pos_weight=pos_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
            v = score(model, x_va)
            joint = joint_or_tpr(v[:, 0], v[:, 1], y_va, 0.01, 51)
            row = {"epoch": epoch, "joint_or_tpr_at_1pct": joint, "seconds": time.time() - t0}
            for k, (b, col) in enumerate((("S", 0), ("C", 1))):
                tpr, _ = own_tpr(v[ben, col], v[mal, col])
                row[f"own_tpr_{b}"] = tpr
                st = br[b]
                if st["stopped"]:
                    continue
                if tpr > st["best"] + 1e-6:
                    st.update(best=tpr, state={kk: vv.clone() for kk, vv in
                                               (getattr(model, b.lower())).state_dict().items()}, epoch=epoch, bad=0)
                else:
                    st["bad"] += 1
                    if st["bad"] >= PATIENCE:
                        st["stopped"] = True
            if epoch == clean_log["best_epoch"]:
                repro["max_abs_diff_vs_clean_val_logits"] = float(np.abs(v - clean_val).max())
            if epoch < len(clean_log["history"]):
                repro.setdefault("joint_history_abs_diff", []).append(
                    abs(joint - clean_log["history"][epoch]["val_or_tpr_at_1pct"]))
            hist.append(row)
            print(f"seed {seed} epoch {epoch}: joint={joint:.4f} S={row['own_tpr_S']:.4f} C={row['own_tpr_C']:.4f}", flush=True)
            if br["S"]["stopped"] and br["C"]["stopped"] and epoch >= len(clean_log["history"]) - 1:
                break
        srec = {"history": hist, "reproduction": {"clean_best_epoch": clean_log["best_epoch"], **repro,
                                                   "max_joint_history_abs_diff": max(repro.get("joint_history_abs_diff", [np.nan]))}}
        for b, col in (("S", 0), ("C", 1)):
            st = br[b]
            getattr(model, b.lower()).load_state_dict(st["state"])
        v = score(model, x_va)
        for b, col in (("S", 0), ("C", 1)):
            name = f"AM{b}_{seed}"
            path = OUT / f"{name}.pt"
            torch.save(getattr(model, b.lower()).state_dict(), path)
            np.save(OUT / f"val_logits_{name}.npy", v[:, col])
            tpr, thr = own_tpr(v[ben, col], v[mal, col])
            srec[name] = {"sha256": sha(path), "selected_epoch": br[b]["epoch"], "own_val_tpr": tpr, "threshold": thr,
                          "clean_best_epoch_joint": clean_log["best_epoch"],
                          "columns": int(len(idx_s if b == "S" else idx_c))}
            print(name, {k: v_ for k, v_ in srec[name].items()}, flush=True)
        record["seeds"][str(seed)] = srec
        (OUT / "ab2_train_record.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
