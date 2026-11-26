"""Resumable stages: prepare, preprocess, train, freeze, evaluate.

Usage: python -m viewdep.pipeline <stage> <dotnet|win64>
Each stage skips work whose outputs already exist; `evaluate` refuses to run unless the frozen
model and threshold record matches the files on disk.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import ingest
from .dependence import bootstrap, contrast, covariance_identity, decide, pair_stats
from .features import Preprocessor, check_extractor_layout
from .model import TwoBranch, score
from .paths import ARTIFACTS, CONFIG, DATA, RAW, RESULTS, fmt_dir, protocol
from .thresholds import at_fpr
from .train import fit


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def stage_prepare(fmt: str) -> None:
    arch = protocol()["dataset"]["archives"][fmt]
    for part in ("train", "test"):
        digest = sha256(RAW / arch[part])
        if digest != arch[f"{part}_sha256"]:
            raise RuntimeError(f"{arch[part]}: SHA256 {digest} does not match the pinned revision")
    check_extractor_layout()
    ingest.ingest(fmt)
    rep = ingest.audit(fmt)
    out = fmt_dir(RESULTS, fmt)
    (out / "cohort_audit.json").write_text(json.dumps(rep, indent=1))
    (DATA / "weeks" / fmt / "counts_week_class.csv").replace(out / "counts_week_class.csv") \
        if not (out / "counts_week_class.csv").exists() else None
    print(json.dumps(rep, indent=1))


def stage_preprocess(fmt: str) -> None:
    out = fmt_dir(DATA / "prep", fmt)
    if (out / "test.npy").exists():
        return
    tr_meta, tr_x = ingest.load_split(fmt, "train")
    masks = _week_masks(tr_meta, tr_x)
    pp = Preprocessor().fit(tr_x, masks)
    np.savez(fmt_dir(ARTIFACTS, fmt) / "preprocessor.npz", **pp.state())
    for split in ("train", "validation", "test"):
        meta, arrays = ingest.load_split(fmt, split)
        m = _week_masks(meta, arrays)
        np.save(out / f"{split}.npy", pp.transform(arrays, m))
        meta[meta.keep].drop(columns="keep").reset_index(drop=True).to_parquet(out / f"{split}_meta.parquet", index=False)
        print(f"{fmt} {split}: {int(meta.keep.sum())} rows", flush=True)


def _week_masks(meta: pd.DataFrame, arrays: list) -> list:
    masks, a = [], 0
    for arr in arrays:
        masks.append(meta.keep.values[a:a + arr.shape[0]])
        a += arr.shape[0]
    assert a == len(meta), "metadata and feature rows are misaligned"
    return masks


def _load_prep(fmt: str, split: str):
    d = DATA / "prep" / fmt
    return np.load(d / f"{split}.npy"), pd.read_parquet(d / f"{split}_meta.parquet")


def stage_train(fmt: str) -> None:
    cfg = protocol()["model"]
    art = fmt_dir(ARTIFACTS, fmt)
    pp = Preprocessor.from_state(np.load(art / "preprocessor.npz"))
    idx_s, idx_c = pp.view_index("structural"), pp.view_index("content")
    x_tr, m_tr = _load_prep(fmt, "train")
    x_va, m_va = _load_prep(fmt, "validation")
    y_tr, y_va = m_tr.label.values.astype(np.int64), m_va.label.values.astype(np.int64)
    for seed in cfg["seeds"]:
        ckpt = art / f"model_s{seed}.pt"
        if ckpt.exists():
            continue
        model, info = fit(x_tr, y_tr, x_va, y_va, idx_s, idx_c, seed, cfg)
        v = score(model, x_va)
        np.save(art / f"val_logits_s{seed}.npy", v)
        (art / f"train_log_s{seed}.json").write_text(json.dumps(info, indent=1))
        torch.save(model.state_dict(), ckpt)


def stage_freeze(fmt: str) -> None:
    """Calibrate branch thresholds on validation benign files and record hashes before evaluation."""
    proto = protocol()
    art = fmt_dir(ARTIFACTS, fmt)
    out = fmt_dir(RESULTS, fmt)
    if (out / "frozen.json").exists():
        return
    _, m_va = _load_prep(fmt, "validation")
    benign = m_va.label.values == 0
    alpha = proto["calibration"]["branch_nominal_fpr"]
    thr, logs = {}, {}
    for seed in proto["model"]["seeds"]:
        v = np.load(art / f"val_logits_s{seed}.npy")
        thr[f"S_{seed}"] = at_fpr(v[benign, 0], alpha)
        thr[f"C_{seed}"] = at_fpr(v[benign, 1], alpha)
        info = json.loads((art / f"train_log_s{seed}.json").read_text())
        logs[str(seed)] = {k: info[k] for k in ("best_epoch", "best_val_or_tpr_at_1pct", "epochs_run", "train_seconds")}
    frozen = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "protocol_sha256": sha256(CONFIG / "protocol.json"),
              "feature_views_sha256": sha256(CONFIG / "feature_views.json"),
              "preprocessor_sha256": sha256(art / "preprocessor.npz"),
              "models_sha256": {str(s): sha256(art / f"model_s{s}.pt") for s in proto["model"]["seeds"]},
              "thresholds": thr, "training": logs,
              "validation_benign": int(benign.sum()), "validation_malware": int((~benign).sum())}
    (out / "frozen.json").write_text(json.dumps(frozen, indent=1))


def stage_evaluate(fmt: str) -> None:
    proto = protocol()
    art, out = fmt_dir(ARTIFACTS, fmt), fmt_dir(RESULTS, fmt)
    frozen = json.loads((out / "frozen.json").read_text())
    checks = {"protocol": (CONFIG / "protocol.json", frozen["protocol_sha256"]),
              "views": (CONFIG / "feature_views.json", frozen["feature_views_sha256"]),
              "preprocessor": (art / "preprocessor.npz", frozen["preprocessor_sha256"])}
    checks.update({f"model {s}": (art / f"model_s{s}.pt", h) for s, h in frozen["models_sha256"].items()})
    for name, (path, h) in checks.items():
        if sha256(path) != h:
            raise RuntimeError(f"{name} changed after freezing")
    pp = Preprocessor.from_state(np.load(art / "preprocessor.npz"))
    idx_s, idx_c = pp.view_index("structural"), pp.view_index("content")
    x_te, m_te = _load_prep(fmt, "test")
    y = m_te.label.values.astype(int)
    scores = m_te[["sha256", "week", "label", "family"]].copy()
    dec = {}
    for seed in proto["model"]["seeds"]:
        model = TwoBranch(idx_s, idx_c)
        model.load_state_dict(torch.load(art / f"model_s{seed}.pt"))
        z = score(model, x_te)
        scores[f"S_{seed}"], scores[f"C_{seed}"] = z[:, 0], z[:, 1]
        dec[f"S_{seed}"] = z[:, 0] > frozen["thresholds"][f"S_{seed}"]
        dec[f"C_{seed}"] = z[:, 1] > frozen["thresholds"][f"C_{seed}"]
    scores.to_parquet(art / "test_scores.parquet", index=False)
    groups = [{"seed_pair": f"{i}-{j}", "same": [(f"S_{i}", f"S_{j}"), (f"C_{i}", f"C_{j}")],
               "crossed": [(f"S_{i}", f"C_{j}"), (f"C_{i}", f"S_{j}")]} for i, j in proto["pairings"]["seed_pairs"]]
    rows = [{"seed_pair": g["seed_pair"], "type": t, "pairing": f"{a}+{b}", **pair_stats(dec[a], dec[b], y)}
            for g in groups for t in ("same", "crossed") for a, b in g[t]]
    pd.DataFrame(rows).to_csv(out / "pairings.csv", index=False)
    point, per_pair = contrast(groups, dec, y)
    inf = proto["inference"]
    bs = bootstrap(groups, dec, y, m_te.family.values, inf["replicates"], inf["rng_seed"])
    weekly = []
    for w in sorted(m_te.week.unique()):
        mw = (m_te.week == w).values
        pw, ppw = contrast(groups, {k: v[mw] for k, v in dec.items()}, y[mw])
        weekly.append({"week": int(w), "files": int(mw.sum()), "malware": int(y[mw].sum()),
                       "contrast_pp": 100 * pw, **{f"pair_{g['seed_pair']}_pp": 100 * v for g, v in zip(groups, ppw)}})
    pd.DataFrame(weekly).to_csv(out / "weekly.csv", index=False)
    branch = [{"branch": k, "fnr": float((~v[y == 1]).mean()), "fpr": float(v[y == 0].mean())} for k, v in dec.items()]
    pd.DataFrame(branch).to_csv(out / "branches.csv", index=False)
    df = pd.DataFrame(rows)
    eoi = proto["decision_rule"]["effect_of_interest_pp"]
    summary = {"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "frozen_at": frozen["frozen_at"],
               "test_files": int(len(y)), "test_malware": int((y == 1).sum()), "test_benign": int((y == 0).sum()),
               "contrast_pp": 100 * point, "ci95_pp": [100 * bs["ci_low"], 100 * bs["ci_high"]],
               "contrast_per_seed_pair_pp": {g["seed_pair"]: 100 * v for g, v in zip(groups, per_pair)},
               "covariance_identity_pp": 100 * covariance_identity(groups, dec, y),
               "bootstrap": bs, "decision": decide(100 * point, 100 * bs["ci_low"], 100 * bs["ci_high"], eoi),
               "weeks_positive": int(sum(r["contrast_pp"] > 0 for r in weekly)), "weeks": len(weekly),
               "mean_by_type": df.groupby("type")[["J", "q1", "q2", "excess", "system_fpr", "joint_misses"]].mean().to_dict(orient="index")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


STAGES = {"prepare": stage_prepare, "preprocess": stage_preprocess, "train": stage_train,
          "freeze": stage_freeze, "evaluate": stage_evaluate}

if __name__ == "__main__":
    STAGES[sys.argv[1]](sys.argv[2])
