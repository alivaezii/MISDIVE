"""MISDIVE Phase 1: one-time vectorisation and scoring of Win64 weeks 47-52.

Steps (protocol V2 section 10, steps 2-3):
  1. verify every digest in FREEZE_RECORD_V2.json (protocol, models, thresholds, scripts) and abort on mismatch;
  2. read Win64_train.zip members 47-52 only (sorted member order, as in viewdep.ingest), keep the first
     occurrence of each SHA-256 listed in data/weeks/win64/unique_<week>.parquet, extract features with the
     same thrember extractor and NaN/inf handling as viewdep.ingest._vectorise;
  3. keep labels in {0, 1} (others excluded and counted); apply the frozen clean-run Win64 preprocessor;
  4. score all 21 detectors (MS/MC clean MLP, LS/LC/FL LightGBM, AMS/AMC AB2) and write
     target_w47_52/target_scores.parquet plus counts by week and class.
No existing file is modified. Outputs go to $MISDIVE_POOL/target_w47_52/ (default work/pool/win64/target_w47_52/).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path
import os

import numpy as np

REPO = Path(os.environ.get("MISDIVE_CLEAN_RUN", Path(__file__).resolve().parents[1] / "single_factor"))
POOL = Path(os.environ.get("MISDIVE_POOL", Path(__file__).resolve().parents[2] / "work" / "pool" / "win64"))
HERE = Path(__file__).resolve().parent
OUT = POOL / "target_w47_52"
SEEDS = (42, 123, 2026)
WEEKS = range(47, 53)
SHA_RE = re.compile(rb'"sha256":\s*"([0-9a-f]{64})"')


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if "--authorised-E" not in sys.argv:
        sys.exit("Refusing: authorisation E (one-time scoring of weeks 47-52) has not been given.")
    import pandas as pd
    import torch
    import lightgbm as lgb
    sys.path.insert(0, str(REPO / "src"))
    from viewdep import ingest
    from viewdep.features import Preprocessor
    from viewdep.model import TwoBranch, mlp, score
    freeze = json.loads((HERE / "FREEZE_RECORD_V2.json").read_text())
    for rel, h in freeze["digests"].items():
        if sha(Path(rel)) != h:
            sys.exit(f"Digest mismatch, aborting: {rel}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, metas, counts = [], [], []
    with zipfile.ZipFile(REPO / "data" / "raw" / "Win64_train.zip") as zf, Pool(8, initializer=ingest._init_worker) as pool:
        members = sorted(zf.namelist())
        for w in WEEKS:
            want = set(pd.read_parquet(REPO / "data" / "weeks" / "win64" / f"unique_{w:02d}.parquet").sha256)
            taken, sel = set(), []
            with zf.open(members[w - 1]) as fh:
                for line in fh:
                    s = SHA_RE.search(line[:400]).group(1).decode()
                    if s in want and s not in taken:
                        taken.add(s)
                        sel.append(line)
            assert len(sel) == len(want), (w, len(sel), len(want))
            res = pool.map(ingest._vectorise, sel, chunksize=32)
            meta = pd.DataFrame([m for m, _ in res])
            meta["week"] = w
            rows.append(np.stack([v for _, v in res]))
            metas.append(meta)
    meta = pd.concat(metas, ignore_index=True)
    X = np.concatenate(rows)
    keep = meta.label.isin([0, 1]).values
    counts = {"excluded_label": int((~keep).sum()),
              "by_week_class": meta[keep].groupby(["week", "label"]).size().unstack().to_dict()}
    art = REPO / "artifacts" / "win64"
    pp = Preprocessor.from_state(np.load(art / "preprocessor.npz"))
    Xp = pp.transform([X], [keep])
    meta = meta[keep].reset_index(drop=True)
    idx_s, idx_c = pp.view_index("structural"), pp.view_index("content")
    out = meta[["sha256", "week", "label", "family"]].copy()
    for s in SEEDS:
        m = TwoBranch(idx_s, idx_c)
        m.load_state_dict(torch.load(art / f"model_s{s}.pt"))
        z = score(m, Xp)
        out[f"MS_{s}"], out[f"MC_{s}"] = z[:, 0], z[:, 1]
        for b, idx in (("S", idx_s), ("C", idx_c)):
            net = mlp(len(idx))
            net.load_state_dict(torch.load(POOL / f"AM{b}_{s}.pt"))
            net.eval()
            with torch.no_grad():
                out[f"AM{b}_{s}"] = net(torch.from_numpy(np.asarray(Xp[:, idx], dtype=np.float32))).squeeze(-1).numpy().astype(np.float64)
        for t, cols in (("LS", idx_s), ("LC", idx_c), ("FL", None)):
            bst = lgb.Booster(model_file=str(POOL / f"{t}_{s}.txt"))
            xx = np.asarray(Xp if cols is None else Xp[:, cols], dtype=np.float32)
            out[f"{t}_{s}"] = bst.predict(xx, raw_score=True).astype(np.float64)
    out.to_parquet(OUT / "target_scores.parquet", index=False)
    (OUT / "target_counts.json").write_text(json.dumps(counts, indent=1, default=str))
    print(counts)


if __name__ == "__main__":
    main()
