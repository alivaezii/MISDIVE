"""Archive ingestion: week mapping, SHA256 deduplication and feature extraction.

Each EMBER2024 JSONL member lists every record twice, so exact SHA256 repeats are collapsed to their
first occurrence in chronological member order. Weeks are written as separate files so that an
interrupted run resumes at the first incomplete week.
"""
from __future__ import annotations

import json
import re
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA, RAW, fmt_dir, protocol

SHA_RE = re.compile(rb'"sha256":\s*"([0-9a-f]{64})"')
META_FIELDS = ("sha256", "label", "family", "first_submission_date", "week_id", "file_type")
_EXTRACTOR = None


def _init_worker():
    global _EXTRACTOR
    import thrember
    _EXTRACTOR = thrember.PEFeatureExtractor()


def _vectorise(line: bytes):
    rec = json.loads(line)
    vec = np.nan_to_num(_EXTRACTOR.process_raw_features(rec).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return {k: rec.get(k) for k in META_FIELDS}, vec


def _member_dates(name: str):
    start, end = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", Path(name).name).groups()
    return start, end


def ingest(fmt: str, workers: int = 8) -> None:
    """Vectorise weeks 1-46 of the training archive and weeks 53-64 of the test archive."""
    proto = protocol()
    arch = proto["dataset"]["archives"][fmt]
    lo_val = proto["windows"]["validation_weeks"][1]
    out = fmt_dir(DATA / "weeks", fmt)
    seen: set[str] = set()
    train_archive: set[str] = set()
    plan = [(arch["train"], 0), (arch["test"], 52)]
    seen_test: set[str] = set()
    with Pool(workers, initializer=_init_worker) as pool:
        for archive, offset in plan:
            with zipfile.ZipFile(RAW / archive) as zf:
                members = sorted(zf.namelist())
                for k, name in enumerate(members, start=1):
                    week = offset + k
                    marker = out / f"week_{week:02d}.json"
                    if marker.exists():
                        shas = set(pd.read_parquet(out / f"unique_{week:02d}.parquet").sha256)
                        seen |= shas
                        if offset == 0:
                            train_archive |= shas
                        continue
                    vectorise = week <= lo_val or offset == 52
                    unique, selected, n_lines, n_overlap = [], [], 0, 0
                    with zf.open(name) as fh:
                        for line in fh:
                            n_lines += 1
                            sha = SHA_RE.search(line[:400]).group(1).decode()
                            if offset == 52 and sha in train_archive and sha not in seen_test:
                                # Test file already present in the training archive: excluded and counted.
                                n_overlap += 1
                                seen_test.add(sha)
                                continue
                            if sha in seen:
                                continue
                            seen.add(sha)
                            if offset == 0:
                                train_archive.add(sha)
                            else:
                                seen_test.add(sha)
                            unique.append(sha)
                            if vectorise:
                                selected.append(line)
                    stats = {"week": week, "member": name, "dates": _member_dates(name), "lines": n_lines,
                             "unique": len(unique), "vectorised": len(selected),
                             "excluded_overlap_with_training_archive": n_overlap}
                    if selected:
                        res = pool.map(_vectorise, selected, chunksize=32)
                        meta = pd.DataFrame([m for m, _ in res])
                        meta["week"] = week
                        np.save(out / f"X_{week:02d}.npy", np.stack([v for _, v in res]))
                        meta.to_parquet(out / f"meta_{week:02d}.parquet", index=False)
                    pd.DataFrame({"sha256": unique}).to_parquet(out / f"unique_{week:02d}.parquet", index=False)
                    marker.write_text(json.dumps(stats))
                    print(f"{fmt} week {week:02d}: lines={n_lines} unique={len(unique)} vectorised={len(selected)}", flush=True)


def audit(fmt: str) -> dict:
    """Week mapping, class/week counts, duplicates and cross-partition overlap."""
    proto = protocol()
    w = proto["windows"]
    out = fmt_dir(DATA / "weeks", fmt)
    stats = [json.loads((out / f"week_{k:02d}.json").read_text()) for k in range(1, 65)]
    metas = []
    for s in stats:
        p = out / f"meta_{s['week']:02d}.parquet"
        if p.exists():
            metas.append(pd.read_parquet(p))
    meta = pd.concat(metas, ignore_index=True)
    meta["split"] = np.select([meta.week <= w["train_weeks"][1], meta.week <= w["validation_weeks"][1]],
                              ["train", "validation"], "test")
    # Week mapping checks: submission date inside the member's date range; week_id consistent.
    ts = pd.to_datetime(meta.first_submission_date, unit="s", utc=True)
    rng = {s["week"]: s["dates"] for s in stats}
    start = meta.week.map(lambda k: pd.Timestamp(rng[k][0], tz="UTC"))
    end = meta.week.map(lambda k: pd.Timestamp(rng[k][1], tz="UTC") + pd.Timedelta(days=1))
    in_range = ((ts >= start) & (ts < end)).mean()
    week_id_offset = (meta.week - meta.week_id).value_counts().to_dict()
    unique_all = {s["week"]: set(pd.read_parquet(out / f"unique_{s['week']:02d}.parquet").sha256) for s in stats}
    train_archive = set().union(*[unique_all[k] for k in range(1, 53)])
    test_set = set().union(*[unique_all[k] for k in range(53, 65)])
    # Historical Win64 sampling audit: a SHA256-prefix threshold does not retain the nominal fraction
    # when prefixes are not uniform within a class. Report realised retention per class.
    u = meta.sha256.str[:8].map(lambda h: int(h, 16) / 2**32)
    from scipy.stats import kstest
    prefix_audit = {}
    for split in ("train", "validation", "test"):
        for lab in (0, 1):
            x = u[(meta.split == split) & (meta.label == lab)]
            prefix_audit[f"{split}_label{lab}"] = {
                "n": int(len(x)), "ks_uniform_stat": float(kstest(x, "uniform").statistic),
                "ks_uniform_p": float(kstest(x, "uniform").pvalue),
                **{f"retained_at_{r}": float((x < r).mean()) for r in (0.2, 0.25, 0.4, 0.5)}}
    counts = meta.groupby(["split", "week", "label"]).size().unstack(fill_value=0)
    counts.to_csv(out / "counts_week_class.csv")
    rep = {
        "format": fmt,
        "lines_total": int(sum(s["lines"] for s in stats)),
        "unique_total": int(sum(s["unique"] for s in stats)),
        "exact_repeats_removed": int(sum(s["lines"] - s["unique"] for s in stats)),
        "fraction_submission_date_in_member_range": float(in_range),
        "week_minus_week_id_counts": {str(k): int(v) for k, v in week_id_offset.items()},
        "test_sha_overlap_with_training_archive_excluded": int(sum(s.get("excluded_overlap_with_training_archive", 0) for s in stats)),
        "test_sha_overlap_after_exclusion": int(len(test_set & train_archive)),
        "sha256_prefix_audit": prefix_audit,
        "labels": {str(k): int(v) for k, v in meta.label.value_counts(dropna=False).items()},
        "file_types": {str(k): int(v) for k, v in meta.file_type.value_counts(dropna=False).items()},
        "per_split": {s: {"files": int((meta.split == s).sum()),
                          "malware": int(((meta.split == s) & (meta.label == 1)).sum()),
                          "benign": int(((meta.split == s) & (meta.label == 0)).sum()),
                          "malware_without_family": int(((meta.split == s) & (meta.label == 1) & meta.family.isna()).sum())}
                      for s in ("train", "validation", "test")},
    }
    (out / "audit.json").write_text(json.dumps(rep, indent=1))
    return rep


def load_split(fmt: str, split: str):
    """Return (meta, list of per-week raw-feature memmaps) for a split, in week order."""
    w = protocol()["windows"]
    lo, hi = {"train": w["train_weeks"], "validation": w["validation_weeks"], "test": w["test_weeks"]}[split]
    out = DATA / "weeks" / fmt
    metas, arrays = [], []
    for k in range(lo, hi + 1):
        metas.append(pd.read_parquet(out / f"meta_{k:02d}.parquet"))
        arrays.append(np.load(out / f"X_{k:02d}.npy", mmap_mode="r"))
    meta = pd.concat(metas, ignore_index=True)
    excluded_label = ~meta.label.isin([0, 1])
    meta["keep"] = ~excluded_label
    return meta, arrays
