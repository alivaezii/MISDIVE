"""Repository layout and frozen configuration."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT.parents[1] / "protocols" / "clean_run"  # frozen protocol and views
DATA = ROOT / "data"            # raw archives and arrays (not tracked by Git)
RAW = DATA / "raw"
ARTIFACTS = ROOT / "artifacts"  # checkpoints and scores (not tracked by Git)
RESULTS = ROOT / "results"      # compact, tracked result tables
FORMATS = ("dotnet", "win64")


def protocol() -> dict:
    return json.loads((CONFIG / "protocol.json").read_text())


def views() -> dict:
    return json.loads((CONFIG / "feature_views.json").read_text())


def fmt_dir(base: Path, fmt: str) -> Path:
    d = base / fmt
    d.mkdir(parents=True, exist_ok=True)
    return d
