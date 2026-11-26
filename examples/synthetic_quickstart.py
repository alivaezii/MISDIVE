"""Quick start: a MISDIVE audit of a small synthetic 2x2 (view x algorithm) detector pool.

The data are simulated (see tests/synthetic.py); the numbers illustrate the API and say nothing about real
detectors. Run from the repository root:  python examples/synthetic_quickstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from misdive import decide, run_audit  # noqa: E402
from synthetic import make_pool  # noqa: E402

design, decisions, labels, families, windows, _ = make_pool(n_files=6000, seed=0)
result = run_audit(design, decisions, labels, clusters=families, windows=windows, n_boot=500, seed=1)

print(result.competence.round(3).to_string(index=False))
print()
print(result.effects_table().round(4).to_string(index=False))
print()
for k, v in decide(result, {"effect_1": 2.0, "effect_2": 1.0, "interaction": 1.0}).items():
    print(k, {a: b for a, b in v.items() if a != "ci95"})
out = ROOT / "work" / "quickstart"
out.mkdir(parents=True, exist_ok=True)
result.to_json(out / "audit.json")
try:
    from misdive.plots import forest
    forest(result, out / "forest.png", mei={"effect_1": 2.0, "effect_2": 1.0, "interaction": 1.0})
except ImportError:
    pass
print(f"\nwrote {out.relative_to(ROOT)}")
