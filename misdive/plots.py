"""Standard forest plot of MISDIVE effects (covariance and correlation scales)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"effect_1": "Factor-1 effect", "effect_2": "Factor-2 effect", "interaction": "Interaction",
          "residual": "Residual", "effect_1_minus_effect_2": "Factor 1 - factor 2"}


def forest(result, path, mei: dict | None = None, title_1: str = "Covariance (pp)", title_2: str = "Correlation (phi)"):
    keys = [k for k in LABELS if k in result.effects["cov_pp"]["point"]]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 0.45 * len(keys) + 1.2), sharey=True)
    for ax, scale, title in ((axes[0], "cov_pp", title_1), (axes[1], "phi", title_2)):
        for i, k in enumerate(keys[::-1]):
            pt = result.effects[scale]["point"][k]
            lo, hi = result.effects[scale]["ci95"].get(k, [pt, pt])
            ax.plot([lo, hi], [i, i], color="#0072B2", lw=1.4)
            ax.plot(pt, i, "o", color="#0072B2")
            if mei and scale == "cov_pp" and k in mei:
                ax.plot([mei[k]] * 2, [i - 0.35, i + 0.35], color="#D55E00", ls="--", lw=1)
        ax.axvline(0, color="0.3", lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([LABELS[k] for k in keys[::-1]], fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
