"""Plot salary-aware PCA groups in their first two principal components."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "development" / "salary_aware_pca"
COLORS = {
    "Profile 1": "#3977A8",
    "Profile 2": "#E0A43A",
    "Profile 3": "#4A8B69",
    "Profile 4": "#C45B55",
}


def main():
    rows = pd.read_csv(RESULTS / "salary_aware_player_groups.csv")
    audit = json.loads((RESULTS / "analysis_audit.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(10, 7), dpi=170)

    for profile, color in COLORS.items():
        for split, marker, size, alpha in (
            ("training", "o", 15, 0.20),
            ("development", "^", 30, 0.60),
        ):
            selected = rows[(rows["profile"] == profile) & (rows["split"] == split)]
            ax.scatter(
                selected["pc1"],
                selected["pc2"],
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                linewidths=0,
                label=f"{profile} — {split.title()}",
            )

    ratios = audit["explained_variance_ratio"]
    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.4)
    ax.axvline(0, color="#777777", linewidth=0.7, alpha=0.4)
    ax.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ratios[1] * 100:.1f}% variance)")
    ax.set_title("Salary-aware PCA player-season groups", fontsize=16, pad=14)
    ax.text(
        0.02,
        0.98,
        "PCA fit on training rows; development rows projected onto same axes\n"
        "Inputs: next-season PIE, WS, VORP, log salary / salary cap",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#444444",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 5},
    )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", frameon=True, fontsize=8, ncol=2)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(RESULTS / "salary_aware_pca_profiles.png", bbox_inches="tight")
    fig.savefig(RESULTS / "salary_aware_pca_profiles.pdf", bbox_inches="tight")
    print(RESULTS / "salary_aware_pca_profiles.png")


if __name__ == "__main__":
    main()
