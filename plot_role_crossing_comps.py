"""Plot bench and starter skill profiles and one example set of comps."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "development" / "role_crossing_comps"


def main():
    coords = pd.read_csv(RESULTS / "skill_pca_coordinates.csv")
    comps = pd.read_csv(RESULTS / "development_bench_to_starter_comps.csv")
    queries = coords[
        (coords["split"] == "development")
        & (coords["role"] == "bench")
    ].copy()
    example = queries[queries["player"].eq("Isaiah Jackson")]
    if example.empty:
        example = queries.head(1)
    query = example.iloc[0]
    selected_comps = comps[
        (comps["bench_player_id"] == query["player_id"])
        & (comps["bench_prior_season"] == query["season"])
    ].sort_values("comp_rank").head(3)
    training_starters = coords[
        (coords["split"] == "training")
        & (coords["role"] == "starter")
    ]
    development_bench = coords[
        (coords["split"] == "development")
        & (coords["role"] == "bench")
    ]
    training_middle = coords[coords["split"].eq("training") & coords["role"].eq("middle")]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=170)
    ax.scatter(
        training_starters["pc1"], training_starters["pc2"],
        s=13, alpha=0.13, color="#888888", label="Training starters",
    )
    ax.scatter(
        training_middle["pc1"], training_middle["pc2"],
        s=10, alpha=0.10, color="#C5A96B", label="Training middle role",
    )
    ax.scatter(
        development_bench["pc1"], development_bench["pc2"],
        s=18, alpha=0.32, color="#3977A8", label="Development bench players",
    )
    ax.scatter(
        query["pc1"], query["pc2"], marker="*", s=240,
        color="#111111", edgecolor="white", linewidth=0.8, zorder=5,
        label=f"{query['player']} (bench query)",
    )

    training_coords = coords[coords["split"].eq("training")]
    for comp in selected_comps.itertuples():
        match = training_coords[
            (training_coords["player_id"] == comp.starter_comp_player_id)
            & (training_coords["target_season"] == comp.comp_outcome_season)
        ]
        if match.empty:
            continue
        point = match.iloc[0]
        ax.plot(
            [query["pc1"], point["pc1"]],
            [query["pc2"], point["pc2"]],
            color="#C45B55", alpha=0.65, linewidth=1.2, zorder=3,
        )
        ax.scatter(
            point["pc1"], point["pc2"], marker="D", s=65,
            color="#C45B55", edgecolor="white", linewidth=0.7, zorder=4,
            label=f"Top starter comp: {comp.starter_comp}",
        )
        ax.annotate(
            comp.starter_comp,
            (point["pc1"], point["pc2"]),
            xytext=(5, 4), textcoords="offset points", fontsize=8,
        )

    ax.annotate(
        str(query["player"]),
        (query["pc1"], query["pc2"]),
        xytext=(7, 8), textcoords="offset points", fontsize=9, weight="bold",
    )
    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.35)
    ax.axvline(0, color="#777777", linewidth=0.7, alpha=0.35)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Bench-to-starter skill-profile comparisons")
    ax.text(
        0.02, 0.98,
        f"{query['player']}, prior season {int(query['season'])}\n"
        "Projection only: nearest comps use all retained PCs, not just PC1/PC2",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 5},
    )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(), unique.keys(), loc="best", fontsize=8,
        frameon=True, ncol=2,
    )
    ax.grid(alpha=0.13)
    fig.tight_layout()
    fig.savefig(RESULTS / "bench_to_starter_skill_pca.png", bbox_inches="tight")
    fig.savefig(RESULTS / "bench_to_starter_skill_pca.pdf", bbox_inches="tight")
    print(RESULTS / "bench_to_starter_skill_pca.png")


if __name__ == "__main__":
    main()
