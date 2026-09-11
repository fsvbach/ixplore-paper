"""Figure: candidate embeddings under three prior variances on Smartvote 2023.

Writes figures/prior_embeddings_smartvote_2023.pdf: the fully fitted
embedding (PCA initialisation, u = 0, T = 20) for tau^2 in {0.05, 0.25, 1e6},
coloured by party and annotated with the train reconstruction MAE and the
boundary fraction. Reads the checkpoints and metrics in
results/smartvote_2023/iteration_effect/.

Usage:
    python -m src.figures.prior_embeddings
"""

import ixplore.visualization as ixplore_vis
import matplotlib.pyplot as plt
import pandas as pd

from src.data import DATASET_DIRS
from src.results import experiment_dir, load_metrics
from src.visualization import apply_defaults, figsize, fmt_tau, panel_labels, save_figure

DATASET = "smartvote_2023"
MODELS_DIR = experiment_dir(DATASET, "iteration_effect") / "models"
ITER_FINAL = 20
PRIOR_PANELS = [0.05, 0.25, 1e6]


def make_figure(train_metrics, info):
    party_colors = info.groupby("party")["color"].first().to_dict()

    # Train reconstruction MAE and boundary fraction at PCA init, u = 0, T = 20.
    tr_f = train_metrics[(train_metrics["init"] == "pca") & (train_metrics["iteration"] == ITER_FINAL)]
    geom_lookup = (tr_f[tr_f["sparsity"] == 0.0]
                   .groupby("tau")[["fit_mae", "boundary"]]
                   .mean())

    fig, axes = plt.subplots(1, 3, figsize=figsize(width_frac=1.0, aspect=0.35))

    for ax, tau in zip(axes, PRIOR_PANELS):
        name = f"tau_{tau}_pca_sp_0.0_iter_{ITER_FINAL}"
        emb = pd.read_csv(MODELS_DIR / name / "embedding.csv", index_col=0)
        emb = emb.copy()
        emb.iloc[:, 0] = -emb.iloc[:, 0]
        panel_colors = (info.loc[emb.index.intersection(info.index), "party"]
                        .map(party_colors).values)
        ixplore_vis.plot_embedding(emb, colors=panel_colors, ax=ax, s=10, alpha=0.85,
                                   edgecolors="white", lw=0.3)

        fit_mae = geom_lookup.loc[tau, "fit_mae"]
        boundary = geom_lookup.loc[tau, "boundary"]
        bbox = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=1.0,
                    edgecolor="0.5", linewidth=0.6)
        ax.text(0.04, 0.04,
                f"$\\bf{{\\tau^2={fmt_tau(tau)}}}$\n"
                f"MAE={fit_mae:.3f}\n"
                f"BF={boundary * 100:.1f}%",
                transform=ax.transAxes, ha="left", va="bottom",
                linespacing=1.4, bbox=bbox)
        ixplore_vis.clean_axis(ax, limits=(-1, 1))

    # Party legend at the top of the middle panel: framed, white background, tight spacing.
    party_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                markerfacecolor=party_colors[p], markersize=4, label=p)
                     for p in sorted(party_colors)]
    leg = axes[1].legend(handles=party_handles, loc="upper center",
                         ncol=9, fontsize=7,
                         handlelength=0.6, handletextpad=0.1,
                         columnspacing=0.5, labelspacing=0.2,
                         borderpad=0.3, framealpha=1.0, facecolor="white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)

    axes[1].set_zorder(axes[0].get_zorder() + 1)
    axes[1].set_zorder(axes[2].get_zorder() + 1)
    axes[1].patch.set_visible(False)
    leg.set_zorder(20)
    panel_labels(axes)
    fig.subplots_adjust(top=0.95, bottom=0.005, left=0.005, right=0.995, wspace=0.05)
    return fig


def main():
    apply_defaults()
    train_metrics = load_metrics(DATASET, "iteration_effect", "train")
    info = pd.read_csv(DATASET_DIRS[DATASET] / "candidates_information.csv", index_col=0)
    save_figure(make_figure(train_metrics, info), f"prior_embeddings_{DATASET}.pdf")


if __name__ == "__main__":
    main()
