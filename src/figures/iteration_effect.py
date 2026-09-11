"""Figure: iteration count and initialisation on Smartvote 2023.

Writes figures/iteration_effect_smartvote_2023.pdf: reconstruction MAE at
u = 0 and imputation MAE at u = 0.6 against the number of iterations, for
PCA and random initialisation, train and test split, at tau^2 = 0.25. Also
prints the per-iteration MAE gap between the two initialisations. Reads
results/smartvote_2023/iteration_effect/.

Usage:
    python -m src.figures.iteration_effect
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from src.results import load_metrics
from src.visualization import apply_defaults, figsize, panel_labels, save_figure

DATASET = "smartvote_2023"
TAU_DEFAULT = 0.25
INITS = ["pca", "random"]
INIT_COLORS = {"pca": "C0", "random": "C1"}
INIT_LABELS = {"pca": "PCA", "random": "Random"}

# Sparsity sets to aggregate over per panel.
FIT_SPARSITIES = [0.0]      # fit MAE: u = 0 only
IMPUTE_SPARSITIES = [0.6]   # impute MAE: undefined at u = 0


def aggregate(df, metric, sparsities):
    """Average per (init, iteration, seed) over the given sparsities, then take
    mean and std across seeds. Returns one row per (init, iteration)."""
    sub = df[(df["tau"] == TAU_DEFAULT) & (df["sparsity"].isin(sparsities))]
    per_seed = sub.groupby(["init", "iteration", "seed"])[metric].mean().reset_index()
    return per_seed.groupby(["init", "iteration"])[metric].agg(["mean", "std"]).reset_index()


def make_figure(train, test):
    fig, axes = plt.subplots(1, 2, figsize=figsize(width_frac=1.0, aspect=0.26),
                             sharex=True, sharey=True)
    panels = [
        ("fit_mae", "Reconstruction", FIT_SPARSITIES),
        ("impute_mae", "Imputation", IMPUTE_SPARSITIES),
    ]
    for ax, (metric, panel_label, sparsities) in zip(axes, panels):
        for init in INITS:
            tr_sub = aggregate(train, metric, sparsities)
            tr_sub = tr_sub[tr_sub["init"] == init]
            te_sub = aggregate(test, metric, sparsities)
            te_sub = te_sub[te_sub["init"] == init]
            ax.errorbar(tr_sub["iteration"], tr_sub["mean"], yerr=tr_sub["std"],
                        marker="o", capsize=3, color=INIT_COLORS[init], linestyle="-")
            ax.errorbar(te_sub["iteration"], te_sub["mean"], yerr=te_sub["std"],
                        marker="s", capsize=3, color=INIT_COLORS[init], linestyle="--")
        ax.set_xlabel("Iteration")
        ax.set_xlim(-0.5, 20.5)
        ax.set_ylim(0.17, 0.34)
        ax.set_ylabel(f"{panel_label} (MAE)")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.yaxis.set_tick_params(labelleft=True)
        ax.grid(alpha=0.3)

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", label="train"),
        Line2D([0], [0], color="black", linestyle="--", marker="s", label="test"),
    ]
    color_handles = [Line2D([0], [0], color=INIT_COLORS[i], lw=2, label=INIT_LABELS[i]) for i in INITS]
    axes[0].legend(handles=style_handles, loc="best", title="Split",
                   handlelength=4, labelspacing=0.3)
    axes[1].legend(handles=color_handles, loc="best", title="Initialization",
                   handlelength=1.5, labelspacing=0.3)
    panel_labels(axes)
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.94], w_pad=0.7)
    return fig


def init_gap(df, metric, sparsities):
    """mae(random) - mae(pca) per iteration at the default prior."""
    sub = df[(df["tau"] == TAU_DEFAULT) & (df["sparsity"].isin(sparsities))]
    grp = sub.groupby(["init", "iteration"])[metric].mean().unstack("init")
    grp["gap"] = (grp["random"] - grp["pca"]).round(4)
    return grp


def main():
    apply_defaults()
    train = load_metrics(DATASET, "iteration_effect", "train")
    test = load_metrics(DATASET, "iteration_effect", "test")
    save_figure(make_figure(train, test), f"iteration_effect_{DATASET}.pdf")

    print("Train fit\n", init_gap(train, "fit_mae", FIT_SPARSITIES))
    print("\nTest fit\n", init_gap(test, "fit_mae", FIT_SPARSITIES))
    print("\nTrain impute\n", init_gap(train, "impute_mae", IMPUTE_SPARSITIES))
    print("\nTest impute\n", init_gap(test, "impute_mae", IMPUTE_SPARSITIES))


if __name__ == "__main__":
    main()
