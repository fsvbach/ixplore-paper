"""Figure: prior-variance trade-off phase diagrams on Smartvote 2023.

Writes figures/prior_phase_smartvote_2023.pdf. For each tau^2 (PCA
initialisation) the iteration trajectory T = 0..20 is traced in two planes:
boundary fraction against train reconstruction MAE (u = 0), and distortion
against test imputation MAE (averaged over u in {0.3, 0.6, 0.9} and seeds).
Marker size grows with T; the arrow points from T = 0 to T = 20. Reads
results/smartvote_2023/iteration_effect/.

Usage:
    python -m src.figures.prior_phase
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from src.results import load_metrics
from src.visualization import apply_defaults, figsize, fmt_tau, panel_labels, save_figure

DATASET = "smartvote_2023"
TAU_VALUES = [0.05, 0.1, 0.25, 0.5, 1.0, 1e6]  # prior variances tau^2


def make_figure(train_metrics, test_metrics):
    # Aggregate per (tau, iteration) at PCA init, u=0; prediction metrics averaged over seeds.
    train_pca0 = (train_metrics[(train_metrics["init"] == "pca")
                                & (train_metrics["sparsity"] == 0.0)]
                  .groupby(["tau", "iteration"])
                  [["fit_mae", "boundary", "distortion_mean"]]
                  .mean()
                  .reset_index())

    # Test-impute MAE: held-out users at u > 0 (artificial mask), averaged over the
    # sparsity grid and seeds for one number per (tau, iteration).
    test_impute = (test_metrics[(test_metrics["init"] == "pca")
                                & (test_metrics["sparsity"] > 0.0)]
                   .groupby(["tau", "iteration"])
                   ["impute_mae"]
                   .mean()
                   .reset_index())

    merged = train_pca0.merge(test_impute, on=["tau", "iteration"])

    tau_cmap = plt.cm.viridis
    tau_color = {s: tau_cmap(i / (len(TAU_VALUES) - 1)) for i, s in enumerate(TAU_VALUES)}
    iter_grid = sorted(merged["iteration"].unique())
    iter_size = {it: 14 + 10 * rank for rank, it in enumerate(iter_grid)}

    def add_phase(ax, x_col, y_col, x_scale=1.0):
        for s in TAU_VALUES:
            traj = merged[merged["tau"] == s].sort_values("iteration")
            c = tau_color[s]
            x = traj[x_col].values * x_scale
            y = traj[y_col].values
            ax.plot(x, y, "-", color=c, lw=1.0, alpha=0.85, zorder=2)
            sizes = traj["iteration"].map(iter_size).values
            ax.scatter(x, y, s=sizes, color=c, edgecolor="white", linewidth=0.4, zorder=3)
            ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[0], y[0]),
                        arrowprops=dict(arrowstyle="->", color=c, alpha=0.4, lw=0.7))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize(width_frac=1.0, aspect=0.28))

    add_phase(axL, x_col="boundary", y_col="fit_mae", x_scale=100.0)
    axL.set_xlabel("Boundary fraction")
    axL.set_ylabel("Train reconstruction (MAE)")

    add_phase(axR, x_col="distortion_mean", y_col="impute_mae")
    axR.set_xlabel("Distortion")
    axR.set_ylabel("Test imputation (MAE)")

    # Consistent grid; per-panel tick spacing matched to each metric range.
    for ax in (axL, axR):
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axL.yaxis.set_major_locator(MultipleLocator(0.001))
    axL.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axR.yaxis.set_major_locator(MultipleLocator(0.005))
    axR.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # Legend embedded in the right panel.
    tau_handles = [plt.Line2D([0], [0], marker="o", color=tau_color[s], lw=1.2,
                              markersize=5, markeredgecolor="white", markeredgewidth=0.4,
                              label=rf"$\tau^2={fmt_tau(s)}$") for s in TAU_VALUES]
    legR = axR.legend(handles=tau_handles, loc="upper left", title=r"Prior Variance",
                      handlelength=0.9, borderpad=0.3, labelspacing=0.0,
                      handletextpad=0.4, framealpha=1.0, ncols=3, columnspacing=0.7)
    legR.get_frame().set_edgecolor("black")
    legR.get_frame().set_linewidth(0.6)

    panel_labels((axL, axR))
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.95], w_pad=0.7)
    return fig


def main():
    apply_defaults()
    train_metrics = load_metrics(DATASET, "iteration_effect", "train")
    test_metrics = load_metrics(DATASET, "iteration_effect", "test")
    save_figure(make_figure(train_metrics, test_metrics), f"prior_phase_{DATASET}.pdf")


if __name__ == "__main__":
    main()
