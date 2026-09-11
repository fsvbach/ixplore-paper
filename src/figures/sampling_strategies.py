"""Figure: Rasch against posterior sampling of pseudo-answers on Smartvote 2023.

Writes figures/sampling_strategies_smartvote_2023.pdf: imputation MAE of the
sample mean and the sample standard deviation on the unobserved items against
the number of observed answers, for Rasch sampling, posterior sampling, and a
uniform random baseline, with the posterior-mean point estimate for
reference. Reads results/ixplore_design/sampling_strategies.csv (written by
src.scripts.run_sampling_strategies).

Usage:
    python -m src.figures.sampling_strategies
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import RESULTS_DIR
from src.visualization import apply_defaults, figsize, panel_labels, save_figure

RESULTS_CSV = RESULTS_DIR / "ixplore_design" / "sampling_strategies.csv"
METHOD_STYLES = [("Rasch", "tab:orange", "-"), ("Posterior", "tab:blue", "-"), ("Random", "gray", ":")]


def make_figure(res_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize(width_frac=1.0, aspect=0.38))

    for method, color, ls in METHOD_STYLES:
        sub = res_df[res_df["method"] == method].sort_values("n_obs")
        ax1.plot(sub["n_obs"], sub["mae"], marker="o", color=color, label=method, markersize=4, ls=ls)
        ax2.plot(sub["n_obs"], sub["mean_std"], marker="o", color=color, label=method, markersize=4, ls=ls)

    point = res_df[res_df["method"] == "Posterior mean"].sort_values("n_obs")
    ax1.plot(point["n_obs"], point["mae"], marker="s", color="black", ls="--", lw=1,
             markersize=3, label="Posterior mean")

    ax1.set_xlabel("Number of observed answers")
    ax1.set_ylabel("Sample imputation MAE")
    ax2.set_xlabel("Number of observed answers")
    ax2.set_ylabel("Sample std. deviation")
    ax1.legend()
    ax2.legend()
    panel_labels((ax1, ax2))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    apply_defaults()
    res_df = pd.read_csv(RESULTS_CSV)
    save_figure(make_figure(res_df), "sampling_strategies_smartvote_2023.pdf")


if __name__ == "__main__":
    main()
