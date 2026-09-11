"""Figure: confidence weights and weight scaling on Smartvote 2023 test users.

Writes figures/weights_geometry_smartvote_2023.pdf: test imputation MAE,
boundary fraction, and round-trip distortion against test sparsity for the
three weight schedules (uniform, provided, extreme), with and without weight
scaling. Reads results/smartvote_2023/weight_effect/.

Usage:
    python -m src.figures.weights_geometry
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from src.results import experiment_dir
from src.visualization import apply_defaults, figsize, panel_labels, save_figure

DATASET = "smartvote_2023"
RESULTS_DIR = experiment_dir(DATASET, "weight_effect")

SCHEDULES = ["uniform", "provided", "extreme"]
SCHEDULE_COLORS = {"uniform": "#888888", "provided": "#4477AA", "extreme": "#CC6677"}
SCHEDULE_DISPLAY = {"uniform": "uniform", "provided": "provided", "extreme": "extreme"}

# Linestyle + marker encode scale_weights everywhere; colour encodes the schedule
# in panels A and B (panel C has no schedule, so a single neutral colour).
SW_STYLE = {False: {"linestyle": "--", "marker": "o"},
            True: {"linestyle": "-", "marker": "s"}}
DIST_COLOR = "0.2"


def make_figure(test_metrics, distortion):
    mae_agg = (
        test_metrics
        .groupby(["schedule", "scale_weights", "sparsity"])["impute_mae"]
        .agg(mae_mean="mean", mae_std="std")
        .reset_index()
    )
    geom = (
        test_metrics
        .groupby(["schedule", "scale_weights", "sparsity"])
        .agg(spread_mean=("spread", "mean"), spread_std=("spread", "std"),
             boundary_mean=("boundary", "mean"), boundary_std=("boundary", "std"))
        .reset_index()
    )
    dist_agg = (
        distortion
        .groupby(["scale_weights", "sparsity"])
        .agg(dist_mean=("distortion_mean", "mean"), dist_std=("distortion_mean", "std"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize(width_frac=1.0, aspect=0.3), sharex=True)

    # Panel A: test impute MAE vs sparsity.
    ax = axes[0]
    for schedule in SCHEDULES:
        color = SCHEDULE_COLORS[schedule]
        for sw in [False, True]:
            sub = mae_agg[(mae_agg["schedule"] == schedule) & (mae_agg["scale_weights"] == sw)]
            ax.errorbar(sub["sparsity"], sub["mae_mean"], yerr=sub["mae_std"],
                        color=color, **SW_STYLE[sw])
    ax.set_ylabel("Imputation error (MAE)")

    # Panel B: boundary fraction.
    ax = axes[1]
    for schedule in SCHEDULES:
        color = SCHEDULE_COLORS[schedule]
        for sw in [False, True]:
            sub = geom[(geom["schedule"] == schedule) & (geom["scale_weights"] == sw)]
            ax.errorbar(sub["sparsity"], sub["boundary_mean"], yerr=sub["boundary_std"],
                        color=color, **SW_STYLE[sw])
    ax.set_ylabel("Boundary fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # Panel C: distortion (scale_weights only, no schedule split).
    ax = axes[2]
    for sw in [False, True]:
        sub = dist_agg[dist_agg["scale_weights"] == sw]
        ax.errorbar(sub["sparsity"], sub["dist_mean"], yerr=sub["dist_std"],
                    color=DIST_COLOR, **SW_STYLE[sw])
    ax.set_ylabel("Distortion")

    for ax in axes:
        ax.set_xlabel("Sparsity")

    # Schedule (colour) legend in panel A, scale_weights (linestyle) legend in panel C.
    schedule_handles = [Line2D([0], [0], color=SCHEDULE_COLORS[s], lw=1.4, label=SCHEDULE_DISPLAY[s])
                        for s in SCHEDULES]
    sw_handles = [Line2D([0], [0], color="black",
                         linestyle=SW_STYLE[sw]["linestyle"], marker=SW_STYLE[sw]["marker"],
                         label=("True" if sw else "False"))
                  for sw in [False, True]]
    leg_schedule = axes[0].legend(handles=schedule_handles, title="Weights", loc="upper left",
                                  handlelength=1.4, borderpad=0.3, labelspacing=0.25, framealpha=1.0)
    leg_schedule.get_frame().set_edgecolor("0.5")
    leg_schedule.get_frame().set_linewidth(0.6)
    axes[0].add_artist(leg_schedule)
    # Longer handles so the dashed pattern is visible.
    leg_sw = axes[2].legend(handles=sw_handles, title="Scaling", loc="upper left",
                            handlelength=3.0, borderpad=0.3, labelspacing=0.25, framealpha=1.0)
    leg_sw.get_frame().set_edgecolor("0.5")
    leg_sw.get_frame().set_linewidth(0.6)

    panel_labels(axes)
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.95], w_pad=0.7)
    return fig


def main():
    apply_defaults()
    test_metrics = pd.read_csv(RESULTS_DIR / "test_metrics.csv")
    distortion = pd.read_csv(RESULTS_DIR / "distortion.csv")
    save_figure(make_figure(test_metrics, distortion), f"weights_geometry_{DATASET}.pdf")


if __name__ == "__main__":
    main()
