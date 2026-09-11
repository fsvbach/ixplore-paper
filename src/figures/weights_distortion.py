"""Figure: round-trip distortion of the latent space on Smartvote 2023.

Writes figures/weights_distortion_smartvote_2023.pdf. Synthetic users on a
30 x 30 grid of the trained latent space are predicted, sparsified at the
test sparsity v, and re-embedded; each line goes from the true grid position
to the re-embedded one. Rows contrast weight scaling off and on, columns the
test sparsity. Loads the model stored in results/smartvote_2023/weight_effect/.

Usage:
    python -m src.figures.weights_distortion
"""

import matplotlib.pyplot as plt
import numpy as np
from ixplore.metrics import compute_distortion

from src.models.ixplore_wrapper import IXPLOREModel
from src.results import experiment_dir
from src.visualization import apply_defaults, figsize, save_figure

DATASET = "smartvote_2023"
RESULTS_DIR = experiment_dir(DATASET, "weight_effect")

RESOLUTION = 30
SEED = 0
SPARSITIES = [0.0, 0.3, 0.6, 0.9]  # fraction of entries masked per synthetic user
ROW_SW = [False, True]             # rows: weight scaling off / on


def make_figure(inner):
    low, high = inner.limits
    fig, axes = plt.subplots(2, 4, figsize=figsize(width_frac=1.0, aspect=0.55),
                             sharex=True, sharey=True)

    letters = iter("ABCDEFGH")
    for r, sw in enumerate(ROW_SW):
        inner.scale_weights = sw
        for c, sp in enumerate(SPARSITIES):
            ax = axes[r, c]
            grid, reembedded, dmean, dstd = compute_distortion(
                inner,
                resolution=RESOLUTION,
                keep_fraction=(1.0 - sp) if sp > 0 else None,
                random_state=SEED,
            )

            cx = (grid[:, 0] - low) / (high - low)
            cy = (grid[:, 1] - low) / (high - low)
            colors = np.column_stack([cx, 0.3 * np.ones(len(grid)), cy])

            for i in range(len(grid)):
                ax.plot([grid[i, 0], reembedded[i, 0]],
                        [grid[i, 1], reembedded[i, 1]],
                        color="k", lw=0.3, alpha=0.4, zorder=1)
            ax.scatter(reembedded[:, 0], reembedded[:, 1], c=colors,
                       s=3, edgecolors="k", lw=0.05, zorder=2)

            ax.set(aspect="equal",
                   xlim=(low - 0.05, high + 0.05),
                   ylim=(low - 0.05, high + 0.05),
                   xticklabels=[], yticklabels=[],
                   xticks=[], yticks=[])
            ax.grid(False)

            # Only the measured distortion stays in-panel; the (scaling, sparsity)
            # configuration is read off the axis labels.
            bbox = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=1.0,
                        edgecolor="0.5", linewidth=0.6)
            ax.text(0.04, 0.04, rf"MAE: {dmean:.2f} $\pm$ {dstd:.2f}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7, bbox=bbox)

            if r == len(ROW_SW) - 1:
                ax.set_xlabel(f"Test sparsity $v={sp:g}$")
            if c == 0:
                ax.set_ylabel(f'Weight-scaling {"on" if sw else "off"}')

            fig.text(0.0, 1.0, next(letters), va="bottom", ha="left",
                     weight="bold", transform=ax.transAxes)

    inner.scale_weights = False

    # Extra top headroom + row gap so the bold panel letters are never clipped.
    fig.subplots_adjust(left=0.025, right=0.995, top=0.95, bottom=0.05, wspace=0.0, hspace=0.15)
    return fig


def main():
    apply_defaults()
    inner = IXPLOREModel.load(RESULTS_DIR)._model
    save_figure(make_figure(inner), f"weights_distortion_{DATASET}.pdf")


if __name__ == "__main__":
    main()
