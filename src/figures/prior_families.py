"""Figure: the Gaussian and the log-barrier prior along one latent axis.

Writes figures/prior_families_smartvote_2023.pdf: both priors evaluated at
x_2 = 0 for the strengths used in the boundary-regularization experiment
(src.scripts.run_boundary_regularization). Needs no stored results.

Usage:
    python -m src.figures.prior_families
"""

import matplotlib.pyplot as plt
import numpy as np
from ixplore.prior import set_gaussian_prior, set_log_barrier_prior

from src.visualization import apply_defaults, panel_labels, save_figure, subplots

GAUSSIAN_TAUS = [0.05, 0.1, 0.25, 0.5, 1.0]
LOG_BARRIER_ALPHAS = [0.1, 0.2, 0.5, 1.0, 2.0]
LIMITS = (-1.0, 1.0)


def make_figure():
    xs = np.linspace(LIMITS[0], LIMITS[1], 401)
    slice_grid = np.column_stack([xs, np.zeros_like(xs)])

    fig, (ax_g, ax_lb) = subplots(1, 2, width_frac=1.0, aspect=0.35, sharey=True)

    cmap_g = plt.get_cmap("Reds")
    for i, tau in enumerate(GAUSSIAN_TAUS):
        p = set_gaussian_prior(slice_grid, prior_variance=tau, log=False)
        color = cmap_g(0.25 + 0.7 * i / max(len(GAUSSIAN_TAUS) - 1, 1))
        ax_g.plot(xs, p, color=color, label=fr"$\tau^2={tau}$")
    ax_g.set_xlabel(r"$x_1$ (at $x_2=0$)")
    ax_g.set_ylabel("prior density")
    ax_g.set_xlim(LIMITS)
    ax_g.set_ylim(bottom=0)
    ax_g.legend(loc="upper right")

    cmap_lb = plt.get_cmap("Blues")
    for i, alpha in enumerate(LOG_BARRIER_ALPHAS):
        p = set_log_barrier_prior(slice_grid, alpha=alpha, limits=LIMITS, log=False)
        color = cmap_lb(0.25 + 0.7 * i / max(len(LOG_BARRIER_ALPHAS) - 1, 1))
        ax_lb.plot(xs, p, color=color, label=fr"$\alpha={alpha}$")
    ax_lb.set_xlabel(r"$x_1$ (at $x_2=0$)")
    ax_lb.set_xlim(LIMITS)
    ax_lb.set_ylim(bottom=0)
    ax_lb.legend(loc="upper right")

    panel_labels((ax_g, ax_lb))
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.95], w_pad=0.7)
    return fig


def main():
    apply_defaults()
    save_figure(make_figure(), "prior_families_smartvote_2023.pdf")


if __name__ == "__main__":
    main()
