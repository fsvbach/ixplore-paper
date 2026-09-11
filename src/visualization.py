"""Default figure parameters matching the report's text width.

The IXPLORE paper (probml.sty) sets `\textwidth = 6.0in`
at 10pt. The figure scripts use these defaults so saved figures match the
report's column width without rescaling in LaTeX.
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.paths import FIGURES_DIR, ROOT

TEXTWIDTH_IN = 6.0
DEFAULT_HEIGHT_IN = 3.2
DEFAULT_FIGSIZE = (TEXTWIDTH_IN, DEFAULT_HEIGHT_IN)


def figsize(width_frac: float = 1.0, aspect: float = DEFAULT_HEIGHT_IN / TEXTWIDTH_IN):
    """Return a (w, h) tuple in inches scaled to a fraction of \\textwidth."""
    w = TEXTWIDTH_IN * width_frac
    return (w, w * aspect)


def apply_defaults():
    """Reset matplotlib to its defaults and apply the paper's rcParams.

    Resetting first makes every figure independent of import order
    (``ixplore.visualization`` changes rcParams globally when imported).
    """
    mpl.rcdefaults()
    mpl.rcParams.update({
        "figure.figsize": DEFAULT_FIGSIZE,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.7",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "errorbar.capsize": 3,
    })


def subplots(nrows=1, ncols=1, width_frac=1.0, aspect=None, **kwargs):
    """plt.subplots wrapper that sizes the figure to the report's textwidth."""
    if aspect is None:
        aspect = DEFAULT_HEIGHT_IN / TEXTWIDTH_IN
    kwargs.setdefault("figsize", figsize(width_frac=width_frac, aspect=aspect))
    return plt.subplots(nrows, ncols, **kwargs)


def fmt_tau(s: float, *, threshold: int = 3, rtol: float = 1e-3) -> str:
    """Format a $\\tau^2$ value for plot labels.

    Returns ``"10^k"`` (LaTeX-ready) when ``s`` is within ``rtol`` of an integer
    power of 10 with absolute exponent ``>= threshold``; otherwise falls back to
    Python's ``:g`` formatting. This keeps small everyday values (0.05, 1.0)
    readable while turning ``1e6`` / ``1e+06`` into ``10^6``.
    """
    if s > 0:
        k = round(np.log10(s))
        if abs(k) >= threshold and abs(s - 10.0 ** k) <= rtol * 10.0 ** k:
            return f"10^{{{int(k)}}}"
    return f"{s:g}"


def make_ellipse(mean: np.ndarray, cov: np.ndarray, n_std: float = 1.0, **kwargs) -> Ellipse:
    """Matplotlib Ellipse patch for an n-sigma contour of a 2D Gaussian."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def ellipse_legend_handler(linewidth: float = 0.5):
    """A HandlerPatch that draws Ellipse legend entries as ellipses (not boxes).

    Use as ``handler_map={Ellipse: ellipse_legend_handler()}`` when calling
    ``ax.legend`` so that ellipse patches show up as miniature ellipses in the
    legend instead of the default rectangle stand-in.
    """
    from matplotlib.legend_handler import HandlerPatch

    class _EllipseHandler(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            center = (width / 2 - xdescent, height / 2 - ydescent)
            patch = Ellipse(xy=center, width=width, height=height * 0.7,
                            angle=0,
                            facecolor=orig_handle.get_facecolor(),
                            edgecolor=orig_handle.get_edgecolor(),
                            linewidth=linewidth,
                            linestyle=orig_handle.get_linestyle())
            patch.set_transform(trans)
            return [patch]

    return _EllipseHandler()


def panel_labels(axes, letters: str = "ABCDEFGH") -> None:
    """Bold panel letters at the top-left corner of each axes, outside the frame."""
    for ax, letter in zip(np.ravel(axes), letters):
        ax.figure.text(0, 1, letter, va="bottom", ha="left", weight="bold",
                       transform=ax.transAxes)


def save_figure(fig, name: str) -> Path:
    """Save *fig* as figures/<name>, close it, and report the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path.relative_to(ROOT)}")
    return path
