"""Figures: train and test MAE curves of all algorithms, one figure per dataset.

Writes figures/baseline_train_test_<dataset>.pdf: the left panel shows the
train split against the train sparsity u, the right panel the test split
against the test sparsity v. Reconstruction is solid with circles,
imputation dashed with squares, one colour per algorithm. Algorithms without
stored metrics are skipped. Reads results/<dataset>/baseline/.

Usage:
    python -m src.figures.baseline_train_test
    python -m src.figures.baseline_train_test --dataset polis
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from src.data import DATASETS
from src.models import ALGORITHM_LABELS
from src.results import load_baseline_metrics
from src.visualization import apply_defaults, figsize, panel_labels, save_figure

SUBSET_KEYS = ["pca-logistic", "vae-2layer", "umap-logistic",
               "ideal", "emirt", "lsirm",
               "ixplore", "ixplore-binarised"]


def _agg(df, model_name):
    """Per-sparsity means and stds of fit_mae and impute_mae, or None if absent."""
    if df.empty:
        return None
    sub = df[df["model"] == model_name]
    if sub.empty:
        return None
    return sub.groupby("sparsity").agg(
        fit_mean=("fit_mae", "mean"), fit_std=("fit_mae", "std"),
        imp_mean=("impute_mae", "mean"), imp_std=("impute_mae", "std"),
    ).reset_index()


def make_figure(dataset):
    train_df = load_baseline_metrics(dataset, "train")
    test_df = load_baseline_metrics(dataset, "test")

    fig, axes = plt.subplots(1, 2, figsize=figsize(width_frac=1.0, aspect=0.4), sharey=True)
    ax_train, ax_test = axes

    for alg in SUBSET_KEYS:
        color = f"C{SUBSET_KEYS.index(alg)}"
        t = _agg(train_df, alg)
        te = _agg(test_df, alg)
        if t is not None:
            ax_train.errorbar(t["sparsity"], t["fit_mean"], yerr=t["fit_std"],
                              marker="o", capsize=3, color=color)
            ax_train.errorbar(t["sparsity"], t["imp_mean"], yerr=t["imp_std"],
                              marker="s", capsize=3, color=color, linestyle="--")
        if te is not None:
            ax_test.errorbar(te["sparsity"], te["fit_mean"], yerr=te["fit_std"],
                             marker="o", capsize=3, color=color)
            ax_test.errorbar(te["sparsity"], te["imp_mean"], yerr=te["imp_std"],
                             marker="s", capsize=3, color=color, linestyle="--")

    ax_train.set_xlabel(r"Train sparsity $u$")
    ax_test.set_xlabel(r"Test sparsity $v$")
    ax_train.set_ylabel("MAE")
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(0.3))
        ax.xaxis.set_minor_locator(MultipleLocator(0.15))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.025))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_tick_params(labelleft=True)

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", label="Reconstruction"),
        Line2D([0], [0], color="black", linestyle="--", marker="s", label="Imputation"),
    ]
    ax_train.legend(handles=style_handles, loc="best", handlelength=4)

    present_algs = [a for a in SUBSET_KEYS
                    if (_agg(train_df, a) is not None) or (_agg(test_df, a) is not None)]
    color_handles = [Line2D([0], [0], color=f"C{SUBSET_KEYS.index(a)}", lw=2,
                            label=ALGORITHM_LABELS[a]) for a in present_algs]
    ax_test.legend(handles=color_handles, loc="best", title="Algorithm",
                   handlelength=1.5, ncols=2, labelspacing=0.15)

    panel_labels(axes)
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.96], w_pad=0.7)
    return fig


def main(args=None):
    parser = argparse.ArgumentParser(description="Per-dataset train/test MAE figures")
    parser.add_argument("--dataset", choices=DATASETS, default=None,
                        help="one dataset (default: all)")
    opts = parser.parse_args(args)
    apply_defaults()
    for dataset in ([opts.dataset] if opts.dataset else DATASETS):
        save_figure(make_figure(dataset), f"baseline_train_test_{dataset}.pdf")


if __name__ == "__main__":
    main()
