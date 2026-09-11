"""Figure: the three IXPLORE kernels against the baselines on Smartvote 2023.

Writes figures/kernel_comparison_<metric>_smartvote_2023.pdf: reconstruction
and imputation error of the linear, polynomial, and RFF kernel across
sparsity levels, train and test split, with the baseline algorithms in grey.
Also prints paired one-sided t-tests of the polynomial and RFF kernel
against the linear kernel, matched on (sparsity, seed). Reads
results/smartvote_2023/feature_effect/ and results/smartvote_2023/baseline/.

Usage:
    python -m src.figures.kernel_comparison
    python -m src.figures.kernel_comparison --metric accuracy
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy import stats

from src.results import load_baseline_metrics, load_metrics
from src.visualization import apply_defaults, figsize, panel_labels, save_figure

DATASET = "smartvote_2023"
KERNELS = ["linear", "polynomial", "rff"]
KERNEL_LABELS = {"linear": "Linear", "polynomial": "Polynomial", "rff": "RFF"}
KERNEL_COLORS = {"linear": "C0", "polynomial": "C1", "rff": "C2"}
SUBSET_MODELS = ["pca-logistic", "vae-2layer", "umap-logistic", "ideal", "lsirm", "ixplore"]
METRIC_LABEL = {"mae": "MAE", "accuracy": "Accuracy"}


def _baseline_line(df, ax, metric, linestyle):
    stats_ = df.groupby("sparsity")[metric].mean().reset_index()
    ax.plot(stats_["sparsity"], stats_[metric],
            color="grey", linestyle=linestyle, linewidth=0.8, alpha=0.5, zorder=0)


def _kernel_errorbar(df, ax, metric, color, linestyle, marker):
    stats_ = df.groupby("sparsity")[metric].agg(["mean", "std"]).reset_index()
    ax.errorbar(stats_["sparsity"], stats_["mean"], yerr=stats_["std"],
                marker=marker, color=color, linestyle=linestyle, zorder=3)


def make_figure(train, test, baseline_train, baseline_test, metric):
    fit_col = f"fit_{metric}"
    impute_col = f"impute_{metric}"

    fig, axes = plt.subplots(1, 2, figsize=figsize(width_frac=1.0, aspect=0.35), sharey=True)

    for alg in SUBSET_MODELS:
        btr = baseline_train[baseline_train["model"] == alg]
        bte = baseline_test[baseline_test["model"] == alg]
        _baseline_line(btr, axes[0], fit_col, "-")
        _baseline_line(btr, axes[1], impute_col, "-")
        if not bte.empty:
            _baseline_line(bte, axes[0], fit_col, "--")
            _baseline_line(bte, axes[1], impute_col, "--")

    for kernel in KERNELS:
        ktr = train[train["model"] == kernel]
        kte = test[test["model"] == kernel]
        color = KERNEL_COLORS[kernel]
        _kernel_errorbar(ktr, axes[0], fit_col, color, "-", "o")
        _kernel_errorbar(ktr, axes[1], impute_col, color, "-", "o")
        if not kte.empty:
            _kernel_errorbar(kte, axes[0], fit_col, color, "--", "s")
            _kernel_errorbar(kte, axes[1], impute_col, color, "--", "s")

    axes[0].set_ylabel(f"Reconstruction ({METRIC_LABEL[metric]})")
    axes[1].set_ylabel(f"Imputation ({METRIC_LABEL[metric]})")
    for ax in axes:
        ax.set_xlabel("Sparsity")
        ax.xaxis.set_major_locator(MultipleLocator(0.3))
        ax.xaxis.set_minor_locator(MultipleLocator(0.15))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.025))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_tick_params(labelleft=True)

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", label="train"),
        Line2D([0], [0], color="black", linestyle="--", marker="s", label="test"),
    ]
    axes[0].legend(handles=style_handles, loc="best", handlelength=4)

    color_handles = (
        [Line2D([0], [0], color=KERNEL_COLORS[k], lw=2, label=KERNEL_LABELS[k]) for k in KERNELS]
        + [Line2D([0], [0], color="grey", lw=0.8, alpha=0.5, label="Baselines")]
    )
    axes[1].legend(handles=color_handles, loc="lower right", title="Model",
                   handlelength=1.5, ncols=2, labelspacing=0.15, facecolor="white")
    panel_labels(axes)
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.95], w_pad=0.7)
    return fig


def paired_test(df, metric, kernel, alternative):
    """Paired one-sided t-test of *kernel* against the linear kernel.

    Pairs are matched on (sparsity, seed) so within-pair variance is removed.
    """
    pivot = (df[df["model"].isin(["linear", kernel])]
             .pivot_table(values=metric, index=["sparsity", "seed"], columns="model"))
    pivot = pivot.dropna()
    diff = pivot[kernel] - pivot["linear"]
    t = stats.ttest_rel(pivot[kernel], pivot["linear"], alternative=alternative)
    return {
        "n_pairs": len(diff),
        "mean_diff": diff.mean(),
        "std_diff": diff.std(),
        "t": t.statistic,
        "p": t.pvalue,
    }


def print_paired_tests(train, test):
    # For MAE: kernel < linear means lower error, so alternative = "less".
    # For accuracy: kernel > linear means higher accuracy, so alternative = "greater".
    configs = [
        ("train", train, "fit_mae", "less"),
        ("train", train, "impute_mae", "less"),
        ("test", test, "fit_mae", "less"),
        ("test", test, "impute_mae", "less"),
        ("train", train, "fit_accuracy", "greater"),
        ("train", train, "impute_accuracy", "greater"),
        ("test", test, "fit_accuracy", "greater"),
        ("test", test, "impute_accuracy", "greater"),
    ]
    rows = []
    for split, df, metric, alt in configs:
        for kernel in ("polynomial", "rff"):
            r = paired_test(df, metric, kernel, alt)
            r.update({"split": split, "metric": metric, "kernel_vs_linear": kernel, "alt": alt})
            rows.append(r)
    result = pd.DataFrame(rows)[["split", "metric", "kernel_vs_linear", "alt",
                                  "n_pairs", "mean_diff", "std_diff", "t", "p"]]
    print("Paired one-sided t-tests against the linear kernel:")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main(args=None):
    parser = argparse.ArgumentParser(description="Kernel comparison figure")
    parser.add_argument("--metric", choices=list(METRIC_LABEL), default="mae")
    opts = parser.parse_args(args)
    apply_defaults()

    train = load_metrics(DATASET, "feature_effect", "train")
    test = load_metrics(DATASET, "feature_effect", "test")
    baseline_train = load_baseline_metrics(DATASET, "train", SUBSET_MODELS)
    baseline_test = load_baseline_metrics(DATASET, "test", SUBSET_MODELS)

    fig = make_figure(train, test, baseline_train, baseline_test, opts.metric)
    save_figure(fig, f"kernel_comparison_{opts.metric}_{DATASET}.pdf")
    print_paired_tests(train, test)


if __name__ == "__main__":
    main()
