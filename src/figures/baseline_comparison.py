"""Figure: baseline comparison on Smartvote 2023.

Writes figures/baseline_comparison_smartvote_2023.pdf: reconstruction and
imputation MAE across sparsity levels, train and test split, for the
algorithms shown in the paper, with the three highlighted IXPLORE points.
Also prints the value range of the shown algorithms at the highlighted
points. Reads results/smartvote_2023/baseline/.

Usage:
    python -m src.figures.baseline_comparison
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from src.models import ALGORITHM_LABELS
from src.results import load_baseline_metrics
from src.visualization import apply_defaults, figsize, panel_labels, save_figure

DATASET = "smartvote_2023"
SUBSET_MODELS = ["pca-logistic", "vae-2layer", "umap-logistic", "ideal", "lsirm", "ixplore"]

# Highlighted IXPLORE points: (panel, split, sparsity, metric, label, text offset, ha).
HIGHLIGHTS = [
    (0, "train", 0.0, "fit_mae", "Embedding candidate responses", (3, -25), "left"),
    (0, "test", 0.6, "fit_mae", "Embedding voter responses", (-43, -13), "center"),
    (1, "test", 0.6, "impute_mae", "Predicting voter responses", (-36, -13), "center"),
]


def _stats(df):
    return df.groupby("sparsity").agg(
        fit_mae_mean=("fit_mae", "mean"), fit_mae_std=("fit_mae", "std"),
        impute_mae_mean=("impute_mae", "mean"), impute_mae_std=("impute_mae", "std"),
    ).reset_index()


def make_figure(train_metrics, test_metrics):
    fig, axes = plt.subplots(1, 2, figsize=figsize(width_frac=1.0, aspect=0.45), sharey=True)

    for i, model_name in enumerate(SUBSET_MODELS):
        color = f"C{i}"
        t_stats = _stats(train_metrics[train_metrics["model"] == model_name])
        axes[0].errorbar(t_stats["sparsity"], t_stats["fit_mae_mean"], yerr=t_stats["fit_mae_std"],
                         marker="o", capsize=3, color=color)
        axes[1].errorbar(t_stats["sparsity"], t_stats["impute_mae_mean"], yerr=t_stats["impute_mae_std"],
                         marker="o", capsize=3, color=color)

        test_sub = test_metrics[test_metrics["model"] == model_name]
        if not test_sub.empty:
            te_stats = _stats(test_sub)
            axes[0].errorbar(te_stats["sparsity"], te_stats["fit_mae_mean"], yerr=te_stats["fit_mae_std"],
                             marker="s", capsize=3, color=color, linestyle="--")
            axes[1].errorbar(te_stats["sparsity"], te_stats["impute_mae_mean"], yerr=te_stats["impute_mae_std"],
                             marker="s", capsize=3, color=color, linestyle="--")

    axes[0].set_ylabel("Reconstruction (MAE)")
    axes[1].set_ylabel("Imputation (MAE)")
    for ax in axes:
        ax.set_xlabel("Sparsity")
        ax.xaxis.set_major_locator(MultipleLocator(0.3))
        ax.xaxis.set_minor_locator(MultipleLocator(0.15))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.025))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_tick_params(labelleft=True)

    # Circle and annotate the three IXPLORE points discussed in the text.
    ixplore = {"train": train_metrics[train_metrics["model"] == "ixplore"],
               "test": test_metrics[test_metrics["model"] == "ixplore"]}
    for panel, split, x, col, text, offset, ha in HIGHLIGHTS:
        ax = axes[panel]
        y = _point(ixplore[split], x, col)
        ax.scatter([x], [y], s=500, facecolors="none", edgecolors="black", linewidths=1.5, zorder=5)
        ax.annotate(text, xy=(x, y), xytext=offset, textcoords="offset points",
                    fontsize=8, ha=ha, va="top",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1,
                                    shrinkA=0, shrinkB=13))

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", label="train"),
        Line2D([0], [0], color="black", linestyle="--", marker="s", label="test"),
    ]
    axes[0].legend(handles=style_handles, loc="best", handlelength=4)

    color_handles = [Line2D([0], [0], color=f"C{i}", lw=2, label=ALGORITHM_LABELS[m])
                     for i, m in enumerate(SUBSET_MODELS)]
    axes[1].legend(handles=color_handles, loc="lower right", title="Algorithm",
                   handlelength=1.5, ncols=2, labelspacing=0.15)
    panel_labels(axes)
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.96], w_pad=0.7)
    return fig


def _point(df, sparsity, col):
    return df[df["sparsity"] == sparsity][col].mean()


def _per_model_value(df, model_name, sparsity, col):
    """Mean of `col` for one model at one sparsity (matches the plotted point)."""
    sub = df[(df["model"] == model_name) & (df["sparsity"] == sparsity)]
    return np.nan if sub.empty else sub[col].mean()


def print_highlight_ranges(train_metrics, test_metrics):
    """Value range across the shown algorithms at the highlighted points."""
    specs = [
        ("Embedding candidate responses", "train", 0.0, "fit_mae"),
        ("Embedding user responses", "test", 0.6, "fit_mae"),
        ("Predicting user responses", "test", 0.6, "impute_mae"),
        ("End of train fit_mae", "train", 0.9, "fit_mae"),
        ("End of test fit_mae", "test", 0.9, "fit_mae"),
        ("End of train impute_mae", "train", 0.9, "impute_mae"),
        ("End of test impute_mae", "test", 0.9, "impute_mae"),
    ]
    print(f"Highlight value ranges across {len(SUBSET_MODELS)} shown algorithms "
          f"({', '.join(ALGORITHM_LABELS[m] for m in SUBSET_MODELS)}):\n")
    for label, split, sp, col in specs:
        src = train_metrics if split == "train" else test_metrics
        vals = {m: _per_model_value(src, m, sp, col) for m in SUBSET_MODELS}
        present = {m: v for m, v in vals.items() if not np.isnan(v)}
        missing = [ALGORITHM_LABELS[m] for m, v in vals.items() if np.isnan(v)]
        print(f"{label}  [{split} {col} @ sparsity {sp}]")
        if present:
            lo_m = min(present, key=present.get)
            hi_m = max(present, key=present.get)
            print(f"  range: {present[lo_m]:.4f} ({ALGORITHM_LABELS[lo_m]})"
                  f"  ->  {present[hi_m]:.4f} ({ALGORITHM_LABELS[hi_m]})")
        else:
            print("  (no data)")
        if missing:
            print(f"  missing: {', '.join(missing)}")
        print()


def main():
    apply_defaults()
    train_metrics = load_baseline_metrics(DATASET, "train")
    test_metrics = load_baseline_metrics(DATASET, "test")
    save_figure(make_figure(train_metrics, test_metrics), f"baseline_comparison_{DATASET}.pdf")
    print_highlight_ranges(train_metrics, test_metrics)


if __name__ == "__main__":
    main()
