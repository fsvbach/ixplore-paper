"""Figure: PCA dimensionality sweep against IXPLORE on Smartvote 2023.

Writes figures/pca_dimensionality_<metric>_smartvote_2023.pdf: reconstruction
and imputation error of linear and logistic PCA as a function of the number
of dimensions d (train solid, test dashed), with the IXPLORE reference
configuration (tau^2 = 0.25, PCA initialisation, T = 10) as horizontal lines.
Vertical ticks mark the smallest d at which a PCA variant beats the reference;
if it never does, the curve's own optimum is marked. Also prints those
crossover dimensions. Reads results/smartvote_2023/pca_sweep/ and
results/smartvote_2023/iteration_effect/.

Usage:
    python -m src.figures.pca_dimensionality
    python -m src.figures.pca_dimensionality --metric accuracy
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from src.models import ALGORITHM_LABELS
from src.results import load_metrics
from src.visualization import apply_defaults, panel_labels, save_figure, subplots

DATASET = "smartvote_2023"
METRICS = ["fit_mae", "fit_accuracy", "impute_mae", "impute_accuracy"]
CELLS = [("train", "fit"), ("train", "impute"), ("test", "fit"), ("test", "impute")]

# IXPLORE reference configuration from the iteration_effect experiment.
REF_TAU, REF_INIT, REF_ITERATION = 0.25, "pca", 10

VARIANTS = ("pca-linear", "pca-logistic")
VARIANT_COLORS = {"pca-linear": "C0", "pca-logistic": "C2", "ixplore": "C3"}
SPLIT_STYLES = {"train": "-", "test": "--"}
METRIC_LABEL = {"mae": "MAE", "accuracy": "Accuracy"}


def ixplore_reference():
    """Metrics of the reference configuration, averaged over seeds and sparsities."""
    ref = {}
    for split in ("train", "test"):
        df = load_metrics(DATASET, "iteration_effect", split)
        sub = df[(df["tau"] == REF_TAU) & (df["init"] == REF_INIT) & (df["iteration"] == REF_ITERATION)]
        assert len(sub) > 0, f"no iteration_effect rows for the reference configuration ({split})"
        for m in METRICS:
            ref[f"{split}_{m}"] = sub[m].mean()
    return ref


def pca_sweep_average():
    """Sweep metrics averaged over seeds and sparsities, one row per (model, k)."""
    train = load_metrics(DATASET, "pca_sweep", "train").groupby(["model", "k"], as_index=False)[METRICS].mean()
    test = load_metrics(DATASET, "pca_sweep", "test").groupby(["model", "k"], as_index=False)[METRICS].mean()
    return (train.rename(columns={m: f"train_{m}" for m in METRICS})
            .merge(test.rename(columns={m: f"test_{m}" for m in METRICS}), on=["model", "k"], how="outer")
            .sort_values(["model", "k"]).reset_index(drop=True))


def crossovers(pca_avg, ref, metric):
    """Smallest k at which each PCA variant beats the reference per cell (None if never),
    and the k and value of each curve's own optimum."""
    better_is_smaller = metric == "mae"
    crossover_k, extreme_k = {}, {}
    for variant in VARIANTS:
        sub = pca_avg[pca_avg["model"] == variant].sort_values("k")
        crossover_k[variant], extreme_k[variant] = {}, {}
        for split, kind in CELLS:
            col = f"{split}_{kind}_{metric}"
            if sub[col].notna().any():
                idx = sub[col].idxmin() if better_is_smaller else sub[col].idxmax()
                extreme_k[variant][(split, kind)] = (int(sub.loc[idx, "k"]), float(sub.loc[idx, col]))
            r = ref.get(col)
            if r is None or pd.isna(r):
                crossover_k[variant][(split, kind)] = None
                continue
            beats = sub[sub[col] < r] if better_is_smaller else sub[sub[col] > r]
            crossover_k[variant][(split, kind)] = None if beats.empty else int(beats["k"].min())
    return crossover_k, extreme_k


def make_figure(pca_avg, ref, crossover_k, extreme_k, metric):
    metric_label = METRIC_LABEL[metric]
    fig, axes = subplots(1, 2, width_frac=1.0, aspect=0.35)
    panels = [
        (axes[0], f"fit_{metric}", f"Reconstruction ({metric_label})"),
        (axes[1], f"impute_{metric}", f"Imputation ({metric_label})"),
    ]
    axes[0].set_ylim(0, 0.3)
    axes[1].set_ylim(0.195, 0.34)

    # First pass: curves and reference lines, so each axes' ylim settles before the
    # crossover markers (which extend down to the x-axis) are drawn.
    for ax, m, ylabel in panels:
        for variant in VARIANTS:
            sub = pca_avg[pca_avg["model"] == variant]
            for split, ls in SPLIT_STYLES.items():
                ax.plot(sub["k"], sub[f"{split}_{m}"], color=VARIANT_COLORS[variant], linestyle=ls)
        for split, ls in SPLIT_STYLES.items():
            v = ref.get(f"{split}_{m}")
            if v is None or pd.isna(v):
                continue
            ax.axhline(v, color=VARIANT_COLORS["ixplore"], linestyle=ls, linewidth=1.2)
        ax.set_xlabel(r"PCA dimensions ($d$)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 74)

    # Second pass: crossover markers with the ylim locked, k labels rotated at the
    # bottom of the axes in the variant colour.
    for ax, m, _ in panels:
        kind = m.split("_")[0]
        y_bottom, y_top = ax.get_ylim()
        ax.set_ylim(y_bottom, y_top)

        crossovers_for_panel = []
        for variant in VARIANTS:
            color = VARIANT_COLORS[variant]
            sub = pca_avg[pca_avg["model"] == variant]
            for split, ls in SPLIT_STYLES.items():
                col = f"{split}_{m}"
                k_cross = crossover_k[variant].get((split, kind))
                if k_cross is not None:
                    y_at_k = float(sub.loc[sub["k"] == k_cross, col].iloc[0])
                    ax.plot([k_cross, k_cross], [y_bottom, y_at_k],
                            color=color, linestyle=ls, linewidth=0.5, alpha=0.9, zorder=4)
                    ax.scatter([k_cross], [y_at_k], s=8, color=color, zorder=6)
                    crossovers_for_panel.append((k_cross, color))
                else:
                    ext = extreme_k[variant].get((split, kind))
                    if ext is not None:
                        k_ext, v_ext = ext
                        ax.scatter([k_ext], [v_ext], s=28, color=color,
                                   edgecolors="white", linewidths=0.8, zorder=6)

        for k_cross, color in crossovers_for_panel:
            ax.annotate(f"$d={k_cross}$",
                        xy=(k_cross, 0), xycoords=("data", "axes fraction"),
                        xytext=(2, 3), textcoords="offset points",
                        color=color, rotation=90, fontsize=7, ha="left", va="bottom", zorder=7)

    split_handles = [Line2D([0], [0], color="black", lw=1.5, linestyle=ls, label=split)
                     for split, ls in SPLIT_STYLES.items()]
    axes[0].legend(handles=split_handles, loc="upper right", ncol=2, frameon=True, framealpha=0.9)

    algo_handles = [Line2D([0], [0], color=VARIANT_COLORS[m], lw=1.5, label=ALGORITHM_LABELS[m])
                    for m in ("pca-linear", "pca-logistic", "ixplore")]
    axes[1].legend(handles=algo_handles, loc="lower right", ncol=1, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(1.0, 0.06))
    panel_labels(axes)
    fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.96], w_pad=0.7)
    return fig


def main(args=None):
    parser = argparse.ArgumentParser(description="PCA dimensionality sweep figure")
    parser.add_argument("--metric", choices=list(METRIC_LABEL), default="mae")
    opts = parser.parse_args(args)
    apply_defaults()

    ref = ixplore_reference()
    pca_avg = pca_sweep_average()
    crossover_k, extreme_k = crossovers(pca_avg, ref, opts.metric)

    fig = make_figure(pca_avg, ref, crossover_k, extreme_k, opts.metric)
    save_figure(fig, f"pca_dimensionality_{opts.metric}_{DATASET}.pdf")

    print(f"IXPLORE reference: tau^2={REF_TAU}, init={REF_INIT}, T={REF_ITERATION}")
    print(pd.Series(ref).round(4).to_string())
    print(f"\nSmallest d at which PCA beats the reference ({METRIC_LABEL[opts.metric]}; 'never' otherwise):")
    rows = {ALGORITHM_LABELS[v]: {f"{split} {kind}": (k if k is not None else "never")
                                  for (split, kind), k in crossover_k[v].items()}
            for v in VARIANTS}
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
