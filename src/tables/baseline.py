"""Tables: baseline comparison, one table per dataset.

Writes tables/baseline_<dataset>.tex: MAE and accuracy of every algorithm
for train and test reconstruction and imputation, each cell the mean over
all sparsity levels (and seeds). Best value per column in bold, runner-up in
italics; IXPLORE variants below the separator. Reads results/<dataset>/baseline/.

Usage:
    python -m src.tables.baseline
    python -m src.tables.baseline --dataset polis
"""

import argparse

import pandas as pd

from src.data import DATASET_LABELS, DATASETS
from src.models import ALGORITHM_LABELS, BASELINE_ALGORITHMS
from src.results import load_baseline_metrics
from src.tables import write_table

ALGORITHMS = list(BASELINE_ALGORITHMS)
IXPLORE_ALGS = {"ixplore", "ixplore-binarised"}
COLUMNS = [
    ("train_fit_mae", "mae"), ("train_fit_accuracy", "accuracy"),
    ("train_impute_mae", "mae"), ("train_impute_accuracy", "accuracy"),
    ("test_fit_mae", "mae"), ("test_fit_accuracy", "accuracy"),
    ("test_impute_mae", "mae"), ("test_impute_accuracy", "accuracy"),
]


def build_summary_table(dataset):
    """One row per algorithm; each cell the mean over sparsity levels of the
    per-sparsity seed means. Warns about (model, sparsity) cells whose seeds
    show zero or undefined spread."""
    train_df = load_baseline_metrics(dataset, "train")
    test_df = load_baseline_metrics(dataset, "test")

    cols = {}
    for split_label, df in [("train", train_df), ("test", test_df)]:
        for setting in ["fit", "impute"]:
            for metric in ["mae", "accuracy"]:
                col = f"{setting}_{metric}"
                key = f"{split_label}_{setting}_{metric}"
                if df.empty or col not in df.columns:
                    cols[key] = pd.Series(dtype=float)
                else:
                    grouped = df.groupby(["model", "sparsity"])[col]
                    per_sp = grouped.mean()
                    cols[key] = per_sp.groupby("model").mean()
                    n_seeds = grouped.size()
                    stds = grouped.std()
                    for (m, sp), s in stds.items():
                        if sp != 0.0 and (s == 0 or pd.isna(s)):
                            kind = ("single-seed (NaN std)" if pd.isna(s)
                                    else f"zero std across {n_seeds.loc[(m, sp)]} seeds")
                            print(f"  WARNING [{dataset}/{split_label}/{col}] {kind}: model={m}, sparsity={sp}")
    return pd.DataFrame(cols).reindex(ALGORITHMS)


def _fmt(value, metric):
    if pd.isna(value):
        return "--"
    if metric == "accuracy":
        return f"{value * 100:.1f}\\%"
    return f"{value:.3f}"


def _rank_decorate(table, col, metric):
    """{alg: latex cell}, bolding the best and italicising the runner-up across
    the full table (baselines and IXPLORE variants combined)."""
    series = table[col].dropna()
    ascending = metric == "mae"
    ordered = series.sort_values(ascending=ascending)
    best = ordered.index[0] if len(ordered) >= 1 else None
    second = ordered.index[1] if len(ordered) >= 2 else None
    out = {}
    for alg in table.index:
        s = _fmt(table.loc[alg, col], metric)
        if alg == best:
            s = r"\textbf{" + s + "}"
        elif alg == second:
            s = r"\textit{" + s + "}"
        out[alg] = s
    return out


def to_latex(table, dataset):
    label = DATASET_LABELS[dataset]
    baseline_algs = [a for a in ALGORITHMS if a not in IXPLORE_ALGS]
    ixplore_algs = [a for a in ALGORITHMS if a in IXPLORE_ALGS]

    caption = (
        rf"\caption{{Performance comparison on {label}."
        r" Each cell is the mean over all sparsity levels."
        r" Reference algorithms are listed in the top block;"
        r" \IXPLORE is run both with the continuous input and the binarized input$^*$."
        r" The best-performing algorithm is shown in \textbf{bold}, the runner-up in \textit{italics}.}"
    )
    header = "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        caption,
        rf"\label{{tab:baseline_{dataset}}}",
        r"\small",
        r"\begin{tabular}{l" + "cc" * 4 + "}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Train Rec.} & \multicolumn{2}{c}{Train Imp.}"
        r" & \multicolumn{2}{c}{Test Rec.} & \multicolumn{2}{c}{Test Imp.} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Algorithm & MAE & ACC & MAE & ACC & MAE & ACC & MAE & ACC \\",
        r"\midrule",
        "",
    ])
    decorated = {col: _rank_decorate(table, col, metric) for col, metric in COLUMNS}

    def _render(algs):
        lines = []
        for alg in algs:
            if alg not in decorated[COLUMNS[0][0]]:
                continue
            cells = [decorated[col][alg] for col, _ in COLUMNS]
            lines.append(f"{ALGORITHM_LABELS[alg]} & " + " & ".join(cells) + r" \\")
        return "\n".join(lines)

    body = _render(baseline_algs) + "\n" + r"\midrule" + "\n" + _render(ixplore_algs) + "\n"
    footer = "\n".join([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return header + body + footer


def main(args=None):
    parser = argparse.ArgumentParser(description="Baseline comparison tables")
    parser.add_argument("--dataset", choices=DATASETS, default=None,
                        help="one dataset (default: all)")
    opts = parser.parse_args(args)
    for dataset in ([opts.dataset] if opts.dataset else DATASETS):
        write_table(f"baseline_{dataset}.tex", to_latex(build_summary_table(dataset), dataset))


if __name__ == "__main__":
    main()
