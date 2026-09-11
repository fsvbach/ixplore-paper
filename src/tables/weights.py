"""Table: confidence-weights summary on Smartvote 2023.

Writes tables/weights_smartvote_2023.tex: for each weight schedule (uniform,
provided, extreme) with and without weight scaling, the test reconstruction
and imputation MAE and accuracy and the latent-geometry diagnostics (spread,
boundary fraction, distortion), averaged over sparsity levels and seeds.
Best value per column in bold, runner-up in italics. Reads
results/smartvote_2023/weight_effect/.

Usage:
    python -m src.tables.weights
"""

import pandas as pd

from src.results import experiment_dir
from src.tables import write_table

DATASET = "smartvote_2023"
RESULTS_DIR = experiment_dir(DATASET, "weight_effect")

DISPLAY_COLS = ["test_fit_mae", "test_fit_acc",
                "test_impute_mae", "test_impute_acc",
                "spread", "boundary", "distortion"]
LOWER_IS_BETTER = {"test_fit_mae", "test_impute_mae", "boundary", "distortion"}
PERCENT_COLS = {"test_fit_acc", "test_impute_acc", "boundary"}
DECIMALS = {"test_fit_mae": 3, "test_impute_mae": 3,
            "spread": 3, "distortion": 3,
            "test_fit_acc": 1, "test_impute_acc": 1, "boundary": 1}
SCHEDULE_LABEL = {"uniform": "Uniform", "provided": "Provided", "extreme": "Extreme"}
SW_LABEL = {False: "Disabled", True: "Enabled"}

CAPTION = (
    r"\caption{Confidence-weights summary on Smartvote 2023. The model is trained once on "
    r"candidates with uniform weights (PCA initialization, $\tau^2 = 0.25$, $T = 10$ iterations, "
    r"grid resolution $R = 100$); each row is an inference-time configuration of weight scaling "
    r"and weight schedule. Cells are averaged across sparsity levels and seeds. The first four "
    r"metric columns report test reconstruction and imputation error; the last three are "
    r"latent-geometry diagnostics: the spread as the total variance of the embedding, the "
    r"boundary fraction (BF), and the distortion (DIS). Best in column \textbf{bold}, runner-up "
    r"in \textit{italics}.}"
)


def summarise(test_metrics, distortion):
    """One row per (scale_weights, schedule), metrics averaged over sparsity and seeds."""
    metrics_per_row = (
        test_metrics
        .groupby(["schedule", "scale_weights"])
        .agg(test_fit_mae=("fit_mae", "mean"),
             test_fit_acc=("fit_accuracy", "mean"),
             test_impute_mae=("impute_mae", lambda s: s.dropna().mean()),
             test_impute_acc=("impute_accuracy", lambda s: s.dropna().mean()),
             spread=("spread", "mean"),
             boundary=("boundary", "mean"))
        .reset_index()
    )
    # Distortion is recorded per (scale_weights, sparsity, seed) and is schedule-independent.
    dist_per_sw = (
        distortion
        .groupby("scale_weights")["distortion_mean"]
        .mean()
        .rename("distortion")
        .reset_index()
    )
    summary = metrics_per_row.merge(dist_per_sw, on="scale_weights")
    summary["_sched_order"] = summary["schedule"].map(dict(uniform=0, provided=1, extreme=2))
    summary["_sw_order"] = summary["scale_weights"].astype(int)
    return (summary.sort_values(["_sw_order", "_sched_order"])
            .drop(columns=["_sched_order", "_sw_order"]))


def _format_value(v, col):
    d = DECIMALS[col]
    if col in PERCENT_COLS:
        return f"{v * 100:.{d}f}\\%"
    return f"{v:.{d}f}"


def _rank_marks(series, lower_is_better):
    """{row_idx: 'bold'|'italic'} for the best and runner-up values."""
    ordered = series.sort_values(ascending=lower_is_better)
    marks = {}
    if len(ordered) >= 1:
        marks[ordered.index[0]] = "bold"
    if len(ordered) >= 2 and ordered.iloc[1] != ordered.iloc[0]:
        marks[ordered.index[1]] = "italic"
    return marks


def _wrap(s, mark):
    if mark == "bold":
        return rf"\textbf{{{s}}}"
    if mark == "italic":
        return rf"\textit{{{s}}}"
    return s


def to_latex(summary):
    formatted = pd.DataFrame(index=summary.index, columns=DISPLAY_COLS, dtype=object)
    for col in DISPLAY_COLS:
        marks = _rank_marks(summary[col], lower_is_better=col in LOWER_IS_BETTER)
        for idx, val in summary[col].items():
            formatted.loc[idx, col] = _wrap(_format_value(val, col), marks.get(idx))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        CAPTION,
        r"\label{tab:weights_smartvote_2023}",
        r"\small",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Test Rec.} & \multicolumn{2}{c}{Test Imp.} & \multicolumn{3}{c}{Latent geometry} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}",
        r"Scaling & Weights & MAE & ACC & MAE & ACC & Spread & BF & DIS \\",
        r"\midrule",
    ]
    prev_sw = None
    for idx, row in summary.iterrows():
        sw = row["scale_weights"]
        if prev_sw is not None and sw != prev_sw:
            lines.append(r"\midrule")
        cells = [formatted.loc[idx, c] for c in DISPLAY_COLS]
        lines.append(f"{SW_LABEL[sw]} & {SCHEDULE_LABEL[row['schedule']]} & " + " & ".join(cells) + r" \\")
        prev_sw = sw
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    test_metrics = pd.read_csv(RESULTS_DIR / "test_metrics.csv")
    distortion = pd.read_csv(RESULTS_DIR / "distortion.csv")
    summary = summarise(test_metrics, distortion)
    print(summary.set_index(["scale_weights", "schedule"])[DISPLAY_COLS].round(4).to_string())
    write_table(f"weights_{DATASET}.tex", to_latex(summary))


if __name__ == "__main__":
    main()
