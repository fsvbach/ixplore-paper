"""Table: held-out MAE of MAP, posterior mean, and full marginalization.

Writes tables/point_estimate_mae_smartvote_2023.tex: predictive MAE on the
held-out answers against the number of observed answers, for the three ways
of summarising a user's grid posterior, at tau^2 = 0.25; best per row in
bold. Reads results/ixplore_design/point_estimates_accuracy.csv (written by
src.scripts.run_point_estimates).

Usage:
    python -m src.tables.point_estimate_mae
"""

import pandas as pd

from src.paths import RESULTS_DIR
from src.tables import write_table

RESULTS_CSV = RESULTS_DIR / "ixplore_design" / "point_estimates_accuracy.csv"


def to_latex(acc_df):
    best = acc_df[["mae_map", "mae_mean", "mae_post"]].min(axis=1)
    lines = [r"\begin{tabular}{rccc}", r"\toprule",
             r"$n_{\text{obs}}$ & MAP & Posterior mean & Full posterior \\", r"\midrule"]
    for idx, r in acc_df.iterrows():
        def cell(v):
            s = f"{v:.4f}"
            return r"\textbf{" + s + "}" if abs(v - best.loc[idx]) < 1e-9 else s
        lines.append(f"{int(r.n_observed)} & {cell(r.mae_map)} & "
                     f"{cell(r.mae_mean)} & {cell(r.mae_post)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    acc_df = pd.read_csv(RESULTS_CSV)
    write_table("point_estimate_mae_smartvote_2023.tex", to_latex(acc_df))


if __name__ == "__main__":
    main()
