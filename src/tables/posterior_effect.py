"""Table: point-estimate against uncertainty-weighted item update on Smartvote 2023.

Writes tables/posterior_effect_smartvote_2023.tex: test imputation MAE of
both item-update variants per test sparsity v in {0.3, 0.6, 0.9}, averaged
over seeds, with the difference; the better variant per row in bold. Reads
results/smartvote_2023/posterior_effect/test_metrics.csv (written by
src.scripts.run_posterior_effect).

Usage:
    python -m src.tables.posterior_effect
"""

import pandas as pd

from src.results import load_metrics
from src.tables import write_table

DATASET = "smartvote_2023"
SPARSITIES = [0.3, 0.6, 0.9]  # imputation is undefined at v = 0


def summarise(test_df):
    def mean_mae(cond, v):
        s = test_df[(test_df["model"] == cond) & (test_df["sparsity"] == v)]["impute_mae"].dropna()
        return s.mean()

    rows = []
    for v in SPARSITIES:
        pe = mean_mae("posterior_mean", v)
        uw = mean_mae("posterior_weighted", v)
        rows.append((v, pe, uw, uw - pe))
    return pd.DataFrame(rows, columns=["sparsity", "pe", "uw", "delta"])


def to_latex(summary):
    # Report convention: plain means, best in column bold (no +/- std).
    lines = [
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"Sparsity & Point-estimate & Uncertainty-weighted & $\Delta$ \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        pe_s = f"{r.pe:.4f}"
        uw_s = f"{r.uw:.4f}"
        if r.pe <= r.uw:
            pe_s = r"\textbf{" + pe_s + "}"
        else:
            uw_s = r"\textbf{" + uw_s + "}"
        lines.append(f"{r.sparsity:g} & {pe_s} & {uw_s} & ${r.delta:+.4f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    summary = summarise(load_metrics(DATASET, "posterior_effect", "test"))
    print(summary.round(4).to_string(index=False))
    write_table(f"posterior_effect_{DATASET}.tex", to_latex(summary))


if __name__ == "__main__":
    main()
