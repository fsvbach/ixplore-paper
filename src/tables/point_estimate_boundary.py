"""Table: boundary fraction and train reconstruction of MAP and posterior mean.

Writes tables/point_estimate_boundary_smartvote_2023.tex: per prior variance
tau^2 (model retrained at that tau^2), the share of embeddings within 5% of
the grid edge and the in-sample reconstruction MAE and accuracy, for the MAP
and the posterior-mean point estimate; the better of the two in bold. Reads
results/ixplore_design/point_estimates_boundary.csv (written by
src.scripts.run_point_estimates).

Usage:
    python -m src.tables.point_estimate_boundary
"""

import pandas as pd

from src.paths import RESULTS_DIR
from src.tables import write_table

RESULTS_CSV = RESULTS_DIR / "ixplore_design" / "point_estimates_boundary.csv"


def _sig(s):
    return r"$10^{6}$" if s >= 1e5 else f"{s:g}"


def _lo_pair(a, b, f):
    sa, sb = f(a), f(b)
    if a < b:
        sa = r"\textbf{" + sa + "}"
    elif b < a:
        sb = r"\textbf{" + sb + "}"
    return sa, sb


def _hi_pair(a, b, f):
    sa, sb = f(a), f(b)
    if a > b:
        sa = r"\textbf{" + sa + "}"
    elif b > a:
        sb = r"\textbf{" + sb + "}"
    return sa, sb


def to_latex(bdf):
    lines = [r"\begin{tabular}{rcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Boundary frac.} & \multicolumn{2}{c}{Recon. MAE} & "
             r"\multicolumn{2}{c}{Recon. Acc.} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"$\tau^2$ & MAP & Mean & MAP & Mean & MAP & Mean \\", r"\midrule"]
    for _, r in bdf.iterrows():
        bfa, bfb = _lo_pair(r.bf_map, r.bf_mean, lambda x: f"{x:.3f}")
        maa, mab = _lo_pair(r.mae_map, r.mae_mean, lambda x: f"{x:.4f}")
        aca, acb = _hi_pair(r.acc_map, r.acc_mean, lambda x: f"{x:.3f}")
        lines.append(f"{_sig(r.tau)} & {bfa} & {bfb} & {maa} & {mab} & {aca} & {acb} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main():
    bdf = pd.read_csv(RESULTS_CSV)
    write_table("point_estimate_boundary_smartvote_2023.tex", to_latex(bdf))


if __name__ == "__main__":
    main()
