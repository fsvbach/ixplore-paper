"""Analysis-side metrics.

Adds the experiment-protocol metrics that depend on a ground-truth full
matrix (`get_masks`, `compute_metrics`) - concepts that don't exist in
production inference. Package primitives live in `ixplore.metrics`.
"""

import numpy as np
from ixplore import metrics


# ---------------------------------------------------------------------------
# Sparsity-driven fit/impute split (requires a full ground-truth matrix)
# ---------------------------------------------------------------------------


def get_masks(
    sparse_reactions: np.ndarray,
    full_reactions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for fit (observed) and impute (artificially masked) cells.

    Parameters
    ----------
    sparse_reactions : array with NaN for missing entries (original + artificial sparsity)
    full_reactions : array with NaN only for originally missing entries

    Returns
    -------
    fit_mask : cells observed in sparse_reactions and valid in full_reactions
    impute_mask : cells masked by sparsity but valid in full_reactions
    """
    valid = ~np.isnan(full_reactions)
    observed = ~np.isnan(sparse_reactions)
    fit_mask = observed & valid
    impute_mask = (~observed) & valid
    return fit_mask, impute_mask


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    sparse_reactions: np.ndarray,
) -> dict[str, float]:
    """Compute fit and impute metrics.

    Fit mask: cells observed in sparse_reactions and valid in ground_truth.
    Impute mask: cells masked by sparsity but valid in ground_truth.

    Returns dict with keys: fit_mae, fit_accuracy, fit_rmse,
    impute_mae, impute_accuracy, impute_rmse (impute_* are NaN if no masked cells).
    """
    fit_mask, impute_mask = get_masks(sparse_reactions, ground_truth)

    fit_true = ground_truth[fit_mask]
    fit_pred = predictions[fit_mask]
    results = {
        "fit_mae": round(metrics.compute_mae(fit_true, fit_pred), 4),
        "fit_accuracy": round(metrics.compute_accuracy(fit_true, fit_pred, neutral_window=0.05), 4),
        "fit_rmse": round(metrics.compute_rmse(fit_true, fit_pred), 4),
    }

    if impute_mask.any():
        imp_true = ground_truth[impute_mask]
        imp_pred = predictions[impute_mask]
        results["impute_mae"] = round(metrics.compute_mae(imp_true, imp_pred), 4)
        results["impute_accuracy"] = round(metrics.compute_accuracy(imp_true, imp_pred, neutral_window=0.05), 4)
        results["impute_rmse"] = round(metrics.compute_rmse(imp_true, imp_pred), 4)
    else:
        results["impute_mae"] = float("nan")
        results["impute_accuracy"] = float("nan")
        results["impute_rmse"] = float("nan")

    return results
