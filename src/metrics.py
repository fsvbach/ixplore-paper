"""Analysis-side metrics.

Adds experiment-protocol metrics that depend on a ground-truth full matrix
(`get_masks`, `compute_metrics`, `compute_recovery`) — concepts that don't
exist in production inference. Package primitives live in `ixplore.metrics`.
"""

import numpy as np
import pandas as pd
from ixplore import IXPLORE, metrics
from ixplore.utils import add_sparsity


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


# ---------------------------------------------------------------------------
# Per-user recovery: is the sparse re-embedding within 1σ of the full posterior?
# ---------------------------------------------------------------------------


def gaussian_moments(posteriors: np.ndarray, X_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match Gaussians to one or a batch of grid posteriors.

    posteriors : (N, G) or (G,)
    X_grid     : (G, 2)
    Returns means and covariances with matching leading shape:
      (N, 2), (N, 2, 2)  for batched input
      (2,),  (2, 2)      for a single posterior
    """
    single = posteriors.ndim == 1
    p = posteriors[None, :] if single else posteriors
    means = p @ X_grid                                                    # (N, 2)
    diff = X_grid[None, :, :] - means[:, None, :]                          # (N, G, 2)
    weighted = p[:, :, None] * diff                                        # (N, G, 2)
    covs = np.einsum("ngi,ngj->nij", weighted, diff)                       # (N, 2, 2)
    if single:
        return means[0], covs[0]
    return means, covs


def within_1sigma(points: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """Mahalanobis ≤ 1 check against moment-matched Gaussian(s).

    Accepts a single (point, mean, cov) triple or aligned batches.
    Returns a scalar bool for single input, or a (N,) bool array for batches.
    """
    single = points.ndim == 1
    pts = points[None, :] if single else points
    mns = means[None, :] if single else means
    cvs = covs[None, :, :] if single else covs
    diff = pts - mns
    cov_invs = np.linalg.pinv(cvs)
    mahal_sq = np.einsum("ni,nij,nj->n", diff, cov_invs, diff)
    out = mahal_sq <= 1.0
    return bool(out[0]) if single else out


def compute_recovery(
    model: IXPLORE,
    test_users: pd.DataFrame,
    sparsity: float | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, float]:
    """For each test user, check if the sparse-data position is within 1σ of the full posterior.

    "Within 1σ" is Mahalanobis distance ≤ 1 under the moment-matched Gaussian
    fit to the user's full-data posterior.

    Parameters
    ----------
    model : IXPLORE
        Fitted model providing item parameters and the latent grid.
    test_users : pd.DataFrame
        Users × items, NaN for already-missing answers.
    sparsity : float | None
        Fraction of items KEPT per user when re-embedding. None ⇒ full.
    random_state : int
        Seed for the sparsity mask.

    Returns
    -------
    mahalanobis : (N,) array of Mahalanobis distances of sparse means under
        each user's full-data posterior Gaussian.
    within_sigma_fraction : fraction of users with distance ≤ 1.
    """
    X_grid = model.X

    if sparsity is not None:
        rng = np.random.default_rng(random_state)
        sparse_users = add_sparsity(test_users, keep_fraction=sparsity, generator=rng)
    else:
        sparse_users = test_users

    full_posts = model.compute_posteriors(test_users)                     # (N, G)
    sparse_posts = model.compute_posteriors(sparse_users)                 # (N, G)

    full_means, full_covs = gaussian_moments(full_posts, X_grid)          # (N, 2), (N, 2, 2)
    sparse_means = sparse_posts @ X_grid                                  # (N, 2)

    cov_invs = np.linalg.pinv(full_covs)                                  # (N, 2, 2)
    diff = sparse_means - full_means                                      # (N, 2)
    mahalanobis = np.sqrt(np.einsum("ni,nij,nj->n", diff, cov_invs, diff))

    within = float((mahalanobis <= 1.0).mean())
    return mahalanobis, within
