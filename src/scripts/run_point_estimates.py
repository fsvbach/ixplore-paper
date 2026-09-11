"""Point estimate from the grid posterior on Smartvote 2023.

Compares three ways of summarising a user's grid posterior for prediction:
MAP, posterior mean, and full marginalization.

Part 1 (results/ixplore_design/point_estimates_accuracy.csv): at the default
prior tau^2 = 0.25, a random subset of every candidate's answers is held out,
the posterior is computed from the rest, and the held-out answers are
predicted under the three estimates; MAE is averaged over all candidates per
number of observed answers.

Part 2 (results/ixplore_design/point_estimates_boundary.csv): for every prior
variance tau^2 a fresh model is trained; the share of embeddings within 5% of
the grid edge and the in-sample reconstruction MAE and accuracy are recorded
for MAP and posterior mean.

Usage:
    python -m src.scripts.run_point_estimates
"""

import numpy as np
import pandas as pd
from ixplore import IXPLORE, metrics
from ixplore.optimization import posterior_maps, posterior_means

from src.data import load_dataset
from src.paths import RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "ixplore_design"
N_OBSERVED_LIST = [5, 10, 15, 30, 45, 60, 70]
TAU_VALUES = [0.05, 0.1, 0.25, 0.5, 1.0, 1e6]  # prior variances tau^2
BOUNDARY_THRESHOLD = 0.05


def heldout_accuracy(reactions):
    """Part 1: held-out predictive MAE against the number of observed answers."""
    n_users, n_items = reactions.shape
    xplore = IXPLORE(reactions, prior_variance=0.25)
    xplore.iterate(n_iterations=10)
    predictions_X = np.exp(xplore._log_p1)  # (G, n_items): P(Y=1 | grid point)

    rng = np.random.default_rng(42)
    results = []
    for n_obs in N_OBSERVED_LIST:
        maes_map, maes_mean, maes_post = [], [], []
        for user_idx in range(n_users):
            ur = reactions.iloc[user_idx]
            obs = rng.choice(n_items, size=n_obs, replace=False)
            hld = np.setdiff1d(np.arange(n_items), obs)
            y = ur.iloc[hld].values

            post = xplore.compute_posteriors(ur.iloc[obs])
            post_2d = post.reshape(1, -1)

            pm = xplore.predict(posterior_maps(post_2d, xplore.X))[0]
            maes_map.append(np.abs(y - pm[hld]).mean())
            mn = xplore.predict(posterior_means(post_2d, xplore.X))[0]
            maes_mean.append(np.abs(y - mn[hld]).mean())
            pp = (predictions_X * post.reshape(-1, 1)).sum(axis=0)
            maes_post.append(np.abs(y - pp[hld]).mean())

        results.append({"n_observed": n_obs,
                        "mae_map": np.mean(maes_map),
                        "mae_mean": np.mean(maes_mean),
                        "mae_post": np.mean(maes_post)})
        print(f"n_obs={n_obs:2d}  MAP {np.mean(maes_map):.4f}  "
              f"Mean {np.mean(maes_mean):.4f}  Full {np.mean(maes_post):.4f}")

    acc_df = pd.DataFrame(results)
    out = OUTPUT_DIR / "point_estimates_accuracy.csv"
    acc_df.to_csv(out, index=False)
    print(f"Saved {out}")


def boundary_and_reconstruction(reactions):
    """Part 2: boundary fraction and train reconstruction per prior variance."""
    Y_true = reactions.values
    obs_mask = ~np.isnan(Y_true)

    rows = []
    for tau in TAU_VALUES:
        m = IXPLORE(reactions, prior_variance=tau)
        m.iterate(n_iterations=10)

        lo, hi = m.limits
        margin = (hi - lo) * BOUNDARY_THRESHOLD

        def bf(emb):
            near = ((emb[:, 0] - lo < margin) | (hi - emb[:, 0] < margin) |
                    (emb[:, 1] - lo < margin) | (hi - emb[:, 1] < margin))
            return near.mean()

        post = m._posteriors()
        emb_map = posterior_maps(post, m.X)
        emb_mean = posterior_means(post, m.X)
        pred_map = m.predict(emb_map)
        pred_mean = m.predict(emb_mean)

        t = Y_true[obs_mask]
        rows.append({
            "tau": tau,
            "bf_map": bf(emb_map), "bf_mean": bf(emb_mean),
            "mae_map": metrics.compute_mae(t, pred_map[obs_mask]),
            "mae_mean": metrics.compute_mae(t, pred_mean[obs_mask]),
            "acc_map": metrics.compute_accuracy(t, pred_map[obs_mask], neutral_window=0.05),
            "acc_mean": metrics.compute_accuracy(t, pred_mean[obs_mask], neutral_window=0.05),
        })
        print(f"tau^2={tau:>8}  BF MAP {rows[-1]['bf_map']:.3f} Mean {rows[-1]['bf_mean']:.3f}"
              f" | MAE MAP {rows[-1]['mae_map']:.4f} Mean {rows[-1]['mae_mean']:.4f}"
              f" | ACC MAP {rows[-1]['acc_map']:.3f} Mean {rows[-1]['acc_mean']:.3f}")

    bdf = pd.DataFrame(rows)
    out = OUTPUT_DIR / "point_estimates_boundary.csv"
    bdf.to_csv(out, index=False)
    print(f"Saved {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reactions = load_dataset("smartvote_2023").train_reactions
    print("data:", reactions.shape[0], "users x", reactions.shape[1], "items")
    heldout_accuracy(reactions)
    boundary_and_reconstruction(reactions)


if __name__ == "__main__":
    main()
