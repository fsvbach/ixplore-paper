"""Sampling pseudo-answers: Rasch against posterior sampling on Smartvote 2023.

For a candidate who has answered only a subset of the 75 questions, two ways
of drawing pseudo-answers are compared: Rasch sampling (category response
functions around the imputed continuous scores, drawn with the Gumbel-max
trick) and posterior sampling (latent positions drawn from the grid
posterior, answers predicted from each position), plus a uniform random
baseline. The model (tau^2 = 0.25, PCA initialisation, 10 iterations) is
fitted once on all candidates; for every candidate and every number of
observed answers a random subset is observed and the MAE of the sample mean
and the mean sample standard deviation are computed on the unobserved items,
averaged over candidates. The posterior-mean point estimate is recorded for
reference. Writes results/ixplore_design/sampling_strategies.csv.

Usage:
    python -m src.scripts.run_sampling_strategies
"""

import logging

import numpy as np
import pandas as pd
from ixplore import IXPLORE
from tqdm import tqdm

from src.data import load_dataset
from src.paths import RESULTS_DIR

OUTPUT_CSV = RESULTS_DIR / "ixplore_design" / "sampling_strategies.csv"
N_OBSERVED_LIST = [5, 10, 15, 30, 45, 60, 70]
NUM_SAMPLES = 200
NUM_OPTIONS = 5   # Rasch CRF: 5-point Smartvote scale
VARIANCE = 0.1    # Rasch CRF bandwidth
METHODS = [("rasch", "Rasch"), ("posterior", "Posterior"), ("random", "Random")]


def main():
    logging.getLogger("ixplore").setLevel(logging.WARNING)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    reactions = load_dataset("smartvote_2023").train_reactions
    print(f"Dataset: {reactions.shape[0]} candidates, {reactions.shape[1]} questions")

    xplore = IXPLORE(reactions, pca_initialization=True, random_state=0,
                     prior_variance=0.25, scale_weights=False)
    xplore.iterate(10)

    Y = reactions.values  # (N, K) scaled answers; candidates fully observed
    n_users, n_items = Y.shape
    rng = np.random.default_rng(42)

    agg = {(label, n): {"mae": [], "std": []} for _, label in METHODS for n in N_OBSERVED_LIST}
    pmean = {n: [] for n in N_OBSERVED_LIST}  # posterior-mean point-estimate MAE

    for u in tqdm(range(n_users), desc="candidates"):
        full = pd.Series(Y[u], index=xplore.items)
        for n in N_OBSERVED_LIST:
            obs_idx = rng.choice(n_items, size=n, replace=False)
            unobs = np.ones(n_items, dtype=bool)
            unobs[obs_idx] = False
            pa = full.iloc[obs_idx]
            y_true = Y[u][unobs]

            # Posterior-mean point estimate (the reference of the point-estimate table).
            pos = xplore.embed(pa)
            pred = xplore.predict(np.asarray(pos).reshape(1, -1))[0]
            pmean[n].append(np.mean(np.abs(pred[unobs] - y_true)))

            for method, label in METHODS:
                kw = dict(method=method, num_samples=NUM_SAMPLES)
                if method == "rasch":
                    kw.update(num_options=NUM_OPTIONS, variance=VARIANCE)
                s = xplore.sample_answers(pa, **kw)
                agg[(label, n)]["mae"].append(np.mean(np.abs(s[:, unobs].mean(axis=0) - y_true)))
                agg[(label, n)]["std"].append(s[:, unobs].std(axis=0).mean())

    rows = []
    for _, label in METHODS:
        for n in N_OBSERVED_LIST:
            rows.append({"n_obs": n, "method": label,
                         "mae": np.mean(agg[(label, n)]["mae"]),
                         "mean_std": np.mean(agg[(label, n)]["std"])})
    for n in N_OBSERVED_LIST:
        rows.append({"n_obs": n, "method": "Posterior mean", "mae": np.mean(pmean[n]), "mean_std": np.nan})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(res_df.pivot(index="n_obs", columns="method", values="mae").to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
