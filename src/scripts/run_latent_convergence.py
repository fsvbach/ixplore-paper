"""Prior-effect: per-tau posterior trajectories for test users.

For each prior tau^2 in {0.05, 0.1, 0.25, 0.5, 1.0, 1e6}, load the corresponding
fully-fit model from `iteration_effect/models/tau_<tau^2>_pca_sp_0.0_iter_20`
(PCA init, sparsity 0, iter 20, seed 0) and compute, for every test user, the
running posterior mean and moment-matched covariance after revealing items
1..k in natural order, k = 0..K.

Default run (no arguments): smartvote_2023, test_fraction=0.15, source models
from results/smartvote_2023/iteration_effect/models, output to
results/smartvote_2023/latent_convergence. The sweep is run twice - once with
`scale_weights=False` (default) and once with `scale_weights=True` - so
downstream code can compare arrival times under both regimes; the scaled run
writes `tau_<tau^2>_scaled.csv`, the unscaled run `tau_<tau^2>.csv`.

Usage:
    python -m src.scripts.run_latent_convergence
    python -m src.scripts.run_latent_convergence --dataset polis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import DATASETS, load_dataset
from src.metrics import gaussian_moments, within_1sigma
from src.models.ixplore_wrapper import IXPLOREModel

TAU_VALUES = [0.05, 0.1, 0.25, 0.5, 1.0, 1e6]  # prior variances tau^2


def trajectory_for_model(model_dir: Path, test: pd.DataFrame,
                         scale_weights: bool) -> pd.DataFrame:
    wrapper = IXPLOREModel.load(model_dir)
    inner = wrapper._model
    inner.scale_weights = scale_weights
    K = test.shape[1]

    final_post = inner.compute_posteriors(test)                          # (N, G)
    final_means, final_covs = gaussian_moments(final_post, inner.X)      # (N, 2), (N, 2, 2)

    rows = []
    for k in range(K + 1):
        post_k = inner.compute_posteriors(test.iloc[:, :k])
        means_k, covs_k = gaussian_moments(post_k, inner.X)
        in_sigma = within_1sigma(means_k, final_means, final_covs)
        std_x = np.sqrt(covs_k[:, 0, 0])
        std_y = np.sqrt(covs_k[:, 1, 1])
        for i, uid in enumerate(test.index):
            rows.append({
                "user": uid, "k": k,
                "mean_x": means_k[i, 0], "mean_y": means_k[i, 1],
                "std_x": std_x[i], "std_y": std_y[i],
                "in_1sigma": bool(in_sigma[i]),
            })
    return pd.DataFrame(rows)


def main(args=None):
    parser = argparse.ArgumentParser(description="Prior-effect trajectories")
    parser.add_argument("--dataset", type=str, default="smartvote_2023", choices=DATASETS)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--source-dir", type=Path,
                        default=None,
                        help="Directory holding the saved models (default: results/<dataset>/iteration_effect/models)")
    parser.add_argument("--output-dir", type=Path, default=None)
    opts = parser.parse_args(args)

    dataset = load_dataset(opts.dataset, test_fraction=opts.test_fraction)
    test = dataset.test_reactions
    if test is None:
        raise SystemExit(f"Dataset {opts.dataset} has no test split.")

    source = opts.source_dir or Path(f"results/{opts.dataset}/iteration_effect/models")
    out_dir = opts.output_dir or Path(f"results/{opts.dataset}/latent_convergence")
    out_dir.mkdir(parents=True, exist_ok=True)

    for scale_weights in (False, True):
        suffix = "_scaled" if scale_weights else ""
        for tau in tqdm(TAU_VALUES, desc=f"tau (scale_weights={scale_weights})"):
            model_dir = source / f"tau_{tau}_pca_sp_0.0_iter_20"
            traj = trajectory_for_model(model_dir, test, scale_weights=scale_weights)
            out = out_dir / f"tau_{tau}{suffix}.csv"
            traj.to_csv(out, index=False)

    print(f"Prior-effect trajectories saved under {out_dir}")


if __name__ == "__main__":
    main()
