"""Iteration effect: joint sweep of prior variance, initialization, and iteration count.

3D sweep over (tau^2, init, iteration), with sparsity/seed as the inner loop.

Default run (no arguments): smartvote_2023, test_fraction=0.15,
tau^2 in {0.05, 0.1, 0.25, 0.5, 1.0, 1e6}, initialization in
{pca (pca_initialization=True), random (pca_initialization=False)},
iteration checkpoints {0, 1, 2, 5, 10, 20}, sparsity levels {0.0, 0.3, 0.6,
0.9}, 5 seeds, sampling_resolution=100. Models start at n_iterations=0 and are
advanced with model.iterate to each checkpoint; the seed=0 model is saved at
every checkpoint. Train metrics, spread (threshold 0.1) and distortion are
recorded at each checkpoint; test metrics come from the sparsity=0.0 model
evaluated against the full test sparsity grid.

Usage:
    python -m src.scripts.run_iteration_effect
    python -m src.scripts.run_iteration_effect --dataset smartvote_2023
    python -m src.scripts.run_iteration_effect --dataset polis --n-seeds 3
    python -m src.scripts.run_iteration_effect --dataset voteview --resolution 50 --output-dir results/custom
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ixplore.metrics import compute_distortion, compute_spread
from src.data import DATASETS, load_dataset
from src.models.ixplore_wrapper import IXPLOREModel

# -- Defaults --
TAU_VALUES = [0.05, 0.1, 0.25, 0.5, 1.0, 1e6]  # prior variances tau^2
INITIALIZATIONS = [("pca", True), ("random", False)]
ITERATION_CHECKPOINTS = [0, 1, 2, 5, 10, 20]
SPARSITY_LEVELS = [0.0, 0.3, 0.6, 0.9]
N_SEEDS = 5
SAMPLING_RESOLUTION = 100
TEST_FRACTION = 0.15  # ignored when the dataset has a natural train/test split
THRESHOLD = 0.1  # for "near border" in compute_spread

def main(args=None):
    parser = argparse.ArgumentParser(description="Iteration effect (prior x init x iteration)")
    parser.add_argument("--dataset", type=str, default="smartvote_2023", choices=DATASETS)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--resolution", type=int, default=SAMPLING_RESOLUTION)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--output-dir", type=Path, default=None)
    opts = parser.parse_args(args)

    dataset = load_dataset(opts.dataset, test_fraction=opts.test_fraction)

    output_dir = opts.output_dir or Path(f"results/{opts.dataset}/iteration_effect")
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_results = []
    test_results = []

    tau_bar = tqdm(TAU_VALUES, desc="Tau", position=0)
    for tau in tau_bar:
        tau_bar.set_postfix(tau=tau)

        init_bar = tqdm(INITIALIZATIONS, desc="Init", position=1, leave=False)
        for init_name, pca_init in init_bar:
            init_bar.set_postfix(init=init_name)

            sparsity_bar = tqdm(SPARSITY_LEVELS, desc="Sparsity", position=2, leave=False)
            for u in sparsity_bar:
                sparsity_bar.set_postfix(sparsity=u)

                train_seeds = range(opts.n_seeds) if u > 0 else [0]
                seed_bar = tqdm(list(train_seeds), desc="Seeds", position=3, leave=False)
                for seed_train in seed_bar:
                    sparse_train = dataset.get_data_with_sparsity("train", u, seed_train)

                    model = IXPLOREModel(
                        prior_variance=tau,
                        n_iterations=0,
                        pca_initialization=pca_init,
                        random_state=seed_train,
                        sampling_resolution=opts.resolution,
                    )
                    model.fit(sparse_train)
                    inner = model._model

                    prev_iter = 0
                    iter_bar = tqdm(ITERATION_CHECKPOINTS, desc="Iteration", position=4, leave=False)
                    for i in iter_bar:
                        if i > prev_iter:
                            model.iterate(i - prev_iter)
                            prev_iter = i

                        train_metrics = model.evaluate(sparse_train, dataset.train_reactions)
                        train_spread = compute_spread(inner.embedding, inner.limits, threshold=THRESHOLD)
                        _, _, dist_mean, dist_std = compute_distortion(inner, random_state=seed_train)
                        train_results.append({
                            **train_metrics, **train_spread,
                            "distortion_mean": round(dist_mean, 4),
                            "distortion_std": round(dist_std, 4),
                            "tau": tau, "init": init_name, "iteration": i,
                            "sparsity": u, "seed": seed_train,
                        })

                        # Save model at every checkpoint at seed 0 so the iteration trajectory
                        # of the embedding can be inspected downstream.
                        if seed_train == 0:
                            model.save(models_dir / f"tau_{tau}_{init_name}_sp_{u}_iter_{i}",
                                       dataset=opts.dataset, sparsity=u, seed=0)

                        # Test loop only at u=0.0 (no randomness in train mask)
                        if u == 0.0:
                            test_sparsities = tqdm(SPARSITY_LEVELS, desc="Test", position=5, leave=False)
                            for v in test_sparsities:
                                test_seeds = range(opts.n_seeds) if v > 0 else [0]
                                for seed_test in test_seeds:
                                    sparse_test = dataset.get_data_with_sparsity("test", v, seed_test)
                                    test_emb = model.embed(sparse_test)
                                    test_metrics = model.evaluate(sparse_test, dataset.test_reactions, test=True)
                                    test_spread = compute_spread(test_emb.values, inner.limits, threshold=THRESHOLD)
                                    test_results.append({
                                        **test_metrics, **test_spread,
                                        "tau": tau, "init": init_name, "iteration": i,
                                        "sparsity": v, "seed": seed_test,
                                    })

    pd.DataFrame(train_results).to_csv(output_dir / "train_metrics.csv", index=False)
    pd.DataFrame(test_results).to_csv(output_dir / "test_metrics.csv", index=False)
    print(f"Iteration effect done ({opts.dataset}). Results in {output_dir}")


if __name__ == "__main__":
    main()
