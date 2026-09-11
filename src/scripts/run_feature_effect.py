"""Feature effect: Effect of feature transform (kernel) on IXPLORE.

Default run (no arguments): smartvote_2023, test_fraction=0.15, kernels
{linear, polynomial, rff}, IXPLORE with n_iterations=10, prior_variance=0.25,
sampling_resolution=100 (PCA init, the wrapper default), 5 seeds, sparsity
levels {0.0, 0.3, 0.6, 0.9}. Train metrics, spread (threshold 0.1) and
distortion are recorded per (kernel, sparsity, seed); the seed=0 model at each
sparsity is saved. Test metrics come from the sparsity=0.0 model evaluated
against the full test sparsity grid.

Usage:
    python -m src.scripts.run_feature_effect
    python -m src.scripts.run_feature_effect --dataset smartvote_2023
    python -m src.scripts.run_feature_effect --dataset polis --n-seeds 3
    python -m src.scripts.run_feature_effect --dataset voteview --resolution 50 --output-dir results/custom
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ixplore.metrics import compute_distortion, compute_spread
from src.data import DATASETS, load_dataset
from src.models.ixplore_wrapper import IXPLOREModel

# -- Defaults --
KERNELS = ["linear", "polynomial", "rff"]
SPARSITY_LEVELS = [0.0, 0.3, 0.6, 0.9]
N_SEEDS = 5
N_ITERATIONS = 10
PRIOR_VARIANCE = 0.25
SAMPLING_RESOLUTION = 100
TEST_FRACTION = 0.15  # ignored when the dataset has a natural train/test split
THRESHOLD = 0.1

def main(args=None):
    parser = argparse.ArgumentParser(description="Feature effect")
    parser.add_argument("--dataset", type=str, default="smartvote_2023", choices=DATASETS)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--resolution", type=int, default=SAMPLING_RESOLUTION)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--output-dir", type=Path, default=None)
    opts = parser.parse_args(args)

    dataset = load_dataset(opts.dataset, test_fraction=opts.test_fraction)

    output_dir = opts.output_dir or Path(f"results/{opts.dataset}/feature_effect")
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_results = []
    test_results = []

    # Use kernel_name to look up from the registry in IXPLOREModel
    kernel_bar = tqdm(KERNELS, desc="Kernel", position=0)
    for kernel_name in kernel_bar:
        kernel_bar.set_postfix(kernel=kernel_name)
        sparsity_bar = tqdm(SPARSITY_LEVELS, desc="Sparsity", position=1, leave=False)
        for u in sparsity_bar:
            sparsity_bar.set_postfix(sparsity=u)

            train_seeds = range(opts.n_seeds) if u > 0 else [0]

            seed_bar = tqdm(list(train_seeds), desc="Seeds", position=2, leave=False)
            for seed_train in seed_bar:
                sparse_train = dataset.get_data_with_sparsity("train", u, seed_train)

                model = IXPLOREModel(
                    kernel_name=kernel_name,
                    n_iterations=N_ITERATIONS,
                    sampling_resolution=opts.resolution,
                    prior_variance=PRIOR_VARIANCE,
                )
                model.fit(sparse_train)

                inner = model._model
                train_metrics = model.evaluate(sparse_train, dataset.train_reactions)
                train_spread = compute_spread(inner.embedding, inner.limits, threshold=THRESHOLD)
                _, _, dist_mean, dist_std = compute_distortion(inner, random_state=seed_train)
                train_results.append({
                    **train_metrics, **train_spread,
                    "distortion_mean": round(dist_mean, 4),
                    "distortion_std": round(dist_std, 4),
                    "model": kernel_name,
                    "sparsity": u, "seed": seed_train,
                })

                if seed_train == 0:
                    model.save(models_dir / f"{kernel_name}_sp_{u}",
                               dataset=opts.dataset, sparsity=u, seed=0)

            # Test only at u=0.0
            if u == 0.0:
                test_sparsities = tqdm(SPARSITY_LEVELS, desc="Test", position=3, leave=False)
                for v in test_sparsities:
                    test_seeds = range(opts.n_seeds) if v > 0 else [0]
                    for seed_test in test_seeds:
                        sparse_test = dataset.get_data_with_sparsity("test", v, seed_test)
                        test_emb = model.embed(sparse_test)
                        test_metrics = model.evaluate(sparse_test, dataset.test_reactions, test=True)
                        test_spread = compute_spread(test_emb.values, inner.limits, threshold=THRESHOLD)
                        test_results.append({
                            **test_metrics, **test_spread,
                            "model": kernel_name,
                            "sparsity": v, "seed": seed_test,
                        })


    pd.DataFrame(train_results).to_csv(output_dir / "train_metrics.csv", index=False)
    pd.DataFrame(test_results).to_csv(output_dir / "test_metrics.csv", index=False)
    print(f"Feature effect done ({opts.dataset}). Results in {output_dir}")


if __name__ == "__main__":
    main()
