"""PCA dimensionality sweep.

For each k in {1, ..., n_items - 1}, fits the PCA-Linear, PCA-Logistic, and
KernelPCA baselines (same defaults as run_baseline.py) at every
(sparsity, seed) combination and stores all metrics. Test metrics are produced
by the seed=0, sparsity=0.0 model evaluated against the test grid, mirroring
the baseline-experiments convention.

Default run (no arguments): smartvote_2023, test_fraction=0.15, all of
pca-linear, pca-logistic and kernel-pca, k from 1 to n_items-1,
5 seeds, sparsity levels {0.0, 0.3, 0.6, 0.9}. Output to
results/smartvote_2023/pca_sweep.

Usage:
    python -m src.scripts.run_pca_sweep
    python -m src.scripts.run_pca_sweep --dataset smartvote_2023
    python -m src.scripts.run_pca_sweep --dataset polis --n-seeds 3
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import DATASETS, load_dataset
from src.models import BASELINE_ALGORITHMS

ALGORITHMS = ["pca-linear", "pca-logistic", "kernel-pca"]
SPARSITY_LEVELS = [0.0, 0.3, 0.6, 0.9]
N_SEEDS = 5


def main(args=None):
    parser = argparse.ArgumentParser(description="PCA dimensionality sweep")
    parser.add_argument("--dataset", type=str, default="smartvote_2023", choices=DATASETS)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--algorithms", type=str, nargs="+", default=ALGORITHMS,
                        choices=ALGORITHMS,
                        help="Subset of sweep algorithms to run (default: all)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory (default: results/<dataset>/pca_sweep)")
    opts = parser.parse_args(args)

    TEST_FRACTION = 0.15
    dataset = load_dataset(opts.dataset, test_fraction=TEST_FRACTION)
    n_items = dataset.train_reactions.shape[1]
    K_VALUES = list(range(1, n_items))

    output_dir = opts.output_dir or Path(f"results/{opts.dataset}/pca_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_results = []
    test_results = []

    algo_bar = tqdm(opts.algorithms, desc="Algorithm", position=0)
    for algorithm in algo_bar:
        algo_bar.set_postfix(algo=algorithm)
        ModelClass = BASELINE_ALGORITHMS[algorithm]

        k_bar = tqdm(K_VALUES, desc="k", position=1, leave=False)
        for k in k_bar:
            k_bar.set_postfix(k=k)

            sparsity_bar = tqdm(SPARSITY_LEVELS, desc="Sparsity", position=2, leave=False)
            for u in sparsity_bar:
                train_seeds = range(opts.n_seeds) if u > 0 else [0]

                for seed_train in train_seeds:
                    sparse_train = dataset.get_data_with_sparsity("train", u, seed_train)

                    model = ModelClass(n_components=k)
                    model.fit(sparse_train)

                    train_metrics = model.evaluate(sparse_train, dataset.train_reactions)
                    train_results.append({
                        **train_metrics,
                        "model": algorithm,
                        "k": k,
                        "sparsity": u,
                        "seed": seed_train,
                    })

                # Test grid: only at u=0.0, using the seed=0 model
                if u == 0.0:
                    for v in SPARSITY_LEVELS:
                        test_seeds = range(opts.n_seeds) if v > 0 else [0]
                        for seed_test in test_seeds:
                            sparse_test = dataset.get_data_with_sparsity("test", v, seed_test)
                            test_metrics = model.evaluate(
                                sparse_test, dataset.test_reactions, test=True,
                            )
                            test_results.append({
                                **test_metrics,
                                "model": algorithm,
                                "k": k,
                                "sparsity": v,
                                "seed": seed_test,
                            })

    _merge_and_write(pd.DataFrame(train_results), output_dir / "train_metrics.csv", opts.algorithms)
    _merge_and_write(pd.DataFrame(test_results), output_dir / "test_metrics.csv", opts.algorithms)
    print(f"PCA sweep done on {opts.dataset}. Results in {output_dir}")


def _merge_and_write(new_df: pd.DataFrame, path: Path, ran_algorithms: list[str]) -> None:
    """Merge new rows with any existing CSV, replacing only the rows for the algorithms just run."""
    if path.exists():
        old_df = pd.read_csv(path)
        kept = old_df[~old_df["model"].isin(ran_algorithms)]
        out = pd.concat([kept, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(path, index=False)


if __name__ == "__main__":
    main()
