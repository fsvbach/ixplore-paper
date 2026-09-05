"""Baseline experiments.

Default run (no arguments): the pca-logistic baseline on smartvote_2023,
test_fraction=0.15, 5 seeds, sparsity levels {0.0, 0.3, 0.6, 0.9}. Train
metrics are recorded at every (sparsity, seed); test metrics come from the
sparsity=0.0, seed=0 model evaluated against the full test sparsity grid.
The seed=0 model at each sparsity is saved under
results/smartvote_2023/baseline/pca-logistic/models/.

Usage:
    python -m src.scripts.run_baseline
    python -m src.scripts.run_baseline --dataset smartvote_2023 --algorithm pca-logistic
    python -m src.scripts.run_baseline --dataset polis --algorithm vae-2layer --n-seeds 3
    python -m src.scripts.run_baseline --algorithm ixplore --output-dir results/custom

Available algorithms: pca-linear, pca-logistic, ideal, vae-2layer, vae-logistic, ixplore
Available datasets: smartvote_2023, smartvote_2019, polis, voteview
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import DATASETS, load_dataset
from src.models import BASELINE_ALGORITHMS

# ── Defaults ──
SPARSITY_LEVELS = [0.0, 0.3, 0.6, 0.9]
N_SEEDS = 5
TEST_FRACTION = 0.15  # ignored when the dataset has a natural train/test split


def main(args=None):
    parser = argparse.ArgumentParser(description="Baseline experiments")
    parser.add_argument("--dataset", type=str, default="smartvote_2023", choices=DATASETS)
    parser.add_argument("--algorithm", type=str, default="pca-logistic", choices=list(BASELINE_ALGORITHMS))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory (default: results/<dataset>/baseline/<algorithm>)")
    opts = parser.parse_args(args)

    dataset = load_dataset(opts.dataset, test_fraction=opts.test_fraction)
    ModelClass = BASELINE_ALGORITHMS[opts.algorithm]

    output_dir = opts.output_dir or Path(f"results/{opts.dataset}/baseline/{opts.algorithm}")
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_results = []
    test_results = []

    sparsity_bar = tqdm(SPARSITY_LEVELS, desc="Sparsity", position=0)
    for u in sparsity_bar:
        sparsity_bar.set_postfix(sparsity=u)

        train_seeds = range(opts.n_seeds) if u > 0 else [0]

        seed_bar = tqdm(list(train_seeds), desc="Seeds", position=1, leave=False)
        for seed_train in seed_bar:
            sparse_train = dataset.get_data_with_sparsity("train", u, seed_train)

            model = ModelClass()
            model.fit(sparse_train)

            train_metrics = model.evaluate(sparse_train, dataset.train_reactions)
            train_results.append({
                **train_metrics,
                "model": opts.algorithm,
                "sparsity": u,
                "seed": seed_train,
            })

            if seed_train == 0:
                model.save(models_dir / f"{opts.algorithm}_sp_{u}",
                           dataset=opts.dataset, sparsity=u, seed=0,
                           test_fraction=opts.test_fraction)

        # Test only at u=0.0 (no randomness in train mask, so test across sparsities)
        if u == 0.0:
            test_sparsities = tqdm(SPARSITY_LEVELS, desc="Test", position=2, leave=False)
            for v in test_sparsities:
                test_seeds = range(opts.n_seeds) if v > 0 else [0]
                for seed_test in test_seeds:
                    sparse_test = dataset.get_data_with_sparsity("test", v, seed_test)
                    test_metrics = model.evaluate(
                        sparse_test, dataset.test_reactions, test=True,
                    )
                    test_results.append({
                        **test_metrics,
                        "model": opts.algorithm,
                        "sparsity": v,
                        "seed": seed_test,
                    })

    pd.DataFrame(train_results).to_csv(output_dir / "train_metrics.csv", index=False)
    pd.DataFrame(test_results).to_csv(output_dir / "test_metrics.csv", index=False)
    print(f"Baseline done ({opts.algorithm} on {opts.dataset}). Results in {output_dir}")


if __name__ == "__main__":
    main()
