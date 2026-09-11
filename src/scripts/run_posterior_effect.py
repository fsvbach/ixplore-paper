"""Item update: point-estimate against uncertainty-weighted on Smartvote 2023.

The IXPLORE item step refits the per-item logistic regressions from the
current user representations. With `use_point_estimates=True` each user
contributes one training point at the posterior mean; with False the full
grid posterior is used as a soft assignment (EM-style M-step).

A single model is fitted per (sparsity u, seed) with n_iterations=1 and
sampling_resolution=100; the posteriors are computed once and only the item
step is rerun under each variant (`fit_posteriors()` then `fit_models()`), so
both conditions share the same user posteriors. Train cells are recorded per
(u, seed); test cells come from the u = 0 model evaluated over the test
sparsity grid. Rows are appended to
results/smartvote_2023/posterior_effect/{train,test}_metrics.csv as soon as
they are computed, and rows already present are skipped on rerun.

Usage:
    python -m src.scripts.run_posterior_effect
"""

import pandas as pd
from tqdm import tqdm

from src.data import load_dataset
from src.models.ixplore_wrapper import IXPLOREModel
from src.paths import RESULTS_DIR

DATASET = "smartvote_2023"
SPARSITY_LEVELS = [0.0, 0.3, 0.6, 0.9]
N_SEEDS = 5
N_ITERATIONS = 1
SAMPLING_RESOLUTION = 100
CONDITIONS = [("posterior_mean", True), ("posterior_weighted", False)]

OUTPUT_DIR = RESULTS_DIR / DATASET / "posterior_effect"
TRAIN_CSV = OUTPUT_DIR / "train_metrics.csv"
TEST_CSV = OUTPUT_DIR / "test_metrics.csv"


def done_keys(csv, cols):
    if not csv.exists():
        return set()
    d = pd.read_csv(csv)
    if d.empty:
        return set()
    return set(map(tuple, d[cols].itertuples(index=False, name=None)))


def append_row(csv, row):
    pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET)

    train_done = done_keys(TRAIN_CSV, ["model", "sparsity", "seed"])
    test_done = done_keys(TEST_CSV, ["model", "sparsity", "seed"])

    for u in tqdm(SPARSITY_LEVELS, desc="train sparsity"):
        train_seeds = range(N_SEEDS) if u > 0 else [0]
        for seed in train_seeds:
            need = [c for c, _ in CONDITIONS if (c, u, seed) not in train_done]
            need_test = (u == 0.0) and any(
                (c, v, s) not in test_done
                for c, _ in CONDITIONS
                for v in SPARSITY_LEVELS
                for s in (range(N_SEEDS) if v > 0 else [0])
            )
            if not need and not need_test:
                continue

            sparse_train = dataset.get_data_with_sparsity("train", u, seed)
            model = IXPLOREModel(n_iterations=N_ITERATIONS, sampling_resolution=SAMPLING_RESOLUTION)
            model.fit(sparse_train)
            model.fit_posteriors()  # user posteriors computed once, shared across conditions

            for condition, point_est in CONDITIONS:
                model.use_point_estimates = point_est
                model._m().use_point_estimates = point_est
                model.fit_models()  # only the item step changes between conditions

                if (condition, u, seed) not in train_done:
                    tm = model.evaluate(sparse_train, dataset.train_reactions)
                    append_row(TRAIN_CSV, {**tm, "model": condition, "sparsity": u, "seed": seed})

                if u == 0.0:
                    for v in SPARSITY_LEVELS:
                        test_seeds = range(N_SEEDS) if v > 0 else [0]
                        for seed_test in test_seeds:
                            if (condition, v, seed_test) in test_done:
                                continue
                            sparse_test = dataset.get_data_with_sparsity("test", v, seed_test)
                            em = model.evaluate(sparse_test, dataset.test_reactions, test=True)
                            append_row(TEST_CSV, {**em, "model": condition, "sparsity": v, "seed": seed_test})

    print("train rows:", len(pd.read_csv(TRAIN_CSV)))
    print("test rows :", len(pd.read_csv(TEST_CSV)))
    print(f"Posterior-effect done. Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
