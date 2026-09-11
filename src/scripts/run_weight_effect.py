"""Weight effect: test-time per-(user, item) weight schedules x scale_weights flag.

Trains a single IXPLORE model on Smartvote 2023 candidates with uniform
weights, then evaluates the same model on voters under three weight schedules
and both `scale_weights` settings.

Default run (no arguments): dataset smartvote_2023; model trained once with
prior_variance=0.25, n_iterations=10, pca_initialization=True,
sampling_resolution=100. Evaluated over schedules {uniform, provided,
extreme} x scale_weights {False, True} x sparsity {0.0, 0.3, 0.6, 0.9},
5 test seeds (1 seed at sparsity 0.0). Output to
results/smartvote_2023/weight_effect.

Schedules
---------
- uniform   : all 1s
- provided  : per-(user, item) weights from data/smartvote/2023/voters_weights.csv
- extreme   : provided ** 2

Distortion of the trained latent space is measured per (scale_weights, sparsity,
seed) using ixplore.metrics.compute_distortion (which uses the model's current
scale_weights flag but does not take a per-cell weight schedule); it is
computed only for the first schedule since it is schedule-independent.

Usage:
    python -m src.scripts.run_weight_effect
    python -m src.scripts.run_weight_effect --resolution 50 --output-dir results/custom
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ixplore.metrics import compute_distortion, compute_spread
from src.data import load_dataset, load_weights
from src.metrics import compute_metrics
from src.models.ixplore_wrapper import IXPLOREModel

# -- Defaults (PCA init; prior_variance and n_iterations chosen in Experiment 1) --
DATASET = "smartvote_2023"
SCHEDULES = ["uniform", "provided", "extreme"]
SCALE_WEIGHTS = [False, True]
SPARSITY_LEVELS = [0.0, 0.3, 0.6, 0.9]
N_SEEDS = 5
SAMPLING_RESOLUTION = 100
N_ITERATIONS = 10
PRIOR_VARIANCE = 0.25
THRESHOLD = 0.1  # for "near border" in compute_spread

def build_schedule(name: str, weights: pd.DataFrame, reactions: pd.DataFrame) -> pd.DataFrame | None:
    """Return a weight DataFrame aligned to `reactions`, or None for the uniform schedule."""
    if name == "uniform":
        return None
    aligned = weights.reindex(index=reactions.index, columns=reactions.columns)
    if name == "provided":
        return aligned
    if name == "extreme":
        return aligned ** 2
    raise ValueError(f"Unknown schedule: {name}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Weight effect (test-time)")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--resolution", type=int, default=SAMPLING_RESOLUTION)
    parser.add_argument("--output-dir", type=Path, default=None)
    opts = parser.parse_args(args)

    dataset = load_dataset(DATASET)
    weights_full = load_weights(DATASET, "test")
    if weights_full is None:
        raise FileNotFoundError(
            f"Expected per-(user, item) weights at data/smartvote/2023/voters_weights.csv"
        )

    output_dir = opts.output_dir or Path(f"results/{DATASET}/weight_effect")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Train once with uniform weights, PCA init, default tau^2, 50 iter --
    model = IXPLOREModel(
        prior_variance=PRIOR_VARIANCE,
        n_iterations=N_ITERATIONS,
        pca_initialization=True,
        sampling_resolution=opts.resolution,
    )
    model.fit(dataset.train_reactions)
    model.save(output_dir, dataset=DATASET, sparsity=0.0, seed=0)
    inner = model._model

    train_spread = compute_spread(inner.embedding, inner.limits, threshold=THRESHOLD)
    print(f"Trained on {DATASET}. Train spread: {train_spread}")

    test_results = []
    distortion_results = []

    schedule_bar = tqdm(SCHEDULES, desc="Schedule", position=0)
    for schedule in schedule_bar:
        schedule_bar.set_postfix(schedule=schedule)

        for sw in SCALE_WEIGHTS:
            inner.scale_weights = sw

            sparsity_bar = tqdm(SPARSITY_LEVELS, desc=f"Sparsity (sw={sw})", position=1, leave=False)
            for v in sparsity_bar:
                sparsity_bar.set_postfix(sparsity=v)
                test_seeds = range(opts.n_seeds) if v > 0 else [0]

                for seed_test in test_seeds:
                    sparse_test = dataset.get_data_with_sparsity("test", v, seed_test)
                    weights = build_schedule(schedule, weights_full, sparse_test)

                    test_emb = model.embed(sparse_test, weights=weights)
                    preds = model.predict(test_emb.values)
                    test_metrics = compute_metrics(
                        preds, dataset.test_reactions.values, sparse_test.values
                    )
                    test_spread = compute_spread(test_emb.values, inner.limits, threshold=THRESHOLD)
                    test_results.append({
                        **test_metrics, **test_spread,
                        "schedule": schedule, "scale_weights": sw,
                        "sparsity": v, "seed": seed_test,
                    })

                    # Distortion: depends on scale_weights and sparsity, not schedule.
                    # Skip redundant computations across schedules.
                    if schedule == SCHEDULES[0]:
                        _, _, dist_mean, dist_std = compute_distortion(
                            inner, keep_fraction=(1.0 - v) if v > 0 else None,
                            random_state=seed_test,
                        )
                        distortion_results.append({
                            "scale_weights": sw, "sparsity": v, "seed": seed_test,
                            "distortion_mean": round(dist_mean, 4),
                            "distortion_std": round(dist_std, 4),
                        })

    pd.DataFrame(test_results).to_csv(output_dir / "test_metrics.csv", index=False)
    pd.DataFrame(distortion_results).to_csv(output_dir / "distortion.csv", index=False)
    print(f"Weight effect done. Results in {output_dir}")


if __name__ == "__main__":
    main()
