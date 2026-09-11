"""Loaders for the stored experiment metrics under results/."""

from pathlib import Path

import pandas as pd

from src.paths import RESULTS_DIR


def experiment_dir(dataset: str, experiment: str) -> Path:
    """Folder of one experiment, e.g. results/smartvote_2023/iteration_effect."""
    return RESULTS_DIR / dataset / experiment


def load_metrics(dataset: str, experiment: str, split: str) -> pd.DataFrame:
    """Read results/<dataset>/<experiment>/<split>_metrics.csv."""
    return pd.read_csv(experiment_dir(dataset, experiment) / f"{split}_metrics.csv")


def load_baseline_metrics(dataset: str, split: str, algorithms=None) -> pd.DataFrame:
    """Concatenate results/<dataset>/baseline/<algorithm>/<split>_metrics.csv.

    Algorithms without a stored file are skipped. Defaults to every algorithm
    in ``src.models.BASELINE_ALGORITHMS``.
    """
    if algorithms is None:
        from src.models import BASELINE_ALGORITHMS
        algorithms = list(BASELINE_ALGORITHMS)
    frames = []
    for alg in algorithms:
        path = experiment_dir(dataset, "baseline") / alg / f"{split}_metrics.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
