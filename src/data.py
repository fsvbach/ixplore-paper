from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ixplore.utils import add_sparsity as _add_sparsity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASET_DIRS = {
    "smartvote_2023": DATA_DIR / "smartvote" / "2023",
    "smartvote_2019": DATA_DIR / "smartvote" / "2019",
    "polis": DATA_DIR / "polis",
    "voteview": DATA_DIR / "voteview",
    "evs": DATA_DIR / "evs" / "2020",
}

DATASET_LABELS = {
    "smartvote_2023": "Smartvote (2023)",
    "smartvote_2019": "Smartvote (2019)",
    "polis": "Polis (vTaiwan)",
    "voteview": "Voteview (S117)",
    "evs": "European Values Study",
}

DATASETS = list(DATASET_DIRS)


@dataclass
class Dataset:
    name: str
    label: str
    train_reactions: pd.DataFrame
    train_info: pd.DataFrame | None
    test_reactions: pd.DataFrame | None
    test_info: pd.DataFrame | None

    def get_data_with_sparsity(
        self, split: str, sparsity: float, seed: int,
    ) -> pd.DataFrame:
        """Return reactions for *split* with artificial sparsity applied.

        Parameters
        ----------
        split : ``"train"`` or ``"test"``
        sparsity : float
            Fraction of entries to mask (0.0 = keep all, 0.9 = drop 90 %).
        seed : int
            Seed for a fresh ``numpy.random.Generator``.
        """
        data = self.test_reactions if split == "test" else self.train_reactions
        if sparsity == 0.0:
            return data
        rng = np.random.default_rng(seed)
        return _add_sparsity(data, 1.0 - sparsity, rng)


def _read(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with string index."""
    df = pd.read_csv(path, index_col=0, **kwargs)
    df.index = df.index.astype(str)
    return df


def _split(reactions, info, test_fraction, seed=0):
    """Random train/test split on rows. Returns (train_r, train_i, test_r, test_i)."""
    if test_fraction == 0.0:
        return reactions, info, None, None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(reactions))
    n_train = int((1.0 - test_fraction) * len(reactions))
    train_idx = reactions.index[idx[:n_train]]
    test_idx = reactions.index[idx[n_train:]]
    return (
        reactions.loc[train_idx],
        info.loc[train_idx] if info is not None else None,
        reactions.loc[test_idx],
        info.loc[test_idx] if info is not None else None,
    )


def load_weights(name: str, split: str) -> pd.DataFrame | None:
    """Load per-(user, item) weights for a dataset split, or None if absent."""
    base = DATASET_DIRS[name]
    if name == "smartvote_2023":
        filename = "voters_weights.csv" if split == "test" else "candidates_weights.csv"
    elif name == "smartvote_2019":
        filename = "candidates_weights.csv"
    elif name == "polis":
        filename = "vTaiwan_weights.csv"
    elif name == "voteview":
        filename = "senators_weights.csv"
    else:
        raise ValueError(f"Unknown dataset: {name}")
    path = base / filename
    if not path.exists():
        return None
    return _read(path)


def load_dataset(name: str, test_fraction: float = 0.0) -> Dataset:
    base = DATASET_DIRS[name]
    label = DATASET_LABELS[name]

    if name == "smartvote_2023":
        reactions = _read(base / "candidates_reactions.csv")
        info = _read(base / "candidates_information.csv")
        test_reactions = _read(base / "voters_reactions.csv")
        test_info = _read(base / "voters_information.csv")
        return Dataset(
            name=name,
            label=label,
            train_reactions=reactions,
            train_info=info,
            test_reactions=test_reactions,
            test_info=test_info,
        )
    elif name == "smartvote_2019":
        all_reactions = _read(base / "candidates_reactions.csv")
        all_info = _read(base / "candidates_information.csv")
        train_r, train_i, test_r, test_i = _split(all_reactions, all_info, test_fraction)
        return Dataset(
            name=name,
            label=label,
            train_reactions=train_r,
            train_info=train_i,
            test_reactions=test_r,
            test_info=test_i,
        )
    elif name == "polis":
        reactions = _read(base / "vTaiwan_reactions.csv")
        info = _read(base / "vTaiwan_information.csv")
        train_r, train_i, test_r, test_i = _split(reactions, info, test_fraction)
        return Dataset(
            name=name,
            label=label,
            train_reactions=train_r,
            train_info=train_i,
            test_reactions=test_r,
            test_info=test_i,
        )
    elif name == "voteview":
        reactions = _read(base / "senators_reactions.csv")
        info = _read(base / "senators_information.csv")
        train_r, train_i, test_r, test_i = _split(reactions, info, test_fraction)
        return Dataset(
            name=name,
            label=label,
            train_reactions=train_r,
            train_info=train_i,
            test_reactions=test_r,
            test_info=test_i,
        )
    elif name == "evs":
        reactions = _read(base / "respondents_reactions.csv")
        info = _read(base / "respondents_information.csv")
        train_r, train_i, test_r, test_i = _split(reactions, info, test_fraction)
        return Dataset(
            name=name,
            label=label,
            train_reactions=train_r,
            train_info=train_i,
            test_reactions=test_r,
            test_info=test_i,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from {DATASETS}")
