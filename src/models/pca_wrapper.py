from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from ixplore.utils import compute_column_means, mean_impute, iterative_pca_impute
from src.models.base import SpatialModel, embedding_columns as _embedding_columns


class PCALinear(SpatialModel):
    """Plain PCA baseline with linear reconstruction.

    Missing values are filled with column means before a single PCA fit.
    Predictions are produced via scores @ components + mean, clipped to [0, 1].
    """

    name = "PCA-Linear"

    def __init__(self, n_components: int = 2, use_iterative: bool = False):
        self.n_components = n_components
        self.use_iterative = use_iterative
        self._pca: PCA | None = None
        self._embedding: pd.DataFrame | None = None
        self._train_col_means: np.ndarray | None = None

    def fit(self, reactions: pd.DataFrame) -> None:
        X = reactions.values.astype(float)
        self._train_col_means = compute_column_means(X, fill_empty_columns=0.5)
        if self.use_iterative:
            X_filled = iterative_pca_impute(X, n_components=self.n_components, fill_empty_columns=0.5)
        else:
            X_filled = mean_impute(X, column_means=self._train_col_means)

        pca = PCA(n_components=self.n_components, random_state=0)
        scores = pca.fit_transform(X_filled)

        self._pca = pca
        self._embedding = pd.DataFrame(
            scores, index=reactions.index, columns=_embedding_columns(self.n_components)
        )

    def get_embedding(self) -> pd.DataFrame:
        return self._embedding

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        assert self._pca is not None
        recon = embedding @ self._pca.components_ + self._pca.mean_
        return np.clip(recon, 0.0, 1.0)

    def embed(self, reactions: pd.DataFrame) -> pd.DataFrame:
        assert self._pca is not None and self._train_col_means is not None
        X_filled = mean_impute(reactions.values.astype(float), column_means=self._train_col_means)
        scores = self._pca.transform(X_filled)
        return pd.DataFrame(scores, index=reactions.index, columns=_embedding_columns(self.n_components))

    def save(self, path: Path, **kwargs) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "pca": self._pca,
                "embedding": self._embedding,
                "train_col_means": self._train_col_means,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "PCALinear":
        with open(Path(path) / "model.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls(n_components=data["n_components"])
        model._pca = data["pca"]
        model._embedding = data["embedding"]
        model._train_col_means = data["train_col_means"]
        return model


class PCALogistic(PCALinear):
    """PCA + Logistic Regression spatial model following Potthoff (2018).

    Stage 1: Iterative PCA imputation for missing data, then extract 2D scores.
    Stage 2: Per-item binary logistic regression on observed entries only.
    """

    name = "PCA-Logistic"

    def __init__(self, n_components: int = 2, max_impute_iter: int = 20, C: float = 1.0,
                 use_iterative: bool = True):
        super().__init__(n_components=n_components)
        self.max_impute_iter = max_impute_iter
        self.C = C
        self.use_iterative = use_iterative
        self._lr_models: list[LogisticRegression | None] = []
        self._item_defaults: np.ndarray = np.array([])

    def fit(self, reactions: pd.DataFrame) -> None:
        X = reactions.values.astype(float)
        self._train_col_means = compute_column_means(X, fill_empty_columns=0.5)

        if self.use_iterative:
            X_filled = iterative_pca_impute(X, n_components=self.n_components, max_iter=self.max_impute_iter, fill_empty_columns=0.5)
        else:
            X_filled = mean_impute(X, column_means=self._train_col_means)

        pca = PCA(n_components=self.n_components, random_state=0)
        scores = pca.fit_transform(X_filled)

        self._pca = pca
        self._embedding = pd.DataFrame(
            scores, index=reactions.index, columns=_embedding_columns(self.n_components)
        )

        self._fit_logistic_decoder(scores, X, C=self.C)

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        return self._predict_logistic(embedding)

    def get_item_parameters(self) -> list[LogisticRegression | None]:
        return self._lr_models

    def save(self, path: Path, **kwargs) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "max_impute_iter": self.max_impute_iter,
                "C": self.C,
                "pca": self._pca,
                "embedding": self._embedding,
                "lr_models": self._lr_models,
                "item_defaults": self._item_defaults,
                "train_col_means": self._train_col_means,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "PCALogistic":
        with open(Path(path) / "model.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls(n_components=data["n_components"],
                     max_impute_iter=data["max_impute_iter"],
                     C=data["C"])
        model._pca = data["pca"]
        model._embedding = data["embedding"]
        model._lr_models = data["lr_models"]
        model._item_defaults = data["item_defaults"]
        model._train_col_means = data["train_col_means"]
        return model
