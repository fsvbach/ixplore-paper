from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from openTSNE import TSNE, TSNEEmbedding
from sklearn.linear_model import LogisticRegression

from ixplore.utils import compute_column_means, mean_impute
from src.models.base import SpatialModel, embedding_columns as _embedding_columns


class TSNELogistic(SpatialModel):
    """t-SNE + per-item logistic decoder, following Bachmann et al. (ECML 2024).

    Uses openTSNE so test users can be embedded into the existing latent space
    via TSNEEmbedding.transform (sklearn's TSNE has no out-of-sample transform).
    """

    name = "tSNE-Logistic"

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        C: float = 1.0,
        random_state: int = 0,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.C = C
        self.random_state = random_state

        self._tsne_embedding: TSNEEmbedding | None = None
        self._embedding: pd.DataFrame | None = None
        self._train_col_means: np.ndarray | None = None
        self._lr_models: list[LogisticRegression | None] = []
        self._item_defaults: np.ndarray = np.array([])

    def fit(self, reactions: pd.DataFrame) -> None:
        X = reactions.values.astype(float)
        self._train_col_means = compute_column_means(X, fill_empty_columns=0.5)
        X_filled = mean_impute(X, column_means=self._train_col_means)

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            random_state=self.random_state,
        )
        self._tsne_embedding = tsne.fit(X_filled)
        scores = np.asarray(self._tsne_embedding)

        self._embedding = pd.DataFrame(
            scores, index=reactions.index, columns=_embedding_columns(self.n_components)
        )
        self._fit_logistic_decoder(scores, X, C=self.C)

    def get_embedding(self) -> pd.DataFrame:
        return self._embedding

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        return self._predict_logistic(embedding)

    def embed(self, reactions: pd.DataFrame) -> pd.DataFrame:
        assert self._tsne_embedding is not None and self._train_col_means is not None
        X_filled = mean_impute(reactions.values.astype(float), column_means=self._train_col_means)
        new_emb = self._tsne_embedding.transform(X_filled)
        return pd.DataFrame(
            np.asarray(new_emb),
            index=reactions.index,
            columns=_embedding_columns(self.n_components),
        )

    def get_item_parameters(self) -> list[LogisticRegression | None]:
        return self._lr_models

    def save(self, path: Path, **kwargs) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "perplexity": self.perplexity,
                "C": self.C,
                "random_state": self.random_state,
                "tsne_embedding": self._tsne_embedding,
                "embedding": self._embedding,
                "lr_models": self._lr_models,
                "item_defaults": self._item_defaults,
                "train_col_means": self._train_col_means,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "TSNELogistic":
        with open(Path(path) / "model.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls(
            n_components=data["n_components"],
            perplexity=data["perplexity"],
            C=data["C"],
            random_state=data["random_state"],
        )
        model._tsne_embedding = data["tsne_embedding"]
        model._embedding = data["embedding"]
        model._lr_models = data["lr_models"]
        model._item_defaults = data["item_defaults"]
        model._train_col_means = data["train_col_means"]
        return model
