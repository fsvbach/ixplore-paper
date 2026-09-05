from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.metrics import compute_metrics


def embedding_columns(n_components: int) -> list[str]:
    """Use (x, y) for 2D embeddings, z0..z{k-1} otherwise."""
    return ["x", "y"] if n_components == 2 else [f"z{i}" for i in range(n_components)]


class SpatialModel(ABC):
    """
    Unified interface for 2D spatial / embedding models on [0,1] reaction data.

    Terminology:
      - reactions: pd.DataFrame, index=users, columns=items, values in [0,1] or NaN
      - embedding: np.ndarray of shape (N, 2), user positions in latent space
    """

    name: str

    @abstractmethod
    def fit(self, reactions: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def get_embedding(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def predict(self, embedding: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def embed(self, reactions: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def save(self, path: Path, **kwargs) -> None:
        ...

    def evaluate(self, sparse_reactions, ground_truth, test=False):
        """Evaluate model on train or test data.

        Args:
            sparse_reactions: DataFrame with NaN for missing + artificially masked entries.
            ground_truth: DataFrame with NaN only for originally missing entries.
            test: If False, predict on training embedding. If True, embed users from
                  sparse_reactions first, then predict.

        Returns:
            dict with keys: fit_mae, fit_accuracy, fit_rmse,
            impute_mae, impute_accuracy, impute_rmse.
        """
        if test:
            embs = self.embed(sparse_reactions)
        else:
            embs = self.get_embedding()
        preds = self.predict(embs.values)
        assert not np.isnan(preds).any(), "Predictions contain NaN values"
        return compute_metrics(preds, ground_truth.values, sparse_reactions.values)

    # ── Shared logistic decoder (used by PCA/UMAP/t-SNE + Logistic wrappers) ──
    # Stage 2 of the "embed-then-decode" pattern: given 2D training scores and the
    # original (NaN-containing) reaction matrix, fit a per-item binary LR on
    # observed entries. Items with only one observed class fall back to a constant.

    def _fit_logistic_decoder(self, scores: np.ndarray, X: np.ndarray, C: float = 1.0) -> None:
        K = X.shape[1]
        self._lr_models: list[LogisticRegression | None] = []
        self._item_defaults: np.ndarray = np.full(K, 0.5)
        for k in range(K):
            # Drop neutral (0.5) votes: match the binarisation convention used by
            # the R-based IRT wrappers (see src/models/_r_utils.py:binarise).
            obs = ~np.isnan(X[:, k]) & (X[:, k] != 0.5)
            y_bin = (X[obs, k] > 0.5).astype(int)
            if len(np.unique(y_bin)) < 2:
                self._lr_models.append(None)
                self._item_defaults[k] = y_bin.mean() if y_bin.size > 0 else 0.5
            else:
                lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
                lr.fit(scores[obs], y_bin)
                self._lr_models.append(lr)

    def _predict_logistic(self, embedding: np.ndarray) -> np.ndarray:
        K = len(self._lr_models)
        preds = np.zeros((len(embedding), K))
        for k in range(K):
            lr = self._lr_models[k]
            if lr is None:
                preds[:, k] = self._item_defaults[k]
            else:
                preds[:, k] = lr.predict_proba(embedding)[:, 1]
        return preds
