from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA as SKKernelPCA
from sklearn.svm import SVC

from ixplore.utils import compute_column_means, mean_impute
from src.models.base import SpatialModel, embedding_columns as _embedding_columns


def _median_gamma(X_filled: np.ndarray, rng: np.random.Generator, max_n: int = 1000) -> float:
    """RBF median heuristic: gamma = 1 / median(pairwise squared distance).

    Subsamples rows when the matrix is large so the pairwise computation stays
    cheap; the median is stable under subsampling.
    """
    n = X_filled.shape[0]
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        S = X_filled[idx]
    else:
        S = X_filled
    sq = np.sum((S[:, None, :] - S[None, :, :]) ** 2, axis=-1)
    med = np.median(sq[np.triu_indices_from(sq, k=1)])
    return 1.0 / med if med > 0 else 1.0


class KernelPCA(SpatialModel):
    """Kernel PCA + per-item Platt-calibrated RBF-SVM decoder, with per-dataset
    gamma tuning.

    Nonlinear counterpart to PCALogistic: an RBF-kernel PCA replaces the linear
    PCA in stage 1, then a per-item RBF SVM (Platt-calibrated to a probability)
    fits on the kernel scores in stage 2.

    The stage-1 RBF bandwidth gamma is the only sensitive hyper-parameter and
    its useful scale is dataset-dependent. If gamma is None (the default), fit()
    selects it automatically: it holds out a random fraction of the observed
    training entries, fits KPCA + the decoder for each candidate gamma on the
    remaining entries, and keeps the gamma with the lowest MAE on the held-out
    entries. The grid is expressed as multipliers of the RBF median heuristic
    (1 / median pairwise squared distance) so it adapts to each dataset's scale.
    Tuning never sees test data: it uses only a split of the reactions passed to
    fit(). An explicit float gamma disables tuning.

    Kernel PCA has no exact inverse for a nonlinear kernel, so there is no
    linear-reconstruction variant; prediction goes through the SVM decoder.
    SVC(probability=True) is used so predict_proba returns a calibrated [0, 1]
    value, matching what compute_metrics scores (MAE / accuracy on {0, 1}).
    Items with a single observed class fall back to a constant default.
    Missing values are mean-imputed before the kernel is computed. sklearn's
    KernelPCA.transform embeds new users against the fitted training points, so
    test users need no refit.
    """

    name = "KernelPCA"

    # Candidate gammas as multipliers of the median-heuristic scale.
    GAMMA_MULTIPLIERS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    # Fraction of observed training entries held out to score candidate gammas.
    TUNE_HOLDOUT = 0.15
    # Above this many users the per-item Platt-calibrated SVM decoder is too
    # slow (probability=True does internal CV per item); fall back to the
    # shared per-item logistic decoder, which scales far better with N.
    SVM_DECODER_MAX_N = 2000

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        gamma: float | None = None,
        svm_C: float = 1.0,
        random_state: int = 0,
    ):
        self.n_components = n_components
        self.kernel = kernel
        # None -> auto-tune in fit(); float -> use as given (no tuning)
        self.gamma = gamma
        # Regularisation strength of the stage-2 RBF SVM decoder.
        self.svm_C = svm_C
        self.random_state = random_state

        # gamma actually used (resolved during fit; equals self.gamma if given)
        self.gamma_ = gamma
        self._kpca: SKKernelPCA | None = None
        self._embedding: pd.DataFrame | None = None
        self._train_col_means: np.ndarray | None = None
        self._svm_models: list[SVC | None] = []
        self._item_defaults: np.ndarray = np.array([])
        # Set in _fit_decoder: True if the logistic fallback was used (N large).
        self._use_logistic_decoder: bool = False

    # -- gamma tuning (training data only) --
    def _score_gamma(self, X_obs: np.ndarray, X_full: np.ndarray,
                     holdout_mask: np.ndarray, gamma: float) -> float:
        """Fit KPCA + decoder on X_obs (holdout entries set NaN), return MAE on
        the held-out entries measured against X_full."""
        col_means = compute_column_means(X_obs, fill_empty_columns=0.5)
        X_filled = mean_impute(X_obs, column_means=col_means)
        kpca = SKKernelPCA(
            n_components=self.n_components, kernel=self.kernel,
            gamma=gamma, random_state=self.random_state,
        )
        scores = kpca.fit_transform(X_filled)
        self._fit_decoder(scores, X_obs)
        preds = self._predict_decoder(scores)
        return float(np.abs(preds[holdout_mask] - X_full[holdout_mask]).mean())

    def _resolve_gamma(self, X: np.ndarray) -> float:
        rng = np.random.default_rng(self.random_state)
        X_filled_full = mean_impute(X, column_means=compute_column_means(X, fill_empty_columns=0.5))
        base = _median_gamma(X_filled_full, rng)
        candidates = [m * base for m in self.GAMMA_MULTIPLIERS]

        observed = np.argwhere(~np.isnan(X))
        if len(observed) < 50:  # too few entries to tune reliably
            return base
        n_hold = max(1, int(self.TUNE_HOLDOUT * len(observed)))
        hold_idx = rng.choice(len(observed), size=n_hold, replace=False)
        holdout_mask = np.zeros(X.shape, dtype=bool)
        rows, cols = observed[hold_idx, 0], observed[hold_idx, 1]
        holdout_mask[rows, cols] = True

        X_obs = X.copy()
        X_obs[holdout_mask] = np.nan

        best_gamma, best_mae = base, np.inf
        for g in candidates:
            mae = self._score_gamma(X_obs, X, holdout_mask, g)
            if mae < best_mae:
                best_mae, best_gamma = mae, g
        return best_gamma

    def fit(self, reactions: pd.DataFrame) -> None:
        X = reactions.values.astype(float)
        self._train_col_means = compute_column_means(X, fill_empty_columns=0.5)

        self.gamma_ = self.gamma if self.gamma is not None else self._resolve_gamma(X)

        X_filled = mean_impute(X, column_means=self._train_col_means)
        kpca = SKKernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma_,
            random_state=self.random_state,
        )
        scores = kpca.fit_transform(X_filled)

        self._kpca = kpca
        self._embedding = pd.DataFrame(
            scores, index=reactions.index, columns=_embedding_columns(self.n_components)
        )
        self._fit_decoder(scores, X)

    # -- stage-2 decoder: per-item Platt-calibrated RBF SVM --
    # For N <= SVM_DECODER_MAX_N users the per-item Platt-calibrated RBF SVM is
    # used. Beyond that its probability=True cross-validation makes a full
    # evaluation sweep impractically slow, so the shared per-item logistic
    # decoder (base.SpatialModel) is used instead.
    def _fit_decoder(self, scores: np.ndarray, X: np.ndarray) -> None:
        self._use_logistic_decoder = scores.shape[0] > self.SVM_DECODER_MAX_N
        if self._use_logistic_decoder:
            self._fit_logistic_decoder(scores, X, C=self.svm_C)
            return

        K = X.shape[1]
        self._svm_models = []
        self._item_defaults = np.full(K, 0.5)
        for k in range(K):
            # Drop neutral (0.5) votes: match the binarisation convention used by
            # the R-based IRT wrappers (see src/models/_r_utils.py:binarise).
            obs = ~np.isnan(X[:, k]) & (X[:, k] != 0.5)
            y_bin = (X[obs, k] > 0.5).astype(int)
            if len(np.unique(y_bin)) < 2:
                self._svm_models.append(None)
                self._item_defaults[k] = y_bin.mean() if y_bin.size > 0 else 0.5
            else:
                svm = SVC(
                    kernel="rbf",
                    C=self.svm_C,
                    gamma="scale",
                    probability=True,
                    random_state=self.random_state,
                )
                svm.fit(scores[obs], y_bin)
                self._svm_models.append(svm)

    def _predict_decoder(self, embedding: np.ndarray) -> np.ndarray:
        if getattr(self, "_use_logistic_decoder", False):
            return self._predict_logistic(embedding)

        K = len(self._svm_models)
        preds = np.zeros((len(embedding), K))
        for k in range(K):
            svm = self._svm_models[k]
            if svm is None:
                preds[:, k] = self._item_defaults[k]
            else:
                preds[:, k] = svm.predict_proba(embedding)[:, 1]
        return preds

    def get_embedding(self) -> pd.DataFrame:
        return self._embedding

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        return self._predict_decoder(embedding)

    def embed(self, reactions: pd.DataFrame) -> pd.DataFrame:
        assert self._kpca is not None and self._train_col_means is not None
        X_filled = mean_impute(reactions.values.astype(float), column_means=self._train_col_means)
        scores = self._kpca.transform(X_filled)
        return pd.DataFrame(
            scores, index=reactions.index, columns=_embedding_columns(self.n_components)
        )

    def get_item_parameters(self) -> list[SVC | None]:
        return self._svm_models

    def save(self, path: Path, **kwargs) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "kernel": self.kernel,
                "gamma": self.gamma,
                "gamma_": self.gamma_,
                "svm_C": self.svm_C,
                "random_state": self.random_state,
                "kpca": self._kpca,
                "embedding": self._embedding,
                "svm_models": self._svm_models,
                "item_defaults": self._item_defaults,
                "train_col_means": self._train_col_means,
                "use_logistic_decoder": self._use_logistic_decoder,
                "lr_models": getattr(self, "_lr_models", None),
            }, f)

    @classmethod
    def load(cls, path: Path) -> "KernelPCA":
        with open(Path(path) / "model.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls(
            n_components=data["n_components"],
            kernel=data["kernel"],
            gamma=data["gamma"],
            svm_C=data["svm_C"],
            random_state=data["random_state"],
        )
        model.gamma_ = data.get("gamma_", data["gamma"])
        model._kpca = data["kpca"]
        model._embedding = data["embedding"]
        model._svm_models = data["svm_models"]
        model._item_defaults = data["item_defaults"]
        model._train_col_means = data["train_col_means"]
        model._use_logistic_decoder = data.get("use_logistic_decoder", False)
        if data.get("lr_models") is not None:
            model._lr_models = data["lr_models"]
        return model
