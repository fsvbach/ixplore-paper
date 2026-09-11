from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ixplore import IXPLORE
from src.data import load_dataset
from src.models._r_utils import binarise
from src.models.base import SpatialModel

logging.getLogger("ixplore").setLevel(logging.WARNING)

# -- Kernel registry --
# All kernels used in experiments must be registered here so that
# models can be saved/loaded by kernel name instead of pickling callables.

def _polynomial_kernel(X):
    return np.column_stack([X[:, 0], X[:, 1], X[:, 0] ** 2, X[:, 1] ** 2])


def _rff_kernel(X, n_features=10, lengthscale=1.0, seed=42):
    rng = np.random.RandomState(seed)
    D = X.shape[1]
    W = rng.randn(D, n_features) / lengthscale
    b = rng.uniform(0, 2 * np.pi, n_features)
    Z = np.sqrt(2 / n_features) * np.cos(X @ W + b)
    return Z


KERNEL_REGISTRY = {
    "linear": None,
    "polynomial": _polynomial_kernel,
    "rff": _rff_kernel,
}


class IXPLOREModel(SpatialModel):
    name = "IXPLORE"

    def __init__(
        self,
        n_iterations: int = 1,
        prior_variance: float = 1.0,
        kernel=None,
        kernel_name: str = "linear",
        pca_initialization: bool = True,
        sampling_resolution: int = 100,
        scale_weights: bool = False,
        use_point_estimates: bool = True,
        model_regularization: float = 1e-8,
        random_state: int = 0,
    ):
        self.n_iterations = n_iterations
        self.prior_variance = prior_variance
        self.kernel_name = kernel_name
        self.kernel = kernel if kernel is not None else KERNEL_REGISTRY.get(kernel_name)
        self.pca_initialization = pca_initialization
        self.sampling_resolution = sampling_resolution
        self.scale_weights = scale_weights
        self.use_point_estimates = use_point_estimates
        self.model_regularization = model_regularization
        self.random_state = random_state
        self._model: IXPLORE | None = None

    def fit(self, reactions: pd.DataFrame) -> None:
        self._model = IXPLORE(
            reactions,
            prior_variance=self.prior_variance,
            kernel=self.kernel,
            pca_initialization=self.pca_initialization,
            sampling_resolution=self.sampling_resolution,
            scale_weights=self.scale_weights,
            use_point_estimates=self.use_point_estimates,
            model_regularization=self.model_regularization,
            random_state=self.random_state,
        )
        if self.n_iterations > 0:
            self._model.iterate(self.n_iterations)

    def _m(self) -> IXPLORE:
        assert self._model is not None, "Model not fitted"
        return self._model

    # --- SpatialModel interface ---

    def get_embedding(self) -> pd.DataFrame:
        return self._m().get_embedding()

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        return self._m().predict(embedding)

    def embed(self, reactions: pd.DataFrame, weights: pd.DataFrame | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            self._m().embed(reactions, weights=weights), index=reactions.index, columns=["x", "y"]
        )

    def get_item_parameters(self) -> pd.DataFrame:
        return self._m().get_parameters()

    # --- IXPLORE-specific pass-throughs (for experiments 3/4) ---

    def fit_posteriors(self):
        self._m().fit_posteriors()

    def fit_models(self):
        self._m().fit_models()

    def iterate(self, n=1):
        self._m().iterate(n_iterations=n)

    # --- Save / Load ---
    # Instead of pickling the IXPLORE object (which contains unpicklable lambdas),
    # we save the reconstruction ingredients: config + embedding + item params.

    def save(self, path: Path, **kwargs: Any) -> None:
        """Save model to a directory with config.json, embedding.csv, items.csv."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "prior_variance": self.prior_variance,
            "kernel_name": self.kernel_name,
            "pca_initialization": self.pca_initialization,
            "sampling_resolution": self.sampling_resolution,
            "scale_weights": self.scale_weights,
            "use_point_estimates": self.use_point_estimates,
            "model_regularization": self.model_regularization,
            "random_state": self.random_state,
            "n_iterations": self.n_iterations,
            "dataset": kwargs.get("dataset", ""),
            "sparsity": kwargs.get("sparsity", 0.0),
            "seed": kwargs.get("seed", 0),
            "test_fraction": kwargs.get("test_fraction", 0.0),
        }
        (path / "config.json").write_text(json.dumps(config))
        self.get_embedding().to_csv(path / "embedding.csv", float_format="%.3f")
        self.get_item_parameters().to_csv(path / "items.csv", float_format="%.3f")

    @classmethod
    def load(cls, path: Path) -> "IXPLOREModel":
        """Load model from a saved directory.

        If *reactions* is not provided, they are reconstructed from the
        dataset/sparsity/seed stored in config.json.
        """
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        ds = load_dataset(config["dataset"], test_fraction=config.get("test_fraction", 0.0))
        reactions = ds.get_data_with_sparsity("train", config["sparsity"], config["seed"])

        embedding = pd.read_csv(path / "embedding.csv", index_col=0)
        items = pd.read_csv(path / "items.csv", index_col=0)

        kernel_name = config["kernel_name"]
        kernel_fn = KERNEL_REGISTRY.get(kernel_name)

        scale_weights = config.get("scale_weights", False)
        use_point_estimates = config.get("use_point_estimates", True)
        model_regularization = config.get("model_regularization", 1e-8)
        model = cls(
            n_iterations=0,
            prior_variance=config["prior_variance"],
            kernel=kernel_fn,
            kernel_name=kernel_name,
            pca_initialization=config["pca_initialization"],
            sampling_resolution=config["sampling_resolution"],
            scale_weights=scale_weights,
            use_point_estimates=use_point_estimates,
            model_regularization=model_regularization,
            random_state=config["random_state"],
        )
        model.n_iterations = config["n_iterations"]
        model._model = IXPLORE(
            reactions,
            prior_variance=config["prior_variance"],
            kernel=kernel_fn,
            sampling_resolution=config["sampling_resolution"],
            scale_weights=scale_weights,
            use_point_estimates=use_point_estimates,
            model_regularization=model_regularization,
            random_state=config["random_state"],
            pretrained_models=items,
            pretrained_embedding=embedding,
        )
        return model


class IXPLOREBinarised(IXPLOREModel):
    """IXPLORE trained and embedded on 0.5-binarised reactions.

    Ablation for the baseline comparison: matches IDEAL's input preprocessing so
    the IXPLORE/IDEAL gap is attributable to the algorithm rather than the input.
    Evaluation uses the original continuous ground truth (inherited from
    SpatialModel.evaluate), so reported MAE/accuracy stay comparable to the
    continuous-input IXPLORE row.
    """

    name = "IXPLORE (binarised)"

    def fit(self, reactions: pd.DataFrame) -> None:
        super().fit(binarise(reactions))

    def embed(self, reactions: pd.DataFrame, weights: pd.DataFrame | None = None) -> pd.DataFrame:
        return super().embed(binarise(reactions), weights=weights)
