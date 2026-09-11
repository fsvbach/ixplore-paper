"""IDEAL model wrapper using the pscl R package.

Calls ``pscl::ideal(rollcall(...), d=2, store.item=TRUE)`` via Rscript,
reads back ``xbar`` (posterior mean ideal points) and ``betabar``
(posterior mean item parameters), then exposes them through the
SpatialModel interface.

Embedding new users also runs in R: item parameters are fixed via
spike priors on ``betabar``, and only the new users' ideal points
are sampled.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.models._r_utils import binarise as _binarise
from src.models._r_utils import run_r_script as _run_r_script
from src.models.base import SpatialModel

_R_FIT_SCRIPT = """\
library(pscl)

reactions <- read.csv("{reactions_path}", header=TRUE, row.names=1)
rc <- rollcall(reactions)
fit <- ideal(rc, d={d}, store.item=TRUE,
             maxiter={maxiter}, burnin={burnin}, thin={thin})

write.csv(fit$xbar, "{xbar_path}", row.names=TRUE)
write.csv(fit$betabar, "{betabar_path}", row.names=TRUE)
"""

_R_EMBED_SCRIPT = """\
library(pscl)

reactions <- read.csv("{reactions_path}", header=TRUE, row.names=1)
betabar <- as.matrix(read.csv("{betabar_path}", header=TRUE, row.names=1))

n <- nrow(reactions)
d <- {d}

rc <- rollcall(reactions)

# pscl drops lopsided items internally; align bp to the surviving items
rc_work <- dropRollCall(rc, dropList=list(lop=0))
kept <- colnames(rc_work$votes)
m <- length(kept)

bp  <- betabar[kept, , drop=FALSE]
bpv <- matrix(1e6, nrow=m, ncol=d+1)

priors <- list(
    xp  = matrix(0, nrow=n, ncol=d),
    xpv = matrix(1, nrow=n, ncol=d),
    bp  = bp,
    bpv = bpv
)

fit <- ideal(rc, d=d, priors=priors, store.item=FALSE,
             startvals="eigen",
             maxiter={maxiter}, burnin={burnin}, thin={thin})

write.csv(fit$xbar, "{xbar_path}", row.names=TRUE)
"""


class IDEAL(SpatialModel):
    """IDEAL (Clinton, Jackman & Rivers 2004) via pscl R package.

    Parameters
    ----------
    n_components : int
        Latent dimensionality (default 2).
    maxiter, burnin, thin : int
        MCMC parameters passed to ``pscl::ideal()``.
    """

    name = "IDEAL"

    def __init__(
        self,
        n_components: int = 2,
        maxiter: int = 10000,
        burnin: int = 5000,
        thin: int = 100,
    ):
        self.n_components = n_components
        self.maxiter = maxiter
        self.burnin = burnin
        self.thin = thin
        self._embedding = None
        self._betabar = None

    def fit(self, reactions: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            reactions_path = tmp / "reactions.csv"
            xbar_path = tmp / "xbar.csv"
            betabar_path = tmp / "betabar.csv"

            _binarise(reactions).to_csv(reactions_path)

            script = _R_FIT_SCRIPT.format(
                reactions_path=reactions_path,
                d=self.n_components,
                maxiter=self.maxiter,
                burnin=self.burnin,
                thin=self.thin,
                xbar_path=xbar_path,
                betabar_path=betabar_path,
            )
            _run_r_script(script)

            xbar = pd.read_csv(xbar_path, index_col=0)
            betabar = pd.read_csv(betabar_path, index_col=0)

        xbar.index = reactions.index
        # Keep pscl's "Vote 1" row names for R interop; store column mapping
        self._item_columns = reactions.columns
        self._embedding = pd.DataFrame(
            xbar.values[:, :self.n_components],
            index=reactions.index,
            columns=["x", "y"],
        )
        # pscl drops lopsided items; pad betabar back to all original items so
        # predictions stay aligned with the input column order. Dropped items
        # get zero discrimination + zero difficulty -> neutral Phi(0) = 0.5.
        full_betabar = pd.DataFrame(
            0.0,
            index=[f"Vote {i+1}" for i in range(len(reactions.columns))],
            columns=betabar.columns,
        )
        full_betabar.loc[betabar.index] = betabar.values
        self._betabar = full_betabar

    # --- SpatialModel interface ---

    def get_embedding(self) -> pd.DataFrame:
        return self._embedding

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        """Predict P(Y=1|X) = Phi(X @ beta - alpha) for each item."""
        assert self._betabar is not None
        beta = self._betabar.values  # (K, d+1): [disc1, disc2, ..., difficulty]
        # Augment embedding with -1 so [x, y, -1] @ [beta1, beta2, alpha]^T = x'beta - alpha
        aug = np.column_stack([embedding, -np.ones(len(embedding))])
        return norm.cdf(aug @ beta.T)

    def embed(self, reactions: pd.DataFrame) -> pd.DataFrame:
        """Embed new users in R with item parameters fixed via spike priors."""
        assert self._betabar is not None
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            reactions_path = tmp / "reactions.csv"
            betabar_path = tmp / "betabar.csv"
            xbar_path = tmp / "xbar.csv"

            # Rename columns to match pscl's "Vote N" convention so
            # the rollcall object aligns with betabar row names
            binary = _binarise(reactions)
            binary.columns = self._betabar.index
            binary.to_csv(reactions_path)
            self._betabar.to_csv(betabar_path)

            script = _R_EMBED_SCRIPT.format(
                reactions_path=reactions_path,
                betabar_path=betabar_path,
                d=self.n_components,
                maxiter=self.maxiter,
                burnin=self.burnin,
                thin=self.thin,
                xbar_path=xbar_path,
            )
            _run_r_script(script)

            xbar = pd.read_csv(xbar_path, index_col=0)

        xbar.index = reactions.index
        return pd.DataFrame(
            xbar.values[:, :self.n_components],
            index=reactions.index,
            columns=["x", "y"],
        )

    def get_item_parameters(self) -> pd.DataFrame:
        assert self._betabar is not None
        result = self._betabar.copy()
        result.index = self._item_columns
        return result

    # --- Save / Load ---

    def save(self, path: Path, **kwargs) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "maxiter": self.maxiter,
                "burnin": self.burnin,
                "thin": self.thin,
                "embedding": self._embedding,
                "betabar": self._betabar,
                "item_columns": self._item_columns,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "IDEAL":
        with open(Path(path) / "model.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls(
            n_components=data["n_components"],
            maxiter=data["maxiter"],
            burnin=data["burnin"],
            thin=data["thin"],
        )
        model._embedding = data["embedding"]
        model._betabar = data["betabar"]
        model._item_columns = data["item_columns"]
        return model
