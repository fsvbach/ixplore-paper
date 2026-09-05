"""emIRT (binary EM) wrapper using the emIRT R package.

Implements the binary ideal-point model of Imai, Lo & Olmsted (2016),
estimated by their fast EM algorithm via ``emIRT::binIRT``. binIRT is
the EM analogue of the Clinton-Jackman-Rivers / IDEAL posterior (Imai
et al. 2016 derive it as exactly that), so this is a direct
EM-vs-MCMC counterpart to the :class:`IDEAL` wrapper on the same
0.5-binarised data.

emIRT is **one-dimensional only** (``binIRT`` hard-errors on
``.D != 1``). To fit the repo's 2D ``SpatialModel`` interface, the
single estimated ideal point is exposed as the x-coordinate and y is
fixed at 0. In 2D embedding plots emIRT therefore appears as a
horizontal line -- the honest visual consequence of a 1D model.

Model / prediction
------------------
``binIRT`` returns ideal points ``x`` (N) and item parameters
``beta`` (J x 2), whose columns are ``d0`` (intercept, = -difficulty)
and ``d1`` (discrimination). The two-parameter probit gives

    P(Y = 1 | x) = Phi(d0 + d1 * x)

which is the same likelihood family as IDEAL's
``Phi(x' beta - alpha)``. ``predict`` returns this probability, a
scalar in (0, 1) directly comparable to the other binarised baselines.

Embedding new users solves a per-user 1D MLE with item parameters
fixed, mirroring the W-NOMINATE / LSIRM wrappers (emIRT has no native
fixed-item embedding pathway).
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from src.models._r_utils import binarise as _binarise
from src.models._r_utils import run_r_script as _run_r_script
from src.models.base import SpatialModel, embedding_columns as _embedding_columns

_R_FIT_SCRIPT = """\
library(emIRT)
library(pscl)

# Binary matrix: 1 = yea, 0 = nay, NA = missing. Rows respondents.
df <- read.csv("{reactions_path}", header=TRUE, row.names=1, check.names=FALSE)
rc <- rollcall(df, legis.names=rownames(df), vote.names=colnames(df))
crc <- convertRC(rc, type="binIRT")

s <- getStarts(crc$n, crc$m, 1)
p <- makePriors(crc$n, crc$m, 1)

fit <- binIRT(.rc = crc, .starts = s, .priors = p, .D = 1L,
              .control = list(threads = 1, verbose = FALSE,
                              thresh = {thresh}))

# x is (N x 1); beta is (J x 2): column d0 = intercept, d1 = discrimination.
write.csv(data.frame(x = fit$means$x[, 1]), "{x_path}", row.names = FALSE)
write.csv(data.frame(d0 = fit$means$beta[, 1],
                      d1 = fit$means$beta[, 2]),
          "{item_path}", row.names = FALSE)
"""


def _prob_yea(x: np.ndarray, d0: np.ndarray, d1: np.ndarray) -> np.ndarray:
    """P(Y=1) = Phi(d0 + d1 * x); shapes x:(N,), d0/d1:(J,) -> (N, J)."""
    eta = d0[None, :] + d1[None, :] * x[:, None]
    return norm.cdf(eta)


class EMIRT(SpatialModel):
    """emIRT binary EM ideal point (Imai, Lo & Olmsted 2016).

    Parameters
    ----------
    n_components : int
        Must be 2 for interface compatibility; the model itself is 1D and
        the second coordinate is fixed at 0.
    thresh : float
        EM convergence threshold (correlation-based, per emIRT).
    """

    name = "emIRT"

    def __init__(
        self,
        n_components: int = 2,
        thresh: float = 1e-6,
    ):
        self.n_components = n_components
        self.thresh = thresh
        self._embedding: pd.DataFrame | None = None
        self._x: np.ndarray | None = None      # (N,) 1D ideal points
        self._d0: np.ndarray | None = None     # (J,) item intercept
        self._d1: np.ndarray | None = None     # (J,) item discrimination
        self._item_columns: pd.Index | None = None

    def _run_r_fit(self, binary: pd.DataFrame, tmp: Path) -> dict:
        reactions_path = tmp / "reactions.csv"
        x_path = tmp / "x.csv"
        item_path = tmp / "item.csv"

        binary.to_csv(reactions_path)

        script = _R_FIT_SCRIPT.format(
            reactions_path=reactions_path,
            x_path=x_path,
            item_path=item_path,
            thresh=self.thresh,
        )
        _run_r_script(script)

        x = pd.read_csv(x_path)["x"].to_numpy(dtype=float)
        item = pd.read_csv(item_path)
        return {
            "x": x,
            "d0": item["d0"].to_numpy(dtype=float),
            "d1": item["d1"].to_numpy(dtype=float),
        }

    def fit(self, reactions: pd.DataFrame) -> None:
        binary = _binarise(reactions)  # {0, 1, NaN}
        with tempfile.TemporaryDirectory() as tmp_str:
            out = self._run_r_fit(binary, Path(tmp_str))

        self._x = out["x"]
        self._d0 = out["d0"]
        self._d1 = out["d1"]
        self._item_columns = reactions.columns

        cols = _embedding_columns(self.n_components)
        emb = np.zeros((len(reactions.index), self.n_components))
        emb[:, 0] = self._x  # x = 1D ideal point; remaining dims stay 0
        self._embedding = pd.DataFrame(emb, index=reactions.index, columns=cols)

    # --- SpatialModel interface ---

    def get_embedding(self) -> pd.DataFrame:
        assert self._embedding is not None
        return self._embedding

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        """P(Y=1) = Phi(d0 + d1 * x) for each (user, item).

        Uses only the first embedding column (the 1D ideal point); the
        second column is the fixed-zero placeholder.
        """
        assert self._d0 is not None
        x = np.asarray(embedding, dtype=float)[:, 0]
        return _prob_yea(x, self._d0, self._d1)

    def embed(self, reactions: pd.DataFrame) -> pd.DataFrame:
        """Per-user 1D MLE with item parameters fixed at the fitted values.

        binIRT has no native fixed-item embedding pathway, so each new
        user's ideal point is found by maximising their binary
        log-likelihood (mirrors the W-NOMINATE / LSIRM wrappers).
        """
        assert self._d0 is not None
        binary = _binarise(reactions).to_numpy()  # (N, J), {0, 1, NaN}
        N = binary.shape[0]
        D = self.n_components
        coords = np.zeros((N, D))

        x0 = float(np.mean(self._x)) if self._x is not None else 0.0

        for i in range(N):
            yi = binary[i]
            obs = ~np.isnan(yi)
            if not obs.any():
                coords[i, 0] = x0
                continue
            y = yi[obs].astype(int)
            d0 = self._d0[obs]
            d1 = self._d1[obs]

            def neg_ll(xi):
                p = norm.cdf(d0 + d1 * xi)
                p = np.clip(p, 1e-9, 1 - 1e-9)
                return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

            res = minimize_scalar(
                neg_ll, bounds=(-10.0, 10.0), method="bounded",
            )
            coords[i, 0] = res.x

        return pd.DataFrame(
            coords, index=reactions.index,
            columns=_embedding_columns(D),
        )

    def get_item_parameters(self) -> pd.DataFrame:
        assert self._d0 is not None
        return pd.DataFrame(
            {"d0": self._d0, "d1": self._d1},
            index=self._item_columns,
        )

    # --- Save / Load ---

    def save(self, path: Path, **kwargs) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "thresh": self.thresh,
                "embedding": self._embedding,
                "x": self._x,
                "d0": self._d0,
                "d1": self._d1,
                "item_columns": self._item_columns,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "EMIRT":
        with open(Path(path) / "model.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls(
            n_components=data["n_components"],
            thresh=data["thresh"],
        )
        model._embedding = data["embedding"]
        model._x = data["x"]
        model._d0 = data["d0"]
        model._d1 = data["d1"]
        model._item_columns = data["item_columns"]
        return model
