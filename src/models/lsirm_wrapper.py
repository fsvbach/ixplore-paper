"""LSIRM (2PL) wrapper using the lsirm12pl R package.

Implements the Latent Space Item Response Model of Jeon et al. (2021); see
also Lee et al. (2025) for the political-science application that motivates
its inclusion as a metric-space alternative to NOMINATE/IDEAL.

Model (binary, 2PL, MAR missingness):

    logit P(Y_ji = 1) = theta_j * alpha_i + beta_i - gamma * ||z_j - w_i||

where z_j is respondent j's 2D latent position (our **embedding**), w_i is
item i's 2D latent position, alpha_i is item discrimination, beta_i is item
difficulty, gamma is the scalar distance weight, and theta_j is a per-
respondent ability/random effect.

The package is MCMC-based; this wrapper calls ``lsirm12pl::lsirm2pl_mar``
via Rscript and reads back the posterior point estimates.

Embedding new users: lsirm12pl does not expose a "fix item parameters and
sample only new respondents" pathway. To embed out-of-sample users this
wrapper refits over the **union** of train + new users, then returns only
the new rows. This is more expensive than IDEAL's spike-prior trick but
keeps the model honest; do not call ``embed`` in inner loops.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models._r_utils import binarise as _binarise
from src.models._r_utils import run_r_script as _run_r_script
from src.models.base import SpatialModel, embedding_columns as _embedding_columns

_R_FIT_SCRIPT = """\
library(lsirm12pl)

# lsirm2pl expects a numeric matrix with NA for missing; row/column order is
# preserved in the returned point estimates.
reactions <- as.matrix(read.csv("{reactions_path}", header=TRUE, row.names=1,
                                check.names=FALSE))
storage.mode(reactions) <- "numeric"

fit <- lsirm2pl(reactions,
                missing_data="mar",
                ndim={ndim},
                niter={niter}, nburn={nburn}, nthin={nthin},
                seed={seed}, verbose=FALSE)

# Posterior point estimates. z is (N, ndim); w is (J, ndim).
write.csv(fit$z_estimate,     "{z_path}",     row.names=FALSE)
write.csv(fit$w_estimate,     "{w_path}",     row.names=FALSE)
write.csv(data.frame(theta=fit$theta_estimate),
                              "{theta_path}", row.names=FALSE)
write.csv(data.frame(alpha=fit$alpha_estimate,
                     beta =fit$beta_estimate),
                              "{ab_path}",    row.names=FALSE)
write.csv(data.frame(gamma=fit$gamma_estimate),
                              "{gamma_path}", row.names=FALSE)
"""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _lsirm_probs(
    z: np.ndarray,         # (N, D)
    theta: np.ndarray,     # (N,)
    w: np.ndarray,         # (J, D)
    alpha: np.ndarray,     # (J,)
    beta: np.ndarray,      # (J,)
    gamma: float,
) -> np.ndarray:
    """P(Y=1) under the 2PL LSIRM, vectorised over (N, J)."""
    # Pairwise Euclidean distance ||z_n - w_j||  → (N, J)
    diff = z[:, None, :] - w[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    logit = theta[:, None] * alpha[None, :] + beta[None, :] - gamma * dist
    return _sigmoid(logit)


class LSIRM(SpatialModel):
    """LSIRM 2PL (Jeon et al. 2021) via the ``lsirm12pl`` R package.

    Parameters
    ----------
    n_components : int
        Latent space dimensionality (default 2).
    niter, nburn, nthin : int
        MCMC controls passed to ``lsirm2pl``. The defaults are lighter than
        the package's own defaults (15000/2500/5) so fitting is tractable
        in evaluation loops; raise them for the final report.
    seed : int
        MCMC seed (passed to the R-side ``seed`` argument).
    """

    name = "LSIRM"

    def __init__(
        self,
        n_components: int = 2,
        niter: int = 5000,
        nburn: int = 1000,
        nthin: int = 5,
        seed: int = 0,
        pr_sd_theta: float = 1.0,
    ):
        self.n_components = n_components
        self.niter = niter
        self.nburn = nburn
        self.nthin = nthin
        self.seed = seed
        # Prior SD on theta used by lsirm2pl training (its default is 1.0).
        # embed() reuses the *same* prior so out-of-sample positions are
        # MAP estimates consistent with the training posterior, not a free
        # MLE. Keep in sync with the R-side pr_sd_theta if ever changed.
        self.pr_sd_theta = pr_sd_theta
        self._embedding: pd.DataFrame | None = None
        self._theta: np.ndarray | None = None  # (N,)
        self._w: np.ndarray | None = None      # (J, D)
        self._alpha: np.ndarray | None = None  # (J,)
        self._beta: np.ndarray | None = None   # (J,)
        self._gamma: float | None = None
        self._item_columns: pd.Index | None = None
        # Cache training data + per-user theta for predict() on the
        # training embedding. embed() refits with new rows appended.
        self._train_index: pd.Index | None = None

    def _run_r_fit(self, reactions: pd.DataFrame, tmp: Path) -> dict:
        reactions_path = tmp / "reactions.csv"
        z_path = tmp / "z.csv"
        w_path = tmp / "w.csv"
        theta_path = tmp / "theta.csv"
        ab_path = tmp / "ab.csv"
        gamma_path = tmp / "gamma.csv"

        _binarise(reactions).to_csv(reactions_path)

        script = _R_FIT_SCRIPT.format(
            reactions_path=reactions_path,
            ndim=self.n_components,
            niter=self.niter, nburn=self.nburn, nthin=self.nthin,
            seed=self.seed,
            z_path=z_path, w_path=w_path,
            theta_path=theta_path, ab_path=ab_path, gamma_path=gamma_path,
        )
        _run_r_script(script)

        return {
            "z": pd.read_csv(z_path).to_numpy(dtype=float),
            "w": pd.read_csv(w_path).to_numpy(dtype=float),
            "theta": pd.read_csv(theta_path)["theta"].to_numpy(dtype=float),
            "alpha": pd.read_csv(ab_path)["alpha"].to_numpy(dtype=float),
            "beta": pd.read_csv(ab_path)["beta"].to_numpy(dtype=float),
            "gamma": float(pd.read_csv(gamma_path)["gamma"].iloc[0]),
        }

    def fit(self, reactions: pd.DataFrame) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            out = self._run_r_fit(reactions, Path(tmp_str))

        cols = _embedding_columns(self.n_components)
        self._embedding = pd.DataFrame(
            out["z"], index=reactions.index, columns=cols,
        )
        self._theta = out["theta"]
        self._w = out["w"]
        self._alpha = out["alpha"]
        self._beta = out["beta"]
        self._gamma = out["gamma"]
        self._item_columns = reactions.columns
        self._train_index = reactions.index

    # --- SpatialModel interface ---

    def get_embedding(self) -> pd.DataFrame:
        assert self._embedding is not None
        return self._embedding

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        """P(Y=1) for the given embedding.

        Each row of ``embedding`` must come from the training fit so that
        the matching per-row ``theta`` can be applied. For embed-then-predict
        on new users, the wrapper's ``embed`` returns positions whose theta
        is included via ``self._theta_new``.
        """
        assert self._w is not None
        embedding = np.asarray(embedding, dtype=float)
        N = embedding.shape[0]
        # Decide which theta vector to use: training embedding (N == N_train)
        # vs out-of-sample embed result (N == N_new).
        if N == len(self._theta):
            theta = self._theta
        elif hasattr(self, "_theta_new") and N == len(self._theta_new):
            theta = self._theta_new
        else:
            raise ValueError(
                f"LSIRM.predict: embedding row count {N} matches neither the "
                f"training fit ({len(self._theta)}) nor the most recent "
                f"out-of-sample embed call."
            )
        return _lsirm_probs(
            embedding, theta, self._w, self._alpha, self._beta, self._gamma,
        )

    def embed(self, reactions: pd.DataFrame) -> pd.DataFrame:
        """Per-user MAP estimate under LSIRM's own prior.

        For each new user, holds w, alpha, beta, gamma fixed at the training
        posterior means and maximises the log-posterior over (z, theta) —
        the binary log-likelihood plus LSIRM's training prior
        ``z ~ N(0, I)``, ``theta ~ N(0, pr_sd_theta^2)``. The prior is
        essential: a free MLE is unidentified for users with few observed
        answers and diverges (positions of magnitude 1e3+), so re-embedding
        the training data would not recover the training coordinates. With
        the prior the estimator matches the regularised regime of the
        training MCMC. theta for new rows is stashed in ``self._theta_new``
        so the next ``predict`` call uses consistent per-row abilities.
        """
        assert self._w is not None and self._alpha is not None
        binary = _binarise(reactions.reindex(columns=self._item_columns)).values
        N, J = binary.shape
        D = self.n_components

        z_init = np.zeros(D)
        theta_init = float(self._theta.mean()) if self._theta is not None else 0.0
        x0 = np.concatenate([z_init, [theta_init]])

        z_new = np.zeros((N, D))
        theta_new = np.zeros(N)

        for i in range(N):
            yi = binary[i]
            obs = ~np.isnan(yi)
            if not obs.any():
                z_new[i] = z_init
                theta_new[i] = theta_init
                continue
            y_obs = yi[obs].astype(int)
            w_obs = self._w[obs]
            a_obs = self._alpha[obs]
            b_obs = self._beta[obs]

            def neg_log_post(params):
                z = params[:D]
                theta = params[D]
                logit = theta * a_obs + b_obs - self._gamma * np.linalg.norm(
                    z[None, :] - w_obs, axis=1,
                )
                p = _sigmoid(logit)
                p = np.clip(p, 1e-9, 1 - 1e-9)
                nll = -(y_obs * np.log(p) + (1 - y_obs) * np.log(1 - p)).sum()
                # Negative log-prior: z ~ N(0, I) per dim, theta ~ N(0, sd^2).
                # Matches lsirm2pl's training prior so the MAP estimate stays
                # in the same regularised regime as the training posterior.
                neg_log_prior = 0.5 * np.dot(z, z) + 0.5 * (
                    theta / self.pr_sd_theta
                ) ** 2
                return nll + neg_log_prior

            res = minimize(neg_log_post, x0, method="L-BFGS-B")
            z_new[i] = res.x[:D]
            theta_new[i] = res.x[D]

        self._theta_new = theta_new
        return pd.DataFrame(
            z_new, index=reactions.index, columns=_embedding_columns(D),
        )

    def get_item_parameters(self) -> pd.DataFrame:
        assert self._w is not None
        cols = (
            [f"w{d + 1}" for d in range(self.n_components)]
            + ["alpha", "beta"]
        )
        data = np.column_stack([self._w, self._alpha, self._beta])
        return pd.DataFrame(data, index=self._item_columns, columns=cols)

    # --- Save / Load ---

    def save(self, path: Path, **kwargs) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "niter": self.niter, "nburn": self.nburn, "nthin": self.nthin,
                "seed": self.seed,
                "pr_sd_theta": self.pr_sd_theta,
                "embedding": self._embedding,
                "theta": self._theta,
                "w": self._w,
                "alpha": self._alpha,
                "beta": self._beta,
                "gamma": self._gamma,
                "item_columns": self._item_columns,
                "train_index": self._train_index,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "LSIRM":
        with open(Path(path) / "model.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls(
            n_components=data["n_components"],
            niter=data["niter"], nburn=data["nburn"], nthin=data["nthin"],
            seed=data["seed"],
            pr_sd_theta=data.get("pr_sd_theta", 1.0),
        )
        model._embedding = data["embedding"]
        model._theta = data["theta"]
        model._w = data["w"]
        model._alpha = data["alpha"]
        model._beta = data["beta"]
        model._gamma = data["gamma"]
        model._item_columns = data["item_columns"]
        model._train_index = data["train_index"]
        return model
