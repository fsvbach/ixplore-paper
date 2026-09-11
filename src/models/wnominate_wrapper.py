"""W-NOMINATE model wrapper using the wnominate R package.

Calls ``wnominate::wnominate(rollcall(...), dims=2, polarity=...)`` via Rscript,
reads back legislator coordinates plus per-roll-call midpoints and spreads,
the dimension weights, and the utility scale beta.

Predictions are computed in Python using the W-NOMINATE Gaussian utility
(Poole & Rosenthal; see Poole 2007 Sec. 1):

    u_yea = beta * exp(-0.5 * sum_k w_k^2 * (x_k - z_yea_k)^2)
    u_nay = beta * exp(-0.5 * sum_k w_k^2 * (x_k - z_nay_k)^2)
    P(yea) = exp(u_yea) / (exp(u_yea) + exp(u_nay))

with z_yea = midpoint - spread/2 and z_nay = midpoint + spread/2.

W-NOMINATE drops legislators (too few votes) and roll calls (lopsided /
non-polarising) by its own internal rules. Those dropped rows/columns have
no estimated parameters, so this wrapper predicts a constant 0.5 for them
rather than fabricating coordinates. ``minvotes`` defaults to 1 so the
drop set is as small as wnominate allows.

Embedding new users (out-of-sample) solves W-NOMINATE's step 2: holding the
item parameters fixed, maximise each user's log-likelihood over their 2D
coordinate using scipy.optimize.
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
library(pscl)
library(wnominate)

reactions <- read.csv("{reactions_path}", header=TRUE, row.names=1, check.names=FALSE)
# Pass row/column names through rollcall() - otherwise wnominate relabels
# legislators "Legislator 1..N" and rollcalls "1..J", losing alignment.
rc <- rollcall(reactions,
               legis.names=rownames(reactions),
               vote.names=colnames(reactions))
fit <- wnominate(rc, dims={dims}, polarity={polarity_r},
                 minvotes={minvotes}, lop={lop}, trials={trials}, verbose=FALSE)

coord_cols <- paste0("coord", 1:{dims}, "D")
write.csv(fit$legislators[, coord_cols, drop=FALSE], "{coord_path}", row.names=TRUE)

mid_cols    <- paste0("midpoint", 1:{dims}, "D")
spread_cols <- paste0("spread",   1:{dims}, "D")
write.csv(fit$rollcalls[, c(mid_cols, spread_cols), drop=FALSE], "{rc_path}", row.names=TRUE)

write.csv(data.frame(beta=as.numeric(fit$beta)), "{scalars_path}", row.names=FALSE)
write.csv(data.frame(weight=as.numeric(fit$weights)), "{weights_path}", row.names=FALSE)
"""


def _polarity_r_literal(polarity: int | list[int], dims: int) -> str:
    """Render polarity as an R vector literal of length `dims`."""
    if isinstance(polarity, int):
        polarity = [polarity] * dims
    if len(polarity) != dims:
        raise ValueError(f"polarity length {len(polarity)} != dims {dims}")
    return "c(" + ", ".join(str(p) for p in polarity) + ")"


def _wn_choice_probs(
    embedding: np.ndarray,
    midpoints: np.ndarray,
    spreads: np.ndarray,
    weights: np.ndarray,
    beta: float,
) -> np.ndarray:
    """P(yea) for each (user, item) under the W-NOMINATE Gaussian-utility model.

    embedding: (N, D)
    midpoints, spreads: (J, D)
    weights: (D,)
    beta: scalar utility scale
    Returns: (N, J) array of P(yea) in (0, 1).
    """
    yea = midpoints - spreads / 2.0  # (J, D)
    nay = midpoints + spreads / 2.0
    w2 = (weights ** 2)  # (D,)
    d2_yea = ((embedding[:, None, :] - yea[None, :, :]) ** 2) * w2
    d2_nay = ((embedding[:, None, :] - nay[None, :, :]) ** 2) * w2
    u_yea = beta * np.exp(-0.5 * d2_yea.sum(axis=-1))
    u_nay = beta * np.exp(-0.5 * d2_nay.sum(axis=-1))
    m = np.maximum(u_yea, u_nay)
    e_yea = np.exp(u_yea - m)
    e_nay = np.exp(u_nay - m)
    return e_yea / (e_yea + e_nay)


class WNOMINATE(SpatialModel):
    """W-NOMINATE (Poole & Rosenthal 1985, 1991) via the ``wnominate`` R package.

    Parameters
    ----------
    n_components : int
        Latent dimensionality (default 2). ``wnominate`` defaults to 2.
    polarity : int | list[int] | None
        Row index (1-based) of the legislator made positive in each
        dimension. When None, the rows with the most observations are
        chosen at fit time so they survive wnominate's ``minvotes`` filter.
    minvotes, lop, trials :
        Forwarded to ``wnominate::wnominate``. ``minvotes`` defaults to 1
        and ``lop`` to 0.0 so wnominate drops as few rows/columns as its
        algorithm permits; whatever it still drops is predicted as 0.5.
    """

    name = "W-NOMINATE"

    def __init__(
        self,
        n_components: int = 2,
        polarity: int | list[int] | None = None,
        minvotes: int = 1,
        lop: float = 0.0,
        trials: int = 3,
        max_sparsity: float = 0.85,
    ):
        # polarity=None -> pick legislators with most observations in fit()
        # (guarantees they survive minvotes filter, no matter the sparsity).
        # max_sparsity: above this missing fraction the matrix is below what
        # W-NOMINATE's alternating estimator can meaningfully (or tractably)
        # fit; fit() then short-circuits to a degenerate 0.5-everywhere model
        # instead of grinding for many minutes per seed.
        self.n_components = n_components
        self.polarity = polarity
        self.minvotes = minvotes
        self.lop = lop
        self.trials = trials
        self.max_sparsity = max_sparsity
        self._degenerate = False
        self._embedding: pd.DataFrame | None = None
        self._midpoints: np.ndarray | None = None  # (J, D)
        self._spreads: np.ndarray | None = None    # (J, D)
        self._weights: np.ndarray | None = None    # (D,)
        self._beta: float | None = None
        self._item_columns: pd.Index | None = None
        # Per-item training positive rate, used as the mean-imputation
        # fallback for rows/cols W-NOMINATE cannot estimate.
        self._item_means: np.ndarray | None = None  # (J,) in [0, 1]
        # Boolean masks marking rows/cols W-NOMINATE dropped (no estimate).
        self._dropped_users: np.ndarray | None = None   # (N,) True = dropped
        self._dropped_items: np.ndarray | None = None    # (J,) True = dropped

    @staticmethod
    def _compute_item_means(reactions: pd.DataFrame) -> np.ndarray:
        """Per-item positive rate on binarised training data.

        Columns with no observed entries fall back to 0.5.
        """
        binary = _binarise(reactions).to_numpy(dtype=float)  # {0, 1, NaN}
        with np.errstate(invalid="ignore"):
            means = np.nanmean(binary, axis=0)
        return np.where(np.isnan(means), 0.5, means)

    def _set_degenerate(self, reactions: pd.DataFrame) -> None:
        """Set up a model whose predictions are per-item mean-imputation.

        Used when the data is too sparse for W-NOMINATE to fit. Every user
        and item is marked dropped, so predict() falls back to the per-item
        training positive rate (mean-imputation) - a fairer "no-model"
        baseline than a flat 0.5.
        """
        D = self.n_components
        J = len(reactions.columns)
        self._degenerate = True
        self._embedding = pd.DataFrame(
            0.0, index=reactions.index, columns=_embedding_columns(D), dtype=float
        )
        self._midpoints = np.zeros((J, D))
        self._spreads = np.zeros((J, D))
        self._weights = np.ones(D)
        self._beta = 1.0
        self._item_columns = reactions.columns
        self._item_means = self._compute_item_means(reactions)
        self._dropped_users = np.ones(len(reactions.index), dtype=bool)
        self._dropped_items = np.ones(J, dtype=bool)

    def fit(self, reactions: pd.DataFrame) -> None:
        # Per-item means are the mean-imputation fallback for anything
        # W-NOMINATE can't estimate (dropped rows/cols, or too-sparse data).
        self._item_means = self._compute_item_means(reactions)

        # Too sparse for W-NOMINATE -> mean-imputation model, skip the R call.
        observed_frac = float((~reactions.isna()).to_numpy().mean())
        if (1.0 - observed_frac) > self.max_sparsity:
            self._set_degenerate(reactions)
            return

        # Polarity legislators must survive wnominate's minvotes filter,
        # otherwise the R-side check fails with "polarity is incorrectly
        # specified". When polarity is None, pick the rows with the most
        # observations - those are guaranteed to survive.
        if self.polarity is None:
            obs_per_user = (~reactions.isna()).sum(axis=1)
            top_rows = obs_per_user.sort_values(ascending=False).index[: self.n_components]
            polarity = [int(reactions.index.get_loc(r)) + 1 for r in top_rows]
        else:
            polarity = self.polarity

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            reactions_path = tmp / "reactions.csv"
            coord_path = tmp / "coords.csv"
            rc_path = tmp / "rollcalls.csv"
            scalars_path = tmp / "scalars.csv"
            weights_path = tmp / "weights.csv"

            _binarise(reactions).to_csv(reactions_path)

            script = _R_FIT_SCRIPT.format(
                reactions_path=reactions_path,
                dims=self.n_components,
                polarity_r=_polarity_r_literal(polarity, self.n_components),
                minvotes=self.minvotes,
                lop=self.lop,
                trials=self.trials,
                coord_path=coord_path,
                rc_path=rc_path,
                scalars_path=scalars_path,
                weights_path=weights_path,
            )
            _run_r_script(script)

            coords = pd.read_csv(coord_path, index_col=0)
            rcs = pd.read_csv(rc_path, index_col=0)
            scalars = pd.read_csv(scalars_path)
            weights = pd.read_csv(weights_path)

        # --- Legislator coordinates --------------------------------------
        # W-NOMINATE drops legislators that fail minvotes. Dropped users get
        # no coordinate; we mark them and predict 0.5 for them later. We
        # still need *some* numeric placeholder so predict() doesn't choke;
        # 0.0 is fine because the mask overrides those predictions anyway.
        coords.columns = _embedding_columns(self.n_components)
        embedding = pd.DataFrame(
            0.0, index=reactions.index, columns=coords.columns, dtype=float
        )
        # CSV round-trip may flip integer-looking indices ("9","16") to int64
        # while reactions.index is object-typed string. Compare on string keys
        # so the alignment is robust to either side's dtype.
        coords_keys = coords.index.astype(str)
        target_keys = embedding.index.astype(str)
        # Index.isin() returns a plain ndarray (no .to_numpy()); keep it as one.
        present_mask = np.asarray(target_keys.isin(coords_keys))
        if present_mask.any():
            coords.index = coords_keys
            embedding.loc[present_mask] = coords.loc[target_keys[present_mask]].values
        self._embedding = embedding
        self._dropped_users = ~present_mask

        # --- Item parameters ---------------------------------------------
        # wnominate ignores vote.names and labels rollcalls "1".."J" by
        # position; re-key to original column names via that 1-based index.
        # Dropped roll calls get no parameters -> mark them, predict 0.5.
        mid_cols = [f"midpoint{d + 1}D" for d in range(self.n_components)]
        sp_cols = [f"spread{d + 1}D" for d in range(self.n_components)]
        full_mid = pd.DataFrame(0.0, index=reactions.columns, columns=mid_cols)
        full_sp = pd.DataFrame(0.0, index=reactions.columns, columns=sp_cols)

        rcs_int_pos = rcs.index.astype(str).astype(int) - 1
        kept_cols = reactions.columns[rcs_int_pos]
        full_mid.loc[kept_cols] = rcs[mid_cols].values
        full_sp.loc[kept_cols] = rcs[sp_cols].values

        dropped_items = np.ones(len(reactions.columns), dtype=bool)
        dropped_items[rcs_int_pos.to_numpy()] = False
        self._dropped_items = dropped_items

        self._item_columns = reactions.columns
        self._midpoints = full_mid.values
        self._spreads = full_sp.values
        self._weights = weights["weight"].to_numpy(dtype=float)
        self._beta = float(scalars["beta"].iloc[0])

    # --- SpatialModel interface ---

    def get_embedding(self) -> pd.DataFrame:
        assert self._embedding is not None
        return self._embedding

    def _predict_raw(self, embedding: np.ndarray) -> np.ndarray:
        return _wn_choice_probs(
            np.asarray(embedding, dtype=float),
            self._midpoints,
            self._spreads,
            self._weights,
            self._beta,
        )

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        """P(yea); per-item mean-imputation for anything W-NOMINATE dropped.

        Dropped roll calls (columns) have no item parameters and dropped
        legislators (rows) have no coordinate - neither is estimable under
        W-NOMINATE's own filtering rules. Rather than fabricate a
        coordinate, those cells fall back to the per-item training positive
        rate (mean-imputation), a fairer "no-model" baseline than 0.5.
        """
        assert self._midpoints is not None
        preds = self._predict_raw(embedding)
        item_means = (
            self._item_means
            if self._item_means is not None
            else np.full(preds.shape[1], 0.5)
        )
        # Dropped items: every user gets that item's training mean.
        if self._dropped_items is not None and self._dropped_items.any():
            preds[:, self._dropped_items] = item_means[self._dropped_items]
        # Dropped users: only applies when predicting on the training
        # embedding (row count matches and row order is the training order).
        if (
            self._dropped_users is not None
            and self._dropped_users.any()
            and preds.shape[0] == self._dropped_users.shape[0]
        ):
            preds[self._dropped_users, :] = item_means[None, :]
        return preds

    def embed(self, reactions: pd.DataFrame) -> pd.DataFrame:
        """Per-user MLE: argmax_x sum_j log P(y_ij | x, item params).

        Mirrors W-NOMINATE's step-2 update (Poole 2007 Sec. 1) but for a single
        user at a time, with item parameters fixed at the fitted values.
        Observations on dropped roll calls are ignored (no parameters).
        """
        assert self._midpoints is not None
        D = self.n_components
        if self._degenerate:
            # No item parameters were estimated -> no information to embed on.
            return pd.DataFrame(
                0.0, index=reactions.index, columns=_embedding_columns(D),
                dtype=float,
            )
        binary = _binarise(reactions).values  # (N, J), {0, 1, NaN}
        N = binary.shape[0]
        coords = np.zeros((N, D))

        # Start each user at the mean of the (non-dropped) training rows.
        valid = ~self._dropped_users if self._dropped_users is not None else slice(None)
        x0 = np.asarray(self._embedding.values[valid].mean(axis=0))
        if not np.all(np.isfinite(x0)):
            x0 = np.zeros(D)

        keep_item = (
            ~self._dropped_items
            if self._dropped_items is not None
            else np.ones(binary.shape[1], dtype=bool)
        )

        for i in range(N):
            yi = binary[i]
            obs = (~np.isnan(yi)) & keep_item  # ignore dropped roll calls
            if not obs.any():
                coords[i] = x0
                continue
            y_obs = yi[obs].astype(int)
            mid_obs = self._midpoints[obs]
            sp_obs = self._spreads[obs]

            def neg_ll(x):
                p = _wn_choice_probs(
                    x[None, :], mid_obs, sp_obs, self._weights, self._beta,
                )[0]
                p = np.clip(p, 1e-9, 1 - 1e-9)
                return -(y_obs * np.log(p) + (1 - y_obs) * np.log(1 - p)).sum()

            res = minimize(neg_ll, x0, method="L-BFGS-B")
            coords[i] = res.x

        return pd.DataFrame(
            coords, index=reactions.index, columns=_embedding_columns(D)
        )

    def get_item_parameters(self) -> pd.DataFrame:
        assert self._midpoints is not None
        D = self.n_components
        cols = (
            [f"midpoint{d + 1}D" for d in range(D)]
            + [f"spread{d + 1}D" for d in range(D)]
        )
        data = np.column_stack([self._midpoints, self._spreads])
        return pd.DataFrame(data, index=self._item_columns, columns=cols)

    # --- Save / Load ---

    def save(self, path: Path, **kwargs) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "n_components": self.n_components,
                "polarity": self.polarity,
                "minvotes": self.minvotes,
                "lop": self.lop,
                "trials": self.trials,
                "max_sparsity": self.max_sparsity,
                "degenerate": self._degenerate,
                "embedding": self._embedding,
                "midpoints": self._midpoints,
                "spreads": self._spreads,
                "weights": self._weights,
                "beta": self._beta,
                "item_columns": self._item_columns,
                "item_means": self._item_means,
                "dropped_users": self._dropped_users,
                "dropped_items": self._dropped_items,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "WNOMINATE":
        with open(Path(path) / "model.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls(
            n_components=data["n_components"],
            polarity=data["polarity"],
            minvotes=data["minvotes"],
            lop=data["lop"],
            trials=data["trials"],
            max_sparsity=data.get("max_sparsity", 0.85),
        )
        model._degenerate = data.get("degenerate", False)
        model._embedding = data["embedding"]
        model._midpoints = data["midpoints"]
        model._spreads = data["spreads"]
        model._weights = data["weights"]
        model._beta = data["beta"]
        model._item_columns = data["item_columns"]
        model._item_means = data.get("item_means")
        model._dropped_users = data["dropped_users"]
        model._dropped_items = data["dropped_items"]
        return model
