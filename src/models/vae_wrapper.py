"""VAE spatial model — thin wrapper around the ECML 2024 implementation.

Architecture and training loop are taken verbatim from:
  Bachmann, Sarasua & Bernstein, "Fast and Adaptive Questionnaires for
  Voting Advice Applications", ECML-PKDD 2024.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from ixplore.utils import compute_column_means, mean_impute
from src.models.base import SpatialModel


# ── ECML model components (verbatim) ─────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, columns, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(len(columns), 64)
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.columns = columns

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mean(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output):
        super(Decoder, self).__init__()
        self.columns = output
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, len(output))

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


class LogisticDecoder(nn.Module):
    def __init__(self, latent_dim, output):
        super(LogisticDecoder, self).__init__()
        self.columns = output
        self.linear = nn.Linear(latent_dim, len(output))

    def forward(self, z):
        return torch.sigmoid(self.linear(z))


class _VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(_VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.columns = decoder.columns

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def embed(self, reactions):
        with torch.no_grad():
            mu, _ = self.encoder(torch.tensor(reactions).float())
        return mu.numpy()


def MaskedLoss(recon_x, x, mu, logvar, mask, beta=1):
    MSE = F.mse_loss(recon_x * mask, x * mask, reduction="sum") / mask.mean()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD


def _impute(reactions: pd.DataFrame, column_means: np.ndarray | None = None) -> pd.DataFrame:
    """Mean-impute, falling back to 0.5 for fully-empty columns."""
    X = reactions.values.astype(float)
    if column_means is None:
        column_means = compute_column_means(X, fill_empty_columns=0.5)
    return pd.DataFrame(
        mean_impute(X, column_means=column_means),
        index=reactions.index,
        columns=reactions.columns,
    )


def _train_vae(encoder_cls, decoder_cls, reactions, beta=1, patience=300,
               lr=1e-3, test_size=0.1, k=2, random_state=0, max_iters=None):
    """ECML training loop.

    Verbatim from ECML 2024 except for the optional ``max_iters`` hard cap on
    optimizer steps. The cap only changes behaviour when the patience-based
    early stopping fails to trigger within ``max_iters`` steps; this matters
    for ``beta=0`` (logistic decoder), where the absent KLD regulariser lets
    the validation loss keep improving marginally and reset the patience
    counter indefinitely. Set ``max_iters=None`` to recover the exact ECML
    loop. Any run with a finite cap must be flagged in the report.
    """
    torch.manual_seed(random_state)
    imputed_reactions = _impute(reactions)
    train, evals, train_mask, eval_mask = train_test_split(
        imputed_reactions, reactions, test_size=test_size, random_state=0,
    )
    train = torch.tensor(train.values).float()
    evals = torch.tensor(evals.values).float()
    train_mask = torch.tensor(~train_mask.isna().values).float()
    eval_mask = torch.tensor(~eval_mask.isna().values).float()

    encoder = encoder_cls(reactions.columns, k)
    decoder = decoder_cls(k, reactions.columns)
    vae = _VAE(encoder, decoder)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    vae.eval_losses = []
    i = stop = 0
    best_state = None

    while np.min(([0] * patience + vae.eval_losses)[-patience:]) <= stop:
        if max_iters is not None and i >= max_iters:
            break
        optimizer.zero_grad()
        train_batch, mu, logvar = vae(train)
        train_loss = MaskedLoss(train_batch, train, mu, logvar, train_mask, beta)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            eval_batch, mu, logvar = vae(evals)
            eval_loss = MaskedLoss(eval_batch, evals, mu, logvar, eval_mask, beta)
            vae.eval_losses.append(eval_loss.item() / evals.shape[0])
            stop = np.min(vae.eval_losses)

            if vae.eval_losses[-1] == stop:
                best_state = {k: v.clone() for k, v in vae.state_dict().items()}
        i += 1

    if best_state is not None:
        vae.load_state_dict(best_state)
    return vae


# ── SpatialModel adapter ─────────────────────────────────────────────────────

class VAE(SpatialModel):
    """VAE baseline from ECML 2024, adapted to the SpatialModel interface."""

    name = "VAE"

    def __init__(
        self,
        n_components: int = 2,
        beta: float = 1.0,
        lr: float = 1e-3,
        patience: int = 300,
        val_fraction: float = 0.1,
        decoder_type: str = "2-layer",
        random_state: int = 0,
        max_iters: int | None = None,
    ):
        self.n_components = n_components
        self.beta = beta
        self.lr = lr
        self.patience = patience
        self.val_fraction = val_fraction
        self.decoder_type = decoder_type
        self.random_state = random_state
        self.max_iters = max_iters

        self._vae: _VAE | None = None
        self._embedding: pd.DataFrame | None = None
        self._train_col_means: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, reactions: pd.DataFrame) -> None:
        encoder_cls = Encoder
        decoder_cls = LogisticDecoder if self.decoder_type == "logistic" else Decoder

        self._vae = _train_vae(
            encoder_cls, decoder_cls, reactions,
            beta=self.beta, patience=self.patience, lr=self.lr,
            test_size=self.val_fraction, k=self.n_components,
            random_state=self.random_state, max_iters=self.max_iters,
        )

        # Store column means for test-time imputation (matching ECML)
        self._train_col_means = compute_column_means(
            reactions.values.astype(float), fill_empty_columns=0.5,
        )

        # Store training embeddings (posterior mean)
        train_data = _impute(reactions, column_means=self._train_col_means)
        emb = self._vae.embed(train_data.values)
        self._embedding = pd.DataFrame(
            emb, index=reactions.index, columns=["x", "y"],
        )

    # ----------------------------------------------------------- interface
    def get_embedding(self) -> pd.DataFrame:
        return self._embedding

    def predict(self, embedding: np.ndarray) -> np.ndarray:
        assert self._vae is not None
        with torch.no_grad():
            z = torch.tensor(embedding, dtype=torch.float32)
            preds = self._vae.decoder(z)
        return preds.numpy()

    def embed(self, reactions: pd.DataFrame) -> pd.DataFrame:
        assert self._vae is not None
        imputed = _impute(reactions, column_means=self._train_col_means)
        emb = self._vae.embed(imputed.values)
        return pd.DataFrame(emb, index=reactions.index, columns=["x", "y"])

    # -------------------------------------------------------- persistence
    def save(self, path: Path, **kwargs) -> None:
        assert self._vae is not None
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(
                {
                    "n_components": self.n_components,
                    "beta": self.beta,
                    "lr": self.lr,
                    "patience": self.patience,
                    "val_fraction": self.val_fraction,
                    "decoder_type": self.decoder_type,
                    "vae_state": self._vae.state_dict(),
                    "columns": self._vae.columns,
                    "embedding": self._embedding,
                    "train_col_means": self._train_col_means,
                    "eval_losses": getattr(self._vae, "eval_losses", None),
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "VAE":
        with open(Path(path) / "model.pkl", "rb") as f:
            data = pickle.load(f)
        model = cls(
            n_components=data["n_components"],
            beta=data["beta"],
            lr=data["lr"],
            patience=data["patience"],
            val_fraction=data["val_fraction"],
            decoder_type=data["decoder_type"],
        )
        model._train_col_means = data["train_col_means"]
        model._embedding = data["embedding"]

        columns = data["columns"]
        encoder = Encoder(columns, data["n_components"])
        decoder: nn.Module
        if data["decoder_type"] == "logistic":
            decoder = LogisticDecoder(data["n_components"], columns)
        else:
            decoder = Decoder(data["n_components"], columns)
        vae = _VAE(encoder, decoder)
        vae.load_state_dict(data["vae_state"])
        vae.eval()
        model._vae = vae
        return model
