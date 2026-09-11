"""Boundary regularization: Gaussian against log-barrier prior on Smartvote 2023.

Part 1 trains IXPLORE on the candidates (PCA initialisation, five iterations)
under the Gaussian prior at five variances and under the log-barrier prior at
five strengths, keeping the model state after every iteration, and records
the fit metrics per iteration in
results/ixplore_design/boundary_regularization_fit_metrics.csv.

Part 2 embeds the voters as new users with 5..75 observed answers under two
Gaussian priors, two log-barrier priors, and a uniform prior, with the item
parameters of the corresponding fully trained model held fixed, and records
the fit MAE on the observed and the impute MAE on the held-out answers in
results/ixplore_design/boundary_regularization_holdout.csv.

Usage:
    python -m src.scripts.run_boundary_regularization
"""

import copy
import logging
import warnings

import numpy as np
import pandas as pd
from ixplore import IXPLORE
from ixplore.prior import set_gaussian_prior, set_log_barrier_prior

from src.data import load_dataset
from src.paths import RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "ixplore_design"

GAUSSIAN_TAUS = [0.05, 0.1, 0.25, 0.5, 1.0]
LOG_BARRIER_ALPHAS = [0.1, 0.2, 0.5, 2.0, 5.0]
N_ITERATIONS = 5
N_OBSERVED_LIST = [5, 10, 15, 20, 30, 50, 75]


def apply_gaussian(model, tau):
    """Swap a model's prior to Gaussian with variance tau."""
    model.log_prior = set_gaussian_prior(model.X, prior_variance=tau, log=True)


def apply_log_barrier(model, alpha):
    """Swap a model's prior to the log-barrier prior with strength alpha."""
    model.log_prior = set_log_barrier_prior(model.X, alpha=alpha, limits=model.limits, log=True)


def uniform_prior(model):
    model.log_prior = np.zeros(model.X.shape[0])


def train_gaussian(reactions, tau):
    """Train with Gaussian prior, keeping the model state after each iteration."""
    model = IXPLORE(reactions, pca_initialization=True, random_state=0,
                    prior_variance=tau, scale_weights=False)
    models_by_iter = {}
    for i in range(1, N_ITERATIONS + 1):
        model.iterate(1)
        models_by_iter[i] = copy.deepcopy(model)
    return models_by_iter


def train_log_barrier(reactions, alpha):
    """Train with log-barrier prior, keeping the model state after each iteration."""
    model = IXPLORE(reactions, pca_initialization=True, random_state=0, scale_weights=False)
    apply_log_barrier(model, alpha)
    models_by_iter = {}
    for i in range(1, N_ITERATIONS + 1):
        model.iterate(1)
        apply_log_barrier(model, alpha)
        models_by_iter[i] = copy.deepcopy(model)
    return models_by_iter


def embedding_progression(reactions):
    """Part 1: per-iteration fit metrics under both priors."""
    print("Training Gaussian models...")
    gaussian_models = {s: train_gaussian(reactions, s) for s in GAUSSIAN_TAUS}
    print("Training log-barrier models...")
    log_barrier_models = {a: train_log_barrier(reactions, a) for a in LOG_BARRIER_ALPHAS}

    fit_rows = []
    for tau, mdict in gaussian_models.items():
        for it, m in mdict.items():
            metrics = m.evaluate(boundary_threshold=0.10)
            fit_rows.append({"Prior": "Gaussian", "param": tau, "iteration": it, **metrics})
    for alpha, mdict in log_barrier_models.items():
        for it, m in mdict.items():
            metrics = m.evaluate(boundary_threshold=0.10)
            fit_rows.append({"Prior": "Log-barrier", "param": alpha, "iteration": it, **metrics})

    fit_metrics_df = pd.DataFrame(fit_rows)
    out = OUTPUT_DIR / "boundary_regularization_fit_metrics.csv"
    fit_metrics_df.round(4).to_csv(out, index=False)
    print(f"Saved {len(fit_metrics_df)} rows to {out}")
    return gaussian_models, log_barrier_models


def holdout_accuracy(reactions, voters, gaussian_models, log_barrier_models):
    """Part 2: fit and impute MAE of new users with n observed answers per prior."""
    rng = np.random.default_rng(42)
    n_items = len(reactions.columns)

    def make_eval_model(ref_model, reg_fn):
        """Clone item params from ref_model and apply the given prior."""
        eval_model = IXPLORE(reactions, pretrained_models=ref_model.get_parameters())
        reg_fn(eval_model)
        return eval_model

    reg_configs = [
        ("Gaussian $\\tau^2=1.0$", lambda m: apply_gaussian(m, 1.0), gaussian_models[1.0][N_ITERATIONS]),
        ("Gaussian $\\tau^2=0.1$", lambda m: apply_gaussian(m, 0.1), gaussian_models[0.1][N_ITERATIONS]),
        ("Log-barrier $\\alpha=0.5$", lambda m: apply_log_barrier(m, 0.5), log_barrier_models[0.5][N_ITERATIONS]),
        ("Log-barrier $\\alpha=2.0$", lambda m: apply_log_barrier(m, 2.0), log_barrier_models[2.0][N_ITERATIONS]),
        ("Uniform (no prior)", uniform_prior, gaussian_models[1.0][N_ITERATIONS]),
    ]

    results = []
    for reg_name, reg_fn, ref_model in reg_configs:
        eval_model = make_eval_model(ref_model, reg_fn)
        print(f"Evaluating: {reg_name}")
        for n_obs in N_OBSERVED_LIST:
            fit_maes, impute_maes = [], []
            for voter_idx in range(len(voters)):
                voter_series = voters.iloc[voter_idx]
                voter_values = voter_series.values

                observed_idx = rng.choice(n_items, size=min(n_obs, n_items), replace=False)
                held_out_idx = np.setdiff1d(np.arange(n_items), observed_idx)

                partial = pd.Series(np.nan, index=voter_series.index, dtype=float)
                partial.iloc[observed_idx] = voter_values[observed_idx]

                # Posterior-mean predictions for ALL items: fit MAE uses observed
                # entries, impute MAE uses held-out entries.
                preds = eval_model.predict_answers(partial).values
                fit_maes.append(np.abs(voter_values[observed_idx] - preds[observed_idx]).mean())
                if len(held_out_idx) > 0:
                    impute_maes.append(np.abs(voter_values[held_out_idx] - preds[held_out_idx]).mean())

            results.append({
                "n_observed": n_obs,
                "regularization": reg_name,
                "fit_mae": np.mean(fit_maes),
                "impute_mae": np.mean(impute_maes) if impute_maes else np.nan,
            })

    results_df = pd.DataFrame(results)
    out = OUTPUT_DIR / "boundary_regularization_holdout.csv"
    results_df.round(4).to_csv(out, index=False)
    print(f"Saved {len(results_df)} rows to {out}")
    return results_df


def main():
    warnings.filterwarnings("ignore")
    logging.getLogger("ixplore").setLevel(logging.WARNING)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("smartvote_2023")
    reactions = ds.train_reactions
    voters = ds.test_reactions
    print(f"Dataset: {reactions.shape[0]} candidates, {voters.shape[0]} voters, {reactions.shape[1]} questions")

    gaussian_models, log_barrier_models = embedding_progression(reactions)
    results_df = holdout_accuracy(reactions, voters, gaussian_models, log_barrier_models)

    print("\nFit MAE (on observed answers):")
    print(results_df.pivot(index="n_observed", columns="regularization", values="fit_mae")
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nImpute MAE (on held-out answers):")
    print(results_df.pivot(index="n_observed", columns="regularization", values="impute_mae")
          .to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
