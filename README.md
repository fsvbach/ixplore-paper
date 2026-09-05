# IXPLORE: Bounded Ideal Point Estimation with Grid-Based Uncertainty Quantification

This repository contains the preprocessed datasets, the experiment scripts, the stored results, and the notebooks that generate every figure and table in the paper.

The IXPLORE package is available here:
[fsvbach/IXPLORE](https://github.com/fsvbach/IXPLORE), also on PyPI as `ixplore`.

The full paper is available on ArXiv:
[ArXiv link to be inserted]()

## Layout

```
data/        preprocessed datasets (reactions, user info, statements, weights)
src/         data loading, metrics, model wrappers, experiment scripts
results/     metrics of all experiments + the IXPLORE checkpoints the notebooks need
notebooks/   one notebook per analysis; each writes its figures/tables directly
figures/     PDF figures as included in the paper
tables/      LaTeX tables as included in the paper
```

## Setup

Python 3.14 with the packages in `requirements.txt` (versions used for the
paper are listed there). Install into a fresh environment:

```bash
pip install -r requirements.txt
```

The IRT baselines (IDEAL, emIRT, LSIRM) call R through `Rscript`, which must be
on the `PATH` with the packages `pscl`, `emIRT`, `lsirm12pl`, and optionally `wnominate` installed
(R 4.5.1 was used). R is only needed to rerun those baselines; all notebooks
work without it.

Run all commands from the repository root. The notebooks use paths relative to
their own directory (`../../results`, `../../figures`), so start Jupyter from
the root or from `notebooks/`.

## Reproducing the figures and tables

All figures and tables are produced by the ten notebooks below from the stored
`results/`. The six under `model_evaluation/` read the stored metrics and run
in about two minutes in total. The four under `ixplore_design/` refit small
IXPLORE models on Smartvote 2023 and take one to two minutes each.

```bash
for nb in notebooks/*/*.ipynb; do
  (cd "$(dirname "$nb")" && jupyter nbconvert --to notebook --execute --inplace "$(basename "$nb")")
done
```

## Reproducing the results from scratch

The stored `results/` were produced by the scripts in `src/scripts/`. Each
script writes `train_metrics.csv` and `test_metrics.csv` (one row per model,
sparsity level, and seed) into its own folder under `results/<dataset>/`.

Evaluation protocol, shared by all scripts: the train set is fitted at
sparsity u in {0, 0.3, 0.6, 0.9} (five seeds for u > 0) and evaluated on its
observed entries (reconstruction) and masked entries (imputation). The u = 0
model then embeds the test users at sparsity v over the same grid. Smartvote
2023 uses candidates as train and voters as test users. The other datasets hold
out 15 percent of users as the test set.

Baseline comparison, one run per dataset and algorithm:

```bash
python -m src.scripts.run_baseline --dataset smartvote_2023 --algorithm ixplore
```

Datasets: `smartvote_2019`, `smartvote_2023`, `voteview`, `polis`, `evs`.
Algorithms: `pca-linear`, `pca-logistic`, `kernel-pca`, `tsne-logistic`,
`umap-logistic`, `vae-2layer`, `vae-logistic`, `ideal`, `emirt`, `lsirm`,
`ixplore`, `ixplore-binarised`. The fitted baseline models themselves are not
stored in this repository, only their metrics.

W-NOMINATE was also run on Smartvote 2023 (`src/models/wnominate_wrapper.py`,
R package `wnominate`); its metrics are in
`results/smartvote_2023/baseline/wnominate/`. It is not part of the paper's
comparison, because its estimation does not handle response matrices with
many missing values well, and it is commented out in `src/models/__init__.py`.

Configuration analysis on Smartvote 2023:

```bash
python -m src.scripts.run_iteration_effect     # prior variance x init x iterations
python -m src.scripts.run_latent_convergence   # needs the iteration_effect checkpoints
python -m src.scripts.run_feature_effect       # linear / polynomial / RFF kernels
python -m src.scripts.run_weight_effect        # confidence-weight schedules x scaling
python -m src.scripts.run_pca_sweep            # PCA baselines over all dimensions
```

`run_latent_convergence` reads the fully fitted models that
`run_iteration_effect` saves under `results/smartvote_2023/iteration_effect/models/`.
Those six checkpoints (PCA init, u = 0, 20 iterations, one per prior variance)
and the three `feature_effect` kernel checkpoints are the only model files kept
in `results/`, because the notebooks load them.

The appendix experiments on point estimates, item updates, boundary
regularization, and sampling strategies run inside their notebooks and write
their intermediate results to `results/ixplore_design/` and
`results/smartvote_2023/posterior_effect/`.

## Datasets

All response values are rescaled to [0, 1]. User information files contain
only an id, a party label, and a plotting color.

| Dataset | Folder | Train users | Test users | Items | Scale | Missing |
|---|---|---|---|---|---|---|
| Smartvote 2019 | `data/smartvote/2019` | 1,912 candidates | -- | 75 | 4-Likert | 0% |
| Smartvote 2023 | `data/smartvote/2023` | 1,029 candidates | 1,121 voters | 75 | 4-Likert | 0% |
| Voteview (S117) | `data/voteview` | 102 senators | -- | 603 | binary | 5.2% |
| Polis (vTaiwan) | `data/polis` | 1,921 participants | -- | 197 | ternary | 86.9% |
| EVS (2017) | `data/evs/2020` | 35,116 respondents | -- | 32 | 10-Likert | 0% |

Sources:

- Smartvote 2019: Bachmann, Sarasua, and Bernstein (2024), "Fast and Adaptive
  Questionnaires for Voting Advice Applications".
- Smartvote 2023: Bachmann, van der Weijden, Sarasua, and Bernstein (2026),
  "Estimating the Recommendation Certainty in Candidate-Based Voting Advice
  Applications". `voters_weights.csv` holds the per-answer confidence weights
  used in the weight experiment.
- Voteview: Boche et al. (2018), "The new Voteview.com", 117th U.S. Senate.
- Polis: Small et al. (2021), "Polis: Scaling deliberation by mapping high
  dimensional opinion spaces", vTaiwan conversation.
- EVS: European Values Study 2017 (GESIS ZA7500), 32 numerical items from the
  Joint EVS/WVS 2017-2022 release (GESIS ZA7505). Only the preprocessed CSVs
  are included, not the original data file.
