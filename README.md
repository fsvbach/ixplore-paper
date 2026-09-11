# IXPLORE: Bounded Ideal Point Estimation with Grid-Based Uncertainty Quantification

This repository contains the preprocessed datasets, the experiment scripts, the stored results, and the scripts that generate every figure and table in the paper.

The IXPLORE package is available here:
[fsvbach/IXPLORE](https://github.com/fsvbach/IXPLORE), also on PyPI as `ixplore`.

A preprint of the full paper is available on ArXiv:
[IXPLORE Report](http://arxiv.org/abs/2609.06018)

## Layout

```
data/          preprocessed datasets (reactions, user info, statements, weights)
src/           data loading, metrics, model wrappers
src/scripts/   experiment scripts; each writes its metrics to results/
src/figures/   one script per figure; each reads results/ and writes its PDF to figures/
src/tables/    one script per table; each reads results/ and writes its .tex to tables/
results/       metrics of all experiments + the IXPLORE checkpoints the figure scripts need
figures/       PDF figures as included in the paper
tables/        LaTeX tables as included in the paper
```

## Setup

Python 3.14 with the packages in `requirements.txt` (versions used for the
paper are listed there). Install into a fresh environment:

```bash
pip install -r requirements.txt
```

The IRT baselines (IDEAL, emIRT, LSIRM) call R through `Rscript`, which must be
on the `PATH` with the packages `pscl`, `emIRT`, `lsirm12pl`, and optionally `wnominate` installed
(R 4.5.1 was used). R is only needed to rerun those baselines; the figure and
table scripts work without it.

All scripts resolve their paths relative to the repository, so they can be
run from any working directory. The commands below assume the repository root.

## Reproducing the figures and tables

Every figure and table is produced from the stored `results/` by one script
in `src/figures/` or `src/tables/`:

```bash
./reproduce.sh
```

runs `python -m src.figures` followed by `python -m src.tables`. Any single
output can be regenerated on its own, for example
`python -m src.figures.prior_phase` or `python -m src.tables.baseline --dataset polis`.
Each script also prints the numbers quoted in the text that belong to its
figure or table (t-tests, crossover dimensions, convergence summary).

| Script | Output |
|---|---|
| `src/figures/baseline_comparison.py` | `figures/baseline_comparison_smartvote_2023.pdf` |
| `src/figures/baseline_train_test.py` | `figures/baseline_train_test_<dataset>.pdf`, all five datasets |
| `src/figures/kernel_comparison.py` | `figures/kernel_comparison_mae_smartvote_2023.pdf` |
| `src/figures/kernels_visual.py` | `figures/kernels_visual_smartvote_2023.pdf` |
| `src/figures/iteration_effect.py` | `figures/iteration_effect_smartvote_2023.pdf` |
| `src/figures/prior_phase.py` | `figures/prior_phase_smartvote_2023.pdf` |
| `src/figures/prior_embeddings.py` | `figures/prior_embeddings_smartvote_2023.pdf` |
| `src/figures/posterior_trajectory.py` | `figures/posterior_trajectory_smartvote_2023.pdf` |
| `src/figures/weights_geometry.py` | `figures/weights_geometry_smartvote_2023.pdf` |
| `src/figures/weights_distortion.py` | `figures/weights_distortion_smartvote_2023.pdf` |
| `src/figures/pca_dimensionality.py` | `figures/pca_dimensionality_mae_smartvote_2023.pdf` |
| `src/figures/prior_families.py` | `figures/prior_families_smartvote_2023.pdf` |
| `src/figures/sampling_strategies.py` | `figures/sampling_strategies_smartvote_2023.pdf` |
| `src/tables/baseline.py` | `tables/baseline_<dataset>.tex`, all five datasets |
| `src/tables/weights.py` | `tables/weights_smartvote_2023.tex` |
| `src/tables/posterior_effect.py` | `tables/posterior_effect_smartvote_2023.tex` |
| `src/tables/point_estimate_mae.py` | `tables/point_estimate_mae_smartvote_2023.tex` |
| `src/tables/point_estimate_boundary.py` | `tables/point_estimate_boundary_smartvote_2023.tex` |

`tables/datasets.tex` and `tables/hyperparameter.tex` are written by hand.

Last verified on 2026-09-11 on macOS 26 (Apple Silicon) with Python 3.14.0:
`./reproduce.sh` completes in under a minute and regenerates the committed
figures and tables exactly (pixel-identical PDFs, byte-identical `.tex`).

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
in `results/`, because the figure scripts load them.

The four small experiments of the design and appendix sections fit IXPLORE
models on Smartvote 2023 in one to two minutes each and write to
`results/ixplore_design/` and `results/smartvote_2023/posterior_effect/`:

```bash
python -m src.scripts.run_point_estimates          # MAP vs. posterior mean vs. full posterior
python -m src.scripts.run_boundary_regularization  # Gaussian vs. log-barrier prior
python -m src.scripts.run_posterior_effect         # point-estimate vs. uncertainty-weighted item update
python -m src.scripts.run_sampling_strategies      # Rasch vs. posterior sampling of pseudo-answers
```

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
  Joint EVS/WVS 2017-2022 release (GESIS ZA7505).
