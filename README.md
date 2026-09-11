# IXPLORE: Bounded Ideal Point Estimation with Grid-Based Uncertainty Quantification

This repository contains the preprocessed datasets, the experiment scripts, the stored results, and the scripts that generate every figure and table of the paper.

The IXPLORE package is available here:
[fsvbach/IXPLORE](https://github.com/fsvbach/IXPLORE), also on PyPI as `ixplore`.

## Layout

```
data/          preprocessed datasets (reactions, user info, statements)
src/           data loading, metrics, model wrappers
src/scripts/   experiment scripts; each writes its metrics to results/
src/figures/   one script per figure; each reads results/ and writes its PDF to figures/
src/tables/    the table script; reads results/ and writes the .tex files to tables/
results/       metrics of all experiments + the IXPLORE checkpoints
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
./reproduce_figures.sh
```

runs `python -m src.figures` followed by `python -m src.tables`. Any single
output can be regenerated on its own, for example
`python -m src.figures.prior_phase` or `python -m src.tables.baseline --dataset polis`.
Each script also prints the numbers quoted in the text that belong to its
figure or table (value ranges, crossover dimensions, t-tests, the selected item).

| Paper | Script | Output | Reads |
|---|---|---|---|
| Figure 1 | `src/figures/baseline_comparison.py` | `figures/baseline_comparison_smartvote_2023.pdf` | `results/smartvote_2023/baseline/` |
| Figure 2 | `src/figures/pca_dimensionality.py` | `figures/pca_dimensionality_mae_smartvote_2023.pdf` | `results/smartvote_2023/pca_sweep/`, `iteration_effect/` |
| Figure 3 | `src/figures/prior_embeddings.py` | `figures/prior_embeddings_smartvote_2023.pdf` | `results/smartvote_2023/iteration_effect/` incl. checkpoints |
| Figure 4 | `src/figures/prior_phase.py` | `figures/prior_phase_smartvote_2023.pdf` | `results/smartvote_2023/iteration_effect/` |
| Figure 5 | `src/figures/iteration_effect.py` | `figures/iteration_effect_smartvote_2023.pdf` | `results/smartvote_2023/iteration_effect/` |
| Figure 6 | `src/figures/kernel_comparison.py` | `figures/kernel_comparison_mae_smartvote_2023.pdf` | `results/smartvote_2023/feature_effect/`, `baseline/` |
| Figure 7 | `src/figures/kernels_visual.py` | `figures/kernels_visual_smartvote_2023.pdf` | `results/smartvote_2023/feature_effect/` incl. checkpoints |
| Tables 2, 4-7 | `src/tables/baseline.py` | `tables/baseline_<dataset>.tex`, all five datasets | `results/<dataset>/baseline/` |
| Replication material | `src/figures/baseline_train_test.py` | `figures/baseline_train_test_<dataset>.pdf`, all five datasets | `results/<dataset>/baseline/` |

Table 1 (datasets) and Table 3 (hyperparameters) are written by hand in the
paper source. The `baseline_train_test` figures are the per-dataset
counterparts of Figure 1 that the paper refers to as replication material.

Last verified on 2026-09-11 on macOS 26 (Apple Silicon) with Python 3.14.0:
`./reproduce_figures.sh` completes in under a minute.

## Reproducing the results from scratch

The stored `results/` were computed from `data/` by the four scripts in
`src/scripts/`. Each writes `train_metrics.csv` and `test_metrics.csv` (one
row per model, sparsity level, and seed) into its own folder under `results/`,
and the figure and table scripts read nothing else.

```bash
./reproduce_results.sh
```

runs all of them and skips result folders that already exist; delete a folder
to recompute it. The IRT baselines need R (see Setup).

Evaluation protocol, shared by all scripts: the train set is fitted at
sparsity u in {0, 0.3, 0.6, 0.9} (five seeds for u > 0) and evaluated on its
observed entries (reconstruction) and masked entries (imputation). The u = 0
model then embeds the test users at sparsity v over the same grid. Smartvote
2023 uses candidates as train and voters as test users. The other datasets hold
out 15 percent of users as the test set.

Each result folder is recreated by one command:

| Results folder | Command | Used by |
|---|---|---|
| `results/smartvote_2023/iteration_effect/` | `python -m src.scripts.run_iteration_effect` | Figures 2 to 5 |
| `results/smartvote_2023/feature_effect/` | `python -m src.scripts.run_feature_effect` | Figures 6 and 7 |
| `results/smartvote_2023/pca_sweep/` | `python -m src.scripts.run_pca_sweep` | Figure 2 |
| `results/<dataset>/baseline/<algorithm>/` | `python -m src.scripts.run_baseline --dataset <dataset> --algorithm <algorithm>` | Figures 1 and 6, Tables 2 and 4-7, replication figures |

Datasets: `smartvote_2019`, `smartvote_2023`, `voteview`, `polis`, `evs`.
Algorithms: `pca-linear`, `pca-logistic`, `kernel-pca`, `tsne-logistic`,
`umap-logistic`, `vae-2layer`, `vae-logistic`, `ideal`, `emirt`, `lsirm`,
`ixplore`, `ixplore-binarised`. IDEAL, LSIRM, and the logistic VAE were not
run on EVS; the EVS table shows `--` for them. The fitted baseline models
themselves are not stored in this repository, only their metrics. All four
scripts take `--n-seeds` and `--output-dir`; the configuration scripts also
take `--dataset`.

W-NOMINATE was also run on Smartvote 2023 (`src/models/wnominate_wrapper.py`,
R package `wnominate`); its metrics are in
`results/smartvote_2023/baseline/wnominate/`. It is not part of the paper's
comparison, because its estimation does not handle response matrices with
many missing values well (Section 4.3 of the paper), and it is commented out
in `src/models/__init__.py`. The stored metrics document this: only two seeds
per sparsity level completed, and at u = 0.9 the train reconstruction MAE
rises to 0.33 from about 0.21 at lower sparsity. It is not part of `reproduce_results.sh`.

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
  Applications".
- Voteview: Boche et al. (2018), "The new Voteview.com", 117th U.S. Senate.
- Polis: Small et al. (2021), "Polis: Scaling deliberation by mapping high
  dimensional opinion spaces", vTaiwan conversation.
- EVS: European Values Study 2017 (GESIS ZA7500), 32 numerical items from the
  Joint EVS/WVS 2017-2022 release (GESIS ZA7505).
