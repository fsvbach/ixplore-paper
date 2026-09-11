#!/usr/bin/env bash
# Recompute the stored results/ from the data: the three configuration
# analyses on Smartvote 2023, then the baseline comparison for every dataset
# and algorithm. Run from anywhere:  ./reproduce_results.sh
#
# A result folder that already contains train_metrics.csv is skipped, so the
# script can be restarted after an interruption. Delete a folder under
# results/ to have it recomputed. See README.md for the requirements (R for
# the IRT baselines) and for recreating a single result.
set -euo pipefail
cd "$(dirname "$0")"

DATASETS="voteview smartvote_2023 smartvote_2019 polis evs"
ALGORITHMS="pca-linear pca-logistic kernel-pca tsne-logistic umap-logistic
            vae-2layer vae-logistic ideal emirt lsirm ixplore ixplore-binarised"
# Not part of the stored results: too expensive on the 35,116 EVS respondents.
EVS_SKIP="ideal lsirm vae-logistic"

echo "== configuration analysis on smartvote_2023"
for experiment in iteration_effect feature_effect pca_sweep; do
  if [ -f "results/smartvote_2023/$experiment/train_metrics.csv" ]; then
    echo "skip $experiment (results exist)"
    continue
  fi
  python -m "src.scripts.run_$experiment"
done

echo "== baseline comparison"
for dataset in $DATASETS; do
  for algorithm in $ALGORITHMS; do
    if [ "$dataset" = "evs" ] && [[ " $EVS_SKIP " == *" $algorithm "* ]]; then
      echo "skip $dataset/$algorithm (not run on EVS)"
      continue
    fi
    if [ -f "results/$dataset/baseline/$algorithm/train_metrics.csv" ]; then
      echo "skip $dataset/$algorithm (results exist)"
      continue
    fi
    python -m src.scripts.run_baseline --dataset "$dataset" --algorithm "$algorithm"
  done
done
echo "== done: results/ complete"
