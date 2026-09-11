"""Regenerate every figure of the paper from the stored results.

Usage:
    python -m src.figures
"""

from src.figures import (
    baseline_comparison,
    baseline_train_test,
    iteration_effect,
    kernel_comparison,
    kernels_visual,
    pca_dimensionality,
    posterior_trajectory,
    prior_embeddings,
    prior_families,
    prior_phase,
    sampling_strategies,
    weights_distortion,
    weights_geometry,
)

MODULES = [
    baseline_comparison,
    baseline_train_test,
    kernel_comparison,
    kernels_visual,
    iteration_effect,
    prior_phase,
    prior_embeddings,
    posterior_trajectory,
    weights_geometry,
    weights_distortion,
    pca_dimensionality,
    prior_families,
    sampling_strategies,
]


def main():
    for module in MODULES:
        print(f"== {module.__name__}")
        module.main()


if __name__ == "__main__":
    main()
