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
    prior_embeddings,
    prior_phase,
)

MODULES = [
    baseline_comparison,
    baseline_train_test,
    kernel_comparison,
    kernels_visual,
    iteration_effect,
    prior_phase,
    prior_embeddings,
    pca_dimensionality,
]


def main():
    for module in MODULES:
        print(f"== {module.__name__}")
        module.main()


if __name__ == "__main__":
    main()
