"""Regenerate every generated table of the paper from the stored results.

tables/datasets.tex and tables/hyperparameter.tex are written by hand and
are not touched.

Usage:
    python -m src.tables
"""

from src.tables import baseline, point_estimate_boundary, point_estimate_mae, posterior_effect, weights

MODULES = [baseline, weights, posterior_effect, point_estimate_mae, point_estimate_boundary]


def main():
    for module in MODULES:
        print(f"== {module.__name__}")
        module.main()


if __name__ == "__main__":
    main()
