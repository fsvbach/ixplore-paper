"""Regenerate every generated table of the paper from the stored results.

Table 1 (datasets) and Table 3 (hyperparameters) are written by hand in the
paper source and are not produced here.

Usage:
    python -m src.tables
"""

from src.tables import baseline

MODULES = [baseline]


def main():
    for module in MODULES:
        print(f"== {module.__name__}")
        module.main()


if __name__ == "__main__":
    main()
