"""One script per table of the paper.

Every module exposes ``main()`` and writes its LaTeX table to ``tables/``
from the stored ``results/``. Run one table with ``python -m src.tables.<name>``
or all of them with ``python -m src.tables``.
"""

from pathlib import Path

from src.paths import ROOT, TABLES_DIR


def write_table(name: str, text: str) -> Path:
    """Write *text* to tables/<name> and report the path."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / name
    path.write_text(text)
    print(f"Wrote {path.relative_to(ROOT)}")
    return path
