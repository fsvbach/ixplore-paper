"""Shared helpers for R-backed model wrappers."""

from __future__ import annotations

import subprocess
import tempfile

import pandas as pd


def run_r_script(script: str) -> subprocess.CompletedProcess:
    """Write the script to a temp file and execute it via Rscript."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(script)
        f.flush()
        result = subprocess.run(
            ["Rscript", f.name], capture_output=True, text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Rscript failed:\n{result.stderr}")
    return result


def binarise(reactions: pd.DataFrame) -> pd.DataFrame:
    """Convert [0,1] reactions to {0, 1, NaN} at threshold 0.5; 0.5 treated as missing."""
    binary = reactions.mask(reactions == 0.5)
    return (binary > 0.5).astype(float).where(binary.notna())
