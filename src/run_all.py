"""Runs the whole pipeline in order (except the data download)."""

import runpy
from pathlib import Path

STEPS = [
    "cointegration_ou.py",
    "signal_generation.py",
    "sensitivity_frequency.py",
    "backtest_engine.py",
    "portfolio_optimization.py",
    "make_figures.py",
]

if __name__ == "__main__":
    here = Path(__file__).parent
    for step in STEPS:
        print(f"\n{'=' * 80}\n{step}\n{'=' * 80}")
        runpy.run_path(str(here / step), run_name="__main__")
