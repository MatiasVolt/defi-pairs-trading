"""
Global settings shared by every module.

The key point is the time split:
  - TRAINING period (in-sample): the oldest 2/3 of the history. Everything that is
    estimated (hedge ratio, ADF test, Ornstein-Uhlenbeck parameters, entry threshold,
    portfolio weights) is estimated on this period ONLY;
  - TEST period (out-of-sample): the most recent 1/3, never used to calibrate anything.
    It is the only period whose performance matters.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_FRAC = 2 / 3            # share of the history used for calibration
EXECUTION_LAG_HOURS = 1       # delay between signal and trade execution (0 = trade at the signal price)
TOTAL_CAPITAL_USD = 100_000.0 # assumed total capital
RANDOM_SEED = 42              # reproducibility of the stochastic gas draws
PERIODS_PER_YEAR = 24 * 365   # hourly data


def split_timestamp(index):
    """First timestamp of the test period."""
    return index[int(len(index) * TRAIN_FRAC)]


def train_mask(index):
    return index < split_timestamp(index)


def test_mask(index):
    return index >= split_timestamp(index)


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    return OUTPUT_DIR
