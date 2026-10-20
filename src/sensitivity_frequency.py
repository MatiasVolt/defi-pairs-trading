"""
Robustness study: net Sharpe ratio (after frictions) as a function of the entry threshold.

Hypothesis tested: a higher threshold lowers trading frequency, and therefore the weight
of costs (fixed gas, proportional fees and slippage) relative to the captured PnL.

The threshold is CHOSEN on the training period only (average net Sharpe of the two
pairs). Test-period results are shown next to it to check that the choice holds out of
sample; they are not used for the selection.
"""

import pandas as pd

from config import EXECUTION_LAG_HOURS, OUTPUT_DIR, TOTAL_CAPITAL_USD, ensure_output_dir
from cointegration_ou import load_volumes, principal_row, run_pipeline
from backtest_engine import PAIR_VOLUME_ASSET, backtest_pair, split_metrics
from signal_generation import ENTRY_Z, TRADABLE_PAIRS, build_positions

ENTRY_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def run_sensitivity(lag=EXECUTION_LAG_HOURS):
    results_df, spreads, _, prices = run_pipeline()
    volumes = load_volumes()
    notional = TOTAL_CAPITAL_USD / len(TRADABLE_PAIRS)
    rows = []
    for pair in TRADABLE_PAIRS:
        hl = float(principal_row(results_df, pair)["half_life_h"])
        for entry_z in ENTRY_GRID:
            exit_z = min(0.5, entry_z / 4)
            d = build_positions(spreads[pair], hl, entry_z, exit_z, lag)
            bt = backtest_pair(pair, spreads[pair], d["position"], notional,
                               volumes[PAIR_VOLUME_ASSET[pair]], prices["ETH"])
            for m in split_metrics(bt, notional):
                rows.append({"pair": pair, "entry_z": entry_z, **m})
    return pd.DataFrame(rows)


def select_threshold(df):
    """Threshold maximising the average net Sharpe of the pairs, on the training period only."""
    train_net = df[(df.period == "train") & (df["mode"] == "net")]
    return train_net.groupby("entry_z")["sharpe"].mean().idxmax()


if __name__ == "__main__":
    ensure_output_dir()
    df = run_sensitivity()
    df.to_csv(OUTPUT_DIR / "sensitivity_frequency.csv", index=False)

    view = (df[df["mode"] == "net"]
            .pivot_table(index=["pair", "entry_z"], columns="period", values=["sharpe", "n_trades"])
            .round(2))
    pd.set_option("display.width", 160)
    print("Net Sharpe ratio and number of trades, by period:\n")
    print(view.to_string())

    best = select_threshold(df)
    print(f"\nThreshold selected on the training period: {best}  (value used in signal_generation.py: {ENTRY_Z})")
    if best != ENTRY_Z:
        print("WARNING: update ENTRY_Z in signal_generation.py.")
