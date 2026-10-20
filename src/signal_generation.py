"""
Module 3: signal generation (dynamic EWMA bands).

Replaces the static z-score thresholds of the original paper (computed once on the
whole history) with an EWMA mean and standard deviation, re-estimated every hour.
The EWMA window is proportional to the pair's OU half-life (estimated on the
training period).

The EWMA bands are causal: the value at hour t only uses data up to t. The signal is
then shifted by EXECUTION_LAG_HOURS: a signal observed at t is only executed at t + lag
(you cannot trade at the price that triggered the signal).

Only pairs whose spread is stationary over the training period are traded:
stETH/ETH and USDC/USDT. AAVE/ETH is excluded (ADF not significant).
"""

import numpy as np
import pandas as pd

from config import EXECUTION_LAG_HOURS
from cointegration_ou import run_pipeline, principal_row

EWMA_HALFLIFE_MULTIPLIER = 10   # EWMA window = 10 x the pair's OU half-life
# Entry threshold chosen on the TRAINING period only
# (see sensitivity_frequency.py); the test period was not used to pick it.
ENTRY_Z = 4.0
EXIT_Z = 0.5
MIN_BURN_IN_HOURS = 48          # no signal until the EWMA has stabilised

TRADABLE_PAIRS = ["stETH_ETH", "USDC_USDT"]
EXCLUDED_PAIRS = {"AAVE_ETH": "ADF not significant on the training period"}


def ewma_bands(spread: pd.Series, halflife_hours: float):
    ewm = spread.ewm(halflife=halflife_hours, adjust=False)
    return ewm.mean(), ewm.std(bias=False)


def generate_signal(spread, mean, std, entry_z=ENTRY_Z, exit_z=EXIT_Z, burn_in=MIN_BURN_IN_HOURS):
    """State machine on the z-score:
       z >  entry_z -> position -1 (short the spread: sell Y, buy X)
       z < -entry_z -> position +1 (long the spread)
       |z| < exit_z -> exit (0); otherwise the position is kept."""
    z = (spread - mean) / std
    position = np.zeros(len(spread), dtype=int)
    pos = 0
    for i, zi in enumerate(z.values):
        if i < burn_in or not np.isfinite(zi):
            position[i] = 0
            continue
        if pos == 0:
            if zi > entry_z:
                pos = -1
            elif zi < -entry_z:
                pos = 1
        elif abs(zi) < exit_z:
            pos = 0
        position[i] = pos
    return z, pd.Series(position, index=spread.index, name="position")


def execute(signal_position: pd.Series, lag=EXECUTION_LAG_HOURS) -> pd.Series:
    """Position actually held: the signal of hour t is executed at t + lag."""
    return signal_position.shift(lag).fillna(0).astype(int)


def trade_stats(position: pd.Series):
    entries = (position != 0) & (position.shift(1).fillna(0) == 0)
    runs, run = [], 0
    for p in position:
        if p != 0:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return {
        "n_trades": int(entries.sum()),
        "pct_time_in_position": float((position != 0).mean()),
        "avg_duration_h": float(np.mean(runs)) if runs else 0.0,
    }


def build_positions(spread, half_life_h, entry_z=ENTRY_Z, exit_z=EXIT_Z, lag=EXECUTION_LAG_HOURS):
    halflife_ewma = half_life_h * EWMA_HALFLIFE_MULTIPLIER
    mean, std = ewma_bands(spread, halflife_ewma)
    z, signal = generate_signal(spread, mean, std, entry_z=entry_z, exit_z=exit_z)
    return {"z": z, "signal": signal, "position": execute(signal, lag),
            "ewma_mean": mean, "ewma_std": std, "halflife_ewma_h": halflife_ewma}


def run_signal_pipeline(entry_z=ENTRY_Z, exit_z=EXIT_Z, lag=EXECUTION_LAG_HOURS):
    results_df, spreads, spreads_all, prices = run_pipeline()
    out = {}
    for pair in TRADABLE_PAIRS:
        hl = float(principal_row(results_df, pair)["half_life_h"])
        d = build_positions(spreads[pair], hl, entry_z, exit_z, lag)
        d.update(spread=spreads[pair], ou_half_life_h=hl, stats=trade_stats(d["position"]))
        out[pair] = d
    return out, results_df, spreads, prices


if __name__ == "__main__":
    out, *_ = run_signal_pipeline()
    print(f"Thresholds: entry |z| > {ENTRY_Z}, exit |z| < {EXIT_Z}, execution lag = {EXECUTION_LAG_HOURS} h")
    for pair, d in out.items():
        s = d["stats"]
        print(f"\n=== {pair} ===")
        print(f"  OU half-life = {d['ou_half_life_h']:.2f} h -> EWMA window = {d['halflife_ewma_h']:.1f} h")
        print(f"  trades = {s['n_trades']} | time in position = {s['pct_time_in_position']*100:.1f}% | "
              f"average duration = {s['avg_duration_h']:.1f} h")
    print("\nExcluded pairs:", EXCLUDED_PAIRS)
