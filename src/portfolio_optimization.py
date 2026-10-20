"""
Module 5: capital allocation between pairs (return / CVaR).

Replaces the original paper's "return / variance" objective with "return / CVaR":
it penalises the size of the worst losses rather than average dispersion, which better
fits the main risk of an arbitrage strategy (a sudden break: depeg, correlation shock).

With 2 lines (stETH/ETH and USDC/USDT), a grid search on w in [0, 1] is exact and
transparent. Weights are chosen on the training period (returns net of frictions) and
then evaluated on the test period.
"""

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, PERIODS_PER_YEAR, ensure_output_dir, train_mask, test_mask
from backtest_engine import run_backtests

ALPHA_CVAR = 0.05
N_GRID = 101


def historical_cvar(returns: pd.Series, alpha=ALPHA_CVAR) -> float:
    """Mean of the worst alpha% returns, returned as a positive loss."""
    tail = returns[returns <= returns.quantile(alpha)]
    return -tail.mean() if len(tail) else 0.0


def net_return_matrix():
    """Hourly net returns of each pair, as a % of the line's notional (reference notional =
    equal-weight case; since gas is a fixed USD cost, this is an approximation)."""
    res = run_backtests()
    return pd.DataFrame({p: r["bt"]["net_pnl_usd"] / r["notional"] for p, r in res.items()})


def portfolio_stats(port_ret: pd.Series) -> dict:
    std = port_ret.std()
    mean_ann = port_ret.mean() * PERIODS_PER_YEAR
    cvar = historical_cvar(port_ret)
    return {
        "annualized_return": mean_ann,
        "annualized_vol": std * np.sqrt(PERIODS_PER_YEAR),
        "cvar_5pct_hourly": cvar,
        "sharpe": port_ret.mean() / std * np.sqrt(PERIODS_PER_YEAR) if std > 0 else np.nan,
        # return per unit of tail risk (both hourly, CVaR not annualised)
        "return_over_cvar": port_ret.mean() / cvar if cvar > 0 else np.nan,
    }


def frontier(returns: pd.DataFrame, n_grid=N_GRID) -> pd.DataFrame:
    a, b = returns.columns
    rows = []
    for w in np.linspace(0, 1, n_grid):
        rows.append({f"w_{a}": w, f"w_{b}": 1 - w,
                     **portfolio_stats(w * returns[a] + (1 - w) * returns[b])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    ensure_output_dir()
    R = net_return_matrix()
    a, b = R.columns
    tr, te = train_mask(R.index), test_mask(R.index)

    f_train = frontier(R[tr])
    f_train.to_csv(OUTPUT_DIR / "frontier_train.csv", index=False)
    best = f_train.loc[f_train["return_over_cvar"].idxmax()]
    criterion = "max return/CVaR"
    if best["annualized_return"] <= 0:
        # With a negative return, maximising return/CVaR rewards RISK (a larger denominator
        # makes the ratio "less negative"). The criterion then makes no sense: fall back on
        # the minimum-CVaR portfolio.
        best = f_train.loc[f_train["cvar_5pct_hourly"].idxmin()]
        criterion = "minimum CVaR (no portfolio with a positive net return)"
    w = best[f"w_{a}"]
    print(f"Weights selected on the training period ({criterion}): {a} = {w:.0%}, {b} = {1 - w:.0%}\n")

    rows = []
    for label, weight in [("optimized", w), ("equal weight", 0.5)]:
        for period, mask in [("train", tr), ("test", te)]:
            rows.append({"portfolio": label, "period": period,
                         **portfolio_stats(weight * R[a][mask] + (1 - weight) * R[b][mask])})
    out = pd.DataFrame(rows)
    out["cvar_5pct_hourly"] *= 1e4   # in basis points
    out = out.rename(columns={"cvar_5pct_hourly": "cvar_5pct_hourly_bp"})
    print(out.round(3).to_string(index=False))
    print("\nNote: optimization does not create performance, it only spreads risk across lines.")
