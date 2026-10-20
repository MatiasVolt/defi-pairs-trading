"""
Module 4: backtest engine, with and without DeFi market frictions.

Instead of the original paper's flat 0.1% fee, three frictions are modelled:
  * swap fees set by the AMM pool fee tier (0.01% / 0.05%);
  * slippage from the constant-product formula (x*y=k);
  * stochastic gas cost (Gwei) on Ethereum mainnet.

Assumptions (keep them in mind when reading the results):
  - no pool-depth data is available: depth is proxied by a multiple of the CoinGecko
    24h volume. This is a rough approximation (volume across all venues, not the
    liquidity of a given pool);
  - gas price drawn from a log-normal distribution centred around ~25 Gwei;
  - total capital: 100,000 USD.

Performance is reported separately on the training and test (out-of-sample) periods;
only the latter measures real performance.
"""

import numpy as np
import pandas as pd

from config import (EXECUTION_LAG_HOURS, PERIODS_PER_YEAR, RANDOM_SEED, TOTAL_CAPITAL_USD,
                    train_mask, test_mask)

FRICTION_PARAMS = {
    "stETH_ETH": {"fee_tier": 0.0005},   # 0.05%: Uniswap V3 tier for correlated assets
    "USDC_USDT": {"fee_tier": 0.0001},   # 0.01%: stablecoin pool tier
}
PAIR_VOLUME_ASSET = {"stETH_ETH": "stETH", "USDC_USDT": "USDC"}  # leg used as liquidity proxy

POOL_DEPTH_VOLUME_MULTIPLIER = 10   # assumed depth = 10 x 24h volume (assumption, see README)
GAS_UNITS_PER_SWAP = 150_000        # typical order of magnitude for a Uniswap V3 swap
GAS_PRICE_BASE_GWEI = 25.0
GAS_PRICE_SIGMA = 0.5


# ---------------------------------------------------------------------------
# 1. Gross PnL
# ---------------------------------------------------------------------------

def raw_returns(spread: pd.Series, position: pd.Series) -> pd.Series:
    """r_t = position_{t-1} * (spread_t - spread_{t-1}) (spread in log prices).
    `position` is the position actually held (already shifted by the execution lag)."""
    return position.shift(1).fillna(0).astype(float) * spread.diff().fillna(0.0)


# ---------------------------------------------------------------------------
# 2. Frictions
# ---------------------------------------------------------------------------

def _trade_events(position: pd.Series) -> pd.Series:
    """1 at each hour where the position changes (opening or closing), 0 otherwise."""
    return position.diff().fillna(position.iloc[0]).abs().clip(upper=1).astype(int)


def constant_product_slippage(notional_usd, pool_depth_usd):
    """Price impact of a swap on an x*y=k pool: notional / (notional + reserve)."""
    pool_depth_usd = pool_depth_usd.clip(lower=1.0)
    return notional_usd / (notional_usd + pool_depth_usd)


def stochastic_gas_cost_usd(n_events, eth_price, rng, n_legs=2):
    z = rng.standard_normal(n_events)
    gas_gwei = GAS_PRICE_BASE_GWEI * np.exp(GAS_PRICE_SIGMA * z - GAS_PRICE_SIGMA ** 2 / 2)
    return gas_gwei * 1e-9 * GAS_UNITS_PER_SWAP * eth_price * n_legs


def apply_frictions(position, notional_usd, fee_tier, volume_usd, eth_price, rng):
    """Total cost (USD) hour by hour: 2 legs at each opening/closing."""
    events = _trade_events(position)
    idx = events[events == 1].index
    costs = pd.DataFrame(0.0, index=position.index, columns=["fee_cost", "slippage_cost", "gas_cost"])
    if len(idx):
        slip = constant_product_slippage(notional_usd, volume_usd.reindex(idx) * POOL_DEPTH_VOLUME_MULTIPLIER)
        costs.loc[idx, "fee_cost"] = notional_usd * fee_tier * 2
        costs.loc[idx, "slippage_cost"] = (notional_usd * slip * 2).values
        costs.loc[idx, "gas_cost"] = stochastic_gas_cost_usd(len(idx), eth_price.reindex(idx).values, rng)
    costs["total_friction"] = costs.sum(axis=1)
    return costs


# ---------------------------------------------------------------------------
# 3. Backtest of one pair
# ---------------------------------------------------------------------------

def backtest_pair(pair, spread, position, notional_usd, volume_usd, eth_price, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    raw_pnl = raw_returns(spread, position) * notional_usd
    frictions = apply_frictions(position, notional_usd, FRICTION_PARAMS[pair]["fee_tier"],
                                volume_usd, eth_price, rng)
    return {"raw_pnl_usd": raw_pnl, "net_pnl_usd": raw_pnl - frictions["total_friction"],
            "frictions": frictions, "position": position}


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def performance_metrics(pnl_usd: pd.Series, position: pd.Series, capital_usd: float) -> dict:
    equity = capital_usd + pnl_usd.cumsum()
    returns = pnl_usd / capital_usd
    std = returns.std()
    sharpe = returns.mean() / std * np.sqrt(PERIODS_PER_YEAR) if std > 0 else np.nan
    drawdown = equity / equity.cummax() - 1

    # PnL per trade (consecutive hours in position, exit costs included)
    trade_pnls, cur, in_pos = [], 0.0, False
    for p, pl in zip(position.values, pnl_usd.values):
        if p != 0 or in_pos:
            cur += pl
        if p != 0:
            in_pos = True
        elif in_pos:
            trade_pnls.append(cur)
            cur, in_pos = 0.0, False
    if in_pos:
        trade_pnls.append(cur)
    trade_pnls = np.array(trade_pnls)
    n_wins = int((trade_pnls > 0).sum())

    return {
        "pnl_usd": pnl_usd.sum(),
        "return_pct": pnl_usd.sum() / capital_usd * 100,
        "sharpe": sharpe,
        "max_dd_pct": drawdown.min() * 100,
        "n_trades": len(trade_pnls),
        "win_rate_pct": n_wins / len(trade_pnls) * 100 if len(trade_pnls) else np.nan,
        "equity_curve": equity,
    }


def split_metrics(bt, notional):
    """Gross and net metrics, on the training and test periods."""
    idx = bt["position"].index
    rows = []
    for period, mask in [("train", train_mask(idx)), ("test", test_mask(idx))]:
        for mode, key in [("gross", "raw_pnl_usd"), ("net", "net_pnl_usd")]:
            m = performance_metrics(bt[key][mask], bt["position"][mask], notional)
            rows.append({"period": period, "mode": mode,
                         **{k: v for k, v in m.items() if k != "equity_curve"}})
    return rows


def run_backtests(lag=EXECUTION_LAG_HOURS, weights=None):
    from cointegration_ou import load_volumes
    from signal_generation import run_signal_pipeline

    out, _, _, prices = run_signal_pipeline(lag=lag)
    volumes = load_volumes()
    weights = weights or {p: 1 / len(out) for p in out}
    results = {}
    for pair, d in out.items():
        notional = TOTAL_CAPITAL_USD * weights[pair]
        bt = backtest_pair(pair, d["spread"], d["position"], notional,
                           volumes[PAIR_VOLUME_ASSET[pair]], prices["ETH"])
        results[pair] = {"bt": bt, "notional": notional, "signal": d}
    return results


if __name__ == "__main__":
    from config import OUTPUT_DIR, ensure_output_dir

    pd.set_option("display.width", 200)
    ensure_output_dir()
    all_rows = []
    for lag in [0, EXECUTION_LAG_HOURS]:
        for pair, r in run_backtests(lag=lag).items():
            for row in split_metrics(r["bt"], r["notional"]):
                all_rows.append({"pair": pair, "lag_h": lag, **row})
    df = pd.DataFrame(all_rows)
    print("Capital split equally between pairs. lag_h = delay between signal and execution.\n")
    print(df.round(2).to_string(index=False))
    df.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
