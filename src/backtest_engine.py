"""
Module 5 : moteur de backtest, avec et sans frictions de marché DeFi.

À la place du frais fixe de 0,1 % de l'article original, trois frictions :
  * frais de swap selon le palier du pool AMM (0,01 % / 0,05 %) ;
  * slippage selon la formule à produit constant (x*y=k) ;
  * coût de gas stochastique (Gwei) sur Ethereum mainnet.

Hypothèses (à garder en tête en lisant les résultats) :
  - aucune donnée de profondeur de pool n'est disponible : la profondeur est
    approchée par un multiple du volume 24h CoinGecko. C'est une approximation
    grossière (volume toutes places confondues, pas la liquidité d'un pool) ;
  - prix du gas tiré d'une loi log-normale centrée sur ~25 Gwei ;
  - capital total : 100 000 USD.

Les performances sont reportées séparément sur la période d'apprentissage et sur la
période de test (out-of-sample), seule cette dernière mesurant la performance réelle.
"""

import numpy as np
import pandas as pd

from config import (EXECUTION_LAG_HOURS, PERIODS_PER_YEAR, RANDOM_SEED, TOTAL_CAPITAL_USD,
                    train_mask, test_mask)

FRICTION_PARAMS = {
    "stETH_ETH": {"fee_tier": 0.0005},   # 0,05 % : palier Uniswap V3 des actifs corrélés
    "USDC_USDT": {"fee_tier": 0.0001},   # 0,01 % : palier des pools de stablecoins
}
PAIR_VOLUME_ASSET = {"stETH_ETH": "stETH", "USDC_USDT": "USDC"}  # jambe servant de proxy de liquidité

POOL_DEPTH_VOLUME_MULTIPLIER = 10   # profondeur supposée = 10 x volume 24h (hypothèse, cf. README)
GAS_UNITS_PER_SWAP = 150_000        # ordre de grandeur d'un swap Uniswap V3
GAS_PRICE_BASE_GWEI = 25.0
GAS_PRICE_SIGMA = 0.5


# ---------------------------------------------------------------------------
# 1. PnL brut
# ---------------------------------------------------------------------------

def raw_returns(spread: pd.Series, position: pd.Series) -> pd.Series:
    """r_t = position_{t-1} * (spread_t - spread_{t-1}) (spread en log-prix).
    `position` est la position effectivement détenue (déjà décalée du délai d'exécution)."""
    return position.shift(1).fillna(0).astype(float) * spread.diff().fillna(0.0)


# ---------------------------------------------------------------------------
# 2. Frictions
# ---------------------------------------------------------------------------

def _trade_events(position: pd.Series) -> pd.Series:
    """Nombre de jambes de position modifiées à chaque heure (ouverture ou clôture = 1)."""
    return position.diff().fillna(position.iloc[0]).abs().clip(upper=1).astype(int)


def constant_product_slippage(notional_usd, pool_depth_usd):
    """Impact de prix d'un swap sur un pool x*y=k : notional / (notional + réserve)."""
    pool_depth_usd = pool_depth_usd.clip(lower=1.0)
    return notional_usd / (notional_usd + pool_depth_usd)


def stochastic_gas_cost_usd(n_events, eth_price, rng, n_legs=2):
    z = rng.standard_normal(n_events)
    gas_gwei = GAS_PRICE_BASE_GWEI * np.exp(GAS_PRICE_SIGMA * z - GAS_PRICE_SIGMA ** 2 / 2)
    return gas_gwei * 1e-9 * GAS_UNITS_PER_SWAP * eth_price * n_legs


def apply_frictions(position, notional_usd, fee_tier, volume_usd, eth_price, rng):
    """Coût total (USD) heure par heure : 2 jambes à chaque ouverture/clôture."""
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
# 3. Backtest d'une paire
# ---------------------------------------------------------------------------

def backtest_pair(pair, spread, position, notional_usd, volume_usd, eth_price, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    raw_pnl = raw_returns(spread, position) * notional_usd
    frictions = apply_frictions(position, notional_usd, FRICTION_PARAMS[pair]["fee_tier"],
                                volume_usd, eth_price, rng)
    return {"raw_pnl_usd": raw_pnl, "net_pnl_usd": raw_pnl - frictions["total_friction"],
            "frictions": frictions, "position": position}


# ---------------------------------------------------------------------------
# 4. Métriques
# ---------------------------------------------------------------------------

def performance_metrics(pnl_usd: pd.Series, position: pd.Series, capital_usd: float) -> dict:
    equity = capital_usd + pnl_usd.cumsum()
    returns = pnl_usd / capital_usd
    std = returns.std()
    sharpe = returns.mean() / std * np.sqrt(PERIODS_PER_YEAR) if std > 0 else np.nan
    drawdown = equity / equity.cummax() - 1

    # PnL par trade (épisodes consécutifs en position, frais de sortie inclus)
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
    n_wins, n_losses = int((trade_pnls > 0).sum()), int((trade_pnls < 0).sum())

    return {
        "pnl_usd": pnl_usd.sum(),
        "rendement_pct": pnl_usd.sum() / capital_usd * 100,
        "sharpe": sharpe,
        "max_dd_pct": drawdown.min() * 100,
        "n_trades": len(trade_pnls),
        "win_rate_pct": n_wins / len(trade_pnls) * 100 if len(trade_pnls) else np.nan,
        "equity_curve": equity,
    }


def split_metrics(bt, notional):
    """Métriques brutes et nettes, sur l'apprentissage et sur le test."""
    idx = bt["position"].index
    rows = []
    for period, mask in [("apprentissage", train_mask(idx)), ("test", test_mask(idx))]:
        for mode, key in [("brut", "raw_pnl_usd"), ("net", "net_pnl_usd")]:
            m = performance_metrics(bt[key][mask], bt["position"][mask], notional)
            rows.append({"periode": period, "mode": mode,
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
                all_rows.append({"paire": pair, "delai_h": lag, **row})
    df = pd.DataFrame(all_rows)
    print("Capital équipondéré entre les paires. delai_h = délai entre signal et exécution.\n")
    print(df.round(2).to_string(index=False))
    df.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
