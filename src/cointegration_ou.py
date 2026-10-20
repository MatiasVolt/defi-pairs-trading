"""
Module 2: cointegration and spread modelling (Ornstein-Uhlenbeck).

Replaces the static Engle-Granger approach of the original paper with:
  1) a hedge ratio estimated by OLS on log prices, or fixed at 1 for pairs that are
     economically pegged 1:1 (stETH/ETH, USDC/USDT);
  2) an Augmented Dickey-Fuller test (implemented by hand with numpy) to check that
     the spread is stationary;
  3) an Ornstein-Uhlenbeck process fitted on the spread to estimate its mean-reversion
     speed (kappa), long-run mean (theta), volatility (sigma) and half-life.

Every estimate uses the TRAINING period only (see config.py). The spread is then
extended over the test period with these frozen parameters; a "control" ADF test is
computed on the test period for information only.
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, ensure_output_dir, train_mask, test_mask

ASSET_FILES = {
    "ETH": "ethereum.csv",
    "stETH": "steth.csv",
    "USDC": "usdc.csv",
    "USDT": "usdt.csv",
    "AAVE": "aave.csv",
}

# (name, asset Y, asset X, fixed beta or None if estimated by OLS)
# stETH/ETH and USDC/USDT are pegged 1:1: beta=1 is the main spread, OLS is kept for
# comparison. AAVE/ETH has no peg: OLS only.
PAIRS = [
    ("stETH_ETH", "stETH", "ETH", 1.0),
    ("AAVE_ETH", "AAVE", "ETH", None),
    ("USDC_USDT", "USDC", "USDT", 1.0),
]

DT_HOURS = 1.0


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def _load_column(column, data_dir=DATA_DIR):
    series = {}
    for key, fname in ASSET_FILES.items():
        df = pd.read_csv(data_dir / fname)
        df = df.sort_values("timestamp_ms").drop_duplicates(subset="timestamp_ms")
        df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        series[key] = df.set_index("datetime_utc")[column]
    return pd.DataFrame(series)


def load_prices(data_dir=DATA_DIR):
    """Prices of the 5 assets, aligned on the same timestamps (checked)."""
    prices = _load_column("price_usd", data_dir)
    if prices.isna().any().any():
        raise ValueError("Misaligned series: timestamps are missing after the join.")
    return prices


def load_volumes(data_dir=DATA_DIR):
    """Rolling 24h volume (CoinGecko), used as a liquidity proxy by the backtest."""
    return _load_column("volume_usd", data_dir).dropna()


# ---------------------------------------------------------------------------
# 2. Hedge ratio
# ---------------------------------------------------------------------------

def ols_hedge_ratio(y_log, x_log):
    """y = alpha + beta * x by least squares (first step of Engle-Granger)."""
    X = np.column_stack([np.ones(len(x_log)), x_log])
    (alpha, beta), *_ = np.linalg.lstsq(X, y_log, rcond=None)
    return alpha, beta


def fixed_beta_hedge_ratio(y_log, x_log, beta=1.0):
    """Fixed ratio; alpha = mean of the raw spread (to centre it like OLS)."""
    return float(np.mean(y_log - beta * x_log)), beta


# ---------------------------------------------------------------------------
# 3. ADF test (hand-written implementation)
# ---------------------------------------------------------------------------

# Asymptotic MacKinnon critical values, regression with constant and no trend.
# Note: for a spread built from an estimated OLS regression, the correct critical values
# are Engle-Granger's (more negative, ~ -3.90 at 1%). For fixed-beta spreads nothing is
# estimated and the standard ADF values apply.
ADF_CRIT = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def adf_test(series, lags=1):
    """Delta x_t = a + rho * x_{t-1} + sum_i gamma_i * Delta x_{t-i} + eps_t
    H0: rho = 0 (unit root); H1: rho < 0 (stationary)."""
    x = np.asarray(series)
    dx = np.diff(x)
    n = len(dx) - lags

    y = dx[lags:]
    X_cols = [np.ones(n), x[lags:-1]]
    for i in range(1, lags + 1):
        X_cols.append(dx[lags - i:len(dx) - i])
    X = np.column_stack(X_cols)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma2 = (resid @ resid) / (n - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t_stat = coef[1] / se[1]

    if t_stat < ADF_CRIT["1%"]:
        verdict = "stationary (1%)"
    elif t_stat < ADF_CRIT["5%"]:
        verdict = "stationary (5%)"
    elif t_stat < ADF_CRIT["10%"]:
        verdict = "stationary (10%)"
    else:
        verdict = "NOT stationary"
    return {"adf_stat": t_stat, "verdict": verdict}


# ---------------------------------------------------------------------------
# 4. Ornstein-Uhlenbeck process
# ---------------------------------------------------------------------------

def fit_ou(spread, dt=DT_HOURS):
    """dX_t = kappa (theta - X_t) dt + sigma dW_t, estimated through the exact AR(1)
    regression on the discretised solution:
        X_t = a + b X_{t-1} + eps_t,  b = exp(-kappa dt),  a = theta (1 - b)."""
    x = np.asarray(spread)
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    (a, b), *_ = np.linalg.lstsq(X, x[1:], rcond=None)
    sigma_eps = (x[1:] - X @ np.array([a, b])).std(ddof=2)

    if not (0 < b < 1):
        return {"kappa": np.nan, "theta": np.nan, "sigma_ou": np.nan, "half_life_hours": np.nan}

    kappa = -np.log(b) / dt
    return {
        "kappa": kappa,
        "theta": a / (1 - b),
        "sigma_ou": sigma_eps * np.sqrt(2 * kappa / (1 - b ** 2)),
        "half_life_hours": np.log(2) / kappa,
    }


# ---------------------------------------------------------------------------
# 5. Pipeline
# ---------------------------------------------------------------------------

def _analyze(name, method, alpha, beta, spread, tr, te):
    adf_train = adf_test(spread[tr])
    adf_test_ = adf_test(spread[te])
    ou = fit_ou(spread[tr])
    return {
        "pair": name, "method": method, "alpha": alpha, "beta": beta,
        "adf_train": adf_train["adf_stat"], "verdict_train": adf_train["verdict"],
        "adf_test": adf_test_["adf_stat"], "verdict_test": adf_test_["verdict"],
        "ou_kappa_per_h": ou["kappa"], "ou_sigma": ou["sigma_ou"],
        "half_life_h": ou["half_life_hours"],
    }


def run_pipeline(data_dir=DATA_DIR):
    """Returns (results table, main spread per pair, all variants, prices)."""
    prices = load_prices(data_dir)
    log_prices = np.log(prices)
    tr, te = train_mask(prices.index), test_mask(prices.index)

    results, spreads, spreads_all = [], {}, {}
    for name, y_key, x_key, fixed_beta in PAIRS:
        y_log, x_log = log_prices[y_key].values, log_prices[x_key].values

        variants = [("OLS", ols_hedge_ratio(y_log[tr], x_log[tr]))]
        if fixed_beta is not None:
            variants.append((f"beta={fixed_beta:g}", fixed_beta_hedge_ratio(y_log[tr], x_log[tr], fixed_beta)))

        for method, (alpha, beta) in variants:
            spread = pd.Series(y_log - alpha - beta * x_log, index=prices.index, name=name)
            results.append(_analyze(name, method, alpha, beta, spread.values, tr, te))
            spreads_all[f"{name}__{method}"] = spread

        # main spread: fixed beta if the pair is pegged, OLS otherwise
        spreads[name] = spreads_all[f"{name}__{variants[-1][0]}"]

    return pd.DataFrame(results), spreads, spreads_all, prices


def principal_row(results_df, pair):
    """Results row of the main spread of a pair."""
    rows = results_df[results_df["pair"] == pair]
    fixed = rows[rows["method"].str.startswith("beta=")]
    return (fixed if len(fixed) else rows).iloc[0]


if __name__ == "__main__":
    results_df, spreads, spreads_all, prices = run_pipeline()
    pd.set_option("display.width", 200)
    print(results_df.round(4).to_string(index=False))
    ensure_output_dir()
    results_df.to_csv(OUTPUT_DIR / "cointegration_ou_results.csv", index=False)
    print(f"\nResults saved in {OUTPUT_DIR.name}/cointegration_ou_results.csv")
