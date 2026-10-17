"""
Module 4 : allocation du capital entre les paires (rendement / CVaR).

Remplace la fonction « rendement / variance » de l'article original par
« rendement / CVaR » : on pénalise l'ampleur des pires pertes plutôt que la
dispersion moyenne, ce qui colle mieux au risque principal d'une stratégie
d'arbitrage (décrochage brutal : depeg, choc de corrélation).

Avec 2 lignes (stETH/ETH et USDC/USDT), une recherche par grille sur w ∈ [0, 1]
est exacte et transparente. Les poids sont choisis sur la période d'apprentissage
(rendements nets de frictions) puis évalués sur la période de test.
"""

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, PERIODS_PER_YEAR, ensure_output_dir, train_mask, test_mask
from backtest_engine import run_backtests

ALPHA_CVAR = 0.05
N_GRID = 101


def historical_cvar(returns: pd.Series, alpha=ALPHA_CVAR) -> float:
    """Moyenne des alpha % pires rendements, renvoyée comme une perte positive."""
    tail = returns[returns <= returns.quantile(alpha)]
    return -tail.mean() if len(tail) else 0.0


def net_return_matrix():
    """Rendements horaires nets de chaque paire, en % du notionnel de la ligne
    (notionnel de référence = cas équipondéré ; le gas étant un coût fixe en USD,
    c'est une approximation)."""
    res = run_backtests()
    return pd.DataFrame({p: r["bt"]["net_pnl_usd"] / r["notional"] for p, r in res.items()})


def portfolio_stats(port_ret: pd.Series) -> dict:
    std = port_ret.std()
    mean_ann = port_ret.mean() * PERIODS_PER_YEAR
    cvar = historical_cvar(port_ret)
    return {
        "rendement_annualise": mean_ann,
        "vol_annualisee": std * np.sqrt(PERIODS_PER_YEAR),
        "cvar_5pct_horaire": cvar,
        "sharpe": port_ret.mean() / std * np.sqrt(PERIODS_PER_YEAR) if std > 0 else np.nan,
        # rendement par unité de risque de queue (les deux en horaire, pas d'annualisation de la CVaR)
        "rendement_sur_cvar": port_ret.mean() / cvar if cvar > 0 else np.nan,
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
    best = f_train.loc[f_train["rendement_sur_cvar"].idxmax()]
    criterion = "max rendement/CVaR"
    if best["rendement_annualise"] <= 0:
        # Avec un rendement négatif, maximiser rendement/CVaR récompense le RISQUE
        # (un dénominateur plus grand rend le ratio « moins négatif »). Le critère n'a
        # alors plus de sens : on se replie sur le portefeuille de CVaR minimale.
        best = f_train.loc[f_train["cvar_5pct_horaire"].idxmin()]
        criterion = "CVaR minimale (aucun portefeuille à rendement net positif)"
    w = best[f"w_{a}"]
    print(f"Poids retenus sur l'apprentissage ({criterion}) : {a} = {w:.0%}, {b} = {1 - w:.0%}\n")

    rows = []
    for label, weight in [("optimisé", w), ("équipondéré", 0.5)]:
        for period, mask in [("apprentissage", tr), ("test", te)]:
            rows.append({"portefeuille": label, "periode": period,
                         **portfolio_stats(weight * R[a][mask] + (1 - weight) * R[b][mask])})
    out = pd.DataFrame(rows)
    out["cvar_5pct_horaire"] *= 1e4   # en points de base
    out = out.rename(columns={"cvar_5pct_horaire": "cvar_5pct_horaire_bp"})
    print(out.round(3).to_string(index=False))
    print("\nNB : l'optimisation ne crée pas de performance, elle répartit le risque entre les lignes.")
