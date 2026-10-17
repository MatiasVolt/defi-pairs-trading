"""
Module 3 : génération du signal (bandes dynamiques EWMA).

Remplace les seuils z-score statiques de l'article original (calculés une fois sur
tout l'historique) par une moyenne et un écart-type EWMA, ré-estimés à chaque heure.
La fenêtre EWMA est proportionnelle à la demi-vie OU de la paire (estimée sur la
période d'apprentissage).

Les bandes EWMA sont causales : la valeur à l'heure t n'utilise que les données
jusqu'à t. Le signal est ensuite décalé de EXECUTION_LAG_HOURS : un signal observé
à t n'est exécuté qu'à t + lag (on ne peut pas trader au prix qui a déclenché le signal).

Seules les paires dont le spread est stationnaire sur la période d'apprentissage
sont tradées : stETH/ETH et USDC/USDT. AAVE/ETH est exclue (ADF non significatif).
"""

import numpy as np
import pandas as pd

from config import EXECUTION_LAG_HOURS
from cointegration_ou import run_pipeline, principal_row

EWMA_HALFLIFE_MULTIPLIER = 10   # fenêtre EWMA = 10 x la demi-vie OU de la paire
# Seuil d'entrée choisi sur la période d'APPRENTISSAGE uniquement
# (cf. sensitivity_frequency.py) ; la période de test n'a pas servi à le choisir.
ENTRY_Z = 4.0
EXIT_Z = 0.5
MIN_BURN_IN_HOURS = 48          # pas de signal tant que l'EWMA n'est pas stabilisée

TRADABLE_PAIRS = ["stETH_ETH", "USDC_USDT"]
EXCLUDED_PAIRS = {"AAVE_ETH": "ADF non significatif sur la période d'apprentissage"}


def ewma_bands(spread: pd.Series, halflife_hours: float):
    ewm = spread.ewm(halflife=halflife_hours, adjust=False)
    return ewm.mean(), ewm.std(bias=False)


def generate_signal(spread, mean, std, entry_z=ENTRY_Z, exit_z=EXIT_Z, burn_in=MIN_BURN_IN_HOURS):
    """Machine à états sur le z-score :
       z >  entry_z -> position -1 (short le spread : vendre Y, acheter X)
       z < -entry_z -> position +1 (long le spread)
       |z| < exit_z -> sortie (0) ; sinon on conserve la position."""
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
    """Position effectivement détenue : le signal de l'heure t est exécuté à t + lag."""
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
        "pct_temps_en_position": float((position != 0).mean()),
        "duree_moyenne_h": float(np.mean(runs)) if runs else 0.0,
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
    print(f"Seuils : entrée |z| > {ENTRY_Z}, sortie |z| < {EXIT_Z}, délai d'exécution = {EXECUTION_LAG_HOURS} h")
    for pair, d in out.items():
        s = d["stats"]
        print(f"\n=== {pair} ===")
        print(f"  demi-vie OU = {d['ou_half_life_h']:.2f} h -> fenêtre EWMA = {d['halflife_ewma_h']:.1f} h")
        print(f"  trades = {s['n_trades']} | temps en position = {s['pct_temps_en_position']*100:.1f} % | "
              f"durée moyenne = {s['duree_moyenne_h']:.1f} h")
    print("\nPaires exclues :", EXCLUDED_PAIRS)
