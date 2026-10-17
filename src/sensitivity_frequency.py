"""
Étude de robustesse : Sharpe net (après frictions) en fonction du seuil d'entrée.

Hypothèse testée : un seuil plus élevé réduit la fréquence de trading, donc le poids
des frais (gas fixe, fee et slippage proportionnels) par rapport au PnL capturé.

Le seuil est CHOISI sur la période d'apprentissage uniquement (Sharpe net moyen des
deux paires). Les résultats sur la période de test sont affichés à côté pour vérifier
que le choix tient hors échantillon — ils ne servent pas à la sélection.
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
                rows.append({"paire": pair, "entry_z": entry_z, **m})
    return pd.DataFrame(rows)


def select_threshold(df):
    """Seuil maximisant le Sharpe net moyen des paires, sur l'apprentissage seulement."""
    train_net = df[(df.periode == "apprentissage") & (df["mode"] == "net")]
    return train_net.groupby("entry_z")["sharpe"].mean().idxmax()


if __name__ == "__main__":
    ensure_output_dir()
    df = run_sensitivity()
    df.to_csv(OUTPUT_DIR / "sensitivity_frequency.csv", index=False)

    view = (df[df["mode"] == "net"]
            .pivot_table(index=["paire", "entry_z"], columns="periode", values=["sharpe", "n_trades"])
            .round(2))
    pd.set_option("display.width", 160)
    print("Sharpe net et nombre de trades, par période :\n")
    print(view.to_string())

    best = select_threshold(df)
    print(f"\nSeuil retenu sur l'apprentissage : {best}  (valeur utilisée dans signal_generation.py : {ENTRY_Z})")
    if best != ENTRY_Z:
        print("ATTENTION : mettre à jour ENTRY_Z dans signal_generation.py.")
