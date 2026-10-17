"""
Paramètres globaux du projet, partagés par tous les modules.

Le point clé est le découpage temporel :
  - période d'APPRENTISSAGE (in-sample) : les 2/3 les plus anciens de l'historique.
    Tout ce qui est estimé (hedge ratio, test ADF, paramètres Ornstein-Uhlenbeck,
    choix du seuil d'entrée, poids du portefeuille) l'est UNIQUEMENT sur cette période ;
  - période de TEST (out-of-sample) : le 1/3 le plus récent, jamais utilisé pour
    calibrer quoi que ce soit. C'est la seule période dont la performance compte.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_FRAC = 2 / 3            # part de l'historique utilisée pour la calibration
EXECUTION_LAG_HOURS = 1       # délai entre le signal et l'exécution du trade (0 = exécution au prix du signal)
TOTAL_CAPITAL_USD = 100_000.0 # capital total supposé
RANDOM_SEED = 42              # reproductibilité du tirage stochastique du gas
PERIODS_PER_YEAR = 24 * 365   # données horaires


def split_timestamp(index):
    """Premier horodatage de la période de test."""
    return index[int(len(index) * TRAIN_FRAC)]


def train_mask(index):
    return index < split_timestamp(index)


def test_mask(index):
    return index >= split_timestamp(index)


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    return OUTPUT_DIR
