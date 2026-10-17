"""
Module 1 : acquisition des données.

Récupère l'historique horaire (prix + volume) de 5 actifs via l'API publique CoinGecko :
    - ETH, stETH   (LST)
    - USDC, USDT   (stablecoins)
    - AAVE         (DeFi)

Contrainte de l'API gratuite : historique limité aux 365 derniers jours, et
granularité horaire uniquement pour des requêtes <= 90 jours (au-delà, l'API renvoie
des points journaliers). Chaque actif est donc récupéré en 5 tronçons de 73 jours.

Remarque : la colonne `volume_usd` renvoyée par CoinGecko est un volume 24h glissant,
agrégé sur toutes les places de marché (pas le volume d'un pool donné).

Usage (depuis la racine du dépôt) :
    python src/fetch_coingecko.py
Les CSV fournis dans data/ ont été obtenus avec ce script (fenêtre sept. 2025 – sept. 2026).
"""

import time

import pandas as pd
import requests

from config import DATA_DIR

COINS = {
    "ethereum": "ethereum",
    "staked-ether": "steth",
    "usd-coin": "usdc",
    "tether": "usdt",
    "aave": "aave",
}

TOTAL_DAYS = 364
CHUNK_DAYS = 73
VS_CURRENCY = "usd"
BASE_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart/range"
REQUEST_DELAY_SECONDS = 8   # limite de débit de l'API gratuite


def fetch_chunk(coin_id: str, from_ts: int, to_ts: int, retries=3, backoff=5) -> dict:
    params = {"vs_currency": VS_CURRENCY, "from": from_ts, "to": to_ts}
    url = BASE_URL.format(id=coin_id)
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "prices" in data:
                return data
        time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Échec de récupération pour {coin_id} sur [{from_ts}, {to_ts}]")


def fetch_coin_hourly(coin_id: str) -> pd.DataFrame:
    now = int(time.time())
    one_day = 86400
    start = now - TOTAL_DAYS * one_day
    rows = []
    to_ts = now
    while to_ts > start:
        from_ts = max(start, to_ts - CHUNK_DAYS * one_day)
        data = fetch_chunk(coin_id, from_ts, to_ts)
        vol_map = {t: v for t, v in data.get("total_volumes", [])}
        for t, p in data["prices"]:
            rows.append((t, p, vol_map.get(t)))
        to_ts = from_ts
        time.sleep(REQUEST_DELAY_SECONDS)

    df = pd.DataFrame(rows, columns=["timestamp_ms", "price_usd", "volume_usd"])
    return df.sort_values("timestamp_ms").drop_duplicates(subset="timestamp_ms").reset_index(drop=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for coin_id, out_name in COINS.items():
        print(f"Récupération de {coin_id}...")
        df = fetch_coin_hourly(coin_id)
        df.to_csv(DATA_DIR / f"{out_name}.csv", index=False)
        print(f"  -> {len(df)} lignes dans data/{out_name}.csv "
              f"({pd.to_datetime(df['timestamp_ms'].iloc[0], unit='ms')} → "
              f"{pd.to_datetime(df['timestamp_ms'].iloc[-1], unit='ms')})")
