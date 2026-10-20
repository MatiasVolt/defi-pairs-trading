"""
Module 1: data acquisition.

Downloads hourly history (price + volume) for 5 assets from the public CoinGecko API:
    - ETH, stETH   (liquid staking token)
    - USDC, USDT   (stablecoins)
    - AAVE         (DeFi)

Free-API constraint: history is limited to the last 365 days, and hourly granularity is
only returned for requests of 90 days or less (beyond that, the API returns daily points).
Each asset is therefore downloaded in 5 chunks of 73 days.

Note: the `volume_usd` column returned by CoinGecko is a rolling 24h volume aggregated
across all venues (not the volume of a given pool).

Usage (from the repository root):
    python src/fetch_coingecko.py
The CSV files provided in data/ were produced with this script (window Sept 2025 - Sept 2026).
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
REQUEST_DELAY_SECONDS = 8   # rate limit of the free API


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
    raise RuntimeError(f"Download failed for {coin_id} on [{from_ts}, {to_ts}]")


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
        print(f"Downloading {coin_id}...")
        df = fetch_coin_hourly(coin_id)
        df.to_csv(DATA_DIR / f"{out_name}.csv", index=False)
        print(f"  -> {len(df)} rows in data/{out_name}.csv "
              f"({pd.to_datetime(df['timestamp_ms'].iloc[0], unit='ms')} -> "
              f"{pd.to_datetime(df['timestamp_ms'].iloc[-1], unit='ms')})")
