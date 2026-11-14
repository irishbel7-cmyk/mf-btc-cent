"""btc-cent-forecast-01 entry: damped-momentum forecast of BTC/USD 4h ahead.

Method v1 (from the method note): work in log-price space on the 72h of
1-minute closes, predict the 4h-ahead log return as a small damped
combination of recent log returns, clip it, and exponentiate back.

    r_pred = 0.15 * r1h + 0.10 * r4h + 0.05 * r24h
    r_pred clipped to [-0.02, +0.02]
    forecast = last_price * exp(r_pred)

Output per case: {"forecasts": [{"case": "<id>", "price": <number>}]}
with the price rounded to 5 decimals and clamped to the scorer's bounds.
Standard library only, no network, deterministic.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

W_1H = 0.15
W_4H = 0.10
W_24H = 0.05
CLIP = 0.02


def price_on_grid(value):
    """Clamp to (0, 10_000_000) and round to 5 decimals."""
    clamped = min(MAX_PRICE, max(MIN_PRICE, value))
    return round(clamped, 5)


def last_finite(prices):
    """Last finite price, or None when there is none."""
    for value in reversed(prices):
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def log_return(prices, minutes_back):
    """log(last / price `minutes_back` minutes ago), or None if unusable."""
    last = last_finite(prices)
    if last is None or last <= 0:
        return None
    index = len(prices) - 1 - minutes_back
    if index < 0:
        return None
    earlier = prices[index]
    if not isinstance(earlier, (int, float)):
        return None
    if not math.isfinite(earlier) or earlier <= 0:
        return None
    return math.log(last / float(earlier))


def forecast(prices):
    """Damped-momentum 4h-ahead price forecast for one market."""
    last = last_finite(prices)
    if last is None:
        raise ValueError("market has no finite price")
    r1h = log_return(prices, 60)
    r4h = log_return(prices, 240)
    r24h = log_return(prices, 1440)
    if r1h is None or r4h is None or r24h is None:
        return price_on_grid(last)
    predicted = W_1H * r1h + W_4H * r4h + W_24H * r24h
    predicted = max(-CLIP, min(CLIP, predicted))
    return price_on_grid(last * math.exp(predicted))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with path.open("r", encoding="utf-8") as handle:
            case = json.load(handle)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{path.name} does not hold exactly one market")
        price = forecast(markets[0].get("prices", []))
        payload = {"forecasts": [{"case": case_id, "price": price}]}
        out_path = OUTPUT_DIR / f"{case_id}.out"
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
