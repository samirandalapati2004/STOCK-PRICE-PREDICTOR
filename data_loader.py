"""
data_loader.py
--------------
Thin wrappers around yfinance for pulling price history and recent news
for a given ticker.
"""

import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV history for a ticker."""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    # yfinance sometimes returns a MultiIndex column even for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    return df


def fetch_multiple(tickers: list, period: str = "1y", interval: str = "1d") -> dict:
    """Fetch history for several tickers at once. Returns {ticker: DataFrame}."""
    result = {}
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        try:
            df = fetch_stock_data(t, period=period, interval=interval)
            if not df.empty:
                result[t] = df
        except Exception:
            continue
    return result


def fetch_news(ticker: str, limit: int = 10):
    """
    Fetch recent news items for a ticker. yfinance's news schema has
    changed across versions, so this normalizes to a flat list of dicts:
    [{"title": ..., "link": ...}, ...]
    """
    t = yf.Ticker(ticker)
    try:
        raw_news = t.news or []
    except Exception:
        raw_news = []

    normalized = []
    for item in raw_news[:limit]:
        # Newer yfinance nests fields under "content"
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title")
        link = None
        click_through = content.get("clickThroughUrl")
        if isinstance(click_through, dict):
            link = click_through.get("url")
        link = link or item.get("link")

        if title:
            normalized.append({"title": title, "link": link})

    return normalized
