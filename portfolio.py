"""
portfolio.py
------------
A lightweight, in-memory portfolio tracker: given a list of holdings
(ticker, shares, average buy price), fetch the current price for each
and compute market value and unrealized P&L.

No persistence -- holdings live only in Streamlit's session_state for
the current browser session. That's intentional for a demo project;
swap in a database if you want it to survive restarts.
"""

import pandas as pd
import yfinance as yf


def get_current_price(ticker: str):
    try:
        df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def compute_portfolio(holdings):
    """
    holdings: list of dicts {"ticker": str, "shares": float, "avg_price": float}
    Returns a DataFrame with current price, market value, cost basis, and P&L.
    """
    rows = []
    for h in holdings:
        ticker = h["ticker"].strip().upper()
        shares = h["shares"]
        avg_price = h["avg_price"]
        current_price = get_current_price(ticker)

        cost_basis = shares * avg_price
        market_value = shares * current_price if current_price is not None else None
        pnl = (market_value - cost_basis) if market_value is not None else None
        pnl_pct = (pnl / cost_basis * 100) if pnl is not None and cost_basis else None

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Avg Buy Price": avg_price,
            "Current Price": current_price,
            "Cost Basis": cost_basis,
            "Market Value": market_value,
            "P&L": pnl,
            "P&L %": pnl_pct,
        })

    return pd.DataFrame(rows)
