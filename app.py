"""
app.py
------
Stock Market Prediction & Analysis Dashboard.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import fetch_multiple, fetch_news, fetch_stock_data
from model import forecast_future, train_and_predict
from portfolio import compute_portfolio
from sentiment import analyze_headlines

st.set_page_config(page_title="Stock Market Prediction & Analysis Dashboard", layout="wide")

st.title("📈 Stock Market Prediction & Analysis Dashboard")
st.caption(
    "Educational demo only -- price predictions and sentiment scores here are "
    "not financial advice."
)

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()
    period = st.selectbox("History period", ["6mo", "1y", "2y", "5y"], index=2)
    epochs = st.slider("LSTM training epochs", min_value=3, max_value=30, value=10)
    forecast_days = st.slider("Days to forecast", min_value=1, max_value=30, value=7)
    fetch_btn = st.button("Fetch / Refresh data")

if "df" not in st.session_state:
    st.session_state.df = None
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "result" not in st.session_state:
    st.session_state.result = None

if fetch_btn or st.session_state.df is None:
    with st.spinner(f"Fetching {ticker} data..."):
        try:
            df = fetch_stock_data(ticker, period=period)
            if df.empty:
                st.error("No data found for that ticker. Check the symbol and try again.")
            else:
                st.session_state.df = df
                st.session_state.ticker = ticker
                st.session_state.result = None  # reset stale predictions on new ticker/data
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")

df = st.session_state.df

# ------------------------------------------------------------------ main --
if df is not None and not df.empty:
    ma50_series = df["Close"].rolling(50).mean()
    ma200_series = df["Close"].rolling(200).mean()
    trend_label, trend_icon = "Not enough history", "⚪"
    if not ma50_series.isna().iloc[-1] and not ma200_series.isna().iloc[-1]:
        if ma50_series.iloc[-1] > ma200_series.iloc[-1]:
            trend_label, trend_icon = "Bullish", "📈"
        else:
            trend_label, trend_icon = "Bearish", "📉"

    last_close = float(df["Close"].iloc[-1])

    h1, h2, h3 = st.columns(3)
    h1.metric(f"{st.session_state.ticker} current price", f"${last_close:.2f}")
    h2.metric("Trend (50 vs 200 day MA)", f"{trend_icon} {trend_label}")
    if st.session_state.result is not None:
        next_day_price = forecast_future(
            st.session_state.result["model"], st.session_state.result["scaler"], df, days=1
        )[0]
        h3.metric("Next-day predicted price", f"${next_day_price:.2f}")
    else:
        h3.metric("Next-day predicted price", "Train model in Prediction tab")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "🔮 Prediction", "⚖️ Compare", "💼 Portfolio", "📰 News Sentiment"]
    )

    # ---- Overview -----------------------------------------------------
    with tab1:
        st.subheader(f"{st.session_state.ticker} price history")

        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price"
        )])
        ma20 = df["Close"].rolling(20).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma20, line=dict(width=1), name="MA20"))
        fig.add_trace(go.Scatter(x=df.index, y=ma50_series, line=dict(width=1), name="MA50"))
        fig.add_trace(go.Scatter(x=df.index, y=ma200_series, line=dict(width=1), name="MA200"))
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        prev_close = float(df["Close"].iloc[-2])
        change = last_close - prev_close
        pct = (change / prev_close * 100) if prev_close else 0.0

        col1.metric("Last close", f"${last_close:.2f}")
        col2.metric("Daily change", f"${change:.2f}", f"{pct:.2f}%")
        col3.metric(f"{period} high", f"${df['Close'].max():.2f}")
        col4.metric(f"{period} low", f"${df['Close'].min():.2f}")

        st.subheader("Volume")
        vol_fig = go.Figure(data=[go.Bar(x=df.index, y=df["Volume"])])
        vol_fig.update_layout(height=250)
        st.plotly_chart(vol_fig, use_container_width=True)

    # ---- Prediction -----------------------------------------------------
    with tab2:
        st.subheader("LSTM price prediction")
        st.caption(
            "Trains a small LSTM on the fetched history, evaluates it on a "
            "held-out test split, then forecasts forward from the last known price."
        )

        if len(df) < 120:
            st.warning(
                "Not much history to train on -- pick a longer period "
                "(1y or more recommended) for a more meaningful model."
            )

        if st.button("Train & Predict"):
            with st.spinner("Training LSTM model... this can take a minute or two"):
                try:
                    st.session_state.result = train_and_predict(
                        df, st.session_state.ticker, epochs=epochs
                    )
                except Exception as e:
                    st.error(f"Training failed: {e}")

        result = st.session_state.result
        if result is not None:
            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=result["dates"], y=result["actual"], name="Actual"))
            pred_fig.add_trace(go.Scatter(x=result["dates"], y=result["predicted"], name="Predicted"))
            pred_fig.update_layout(height=450, title="Actual vs predicted (test split)")
            st.plotly_chart(pred_fig, use_container_width=True)

            rmse = float(np.sqrt(np.mean((result["actual"] - result["predicted"]) ** 2)))
            mape = float(np.mean(np.abs((result["actual"] - result["predicted"]) / result["actual"])) * 100)
            c1, c2 = st.columns(2)
            c1.metric("Test RMSE", f"{rmse:.2f}")
            c2.metric("Test MAPE", f"{mape:.2f}%")

            future_prices = forecast_future(
                result["model"], result["scaler"], df, days=forecast_days
            )
            future_dates = pd.bdate_range(start=df.index[-1], periods=forecast_days + 1)[1:]

            future_fig = go.Figure()
            recent = df["Close"].iloc[-60:]
            future_fig.add_trace(go.Scatter(x=recent.index, y=recent, name="Recent actual"))
            future_fig.add_trace(go.Scatter(
                x=future_dates, y=future_prices, name="Forecast", line=dict(dash="dash")
            ))
            future_fig.update_layout(height=450, title=f"{forecast_days}-day forecast")
            st.plotly_chart(future_fig, use_container_width=True)

            st.caption(
                "Forecast is generated recursively (each predicted day feeds into the "
                "next), so accuracy degrades the further out it goes."
            )

    # ---- Compare --------------------------------------------------------
    with tab3:
        st.subheader("Compare multiple companies")
        st.caption("Normalized to 100 at the start of the period so you can compare relative performance regardless of share price.")

        default_tickers = f"{st.session_state.ticker}, MSFT, GOOGL"
        tickers_input = st.text_input("Tickers (comma-separated)", value=default_tickers)
        compare_btn = st.button("Compare")

        if compare_btn:
            tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
            with st.spinner("Fetching comparison data..."):
                data = fetch_multiple(tickers, period=period)

            if not data:
                st.error("Couldn't fetch data for any of those tickers.")
            else:
                comp_fig = go.Figure()
                for t, tdf in data.items():
                    normalized = tdf["Close"] / tdf["Close"].iloc[0] * 100
                    comp_fig.add_trace(go.Scatter(x=tdf.index, y=normalized, name=t))
                comp_fig.update_layout(height=500, title="Normalized price comparison (start = 100)")
                st.plotly_chart(comp_fig, use_container_width=True)

                summary_rows = []
                for t, tdf in data.items():
                    total_return = (tdf["Close"].iloc[-1] / tdf["Close"].iloc[0] - 1) * 100
                    summary_rows.append({
                        "Ticker": t,
                        "Start price": f"${tdf['Close'].iloc[0]:.2f}",
                        "Latest price": f"${tdf['Close'].iloc[-1]:.2f}",
                        f"Return over {period}": f"{total_return:.2f}%",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ---- Portfolio --------------------------------------------------------
    with tab4:
        st.subheader("Portfolio tracker")
        st.caption(
            "Add your holdings below to see live market value and unrealized P&L. "
            "This is session-only -- it resets if you reload the page."
        )

        if "holdings" not in st.session_state:
            st.session_state.holdings = []

        with st.form("add_holding", clear_on_submit=True):
            fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
            h_ticker = fc1.text_input("Ticker")
            h_shares = fc2.number_input("Shares", min_value=0.0, step=1.0)
            h_price = fc3.number_input("Avg buy price ($)", min_value=0.0, step=0.01)
            submitted = fc4.form_submit_button("Add holding")
            if submitted and h_ticker.strip() and h_shares > 0:
                st.session_state.holdings.append({
                    "ticker": h_ticker.strip().upper(),
                    "shares": h_shares,
                    "avg_price": h_price,
                })

        if st.session_state.holdings:
            if st.button("Clear all holdings"):
                st.session_state.holdings = []
                st.rerun()

            with st.spinner("Fetching current prices..."):
                portfolio_df = compute_portfolio(st.session_state.holdings)

            st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

            total_value = portfolio_df["Market Value"].dropna().sum()
            total_cost = portfolio_df["Cost Basis"].sum()
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0.0

            p1, p2, p3 = st.columns(3)
            p1.metric("Total market value", f"${total_value:,.2f}")
            p2.metric("Total cost basis", f"${total_cost:,.2f}")
            p3.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")
        else:
            st.info("No holdings added yet -- use the form above.")

    # ---- News sentiment ---------------------------------------------------
    with tab5:
        st.subheader("Recent news & sentiment")

        with st.spinner("Fetching news..."):
            try:
                news_items = fetch_news(st.session_state.ticker)
            except Exception as e:
                news_items = []
                st.error(f"Failed to fetch news: {e}")

        if not news_items:
            st.info("No recent news found for this ticker.")
        else:
            analyzed = analyze_headlines([item["title"] for item in news_items])
            for a, item in zip(analyzed, news_items):
                a["link"] = item.get("link")

            pos = sum(1 for a in analyzed if a["label"] == "Positive")
            neu = sum(1 for a in analyzed if a["label"] == "Neutral")
            neg = sum(1 for a in analyzed if a["label"] == "Negative")

            c1, c2, c3 = st.columns(3)
            c1.metric("Positive", pos)
            c2.metric("Neutral", neu)
            c3.metric("Negative", neg)

            icon = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
            for a in analyzed:
                line = f"{icon[a['label']]} "
                line += f"[{a['headline']}]({a['link']})" if a["link"] else a["headline"]
                line += f"  \n*Sentiment: {a['label']} ({a['score']:.2f})*"
                st.markdown(line)
else:
    st.info("Enter a ticker and click **Fetch / Refresh data** in the sidebar to get started.")
