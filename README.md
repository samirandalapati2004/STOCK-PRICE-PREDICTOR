# Stock Market Prediction & Analysis Dashboard

A Streamlit dashboard that combines:
- **Stock visualization** -- candlestick chart, 20/50/200-day moving averages, volume
- **Trend signal** -- Bullish/Bearish badge from the 50-day vs 200-day MA cross
- **Trend prediction** -- an LSTM trained on closing-price history, evaluated
  on a held-out test split, plus a recursive multi-day forecast
- **Multi-company comparison** -- normalized relative performance across
  several tickers at once
- **Portfolio tracker** -- add holdings (ticker/shares/buy price), see live
  market value and unrealized P&L
- **News sentiment analysis** -- recent headlines for the ticker scored with
  NLTK's VADER sentiment analyzer

> **Disclaimer:** This is an educational/portfolio project. The predictions
> and sentiment scores are simplified demonstrations of the techniques
> involved, not investment advice.

## Project structure

```
stock-dashboard/
├── app.py           # Streamlit UI -- run this
├── data_loader.py   # Yahoo Finance price + news fetching (single + multi-ticker)
├── model.py         # LSTM build/train/predict/forecast
├── portfolio.py      # Live portfolio value / P&L tracker
├── sentiment.py      # VADER-based headline sentiment scoring
├── requirements.txt
├── README.md
└── models/           # trained models + scalers get saved here (per ticker)
```

## Setup

```bash
cd stock-dashboard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then, in the browser tab that opens:
1. Enter a ticker symbol (e.g. `AAPL`, `MSFT`, `TSLA`) in the sidebar and pick
   a history period, then click **Fetch / Refresh data**.
2. **Overview** tab -- candlestick chart with 20/50-day moving averages,
   key stats, and volume.
3. **Prediction** tab -- click **Train & Predict** to train the LSTM on the
   fetched history (takes roughly 10-60 seconds depending on `epochs` and
   history length) and see actual-vs-predicted accuracy plus a forward
   forecast. Once trained, the next-day predicted price also appears in the
   summary header at the top of the page.
4. **Compare** tab -- enter a comma-separated list of tickers to see their
   normalized relative performance on one chart.
5. **Portfolio** tab -- add holdings (ticker, shares, average buy price) to
   see live market value and unrealized P&L. This is session-only and resets
   on page reload -- there's no database behind it.
6. **News Sentiment** tab -- recent headlines for the ticker with a
   Positive/Neutral/Negative sentiment label and score.

## How the pieces work

- **Data**: `yfinance` pulls OHLCV price history and recent news for any
  ticker Yahoo Finance covers -- no API key needed.
- **Prediction**: closing prices are scaled to [0, 1], turned into
  overlapping 60-day windows, and fed into a 2-layer LSTM that predicts the
  next day's price. The model is evaluated on the last 20% of the history
  (never seen during training), then used to forecast forward one day at a
  time (each predicted day feeds back in as input for the next).
- **Sentiment**: headlines are scored with NLTK's VADER lexicon, which is
  tuned for short informal text and needs no heavy model download. Swap in
  a finance-tuned model (e.g. FinBERT) later if you want more domain-specific
  accuracy -- `sentiment.py`'s `analyze_headlines()` interface can stay the
  same.

## Ideas not built yet (good next steps for a portfolio piece)

These were part of a fuller project spec but are deliberately left out for
now, since they're separate bodies of work:
- **Real-time (intraday/streaming) updates** -- current version fetches on
  demand, not on a live feed.
- **AI chatbot explaining market trends** -- would need an LLM API call
  wired to the fetched data/sentiment as context.
- **Docker deployment** -- straightforward to add (a `Dockerfile` running
  `streamlit run app.py`) if you want a containerized version to ship.
- **Persistent portfolio storage** -- currently in-memory only; swapping in
  SQLite or a small database would make holdings survive a restart.

## Notes / things to try next

- Model + scaler are saved to `models/<TICKER>_lstm.keras` and
  `models/<TICKER>_scaler.pkl` after each training run, so you could add a
  "load saved model" option to skip retraining.
- Longer history periods (`2y`/`5y`) generally give the LSTM more to learn
  from than `6mo`.
- The forecast is recursive, so accuracy naturally degrades further out --
  that's expected, not a bug.
- yfinance's `news` response schema has changed across versions; if
  `fetch_news` ever returns an empty list unexpectedly, check
  `yfinance.__version__` and the actual shape of `Ticker(ticker).news`.
