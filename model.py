"""
model.py
--------
Trains a small LSTM on a ticker's closing-price history to (a) evaluate
next-day prediction accuracy on a held-out test split, and (b) forecast
a number of days forward.

This is a simple univariate time-series model meant for demonstration/
learning purposes -- not a trading signal. See the disclaimer in
README.md.
"""

import os
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

WINDOW = 60  # number of past days used to predict the next day


def create_sequences(data, window=WINDOW):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(window=WINDOW):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_and_predict(df, ticker, epochs=10, window=WINDOW, models_dir="models"):
    """
    Trains on the first 80% of the history and evaluates on the last 20%.
    Saves the trained model + scaler to disk (keyed by ticker) so they
    can be reused without retraining every time.
    """
    os.makedirs(models_dir, exist_ok=True)

    close_prices = df[["Close"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    split = int(len(scaled) * 0.8)
    train_data = scaled[:split]
    test_data = scaled[split - window:]  # include lookback so first test point is predictable

    X_train, y_train = create_sequences(train_data, window)
    X_test, y_test = create_sequences(test_data, window)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_model(window)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    predictions = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    model.save(os.path.join(models_dir, f"{ticker}_lstm.keras"))
    with open(os.path.join(models_dir, f"{ticker}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return {
        "history": history.history,
        "dates": df.index[split:],
        "actual": actual.flatten(),
        "predicted": predictions.flatten(),
        "model": model,
        "scaler": scaler,
    }


def forecast_future(model, scaler, df, days=7, window=WINDOW):
    """Recursively forecast `days` steps forward from the end of df."""
    close_prices = df[["Close"]].values
    scaled = scaler.transform(close_prices)
    current_window = scaled[-window:].reshape(1, window, 1)

    preds = []
    for _ in range(days):
        next_scaled = model.predict(current_window, verbose=0)[0]
        preds.append(next_scaled[0])
        current_window = np.append(current_window[:, 1:, :], [[next_scaled]], axis=1)

    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()
