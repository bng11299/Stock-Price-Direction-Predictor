# main_yf.py

import yfinance as yf
import pandas as pd

from src.features import add_features, save_processed
from src.model import logistic_model, train_lstm, boosting_model, xgboost_model

tickers = [
    'AAPL',  # tech
    'MSFT',  # tech
    'XOM',   # energy
    'JPM',   # finance
    'UNH',   # healthcare
    'WMT',   # retail
    'CAT',   # industrials
    'KO',    # consumer defensive
    'BA',    # aerospace
    'NVDA'   # high-vol tech
]   

results = []

for ticker in tickers:
    print(f"\n===== {ticker} =====")

    # --- Step 1: Download ---
    df = yf.download(ticker, start="2010-01-01", end="2023-12-31")

    # Fix Yahoo multi-index issue
    df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # --- Step 2: Feature engineering ---
    df = add_features(df)

    # Save for models
    save_processed(df)

    # --- Step 3: Models ---
    log_acc = logistic_model()
    boost_acc = boosting_model()
    lstm_acc = train_lstm(seq_len=5, epochs=10, batch_size=32)
    xgb_acc = xgboost_model()
    print(f"Results → Log: {log_acc:.3f}, Boost: {boost_acc:.3f}, XGB: {xgb_acc:.3f}, LSTM: {lstm_acc:.3f}")

    results.append({
        'ticker': ticker,
        'logistic': log_acc,
        'boosting': boost_acc,
        'lstm': lstm_acc,
        'xgboost' : xgb_acc
    })

# --- Final results ---
results_df = pd.DataFrame(results)
print("\nFINAL RESULTS:")
print(results_df)

results_df.to_csv('results/metrics/all_results.csv', index=False)