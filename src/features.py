import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from src.data_loader import load_stock

def add_features(df):
    import numpy as np

    # Existing features (keep yours)
    df['return'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)

    df['ma_5'] = df['Close'].rolling(5).mean()

    # --- NEW FEATURES ---

    # Momentum (cleaner than return_5d)
    df['momentum'] = df['Close'] / df['Close'].shift(5) - 1

    # Rolling position (range-based signal)
    df['roll_max_10'] = df['Close'].rolling(10).max()
    df['roll_min_10'] = df['Close'].rolling(10).min()
    df['pos_in_range'] = (
        (df['Close'] - df['roll_min_10']) /
        (df['roll_max_10'] - df['roll_min_10'])
    )

    # Volume signal
    df['vol_ma5'] = df['Volume'].rolling(5).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_ma5']

    # Volatility
    df['volatility'] = df['return'].rolling(5).std()
    df['volatility_change'] = df['volatility'].pct_change()

    # Target (keep yours)
    df['target'] = (df['return'].shift(-1) > 0).astype(int)

    df['rsi'] = compute_rsi(df['Close'])

    df['volatility'] = df['return'].rolling(5).std()
    df['vol_z'] = (df['volatility'] - df['volatility'].rolling(20).mean()) / df['volatility'].rolling(20).std()

    #Price-Based
    df['trend_10'] = df['Close'] / df['Close'].rolling(10).mean() - 1
    df['trend_20'] = df['Close'] / df['Close'].rolling(20).mean() - 1

    df['high_10'] = df['Close'].rolling(10).max()
    df['low_10'] = df['Close'].rolling(10).min()

    df['breakout_up'] = (df['Close'] > df['high_10'].shift(1)).astype(int)
    df['breakout_down'] = (df['Close'] < df['low_10'].shift(1)).astype(int)

    #Volatility Structure
    df['dist_ma20'] = df['Close'] - df['Close'].rolling(20).mean()

    df['vol_5'] = df['return'].rolling(5).std()
    df['vol_20'] = df['return'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / df['vol_20']

    df['vol_spike'] = (df['vol_5'] > df['vol_20']).astype(int)

    #Volume-Based
    df['vol_ma10'] = df['Volume'].rolling(10).mean()
    df['vol_surge'] = df['Volume'] / df['vol_ma10']

    df['price_vol'] = df['return'] * df['vol_surge']

    #Momentum
    df['mom_3'] = df['Close'] / df['Close'].shift(3) - 1
    df['mom_10'] = df['Close'] / df['Close'].shift(10) - 1 

    # Final cleanup
    df = df.dropna()

    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def save_processed(df):
    df.to_csv('data/processed/features.csv', index=True)
    print("Processed features saved to data/processed/features.csv")



if __name__ == "__main__":
    amzn = load_stock("AMZN")
    spy = load_stock("SPY")
    df = add_features(amzn, spy, threshold=0.002)  # optional: lower threshold to keep more rows
    save_processed(df)