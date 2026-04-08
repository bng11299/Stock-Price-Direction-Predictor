import pandas as pd

# Handles Loading and Cleans Raw Data
def load_stock(symbol="AMZN"):
    df = pd.read_csv("data/raw/sp500_stocks.csv")

    df = df[df['Symbol'] == symbol]
    df = df.dropna(subset=['Close', 'Volume'])

    df = df.sort_values('Date')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    return df[['Close', 'Volume']]