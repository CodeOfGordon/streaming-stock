"""
Simple Alpaca Historical Data Downloader
=========================================

Usage:
    python download_alpaca_data.py

Downloads 2 years of 1-minute data for specified symbols and saves to CSV.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from pathlib import Path


# Configuration
SYMBOLS      = ['AAPL', 'GOOGL', 'MSFT']
PERIOD_YEARS = 2
TIMEFRAME    = '1Min'   # Changed from '1Hour' — matches alpaca_kafka_bridge.py
OUTPUT_DIR   = 'data/raw'

# Alpaca credentials (set as environment variables)
API_KEY    = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')


def download_data():
    """Download historical data from Alpaca"""

    if not API_KEY or not SECRET_KEY:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365 * PERIOD_YEARS)

    print(f"Downloading {PERIOD_YEARS} years of data ({start_date.date()} to {end_date.date()})")
    print(f"Symbols:   {SYMBOLS}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Note: 1-minute bars over 2 years ≈ 200k bars per symbol\n")

    for symbol in SYMBOLS:
        print(f"Downloading {symbol}...", end=' ', flush=True)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,  # Changed from TimeFrame.Hour
            start=start_date,
            end=end_date,
            feed='iex'
        )

        bars = client.get_stock_bars(request)
        df   = bars.df.reset_index()

        # Keep only OHLCV columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        filename = f"{symbol}_{PERIOD_YEARS}y_{TIMEFRAME}.csv"
        filepath = Path(OUTPUT_DIR) / filename
        df.to_csv(filepath, index=False)

        print(f"✓ {len(df):,} bars → {filepath}")

    print(f"\nDone! Data saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    download_data()