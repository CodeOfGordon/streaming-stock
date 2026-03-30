"""
Stock Feature Engineering
=========================

Converts raw OHLCV data into technical features for LSTM.

All windows are calibrated for 1-minute bars:
    5 bars   =  5 minutes
    15 bars  = 15 minutes
    60 bars  =  1 hour      (≈ one full lookback window)
    390 bars =  1 trading day  (6.5h × 60min)

Annualisation constant uses trading minutes per year: 252 × 390 = 98,280.

Features calculated:
- Price:      Returns, log returns, gaps, intraday range
- MA:         SMA, EMA
- Momentum:   RSI, MACD
- Volatility: Historical, Parkinson, Garman-Klass
- Bollinger:  Width, position
- Volume:     Relative volume, VWAP (daily reset), OBV, money flow
- Temporal:   Hour, minute, day-of-week (cyclical), session flags
- Lags:       Returns, relative volume, RSI
"""

import pandas as pd
import numpy as np


# Annualisation factor for 1-minute bars: 252 trading days × 390 minutes per day.
# Volatility is scaled by sqrt(periods per year) to express as annualised vol.
TRADING_MINS_PER_YEAR = 252 * 390   # 98,280


def add_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV data (1-minute bar resolution).

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]

    Returns:
        DataFrame with original columns + engineered features, NaN rows dropped.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print("Engineering features...")


    # =========================================================
    # PRICE FEATURES
    # =========================================================

    # 1-bar return = 1-minute return. 5 and 15 capture short-term momentum.
    df['return_1']  = df['close'].pct_change()
    df['return_5']  = df['close'].pct_change(5)
    df['return_15'] = df['close'].pct_change(15)

    # Log return: smoother than pct_change, better for volatility calcs
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Gap: open vs previous close — captures pre-market moves.
    # Only non-zero on the first bar of each session.
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    # Intraday range: bar-level volatility, normalised by close
    df['intraday_range'] = (df['high'] - df['low']) / df['close']

    # Close position within the bar's range [0=low, 1=high]
    # Bullish bars close near 1, bearish bars near 0
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)


    # =========================================================
    # MOVING AVERAGES
    # =========================================================

    # Window rationale:
    #   5   → very short-term noise filter (5 min)
    #  15   → quarter-hour micro-trend
    #  60   → 1-hour trend (matches lookback window)
    # 390   → full trading day average

    for window in [5, 15, 60, 390]:
        df[f'sma_{window}']          = df['close'].rolling(window).mean()
        df[f'price_to_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']

    # Intraday trend signal: 1-hour SMA vs full-day SMA.
    # When sma_60 > sma_390, the last hour is running above the daily average → intraday bullish.
    df['sma_60_390_cross'] = df['sma_60'] / (df['sma_390'] + 1e-10) - 1

    # EMA windows: 12 and 26 for MACD; 60 as medium-term anchor.
    for window in [12, 26, 60]:
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()


    # =========================================================
    # MOMENTUM INDICATORS
    # =========================================================

    # RSI over 60 bars (1 hour): a shorter window is dominated by microstructure
    # noise — a single large tick can push RSI to extremes and back within minutes.
    rsi_window = 60
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(rsi_window).mean()
    loss  = -delta.where(delta < 0, 0).rolling(rsi_window).mean()
    rs    = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD: 12/26/9 are the standard spans for intraday minute charts.
    # Interpretation: 12-min EMA crossing the 26-min EMA signals short-term momentum shift.
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']           = ema_12 - ema_26
    df['macd_signal']    = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']


    # =========================================================
    # VOLATILITY
    # =========================================================

    # All volatility features annualise using sqrt(TRADING_MINS_PER_YEAR).
    # The scalar for minute-resolution data is sqrt(252 × 390) ≈ 313.5.

    # Historical volatility (close-to-close log returns)
    for window in [15, 30, 60]:
        df[f'volatility_{window}'] = (
            df['log_return'].rolling(window).std() * np.sqrt(TRADING_MINS_PER_YEAR)
        )

    # Parkinson volatility: uses high-low range; 5× more efficient than close-to-close
    for window in [15, 30, 60]:
        hl = np.log(df['high'] / df['low']) ** 2
        df[f'parkinson_vol_{window}'] = (
            np.sqrt(hl.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(TRADING_MINS_PER_YEAR)
        )

    # Garman-Klass volatility: extends Parkinson with open/close info; most efficient estimator
    hl = np.log(df['high'] / df['low']) ** 2
    co = np.log(df['close'] / df['open']) ** 2
    gk = 0.5 * hl - (2 * np.log(2) - 1) * co
    df['gk_vol_60'] = np.sqrt(gk.rolling(60).mean()) * np.sqrt(TRADING_MINS_PER_YEAR)


    # =========================================================
    # BOLLINGER BANDS
    # =========================================================

    # 60-bar window (1 hour) gives a stable volatility envelope.
    # Shorter windows react too aggressively to individual noisy bars.
    bb_window  = 60
    bb_middle  = df['close'].rolling(bb_window).mean()
    bb_std     = df['close'].rolling(bb_window).std()
    bb_upper   = bb_middle + (2 * bb_std)
    bb_lower   = bb_middle - (2 * bb_std)

    df['bb_width']    = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)


    # =========================================================
    # VOLUME FEATURES
    # =========================================================

    # Volume MAs at 15, 30, 60 minutes
    for window in [15, 30, 60]:
        df[f'volume_ma_{window}']       = df['volume'].rolling(window).mean()
        df[f'relative_volume_{window}'] = df['volume'] / (df[f'volume_ma_{window}'] + 1e-10)

    # VWAP — resets daily at market open, matching how institutional traders use it.
    # Early in the session VWAP is unstable (low accumulated volume);
    # by midday it's a strong benchmark for institutional buy/sell decisions.
    df['date']       = df['timestamp'].dt.date
    typical_price    = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol']     = typical_price * df['volume']
    df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
    df['cum_vol']    = df.groupby('date')['volume'].cumsum()
    df['vwap']       = df['cum_tp_vol'] / (df['cum_vol'] + 1e-10)
    df['price_to_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
    df.drop(columns=['date', 'tp_vol', 'cum_tp_vol', 'cum_vol'], inplace=True)

    # OBV: cumulative volume-weighted direction.
    # obv_ma_60 smooths over 1 hour; divergence shows when volume is outpacing price.
    obv              = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv']        = obv
    df['obv_ma_60']  = obv.rolling(60).mean()
    df['obv_divergence'] = (obv - df['obv_ma_60']) / (obv.abs() + 1e-10)

    # Money flow ratio: current 60-min average vs prior 60-min average (lagged by 60)
    df['money_flow']          = df['close'] * df['volume']
    df['money_flow_ratio_60'] = (
        df['money_flow'].rolling(60).mean() / (df['money_flow'].rolling(120).mean() + 1e-10)
    )

    # Volume-weighted return: return magnitude scaled by volume backing
    df['vw_return'] = df['return_1'] * df['volume']


    # =========================================================
    # LAG FEATURES
    # =========================================================

    # Lags of 1/3/5 minutes capture very recent order flow patterns
    for lag in [1, 3, 5]:
        df[f'return_lag_{lag}'] = df['return_1'].shift(lag)

    df['volume_lag_1']          = df['volume'].shift(1)
    df['relative_volume_lag_1'] = df['relative_volume_60'].shift(1)
    df['rsi_lag_1']             = df['rsi'].shift(1)


    # =========================================================
    # TEMPORAL FEATURES
    # =========================================================

    # Hour of day: cyclical encoding so 23:00 and 00:00 are 1 hour apart, not 23
    df['hour']     = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Minute within the hour: captures intraday patterns like the :00 candle
    # (typically high volume) vs mid-hour bars (typically quieter)
    df['minute']     = df['timestamp'].dt.minute
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Day of week: cyclical encoding
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['dow_sin']     = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']     = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Day of month: model learns end-of-month tendencies
    df['dom']     = df['timestamp'].dt.day
    df['dom_sin'] = np.sin(2 * np.pi * df['dom'] / 31)
    df['dom_cos'] = np.cos(2 * np.pi * df['dom'] / 31)

    # Session flags: granular at 1-min resolution — each of the 30 opening bars
    # is individually flagged, letting the model learn the opening volatility pattern
    df['is_market_open']   = ((df['hour'] >= 9) & (df['hour'] < 16) & (df['day_of_week'] < 5)).astype(int)
    df['is_opening_30min'] = ((df['hour'] == 9) & (df['minute'] >= 30)).astype(int)
    df['is_closing_30min'] = ((df['hour'] == 15) & (df['minute'] >= 30)).astype(int)

    # Drop raw time columns — only the cyclical encodings are passed to the model
    df.drop(columns=['hour', 'minute', 'day_of_week', 'dom'], inplace=True)


    # =========================================================
    # CLEANUP
    # =========================================================

    initial_len = len(df)
    null_per_col = df.isnull().sum()
    null_per_col = null_per_col[null_per_col > 0].sort_values(ascending=False)
    nan_rows = df.isnull().any(axis=1).sum()

    # sma_390 is the largest window, so ~390 rows (~1 trading day) are dropped as warmup
    print(f"Generated {len(df.columns)} total columns")
    print(f"NaN rows: {nan_rows} / {initial_len}")
    if not null_per_col.empty:
        print(f"Top NaN columns:\n{null_per_col.head(10).to_string()}")
    print(f"Final shape (before any drop): {df.shape}")

    if drop_na:
        df = df.dropna()
        print(f"Final shape (after dropna): {df.shape}")

    return df


if __name__ == "__main__":
    symbols  = ['AAPL', 'GOOGL', 'MSFT']
    data_dir = 'data/raw'

    for symbol in symbols:
        path = f'{data_dir}/{symbol}_2y_1Min.csv'
        print(f"\nProcessing {symbol}...")

        df          = pd.read_csv(path)
        featured_df = add_features(df)

        out_path = f'data/features/{symbol}_features.parquet'
        featured_df.to_parquet(out_path, index=False)
        print(f"Saved → {out_path}")

    print("\nFeature engineering complete!")