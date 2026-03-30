"""
Kafka Inference Service
========================

Consumes OHLCV bars from Kafka, runs LSTM predictions, publishes results.

Flow:
- Kafka (stock_ohlcv) → Feature Engineering → LSTM → Kafka (stock_predictions)

On startup, buffers are pre-seeded with historical 1-minute bars from Alpaca
so predictions begin on the first live bar rather than after a ~390-minute warmup.

Usage:
    python src/kafka_inference_service.py
"""

import torch
import numpy as np
import pandas as pd
import json
import pickle
import os
from collections import deque
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.feature_eng import add_features
from training.lstm_model import StockLSTM, LSTMConfig

# Load .env from project root (for ALPACA credentials)
load_dotenv(Path(__file__).resolve().parents[2] / '.env')


# ============================================================
# CONFIGURATION
# ============================================================

# Kafka settings
KAFKA_BOOTSTRAP = 'localhost:9092'
INPUT_TOPIC     = 'stock_ohlcv'
OUTPUT_TOPIC    = 'stock_predictions'
CONSUMER_GROUP  = 'lstm_inference_group'

# Model settings
MODEL_PATH = 'models/checkpoints/best_model.pt'
SCALER_PATH = 'models/scalers.pkl'
LOOKBACK_WINDOW = 60    # LSTM sequence length — must match training

# Buffer sizing:
# sma_390 (largest feature window) + LOOKBACK_WINDOW (60) = 450 bars minimum.
# 500 gives comfortable headroom above that.
BUFFER_SIZE    = 700
FEATURE_WARMUP = 450    # sma_390 (390) + LOOKBACK_WINDOW (60)

# Historical seeding: fetch this many recent 1-minute bars on startup
# to pre-fill buffers and avoid the ~390-minute live warmup.
SEED_BARS = 600 #  ~450 NaN warmup = ~150 valid rows ≥ LOOKBACK_WINDOW (60)

# Symbols to seed (should match alpaca_kafka_bridge.py)
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT']

# Alpaca credentials for historical seeding
ALPACA_API_KEY    = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================


def seed_buffer_from_history(symbol: str, n_bars: int) -> list[dict]:
    """
    Fetch the last n_bars of 1-minute historical bars from Alpaca REST API.

    This pre-fills the symbol buffer so feature engineering has enough history
    to produce valid features immediately, rather than waiting ~390 live minutes.

    Returns a list of bar dicts in the same format as the live Kafka messages.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY
    )

    end   = datetime.utcnow()
    # Fetch extra days to account for weekends/holidays where market is closed
    start = end - timedelta(days=14)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed='iex'
    )

    bars = client.get_stock_bars(request).df.reset_index()

    # Explicitly keep only OHLCV columns — Alpaca's REST response includes extra
    # fields (timeframe, trade_count, vwap) that don't exist in live Kafka messages.
    # Any extra column here would appear as NaN for all seed rows in the inference
    # buffer, causing dropna() to wipe out the entire seed history.
    bars = bars[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Trim to the last n_bars of trading activity
    bars = bars.tail(n_bars)

    # Convert to the same dict format the live Kafka messages use
    return [
        {
            'symbol':    symbol,
            'timestamp': row['timestamp'].isoformat(),
            'open':      float(row['open']),
            'high':      float(row['high']),
            'low':       float(row['low']),
            'close':     float(row['close']),
            'volume':    int(row['volume']),
        }
        for _, row in bars.iterrows()
    ]


class InferenceService:
    """
    Real-time LSTM inference service.

    Maintains state per symbol, runs predictions, publishes to Kafka.
    Buffers are pre-seeded with historical data on startup so predictions
    begin on the first live bar.
    """

    def __init__(self):
        print("Initializing Inference Service...")

        # Load model
        print(f"Loading model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        self.model = StockLSTM(LSTMConfig(**checkpoint['model_config']))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(DEVICE)
        self.model.eval()
        print(f"✓ Model loaded (device: {DEVICE})")

        # Load scalers
        print(f"Loading scalers from {SCALER_PATH}...")
        with open(SCALER_PATH, 'rb') as f:
            scalers = pickle.load(f)
        self.feature_scaler = scalers['feature_scaler']
        self.target_scaler  = scalers['target_scaler']
        self.feature_names  = scalers['feature_names']
        print(f"✓ Scalers loaded ({len(self.feature_names)} features)")

        # State per symbol: deque of recent OHLCV bars.
        # Buffer must exceed largest feature window (sma_390) + lookback (60).
        self.symbol_buffers = {}

        # Pre-seed buffers with historical data so we don't wait ~390 live minutes
        self._seed_buffers()

        # Kafka consumer
        self.consumer = KafkaConsumer(
            INPUT_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=CONSUMER_GROUP,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )

        # Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda m: json.dumps(m).encode('utf-8'),
            compression_type='gzip'
        )

        # Stats
        self.predictions_made = 0
        self.latencies = deque(maxlen=100)

        print(f"✓ Kafka connected")
        print(f"  Input topic:  {INPUT_TOPIC}")
        print(f"  Output topic: {OUTPUT_TOPIC}")
        print("\nReady for inference!\n")

    def _seed_buffers(self):
        """
        Pre-fill each symbol's buffer with historical bars from Alpaca.

        Without seeding, the service must wait for ~390 live bars before
        feature engineering can produce non-NaN values. Seeding eliminates
        that wait — the first live bar will trigger a real prediction.
        """
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            print("WARNING: No Alpaca credentials found — skipping historical seeding.")
            print(f"         Predictions will begin after {FEATURE_WARMUP} live bars (~{FEATURE_WARMUP} minutes).")
            for symbol in SYMBOLS:
                self.symbol_buffers[symbol] = deque(maxlen=BUFFER_SIZE)
            return

        print(f"\nSeeding buffers with last {SEED_BARS} bars per symbol (~{SEED_BARS - 450} valid rows after NaN warmup)...")
        for symbol in SYMBOLS:
            self.symbol_buffers[symbol] = deque(maxlen=BUFFER_SIZE)
            try:
                bars = seed_buffer_from_history(symbol, SEED_BARS)
                self.symbol_buffers[symbol].extend(bars)
                print(f"  ✓ {symbol}: seeded with {len(bars)} bars")
            except Exception as e:
                # Non-fatal — fall back to live warmup for this symbol
                print(f"  ✗ {symbol}: seeding failed ({e}), will warm up from live data")
        print()

    def process_message(self, message):
        """
        Process incoming OHLCV bar and make prediction.

        Args:
            message: Kafka message with OHLCV bar
        """
        start_time = datetime.now()

        bar    = message.value
        symbol = bar['symbol']

        # Initialize buffer for any symbol not covered by seeding
        if symbol not in self.symbol_buffers:
            self.symbol_buffers[symbol] = deque(maxlen=BUFFER_SIZE)

        # Add bar to buffer
        self.symbol_buffers[symbol].append(bar)

        # Need enough bars for feature engineering warmup before predicting.
        # sma_390 requires 390 bars; plus 60 for the LSTM sequence = 450 minimum.
        if len(self.symbol_buffers[symbol]) < FEATURE_WARMUP:
            print(f"{symbol}: Warming up... {len(self.symbol_buffers[symbol])}/{FEATURE_WARMUP}")
            return

        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.symbol_buffers[symbol]))

        # Feature engineering
        try:
            featured_df = add_features(df, drop_na=False)  # keep all rows; NaNs only at head
            featured_df = featured_df.dropna()             # drop NaN warmup rows explicitly
        except Exception as e:
            print(f"Feature engineering error for {symbol}: {e}")
            return

        # Get latest row with all features
        if len(featured_df) < LOOKBACK_WINDOW:
            print(f"{symbol}: Not enough valid features after engineering")
            return

        # Exclude raw OHLCV, metadata, and target columns.
        # 'symbol' is explicitly excluded here — it's a string present in the inference
        # buffer dicts but absent from training CSVs, which caused a spurious feature
        # mismatch warning. All remaining columns are numeric engineered features.
        exclude = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [col for col in featured_df.columns if col not in exclude]

        # Verify features match training — should be an exact match now that 'symbol'
        # is excluded. A mismatch here means a genuine schema divergence (e.g. a new
        # feature was added to feature_eng.py after the model was trained).
        if set(feature_cols) != set(self.feature_names):
            missing = set(self.feature_names) - set(feature_cols)
            extra   = set(feature_cols) - set(self.feature_names)
            print(f"WARNING: Feature mismatch for {symbol} — missing: {missing}, extra: {extra}")
            # Align to training feature order; drop any extras, warn on any missing
            feature_cols = [col for col in self.feature_names if col in feature_cols]

        # Select features in the exact column order the scaler was fit on
        features = featured_df[feature_cols].tail(LOOKBACK_WINDOW).values

        # Scale features
        features_scaled = self.feature_scaler.transform(features)

        # Create sequence: (1, lookback, features)
        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(DEVICE)

        # LSTM inference
        with torch.no_grad():
            prediction_scaled, _ = self.model(X)

        # Capture raw scaled output before inverse transform for diagnostics.
        # RobustScaler maps the training median return → 0, so a consistently
        # near-zero scaled value here means the model is predicting "median return"
        # every time — a sign of underfitting or a saturated output layer.
        prediction_scaled_val = prediction_scaled.item()

        # Inverse scale prediction back to actual return space
        prediction_scaled_np = prediction_scaled.cpu().numpy().reshape(-1, 1)
        prediction = self.target_scaler.inverse_transform(prediction_scaled_np)[0, 0]

        # Calculate predicted price
        current_price   = float(bar['close'])
        predicted_price = current_price * (1 + prediction)

        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.latencies.append(latency_ms)

        # Create prediction message
        prediction_msg = {
            'symbol':          symbol,
            'timestamp':       bar['timestamp'],
            'current_price':   current_price,
            'prediction':      float(prediction),
            'predicted_price': float(predicted_price),
            'prediction_type': 'return',
            'model_version':   'v1.0',
            'latency_ms':      latency_ms
        }

        # Publish to Kafka
        self.producer.send(
            OUTPUT_TOPIC,
            value=prediction_msg,
            key=symbol.encode('utf-8')
        )

        self.predictions_made += 1

        # Use 4 decimal places for the return percentage — +0.0000% vs +0.0312% are
        # very different model behaviours but both round to +0.00% at 2dp.
        # Also log the raw scaled output: if it's consistently exactly 0.000000 the
        # output layer may be saturated (e.g. wrong activation) rather than the model
        # just predicting a small return.
        direction = "UP ↑" if prediction > 0 else "DOWN ↓"
        print(f"{symbol}: {direction} {prediction:+.4%} "
              f"(${current_price:.2f} → ${predicted_price:.4f}) "
              f"[scaled: {prediction_scaled_val:.6f}] "
              f"[{latency_ms:.1f}ms]")

        # Log stats every 10 predictions
        if self.predictions_made % 10 == 0:
            avg_latency = sum(self.latencies) / len(self.latencies)
            print(f"\nStats: {self.predictions_made} predictions, "
                  f"avg latency: {avg_latency:.1f}ms, "
                  f"{len(self.symbol_buffers)} symbols\n")

    def run(self):
        """Main loop - consume messages and make predictions"""
        print("Listening for OHLCV bars...\n")

        try:
            for message in self.consumer:
                self.process_message(message)

        except KeyboardInterrupt:
            print("\nStopping inference service...")

        finally:
            self.consumer.close()
            self.producer.flush()
            self.producer.close()
            print(f"Shutdown complete. Made {self.predictions_made} predictions.")


if __name__ == '__main__':
    service = InferenceService()
    service.run()