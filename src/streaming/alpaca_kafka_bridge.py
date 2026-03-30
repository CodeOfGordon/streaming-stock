"""
Alpaca WebSocket → Kafka Bridge
================================

Streams live market data from Alpaca to Kafka.

Flow:
- Alpaca WebSocket (live ticks) → Bar Aggregator → Kafka (1-minute OHLCV bars)

Usage:
    python src/alpaca_kafka_bridge.py
"""

import asyncio
import json
import os
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from kafka import KafkaProducer
from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.models import Trade

# Load environment variables from .env file at project root
load_dotenv(Path(__file__).resolve().parents[2] / '.env')


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOLS        = ['AAPL', 'GOOGL', 'MSFT']
TIMEFRAME      = '1Min'          # 1-minute bars — 60-bar lookback = 1hr warmup
KAFKA_BOOTSTRAP = 'localhost:9092'
KAFKA_TOPIC    = 'stock_ohlcv'
ALPACA_FEED    = DataFeed.IEX          # 'iex' (free) or 'sip' (paid)

# Alpaca credentials (set as environment variables)
API_KEY    = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# ============================================================


class BarAggregator:
    """
    Aggregates trade ticks into 1-minute OHLCV bars.

    Maintains current bar per symbol. When the minute changes, emits the
    completed bar and starts a fresh one.
    """

    def __init__(self):
        self.current_bars    = defaultdict(dict)
        self.last_bar_minute = defaultdict(lambda: None)  # tracks minute boundary per symbol

    def process_trade(self, trade: Trade):
        """
        Process a trade tick and update the current bar.

        Returns:
            Completed bar dict if the minute rolled over, None otherwise.
        """
        symbol    = trade.symbol
        price     = float(trade.price)
        size      = int(trade.size)
        timestamp = trade.timestamp

        # Truncate to the current minute boundary
        current_minute = timestamp.replace(second=0, microsecond=0)
        last_minute    = self.last_bar_minute[symbol]

        if last_minute is not None and current_minute > last_minute:
            # Minute rolled over — close the previous bar and start a new one
            completed_bar = self._create_bar_dict(symbol, last_minute)
            self._start_new_bar(symbol, price, size, current_minute)
            return completed_bar

        # Same minute (or first tick for this symbol) — update running bar
        if last_minute is None:
            self._start_new_bar(symbol, price, size, current_minute)
        else:
            self._update_bar(symbol, price, size)

        return None

    def flush_open_bars(self):
        """
        Emit all in-progress bars as-is.
        Call this on shutdown so the final partial minute isn't silently dropped.
        """
        completed = []
        for symbol, minute in self.last_bar_minute.items():
            if minute is not None and self.current_bars[symbol]:
                completed.append(self._create_bar_dict(symbol, minute))
        return completed

    # ── Private helpers ───────────────────────────────────────

    def _start_new_bar(self, symbol, price, size, minute):
        self.current_bars[symbol] = {
            'open':        price,
            'high':        price,
            'low':         price,
            'close':       price,
            'volume':      size,
            'trade_count': 1,
            'vwap_sum':    price * size,
        }
        self.last_bar_minute[symbol] = minute

    def _update_bar(self, symbol, price, size):
        bar = self.current_bars[symbol]
        bar['high']        = max(bar['high'], price)
        bar['low']         = min(bar['low'],  price)
        bar['close']       = price
        bar['volume']      += size
        bar['trade_count'] += 1
        bar['vwap_sum']    += price * size

    def _create_bar_dict(self, symbol, timestamp):
        bar = self.current_bars[symbol]
        # Publish only the OHLCV fields that feature_eng.py expects.
        # Internal fields (trade_count, vwap_sum, timeframe) are intentionally
        # excluded — they are not part of the feature schema and would appear as
        # NaN columns in the inference buffer, wiping out all seed rows on dropna().
        return {
            'symbol':    symbol,
            'timestamp': timestamp.isoformat(),
            'open':      bar['open'],
            'high':      bar['high'],
            'low':       bar['low'],
            'close':     bar['close'],
            'volume':    bar['volume'],
        }


class AlpacaKafkaBridge:
    """
    Bridge between Alpaca WebSocket and Kafka.

    Subscribes to live trades, aggregates to 1-minute bars, publishes to Kafka.
    """

    def __init__(self):
        if not API_KEY or not SECRET_KEY:
            raise ValueError(
                "Set environment variables:\n"
                "  export ALPACA_API_KEY='your_key'\n"
                "  export ALPACA_SECRET_KEY='your_secret'"
            )

        self.aggregator = BarAggregator()

        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda m: json.dumps(m).encode('utf-8'),
            compression_type='gzip'
        )

        self.stream = StockDataStream(
            api_key=API_KEY,
            secret_key=SECRET_KEY,
            feed=ALPACA_FEED
        )

        self.trades_processed = 0
        self.bars_published   = 0

        print(f"Alpaca → Kafka Bridge initialized")
        print(f"Symbols:   {SYMBOLS}")
        print(f"Timeframe: {TIMEFRAME}")
        print(f"Topic:     {KAFKA_TOPIC}")

    async def handle_trade(self, trade: Trade | dict) -> None:
        """Handle incoming trade tick from Alpaca"""
        self.trades_processed += 1

        completed_bar = self.aggregator.process_trade(trade)
        if completed_bar:
            self._publish_bar(completed_bar)

        if self.trades_processed % 100 == 0:
            print(f"Trades: {self.trades_processed}  |  Bars published: {self.bars_published}")

    def _publish_bar(self, bar):
        """Publish a completed OHLCV bar to Kafka, partitioned by symbol"""
        self.producer.send(
            KAFKA_TOPIC,
            value=bar,
            key=bar['symbol'].encode('utf-8')  # partition key — keeps symbol order guaranteed
        )
        self.bars_published += 1
        print(f"{bar['symbol']} [{bar['timestamp']}] "
              f"O={bar['open']:.2f} H={bar['high']:.2f} "
              f"L={bar['low']:.2f} C={bar['close']:.2f} V={bar['volume']}")

    def run(self):
        print(f"\nStarting bridge — streaming {TIMEFRAME} bars...\n")
        for symbol in SYMBOLS:
            self.stream.subscribe_trades(self.handle_trade, symbol)
        self.stream.run()

    async def stop(self):
        """
        Graceful shutdown: flush any open (partial) bars before closing.
        Without this the last in-progress minute bar is silently lost.
        """
        open_bars = self.aggregator.flush_open_bars()
        for bar in open_bars:
            self._publish_bar(bar)
            print(f"Flushed partial bar: {bar['symbol']} [{bar['timestamp']}]")

        self.producer.flush()
        self.producer.close()
        print(f"\nShutdown complete. Trades: {self.trades_processed}  Bars: {self.bars_published}")


def main():
    bridge = AlpacaKafkaBridge()
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("\nStopping bridge...")
        asyncio.run(bridge.stop())


if __name__ == '__main__':
    main()