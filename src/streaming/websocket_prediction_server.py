"""
WebSocket Prediction Server
============================

Streams predictions from Kafka to browser clients via WebSocket.

Flow:
- Kafka (stock_predictions) → WebSocket Server → Browser Clients

Also provides REST API for historical queries.

Usage:
    python src/websocket_prediction_server.py
"""

import asyncio
import json
import threading
from collections import deque
from datetime import datetime
from kafka import KafkaConsumer
import websockets
from aiohttp import web


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP = 'localhost:9092'
KAFKA_TOPIC = 'stock_predictions'
CONSUMER_GROUP = 'websocket_prediction_server'

WEBSOCKET_HOST = '0.0.0.0'
WEBSOCKET_PORT = 8765

HTTP_HOST = '0.0.0.0'
HTTP_PORT = 8000

# ============================================================


class PredictionServer:
    """
    WebSocket server for real-time prediction streaming.
    
    Maintains buffer of recent predictions and broadcasts to connected clients.
    """
    
    def __init__(self):
        # Connected WebSocket clients
        self.clients = set()
        
        # Client subscriptions: {client: set of symbols}
        self.subscriptions = {}
        
        # Prediction buffer: {symbol: deque of predictions}
        self.buffer = {}
        
        # Stats
        self.predictions_received = 0
        self.messages_sent = 0
        
        print("WebSocket Prediction Server initialized")
    
    # ===== Kafka Consumer (Background Thread) =====
    
    def start_kafka_consumer(self):
        """Start Kafka consumer in background thread"""
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=CONSUMER_GROUP,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )
        
        def consume_loop():
            print(f"Kafka consumer listening on topic: {KAFKA_TOPIC}\n")
            for message in consumer:
                prediction = message.value
                self.predictions_received += 1
                
                # Add to buffer
                symbol = prediction['symbol']
                if symbol not in self.buffer:
                    self.buffer[symbol] = deque(maxlen=100)
                self.buffer[symbol].append(prediction)
                
                # Broadcast to subscribed clients
                asyncio.run(self.broadcast_prediction(prediction))
                
                # Log every 50 predictions
                if self.predictions_received % 50 == 0:
                    print(f"Received {self.predictions_received} predictions, "
                          f"sent {self.messages_sent} messages to {len(self.clients)} clients")
        
        thread = threading.Thread(target=consume_loop, daemon=True)
        thread.start()
    
    async def broadcast_prediction(self, prediction):
        """Broadcast prediction to subscribed WebSocket clients"""
        symbol = prediction['symbol']
        
        # Find subscribed clients
        subscribed = [
            client for client, symbols in self.subscriptions.items()
            if symbol in symbols or '*' in symbols
        ]
        
        if not subscribed:
            return
        
        # Enhance prediction
        enhanced = prediction.copy()
        enhanced['direction'] = 'UP' if prediction['prediction'] > 0 else 'DOWN'
        
        message = json.dumps({
            'type': 'prediction',
            'data': enhanced
        })
        
        # Send to all subscribed clients
        disconnected = []
        for client in subscribed:
            try:
                await client.send(message)
                self.messages_sent += 1
            except:
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            await self.unregister_client(client)
    
    # ===== WebSocket Handlers =====
    
    async def register_client(self, websocket):
        """Register new WebSocket client"""
        self.clients.add(websocket)
        self.subscriptions[websocket] = set()
        
        print(f"Client connected from {websocket.remote_address}. Total: {len(self.clients)}")
        
        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'connected',
            'message': 'Connected to prediction stream',
            'available_symbols': list(self.buffer.keys())
        }))
    
    async def unregister_client(self, websocket):
        """Unregister disconnected client"""
        self.clients.discard(websocket)
        self.subscriptions.pop(websocket, None)
        print(f"Client disconnected. Total: {len(self.clients)}")
    
    async def handle_client_message(self, websocket, message):
        """
        Handle messages from client.
        
        Supported messages:
        - {"type": "subscribe", "symbols": ["AAPL", "GOOGL"]}
        - {"type": "unsubscribe", "symbols": ["AAPL"]}
        - {"type": "get_latest", "symbol": "AAPL"}
        """
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'subscribe':
                symbols = data.get('symbols', [])
                self.subscriptions[websocket].update(symbols)
                
                # Send latest predictions for subscribed symbols
                for symbol in symbols:
                    if symbol in self.buffer and len(self.buffer[symbol]) > 0:
                        latest = self.buffer[symbol][-1]
                        await websocket.send(json.dumps({
                            'type': 'latest',
                            'data': latest
                        }))
                
                await websocket.send(json.dumps({
                    'type': 'subscribed',
                    'symbols': list(self.subscriptions[websocket])
                }))
            
            elif msg_type == 'unsubscribe':
                symbols = data.get('symbols', [])
                self.subscriptions[websocket].difference_update(symbols)
                
                await websocket.send(json.dumps({
                    'type': 'unsubscribed',
                    'symbols': symbols
                }))
            
            elif msg_type == 'get_latest':
                symbol = data.get('symbol')
                latest = self.buffer.get(symbol, [])
                latest = latest[-1] if latest else None
                
                await websocket.send(json.dumps({
                    'type': 'latest',
                    'symbol': symbol,
                    'data': latest
                }))
        
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def websocket_handler(self, websocket):
        """Handle WebSocket connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except:
            pass
        finally:
            await self.unregister_client(websocket)
    
    # ===== REST API =====
    
    async def http_get_symbols(self, request):
        """GET /api/symbols"""
        return web.json_response({'symbols': list(self.buffer.keys())})
    
    async def http_get_latest(self, request):
        """GET /api/latest/{symbol}"""
        symbol = request.match_info['symbol']
        
        if symbol in self.buffer and len(self.buffer[symbol]) > 0:
            return web.json_response(self.buffer[symbol][-1])
        else:
            return web.json_response(
                {'error': f'No predictions for {symbol}'},
                status=404
            )
    
    async def http_get_history(self, request):
        """GET /api/history/{symbol}?limit=20"""
        symbol = request.match_info['symbol']
        limit = int(request.query.get('limit', 20))
        
        if symbol in self.buffer:
            history = list(self.buffer[symbol])[-limit:]
            return web.json_response({'symbol': symbol, 'predictions': history})
        else:
            return web.json_response(
                {'error': f'No predictions for {symbol}'},
                status=404
            )
    
    async def http_get_stats(self, request):
        """GET /api/stats"""
        return web.json_response({
            'predictions_received': self.predictions_received,
            'messages_sent': self.messages_sent,
            'connected_clients': len(self.clients),
            'tracked_symbols': len(self.buffer)
        })
    
    def create_http_app(self):
        """Create HTTP REST API app"""
        app = web.Application()
        
        app.router.add_get('/api/symbols', self.http_get_symbols)
        app.router.add_get('/api/latest/{symbol}', self.http_get_latest)
        app.router.add_get('/api/history/{symbol}', self.http_get_history)
        app.router.add_get('/api/stats', self.http_get_stats)
        
        # CORS
        @web.middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
        
        app.middlewares.append(cors_middleware)
        
        return app
    
    # ===== Main =====
    
    async def start(self):
        """Start WebSocket and HTTP servers"""
        # Start Kafka consumer
        self.start_kafka_consumer()
        
        # Start WebSocket server
        print(f"WebSocket server: ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        websocket_server = await websockets.serve(
            self.websocket_handler,
            WEBSOCKET_HOST,
            WEBSOCKET_PORT
        )
        
        # Start HTTP API server
        print(f"HTTP API server: http://{HTTP_HOST}:{HTTP_PORT}")
        http_app = self.create_http_app()
        http_runner = web.AppRunner(http_app)
        await http_runner.setup()
        http_site = web.TCPSite(http_runner, HTTP_HOST, HTTP_PORT)
        await http_site.start()
        
        print("\nServers started!")
        print("Waiting for predictions...\n")
        
        # Keep running
        await asyncio.Future()


async def main():
    """Entry point"""
    server = PredictionServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    asyncio.run(main())