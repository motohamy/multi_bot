"""
Configuration file for CryptoPrime Signal Generator.
Edit this file to configure webhooks and ticker formats for different cryptocurrencies.
"""

# Webhook URLs for different cryptocurrencies
WEBHOOK_URLS = {
    "BTC/USDT": "https://api.primeautomation.ai/webhook/ChartPrime/55f85bed-fe99-4a33-82be-8243a3ee8e15",
    "SOL/USDT": "https://api.primeautomation.ai/webhook/ChartPrime/0106285e-b00f-44f9-b811-f80c2d3a17d4",
    "default": "https://api.primeautomation.ai/webhook/ChartPrime/55f85bed-fe99-4a33-82be-8243a3ee8e15"
}

# Map from trading pairs (exchange format) to API tickers
TICKER_MAP = {
    "BTC/USDT": "BTCUSDT",
    "SOL/USDT": "SOLUSDT"
}

# Dictionary to store successful formats during runtime
SUCCESSFUL_FORMATS = {
    "BTC/USDT": "BTCUSDT",
    "SOL/USDT": "SOLUSDT"
}
