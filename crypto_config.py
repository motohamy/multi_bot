"""
Configuration file for CryptoPrime Signal Generator.
Edit this file to configure webhooks and ticker formats for different cryptocurrencies.
"""

# Webhook URLs for different cryptocurrencies
WEBHOOK_URLS = {
    "BTC/USDT": "https://api.primeautomation.ai/webhook/ChartPrime/9c3a902a-83e4-48c7-a717-632b3e585de7",
    "SOL/USDT": "https://api.primeautomation.ai/webhook/ChartPrime/6091d974-83af-4b10-bcb0-6953eaff129a",
    "default": "https://api.primeautomation.ai/webhook/ChartPrime/9c3a902a-83e4-48c7-a717-632b3e585de7"
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
