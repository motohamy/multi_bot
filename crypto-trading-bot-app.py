import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import hashlib
import hmac
import urllib.parse
import requests
import datetime
import threading
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

class KrakenTradingBot:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = base64.b64decode(api_secret)
        self.base_url = "https://api.kraken.com"
        self.api_version = "0"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Kraken Trading Bot v1.0"})
        
        # Trading pair configurations
        self.VALID_PAIRS = {
            'XBTUSD': {'min': 0.0002, 'decimals': 5, 'base': 'XBT', 'quote': 'USD', 'display': 'BTC/USD'},
            'ETHUSD': {'min': 0.02, 'decimals': 4, 'base': 'ETH', 'quote': 'USD', 'display': 'ETH/USD'},
            'XRPUSD': {'min': 30.0, 'decimals': 0, 'base': 'XRP', 'quote': 'USD', 'display': 'XRP/USD'},
            'DOTUSD': {'min': 1.0, 'decimals': 2, 'base': 'DOT', 'quote': 'USD', 'display': 'DOT/USD'},
            'ADAUSD': {'min': 50.0, 'decimals': 1, 'base': 'ADA', 'quote': 'USD', 'display': 'ADA/USD'},
            'LTCUSD': {'min': 0.1, 'decimals': 3, 'base': 'LTC', 'quote': 'USD', 'display': 'LTC/USD'},
            'LINKUSD': {'min': 0.5, 'decimals': 2, 'base': 'LINK', 'quote': 'USD', 'display': 'LINK/USD'},
            'UNIUSD': {'min': 0.5, 'decimals': 2, 'base': 'UNI', 'quote': 'USD', 'display': 'UNI/USD'},
            
            'XBTGBP': {'min': 0.0002, 'decimals': 5, 'base': 'XBT', 'quote': 'GBP', 'display': 'BTC/GBP'},
            'ETHGBP': {'min': 0.02, 'decimals': 4, 'base': 'ETH', 'quote': 'GBP', 'display': 'ETH/GBP'},
            'XRPGBP': {'min': 30.0, 'decimals': 0, 'base': 'XRP', 'quote': 'GBP', 'display': 'XRP/GBP'},
            
            'XBTEUR': {'min': 0.0002, 'decimals': 5, 'base': 'XBT', 'quote': 'EUR', 'display': 'BTC/EUR'},
            'ETHEUR': {'min': 0.02, 'decimals': 4, 'base': 'ETH', 'quote': 'EUR', 'display': 'ETH/EUR'},
            'XRPEUR': {'min': 30.0, 'decimals': 0, 'base': 'XRP', 'quote': 'EUR', 'display': 'XRP/EUR'},
        }
        
        # Timeframe mapping for OHLC data
        self.TIMEFRAMES = {
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
        }
        
        # Keep track of open positions
        self.open_positions = {}
        
        # Track last trade times for cooldown
        self.last_trade_time = {}
        
        # Track price history for each pair
        self.price_history = {}
        
        # Track highest prices after position entry for trailing stop
        self.highest_prices = {}
        
        # Flag to control the bot running state
        self.running = False
        
        # Trading status for UI
        self.status_messages = []
        self.trade_history = []

    def _get_nonce(self):
        """Generate a unique nonce for API requests"""
        return int(time.time() * 1000)

    def _generate_signature(self, urlpath, data):
        """Generate API request signature"""
        post_data = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + post_data).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(self.api_secret, message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()

    def _private_request(self, method, data=None):
        """Make a private API request"""
        if data is None:
            data = {}
        urlpath = f"/{self.api_version}/private/{method}"
        data["nonce"] = self._get_nonce()
        signature = self._generate_signature(urlpath, data)
        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature
        }
        try:
            response = self.session.post(
                self.base_url + urlpath,
                data=data,
                headers=headers
            )
            return response.json()
        except Exception as e:
            self.add_status(f"API error: {str(e)}", "error")
            return {"error": [str(e)], "result": {}}

    def _public_request(self, method, data=None):
        """Make a public API request"""
        if data is None:
            data = {}
        urlpath = f"/{self.api_version}/public/{method}"
        try:
            response = self.session.get(
                self.base_url + urlpath,
                params=data
            )
            return response.json()
        except Exception as e:
            self.add_status(f"API error: {str(e)}", "error")
            return {"error": [str(e)], "result": {}}

    def add_status(self, message, level="info"):
        """Add a status message with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_messages.append({"time": timestamp, "message": message, "level": level})
        
        # Keep only the latest 100 messages
        if len(self.status_messages) > 100:
            self.status_messages = self.status_messages[-100:]

    def get_account_balance(self):
        """Get all account balances"""
        response = self._private_request("Balance")
        if 'error' in response and response['error']:
            self.add_status(f"Error getting balance: {response['error']}", "error")
            return {}
            
        if 'result' not in response:
            self.add_status("Invalid balance response", "error")
            return {}
            
        # Filter for non-zero balances and format for display
        balances = {k: float(v) for k, v in response['result'].items() if float(v) > 0}
        
        # Map asset codes to readable names
        readable_balances = {}
        for asset, amount in balances.items():
            # Handle asset name prefixes
            clean_asset = asset
            if asset.startswith('X') and len(asset) > 1:
                clean_asset = asset[1:]
            if asset.startswith('Z') and len(asset) > 1:
                clean_asset = asset[1:]
                
            readable_balances[clean_asset] = amount
            
        return readable_balances

    def get_tradable_assets(self, quote_currency):
        """Get list of valid trading pairs for the given quote currency"""
        assets = []
        for pair, info in self.VALID_PAIRS.items():
            if info['quote'] == quote_currency:
                assets.append({
                    'pair': pair,
                    'asset': info['base'],
                    'display': info['display']
                })
        return assets

    def get_ticker_price(self, pair):
        """Get current market price for a trading pair"""
        data = self._public_request("Ticker", {"pair": pair})
        if 'error' in data and data['error']:
            self.add_status(f"Error getting price for {pair}: {data['error']}", "error")
            return None
            
        if pair not in data['result'] and len(data['result']) > 0:
            # Try to get the first result if pair key doesn't match exactly
            pair = list(data['result'].keys())[0]
            
        try:
            # Get last price (c), ask price (a), and bid price (b)
            last_price = float(data['result'][pair]['c'][0])
            ask_price = float(data['result'][pair]['a'][0])
            bid_price = float(data['result'][pair]['b'][0])
            return {
                'last': last_price,
                'ask': ask_price,
                'bid': bid_price
            }
        except (KeyError, IndexError):
            self.add_status(f"Invalid price data for {pair}", "error")
            return None

    def get_ohlc_data(self, pair, timeframe='1h', count=100):
        """Get OHLC (candlestick) data for a pair"""
        interval = self.TIMEFRAMES.get(timeframe, 60)  # Default to 1h
        data = self._public_request("OHLC", {
            "pair": pair,
            "interval": interval
        })
        
        if 'error' in data and data['error']:
            self.add_status(f"Error getting OHLC data: {data['error']}", "error")
            return pd.DataFrame()
            
        # Check if pair is in result
        result_key = pair
        if pair not in data['result']:
            # Try to get the first key that's not 'last'
            for key in data['result']:
                if key != 'last':
                    result_key = key
                    break
        
        if result_key not in data['result']:
            self.add_status(f"No OHLC data found for {pair}", "error")
            return pd.DataFrame()
            
        ohlc_data = data['result'][result_key]
        
        # Create DataFrame with column names
        df = pd.DataFrame(ohlc_data, columns=[
            'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Take only the most recent 'count' entries
        if len(df) > count:
            df = df.tail(count)
            
        return df

    def update_price_history(self, pair, timeframe):
        """Update price history for a pair"""
        df = self.get_ohlc_data(pair, timeframe)
        if not df.empty:
            self.price_history[pair] = df
            return True
        return False

    def detect_price_drop(self, pair, drop_percentage, timeframe):
        """Detect if price has dropped by the specified percentage in the timeframe"""
        if pair not in self.price_history:
            if not self.update_price_history(pair, timeframe):
                return False
                
        df = self.price_history[pair]
        if df.empty:
            return False
            
        # Get current price
        ticker = self.get_ticker_price(pair)
        if ticker is None:
            return False
            
        current_price = ticker['last']
        
        # Find the highest price in the timeframe
        highest_price = df['high'].max()
        
        # Calculate percentage drop
        drop = (highest_price - current_price) / highest_price * 100
        
        return drop >= drop_percentage

    def calculate_position_size(self, pair, percentage, quote_currency):
        """Calculate position size based on available balance and percentage"""
        # Get available balance for quote currency
        balances = self.get_account_balance()
        
        currency_key = quote_currency
        if currency_key not in balances:
            self.add_status(f"No balance available for {quote_currency}", "error")
            return 0
            
        available_balance = balances[currency_key]
        
        # Calculate amount to use
        amount_to_use = available_balance * (percentage / 100)
        
        # Get current price
        ticker = self.get_ticker_price(pair)
        if ticker is None:
            return 0
            
        # Calculate position size in base currency
        position_size = amount_to_use / ticker['ask']
        
        # Round to appropriate decimals
        if pair in self.VALID_PAIRS:
            decimals = self.VALID_PAIRS[pair]['decimals']
            min_size = self.VALID_PAIRS[pair]['min']
            position_size = round(position_size, decimals)
            
            # Check if position size meets minimum requirement
            if position_size < min_size:
                self.add_status(f"Calculated position size {position_size} is below minimum {min_size} for {pair}", "warning")
                return 0
                
        return position_size

    def create_buy_order(self, pair, volume):
        """Create a market buy order"""
        if pair not in self.VALID_PAIRS:
            self.add_status(f"Invalid trading pair: {pair}", "error")
            return None
            
        pair_info = self.VALID_PAIRS[pair]
        decimals = pair_info['decimals']
        volume = round(volume, decimals)
        
        # Check if volume meets minimum requirement
        if volume < pair_info['min']:
            self.add_status(f"Order volume {volume} below minimum {pair_info['min']} for {pair}", "error")
            return None
        
        data = {
            "pair": pair,
            "type": "buy",
            "ordertype": "market",
            "volume": str(volume)
        }
        
        self.add_status(f"Creating buy order for {volume} {pair_info['base']} of {pair}", "info")
        response = self._private_request("AddOrder", data)
        
        if 'error' in response and response['error']:
            self.add_status(f"Buy order failed: {response['error']}", "error")
            return None
            
        # Record trade in history
        ticker = self.get_ticker_price(pair)
        price = ticker['ask'] if ticker else 0
        
        order_info = {
            'type': 'buy',
            'pair': pair,
            'volume': volume,
            'price': price,
            'time': datetime.datetime.now(),
            'txid': response['result']['txid'][0] if 'result' in response and 'txid' in response['result'] else 'unknown'
        }
        
        self.trade_history.append(order_info)
        
        # Update open positions
        self.open_positions[pair] = {
            'volume': volume,
            'entry_price': price,
            'entry_time': datetime.datetime.now(),
            'highest_price': price,
            'stop_loss': None
        }
        
        self.last_trade_time[pair] = datetime.datetime.now()
        self.add_status(f"Buy order executed: {volume} {pair_info['base']} at {price} {pair_info['quote']}", "success")
        
        return response

    def create_sell_order(self, pair, volume):
        """Create a market sell order"""
        if pair not in self.VALID_PAIRS:
            self.add_status(f"Invalid trading pair: {pair}", "error")
            return None
            
        pair_info = self.VALID_PAIRS[pair]
        decimals = pair_info['decimals']
        volume = round(volume, decimals)
        
        # Check if volume meets minimum requirement
        if volume < pair_info['min']:
            self.add_status(f"Order volume {volume} below minimum {pair_info['min']} for {pair}", "error")
            return None
        
        data = {
            "pair": pair,
            "type": "sell",
            "ordertype": "market",
            "volume": str(volume)
        }
        
        self.add_status(f"Creating sell order for {volume} {pair_info['base']} of {pair}", "info")
        response = self._private_request("AddOrder", data)
        
        if 'error' in response and response['error']:
            self.add_status(f"Sell order failed: {response['error']}", "error")
            return None
            
        # Record trade in history
        ticker = self.get_ticker_price(pair)
        price = ticker['bid'] if ticker else 0
        
        order_info = {
            'type': 'sell',
            'pair': pair,
            'volume': volume,
            'price': price,
            'time': datetime.datetime.now(),
            'txid': response['result']['txid'][0] if 'result' in response and 'txid' in response['result'] else 'unknown'
        }
        
        self.trade_history.append(order_info)
        
        # Remove from open positions
        if pair in self.open_positions:
            entry_price = self.open_positions[pair]['entry_price']
            profit_pct = (price - entry_price) / entry_price * 100
            del self.open_positions[pair]
            self.add_status(f"Sell order executed: {volume} {pair_info['base']} at {price} {pair_info['quote']} (P/L: {profit_pct:.2f}%)", "success")
        else:
            self.add_status(f"Sell order executed: {volume} {pair_info['base']} at {price} {pair_info['quote']}", "success")
        
        return response

    def update_trailing_stop_loss(self, pair, current_price, profit_threshold, stop_loss_pct):
        """Update trailing stop loss for a position"""
        if pair not in self.open_positions:
            return None
            
        position = self.open_positions[pair]
        entry_price = position['entry_price']
        
        # Calculate current profit percentage
        profit_pct = (current_price - entry_price) / entry_price * 100
        
        # Update highest price if current price is higher
        if current_price > position['highest_price']:
            position['highest_price'] = current_price
            
        # Only activate trailing stop if profit threshold is reached
        if profit_pct >= profit_threshold:
            # Calculate stop loss price
            stop_loss = position['highest_price'] * (1 - stop_loss_pct / 100)
            
            # Update stop loss if it's higher than current stop loss or not set
            if position['stop_loss'] is None or stop_loss > position['stop_loss']:
                position['stop_loss'] = stop_loss
                
        return position['stop_loss']

    def check_stop_loss_triggered(self, pair, current_price):
        """Check if stop loss is triggered"""
        if pair not in self.open_positions:
            return False
            
        position = self.open_positions[pair]
        stop_loss = position.get('stop_loss')
        
        if stop_loss is not None and current_price <= stop_loss:
            return True
            
        return False

    def check_max_time_expired(self, pair, max_hours):
        """Check if position has been open for longer than max_hours"""
        if pair not in self.open_positions:
            return False
            
        position = self.open_positions[pair]
        entry_time = position['entry_time']
        
        time_held = (datetime.datetime.now() - entry_time).total_seconds() / 3600
        
        return time_held >= max_hours

    def check_cooldown(self, pair, cooldown_minutes):
        """Check if a pair is in cooldown period"""
        if pair not in self.last_trade_time:
            return False
            
        last_time = self.last_trade_time[pair]
        time_since_last = (datetime.datetime.now() - last_time).total_seconds() / 60
        
        return time_since_last < cooldown_minutes

    def process_trading_logic(self, config):
        """Process trading logic for all selected pairs"""
        try:
            # Update status
            self.add_status("Processing trading logic...", "info")
            
            # Get selected pairs based on quote currency
            quote_currency = config['quote_currency']
            selected_cryptos = config['cryptos']
            
            # Get all available assets for the quote currency
            available_assets = self.get_tradable_assets(quote_currency)
            
            # Filter for selected cryptos
            pairs_to_check = []
            for asset in available_assets:
                if asset['asset'] in selected_cryptos:
                    pairs_to_check.append(asset['pair'])
            
            # Process each pair
            for pair in pairs_to_check:
                # Update price history for drop detection
                timeframe = config['timeframe']
                self.update_price_history(pair, timeframe)
                
                # Get current price
                ticker = self.get_ticker_price(pair)
                if ticker is None:
                    continue
                current_price = ticker['last']
                
                # Check open positions for this pair
                if pair in self.open_positions:
                    position = self.open_positions[pair]
                    
                    # Update trailing stop loss
                    self.update_trailing_stop_loss(
                        pair, 
                        current_price, 
                        config['profit_threshold'], 
                        config['stop_loss_percentage']
                    )
                    
                    # Check if stop loss is triggered
                    if self.check_stop_loss_triggered(pair, current_price):
                        # Sell position
                        volume = position['volume']
                        self.add_status(f"Stop loss triggered for {pair} at {current_price}", "info")
                        self.create_sell_order(pair, volume)
                        continue
                    
                    # Check if max time expired
                    if config['max_hours'] > 0 and self.check_max_time_expired(pair, config['max_hours']):
                        # Sell position
                        volume = position['volume']
                        self.add_status(f"Max time expired for {pair}", "info")
                        self.create_sell_order(pair, volume)
                        continue
                    
                    # If we're still holding, continue to next pair
                    continue
                
                # Check if we can open a new position
                if len(self.open_positions) >= config['max_positions']:
                    continue
                    
                # Check if pair is in cooldown
                if self.check_cooldown(pair, config['cooldown_minutes']):
                    continue
                    
                # Check for price drop
                if self.detect_price_drop(pair, config['drop_percentage'], timeframe):
                    self.add_status(f"Price drop detected for {pair}: {config['drop_percentage']}%", "info")
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        pair, 
                        config['balance_percentage'], 
                        quote_currency
                    )
                    
                    if position_size > 0:
                        # Create buy order
                        self.create_buy_order(pair, position_size)
            
            # Add status update
            self.add_status("Trading logic completed", "info")
            
        except Exception as e:
            self.add_status(f"Error in trading logic: {str(e)}", "error")

    def start_bot(self, config):
        """Start the trading bot with the given configuration"""
        self.running = True
        self.add_status("Trading bot started", "success")
        
        # Start bot in a separate thread
        def run_bot():
            while self.running:
                try:
                    self.process_trading_logic(config)
                    # Wait for the configured interval before next check
                    time.sleep(config['interval_seconds'])
                except Exception as e:
                    self.add_status(f"Bot error: {str(e)}", "error")
                    time.sleep(30)  # Wait before retry
        
        bot_thread = threading.Thread(target=run_bot)
        bot_thread.daemon = True
        bot_thread.start()

    def stop_bot(self):
        """Stop the trading bot"""
        self.running = False
        self.add_status("Trading bot stopped", "warning")

    def is_running(self):
        """Check if the bot is running"""
        return self.running

def create_app():
    """Create the Streamlit application"""
    # Set page config
    st.set_page_config(
        page_title="Kraken Trading Bot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for bot instance
    if 'bot' not in st.session_state:
        st.session_state.bot = None
        
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if 'price_charts' not in st.session_state:
        st.session_state.price_charts = {}
        
    # App title
    st.title("Kraken Crypto Trading Bot")
    
    # Sidebar for configuration
    st.sidebar.header("Bot Configuration")
    
    # Authentication section
    with st.sidebar.expander("API Configuration", expanded=not st.session_state.authenticated):
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        
        if st.button("Connect to Kraken"):
            if api_key and api_secret:
                # Initialize bot with credentials
                try:
                    bot = KrakenTradingBot(api_key, api_secret)
                    # Test connection by getting balance
                    balances = bot.get_account_balance()
                    if balances:
                        st.session_state.bot = bot
                        st.session_state.authenticated = True
                        st.success("Successfully connected to Kraken!")
                    else:
                        st.error("Failed to connect to Kraken. Please check your API credentials.")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
            else:
                st.warning("Please enter your API key and secret.")
    
    # Main configuration (only if authenticated)
    if st.session_state.authenticated and st.session_state.bot:
        bot = st.session_state.bot
        
        # Trading configuration
        with st.sidebar.expander("Trading Settings", expanded=True):
            quote_currencies = ["USD", "GBP", "EUR"]
            quote_currency = st.selectbox("Quote Currency", quote_currencies)
            
            # Get available cryptocurrencies for the selected quote currency
            available_assets = bot.get_tradable_assets(quote_currency)
            crypto_options = [asset['asset'] for asset in available_assets]
            
            selected_cryptos = st.multiselect(
                "Select Cryptocurrencies", 
                crypto_options,
                default=crypto_options[:3] if crypto_options else []
            )
            
            balance_percentage = st.slider(
                "Percentage of Balance per Trade", 
                min_value=1.0, 
                max_value=25.0, 
                value=5.0,
                step=0.5
            )
            
            # Price drop settings
            drop_percentage = st.slider(
                "Price Drop Percentage", 
                min_value=1.0, 
                max_value=20.0, 
                value=5.0,
                step=0.5
            )
            
            timeframe_options = list(bot.TIMEFRAMES.keys())
            timeframe = st.selectbox("Timeframe", timeframe_options, index=timeframe_options.index("1h"))
            
            interval_options = {
                "30 seconds": 30,
                "1 minute": 60,
                "5 minutes": 300,
                "15 minutes": 900,
                "30 minutes": 1800
            }
            interval_selection = st.selectbox("Check Interval", list(interval_options.keys()), index=1)
            interval_seconds = interval_options[interval_selection]
        
        # Buy/Sell settings
        with st.sidebar.expander("Position Management"):
            max_positions = st.slider(
                "Maximum Open Positions", 
                min_value=1, 
                max_value=10, 
                value=3
            )
            
            cooldown_minutes = st.slider(
                "Cooldown Between Trades (minutes)", 
                min_value=0, 
                max_value=1440, 
                value=60
            )
            
            one_position_per_crypto = st.checkbox("One Position Per Cryptocurrency", value=True)
            
            max_hours = st.slider(
                "Maximum Hours to Keep Position", 
                min_value=0, 
                max_value=168, 
                value=48,
                help="0 means no time limit"
            )
            
            profit_threshold = st.slider(
                "Profit Threshold to Activate Stop-Loss (%)", 
                min_value=0.5, 
                max_value=20.0, 
                value=3.0,
                step=0.5
            )
            
            stop_loss_percentage = st.slider(
                "Trailing Stop-Loss Percentage", 
                min_value=0.5, 
                max_value=10.0, 
                value=2.0,
                step=0.5
            )
            
            only_sell_with_profit = st.checkbox("Only Sell Positions with Profit", value=False)
        
        # Compile configuration
        config = {
            'quote_currency': quote_currency,
            'cryptos': selected_cryptos,
            'balance_percentage': balance_percentage,
            'drop_percentage': drop_percentage,
            'timeframe': timeframe,
            'interval_seconds': interval_seconds,
            'max_positions': max_positions,
            'cooldown_minutes': cooldown_minutes,
            'one_position_per_crypto': one_position_per_crypto,
            'max_hours': max_hours,
            'profit_threshold': profit_threshold,
            'stop_loss_percentage': stop_loss_percentage,
            'only_sell_with_profit': only_sell_with_profit
        }
        
        # Start/Stop bot
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if not bot.is_running():
                if st.button("Start Bot", type="primary"):
                    bot.start_bot(config)
        
        with col2:
            if bot.is_running():
                if st.button("Stop Bot", type="secondary"):
                    bot.stop_bot()
        
        # Main content area
        # Display account balances
        st.header("Account Balance")
        balances = bot.get_account_balance()
        
        if balances:
            balance_df = pd.DataFrame([
                {"Currency": currency, "Balance": amount}
                for currency, amount in balances.items()
            ])
            st.dataframe(balance_df, use_container_width=True)
        else:
            st.warning("No balance information available")
        
        # Display current prices for selected cryptocurrencies
        st.header("Current Prices")
        
        price_data = []
        for crypto in selected_cryptos:
            for asset in available_assets:
                if asset['asset'] == crypto:
                    pair = asset['pair']
                    ticker = bot.get_ticker_price(pair)
                    if ticker:
                        price_data.append({
                            "Pair": asset['display'],
                            "Last Price": ticker['last'],
                            "Ask": ticker['ask'],
                            "Bid": ticker['bid']
                        })
        
        if price_data:
            price_df = pd.DataFrame(price_data)
            st.dataframe(price_df, use_container_width=True)
        else:
            st.warning("No price data available")
        
        # Display open positions
        st.header("Open Positions")
        
        if bot.open_positions:
            position_data = []
            for pair, position in bot.open_positions.items():
                pair_info = bot.VALID_PAIRS.get(pair, {})
                display_name = pair_info.get('display', pair)
                
                ticker = bot.get_ticker_price(pair)
                current_price = ticker['last'] if ticker else 0
                
                entry_price = position['entry_price']
                profit_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                
                entry_time = position['entry_time']
                time_held = (datetime.datetime.now() - entry_time).total_seconds() / 3600
                
                position_data.append({
                    "Pair": display_name,
                    "Volume": position['volume'],
                    "Entry Price": entry_price,
                    "Current Price": current_price,
                    "P/L %": profit_pct,
                    "Stop Loss": position.get('stop_loss', 'Not Set'),
                    "Time Held (h)": round(time_held, 1)
                })
            
            position_df = pd.DataFrame(position_data)
            st.dataframe(position_df, use_container_width=True)
        else:
            st.info("No open positions")
        
        # Display price charts
        st.header("Price Charts")
        
        # Select crypto for chart
        chart_crypto = st.selectbox(
            "Select Cryptocurrency for Chart",
            selected_cryptos if selected_cryptos else ["BTC"]
        )
        
        # Find pair for selected crypto
        chart_pair = None
        for asset in available_assets:
            if asset['asset'] == chart_crypto:
                chart_pair = asset['pair']
                break
        
        if chart_pair:
            # Get OHLC data
            df = bot.get_ohlc_data(chart_pair, timeframe)
            
            if not df.empty:
                # Create candlestick chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   row_heights=[0.7, 0.3],
                                   vertical_spacing=0.05)
                
                # Add candlestick trace
                fig.add_trace(
                    go.Candlestick(
                        x=df['time'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Add volume trace
                fig.add_trace(
                    go.Bar(
                        x=df['time'],
                        y=df['volume'],
                        name="Volume"
                    ),
                    row=2, col=1
                )
                
                # Add buy/sell markers if this pair has trades
                trades_for_pair = [t for t in bot.trade_history if t['pair'] == chart_pair]
                if trades_for_pair:
                    buy_times = [t['time'] for t in trades_for_pair if t['type'] == 'buy']
                    buy_prices = [t['price'] for t in trades_for_pair if t['type'] == 'buy']
                    
                    sell_times = [t['time'] for t in trades_for_pair if t['type'] == 'sell']
                    sell_prices = [t['price'] for t in trades_for_pair if t['type'] == 'sell']
                    
                    if buy_times:
                        fig.add_trace(
                            go.Scatter(
                                x=buy_times,
                                y=buy_prices,
                                mode='markers',
                                marker=dict(color='green', size=10, symbol='triangle-up'),
                                name="Buy"
                            ),
                            row=1, col=1
                        )
                    
                    if sell_times:
                        fig.add_trace(
                            go.Scatter(
                                x=sell_times,
                                y=sell_prices,
                                mode='markers',
                                marker=dict(color='red', size=10, symbol='triangle-down'),
                                name="Sell"
                            ),
                            row=1, col=1
                        )
                
                # Add stop loss line if there's an open position
                if chart_pair in bot.open_positions:
                    position = bot.open_positions[chart_pair]
                    if position.get('stop_loss'):
                        fig.add_trace(
                            go.Scatter(
                                x=[df['time'].iloc[-1]],
                                y=[position['stop_loss']],
                                mode='markers+lines',
                                line=dict(color='red', dash='dash'),
                                name="Stop Loss"
                            ),
                            row=1, col=1
                        )
                
                # Update layout
                fig.update_layout(
                    title=f"{chart_crypto}/{quote_currency} ({timeframe} timeframe)",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No chart data available for {chart_crypto}")
        
        # Display trade history
        st.header("Trade History")
        
        if bot.trade_history:
            history_df = pd.DataFrame([
                {
                    "Time": trade['time'].strftime("%Y-%m-%d %H:%M:%S"),
                    "Type": trade['type'].capitalize(),
                    "Pair": trade['pair'],
                    "Volume": trade['volume'],
                    "Price": trade['price'],
                    "Transaction ID": trade['txid']
                }
                for trade in bot.trade_history
            ])
            st.dataframe(history_df.sort_values("Time", ascending=False), use_container_width=True)
        else:
            st.info("No trade history available")
        
        # Display bot status
        st.header("Bot Status Log")
        
        if bot.status_messages:
            log_df = pd.DataFrame([
                {
                    "Time": msg['time'],
                    "Message": msg['message'],
                    "Level": msg['level']
                }
                for msg in bot.status_messages
            ])
            
            # Style status based on level
            def highlight_status(s):
                if s['Level'] == 'error':
                    return ['background-color: #FFCDD2'] * len(s)
                elif s['Level'] == 'warning':
                    return ['background-color: #FFECB3'] * len(s)
                elif s['Level'] == 'success':
                    return ['background-color: #C8E6C9'] * len(s)
                else:
                    return [''] * len(s)
            
            st.dataframe(
                log_df.sort_values("Time", ascending=False).style.apply(highlight_status, axis=1),
                use_container_width=True
            )
        else:
            st.info("No status messages")
    
    # Display login screen if not authenticated
    elif not st.session_state.authenticated:
        st.info("Please enter your Kraken API credentials in the sidebar to get started.")
        
        # Usage instructions
        with st.expander("Bot Usage Instructions"):
            st.markdown("""
            ### How to use this trading bot:
            
            1. **API Configuration**
               - Generate API keys on your Kraken account with trading permissions
               - Enter the API key and secret in the sidebar
               - Click "Connect to Kraken" to authenticate
            
            2. **Trading Settings**
               - Select your quote currency (USD, GBP, EUR)
               - Choose which cryptocurrencies to trade
               - Set the balance percentage to use per trade
               - Configure the price drop threshold and timeframe
               - Set the monitoring interval
            
            3. **Position Management**
               - Set maximum concurrent positions
               - Configure cooldown between trades
               - Set trailing stop-loss parameters
               - Configure maximum holding time
            
            4. **Start the Bot**
               - Click "Start Bot" to begin automated trading
               - Monitor results in the dashboard
               - Stop the bot at any time
            
            ### Trading Strategy Description
            
            This bot implements a "buy the dip" strategy with trailing stop-loss:
            
            - **Entry Strategy:** The bot buys when a cryptocurrency drops by your configured percentage within the selected timeframe
            - **Exit Strategy:** The bot uses a trailing stop-loss that activates once profit reaches your threshold
            - **Risk Management:** Maximum positions, cooldown periods, and position sizing protect your capital
            """)

if __name__ == "__main__":
    create_app()
