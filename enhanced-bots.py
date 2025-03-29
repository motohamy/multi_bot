import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import time
import json
import logging
import requests
import sqlite3
from datetime import datetime, timedelta
import ccxt
import talib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union
import warnings
import importlib.util
import sys
import threading

# Suppress warnings
warnings.filterwarnings('ignore')

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_trading.log'),
        logging.StreamHandler()
    ]
)

# Constants - Adjusted for shorter-term trading
SEQ_LENGTH = 40
HIDDEN_SIZE = 256
DROPOUT = 0.4
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 2e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PREDICTION_HORIZON = 3  # Reduced from 5 for shorter-term trades
RETRAINING_INTERVAL = 24  # Retrain every 24 hours

# Default webhook URL (will be overridden by config)
DEFAULT_WEBHOOK_URL = 'https://api.primeautomation.ai/webhook/ChartPrime/9c3a902a-83e4-48c7-a717-632b3e585de7'

# Default webhook mappings in case config file doesn't exist
DEFAULT_WEBHOOK_URLS = {
    'BTC/USDT': 'https://api.primeautomation.ai/webhook/ChartPrime/9c3a902a-83e4-48c7-a717-632b3e585de7',
    'SOL/USDT': 'https://api.primeautomation.ai/webhook/ChartPrime/6091d974-83af-4b10-bcb0-6953eaff129a',
    'default': 'https://api.primeautomation.ai/webhook/ChartPrime/9c3a902a-83e4-48c7-a717-632b3e585de7'
}

# Default ticker map
DEFAULT_TICKER_MAP = {
    'BTC/USDT': 'BTCUSDT',
    'SOL/USDT': 'SOL',
}

# Load configuration
CONFIG_FILE = 'crypto_config.py'

try:
    if os.path.exists(CONFIG_FILE):
        spec = importlib.util.spec_from_file_location("crypto_config", CONFIG_FILE)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Get webhook URLs and ticker mappings
        WEBHOOK_URLS = getattr(config_module, 'WEBHOOK_URLS', DEFAULT_WEBHOOK_URLS)
        TICKER_MAP = getattr(config_module, 'TICKER_MAP', DEFAULT_TICKER_MAP)
        SUCCESSFUL_FORMATS = getattr(config_module, 'SUCCESSFUL_FORMATS', {})
        
        logging.info(f"Loaded configuration from {CONFIG_FILE}")
    else:
        WEBHOOK_URLS = DEFAULT_WEBHOOK_URLS
        TICKER_MAP = DEFAULT_TICKER_MAP
        SUCCESSFUL_FORMATS = {}
        
        # Create a default config file
        with open(CONFIG_FILE, 'w') as f:
            f.write('"""' + '\n')
            f.write('Configuration file for CryptoPrime Signal Generator.' + '\n')
            f.write('Edit this file to configure webhooks and ticker formats for different cryptocurrencies.' + '\n')
            f.write('"""' + '\n\n')
            f.write('# Webhook URLs for different cryptocurrencies' + '\n')
            f.write('WEBHOOK_URLS = ' + json.dumps(WEBHOOK_URLS, indent=4) + '\n\n')
            f.write('# Map from trading pairs (exchange format) to API tickers' + '\n')
            f.write('TICKER_MAP = ' + json.dumps(TICKER_MAP, indent=4) + '\n\n')
            f.write('# Dictionary to store successful formats during runtime' + '\n')
            f.write('SUCCESSFUL_FORMATS = {}' + '\n')
        
        logging.info(f"Created default configuration file: {CONFIG_FILE}")
except Exception as e:
    logging.error(f"Error loading configuration: {str(e)}")
    WEBHOOK_URLS = DEFAULT_WEBHOOK_URLS
    TICKER_MAP = DEFAULT_TICKER_MAP
    SUCCESSFUL_FORMATS = {}

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

#################################
# 1. MODEL ARCHITECTURE
#################################

class FeatureAttention(nn.Module):
    """
    Dynamic feature importance weighting module
    """
    def __init__(self, num_features):
        super().__init__()
        self.importance_weights = nn.Parameter(torch.ones(num_features))
        self.selector = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # Calculate feature importance scores
        feature_scores = self.selector(x.mean(dim=1))
        # Apply importance weighting
        return x * feature_scores.unsqueeze(1) * self.importance_weights

class MultiScaleProcessor(nn.Module):
    """
    Multi-scale temporal feature extractor
    """
    def __init__(self, input_size):
        super().__init__()
        self.short_term = nn.Conv1d(input_size, 96, kernel_size=3, padding=1)
        self.medium_term = nn.Conv1d(input_size, 96, kernel_size=7, padding=3)
        self.long_term = nn.Conv1d(input_size, 96, kernel_size=15, padding=7)
        self.very_long_term = nn.Conv1d(input_size, 96, kernel_size=30, padding=15)
        
        # Add batch normalization
        self.bn_short = nn.BatchNorm1d(96)
        self.bn_medium = nn.BatchNorm1d(96)
        self.bn_long = nn.BatchNorm1d(96)
        self.bn_very_long = nn.BatchNorm1d(96)
        
    def forward(self, x):
        # Convert to shape [batch, features, sequence]
        x = x.permute(0, 2, 1)
        
        # Apply different scale convolutions with batch norm
        short = F.leaky_relu(self.bn_short(self.short_term(x)))
        medium = F.leaky_relu(self.bn_medium(self.medium_term(x)))
        long = F.leaky_relu(self.bn_long(self.long_term(x)))
        very_long = F.leaky_relu(self.bn_very_long(self.very_long_term(x)))
        
        # Apply pooling for different time horizons
        short = F.max_pool1d(short, 2)
        medium = F.avg_pool1d(medium, 2)
        long = F.max_pool1d(long, 2)
        very_long = F.adaptive_avg_pool1d(very_long, short.size(2))
        
        # Combine multi-scale features
        multi_scale = torch.cat([short, medium, long, very_long], dim=1)
        
        # Convert back to [batch, sequence, features]
        return multi_scale.permute(0, 2, 1)

class EnhancedTemporalModel(nn.Module):
    """
    Transformer-LSTM hybrid architecture
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # LSTM component for sequential processing
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True,
            dropout=DROPOUT
        )
        
        # Multi-head attention for improved context modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size*2, 
            num_heads=8,
            dropout=DROPOUT
        )
        
        # Layer normalization for stability
        self.temporal_norm1 = nn.LayerNorm(hidden_size*2)
        self.temporal_norm2 = nn.LayerNorm(hidden_size*2)
        
        # Feed-forward network for transformer-style processing
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size*4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size*4, hidden_size*2)
        )
        
    def forward(self, x):
        # Initial projection with normalization
        x = self.input_norm(F.gelu(self.input_projection(x)))
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Apply normalization
        norm_lstm_out = self.temporal_norm1(lstm_out)
        
        # Self-attention mechanism (transformer style)
        attn_out, _ = self.attention(
            norm_lstm_out.transpose(0, 1),
            norm_lstm_out.transpose(0, 1),
            norm_lstm_out.transpose(0, 1)
        )
        
        # First residual connection and normalization
        attn_out = norm_lstm_out + attn_out.transpose(0, 1)
        norm_attn_out = self.temporal_norm2(attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(norm_attn_out)
        
        # Second residual connection
        output = norm_attn_out + ffn_out
        
        return output

class MainModel(nn.Module):
    """
    Complete trading model with all enhancements
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        
        # Feature attention module
        self.feature_attention = FeatureAttention(input_size)
        
        # Multi-scale temporal processing
        self.multi_scale = MultiScaleProcessor(input_size)
        
        # Combined feature dimension after multi-scale processing
        multi_scale_size = 96 * 4  # short, medium, long, very long term features
        
        # Enhanced temporal model with transformer-LSTM hybrid
        self.temporal_model = EnhancedTemporalModel(multi_scale_size, hidden_size)
        
        # Direction prediction head (Buy, Hold, Sell)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(DROPOUT),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(DROPOUT/2),
            nn.GELU(),
            nn.Linear(hidden_size//2, 3)
        )
        
        # Stop loss prediction head
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(DROPOUT),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(DROPOUT/2),
            nn.GELU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
        
        # Take profit prediction head
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(DROPOUT),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(DROPOUT/2),
            nn.GELU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
        
        # Trade duration prediction head (new)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(DROPOUT),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.Dropout(DROPOUT/2),
            nn.GELU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply feature attention
        x_attended = self.feature_attention(x)
        
        # Apply multi-scale processing
        x_multi = self.multi_scale(x_attended)
        
        # Enhanced temporal processing
        temporal_features = self.temporal_model(x_multi)
        
        # Use the last timestep features for prediction
        last_features = temporal_features[:, -1]
        
        # Direction prediction (Buy, Hold, Sell)
        direction_logits = self.direction_head(last_features)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Stop loss prediction (0-10% range for shorter-term trades)
        stop_loss = self.sl_head(last_features) * 0.2
        
        # Take profit prediction (0-15% range for shorter-term trades)
        take_profit = self.tp_head(last_features) * 0.2
        
        # Trade duration prediction (1-48 hours)
        duration = self.duration_head(last_features) * 48 + 1
        
        return direction_probs, stop_loss, take_profit, duration, last_features

#################################
# 2. TRADE FEEDBACK SYSTEM
#################################

class TradeFeedbackSystem:
    """
    System to track trade outcomes and provide feedback for model improvement
    """
    def __init__(self, db_path='trades.db'):
        self.db_path = db_path
        self.conn = self._init_db()
        
    def _init_db(self):
        """Initialize database connection and create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            action TEXT,
            entry_price REAL,
            exit_price REAL,
            profit_loss REAL,
            profit_pct REAL,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            duration_hours REAL,
            exit_reason TEXT,
            success INTEGER,
            predicted_sl REAL,
            predicted_tp REAL,
            predicted_duration REAL
        )
        ''')
        
        # Create feedback data table for model improvements
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_feedback (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            timestamp TIMESTAMP,
            feature_data BLOB,
            actual_outcome INTEGER,
            confidence REAL
        )
        ''')
        
        conn.commit()
        return conn
    
    def record_trade_entry(self, symbol, action, entry_price, predicted_sl, predicted_tp, predicted_duration):
        """Record a new trade entry"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO trades 
            (symbol, action, entry_price, entry_time, predicted_sl, predicted_tp, predicted_duration) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, action, entry_price, datetime.now(), predicted_sl, predicted_tp, predicted_duration)
        )
        
        self.conn.commit()
        return cursor.lastrowid
    
    def record_trade_exit(self, trade_id, exit_price, exit_reason="manual"):
        """Record a trade exit and calculate profit/loss"""
        cursor = self.conn.cursor()
        
        # Get trade information
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = cursor.fetchone()
        
        if not trade:
            logging.error(f"Trade ID {trade_id} not found")
            return False
        
        # Extract trade data
        symbol = trade[1]
        action = trade[2]
        entry_price = trade[3]
        entry_time = datetime.fromisoformat(trade[7])
        exit_time = datetime.now()
        
        # Calculate duration in hours
        duration_hours = (exit_time - entry_time).total_seconds() / 3600
        
        # Calculate profit/loss
        if action == 'buy':
            profit_loss = exit_price - entry_price
            profit_pct = (profit_loss / entry_price) * 100
        elif action == 'sell':
            profit_loss = entry_price - exit_price
            profit_pct = (profit_loss / entry_price) * 100
        else:
            profit_loss = 0
            profit_pct = 0
        
        # Determine success (1 = profitable, 0 = loss)
        success = 1 if profit_pct > 0 else 0
        
        # Update trade record
        cursor.execute(
            """
            UPDATE trades 
            SET exit_price = ?, exit_time = ?, profit_loss = ?, profit_pct = ?, 
                duration_hours = ?, exit_reason = ?, success = ?
            WHERE id = ?
            """,
            (exit_price, exit_time, profit_loss, profit_pct, duration_hours, 
             exit_reason, success, trade_id)
        )
        
        self.conn.commit()
        
        logging.info(f"Trade exit recorded - Symbol: {symbol}, Action: {action}, "
                     f"P/L: {profit_pct:.2f}%, Duration: {duration_hours:.1f}h, Reason: {exit_reason}")
        
        return True
    
    def save_model_feedback(self, symbol, feature_data, actual_outcome, confidence=1.0):
        """Save feature data with actual outcome for model retraining"""
        cursor = self.conn.cursor()
        
        # Serialize feature data
        feature_data_bytes = json.dumps(feature_data.tolist()).encode()
        
        cursor.execute(
            """
            INSERT INTO model_feedback
            (symbol, timestamp, feature_data, actual_outcome, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (symbol, datetime.now(), feature_data_bytes, actual_outcome, confidence)
        )
        
        self.conn.commit()
    
    def get_feedback_data(self, symbol, limit=1000):
        """Retrieve feedback data for model retraining"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """
            SELECT feature_data, actual_outcome, confidence 
            FROM model_feedback
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, limit)
        )
        
        feedback_data = []
        for row in cursor.fetchall():
            feature_data = np.array(json.loads(row[0]))
            actual_outcome = row[1]
            confidence = row[2]
            feedback_data.append((feature_data, actual_outcome, confidence))
        
        return feedback_data
    
    def get_performance_metrics(self, symbol, days=30):
        """Get performance metrics for a symbol over a given time period"""
        cursor = self.conn.cursor()
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as winning_trades,
                AVG(profit_pct) as avg_profit,
                AVG(CASE WHEN success = 1 THEN profit_pct ELSE NULL END) as avg_win,
                AVG(CASE WHEN success = 0 THEN profit_pct ELSE NULL END) as avg_loss,
                AVG(duration_hours) as avg_duration
            FROM trades
            WHERE symbol = ? AND entry_time >= ?
            """,
            (symbol, since_date)
        )
        
        return cursor.fetchone()
    
    def get_model_adjustment_factors(self, symbol):
        """Calculate model adjustment factors based on trade history"""
        cursor = self.conn.cursor()
        
        # Get win rate for different actions
        cursor.execute(
            """
            SELECT action, 
                   COUNT(*) as total,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as wins
            FROM trades
            WHERE symbol = ? AND exit_price IS NOT NULL
            GROUP BY action
            """,
            (symbol,)
        )
        
        action_stats = {}
        for row in cursor.fetchall():
            action, total, wins = row
            win_rate = wins / total if total > 0 else 0.5
            # Calculate adjustment factor based on win rate
            # If win rate is high, we should encourage this action
            # If win rate is low, we should discourage it
            adjustment = (win_rate - 0.5) * 2  # Scale to [-1, 1]
            action_stats[action] = {
                'win_rate': win_rate,
                'adjustment': adjustment
            }
        
        # Get SL/TP effectiveness
        cursor.execute(
            """
            SELECT 
                AVG(CASE WHEN exit_reason = 'tp_hit' THEN 1 ELSE 0 END) as tp_rate,
                AVG(CASE WHEN exit_reason = 'sl_hit' THEN 1 ELSE 0 END) as sl_rate,
                AVG(predicted_tp) as avg_predicted_tp,
                AVG(predicted_sl) as avg_predicted_sl,
                AVG(CASE WHEN success = 1 THEN profit_pct ELSE NULL END) / 100 as avg_actual_win
            FROM trades
            WHERE symbol = ? AND exit_price IS NOT NULL
            """,
            (symbol,)
        )
        
        sl_tp_stats = cursor.fetchone()
        if sl_tp_stats and sl_tp_stats[0] is not None:
            tp_rate, sl_rate, avg_predicted_tp, avg_predicted_sl, avg_actual_win = sl_tp_stats
            
            # Calculate TP adjustment factor
            if avg_actual_win and avg_predicted_tp:
                tp_adjustment = avg_actual_win / avg_predicted_tp if avg_predicted_tp > 0 else 1.0
            else:
                tp_adjustment = 1.0
                
            # Calculate SL adjustment factor
            if sl_rate > 0.3:  # If stop losses are hit too often
                sl_adjustment = 1.2  # Suggest wider stop losses
            elif sl_rate < 0.1:  # If stop losses are rarely hit
                sl_adjustment = 0.9  # Suggest tighter stop losses
            else:
                sl_adjustment = 1.0
        else:
            tp_adjustment = 1.0
            sl_adjustment = 1.0
        
        return {
            'action_adjustments': action_stats,
            'tp_adjustment': tp_adjustment,
            'sl_adjustment': sl_adjustment
        }
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

#################################
# 3. SIGNAL DISPATCHER
#################################

class SignalDispatcher:
    """
    Enhanced system for reliable signal delivery
    """
    def __init__(self):
        self.webhook_urls = WEBHOOK_URLS
        self.ticker_map = TICKER_MAP
        self.max_retries = 3
        self.backoff_factor = 2
        self.active_trades = {}  # To track active trades for each symbol
    
    def get_webhook_url(self, symbol):
        """Get the correct webhook URL for the given symbol"""
        if symbol in self.webhook_urls:
            return self.webhook_urls[symbol]
        return self.webhook_urls.get('default', DEFAULT_WEBHOOK_URL)
    
    def get_api_ticker(self, symbol):
        """Convert the trading symbol to the API-compatible ticker format"""
        # First check if we have a mapping
        if symbol in self.ticker_map:
            return self.ticker_map[symbol]
        
        # Check if we've successfully used a format before
        if symbol in SUCCESSFUL_FORMATS:
            return SUCCESSFUL_FORMATS[symbol]
        
        # Default to just the base currency
        return symbol.split('/')[0]
    
    def send_webhook(self, symbol, action, price, **kwargs):
        """Send a trading signal to the appropriate webhook with enhanced reliability"""
        # Get the API-compatible ticker format
        api_ticker = self.get_api_ticker(symbol)
        
        # Get the correct webhook URL for this symbol
        webhook_url = self.get_webhook_url(symbol)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format the payload based on action type
        if action == "buy":
            payload = {
                "ticker": api_ticker,
                "action": "buy",
                "price": str(price),
                "time": current_time
            }
            message = "BUY"
        elif action == "sell":
            payload = {
                "ticker": api_ticker,
                "action": "sell",
                "price": str(price),
                "time": current_time
            }
            message = "SELL"
        elif action == "exit_buy":
            payload = {
                "ticker": api_ticker,
                "action": "exit_buy",
                "price": str(price),
                "time": current_time,
                "size": str(kwargs.get("size", "")),
                "per": str(kwargs.get("per", "")),
                "sl": str(kwargs.get("sl", "")),
                "tp": str(kwargs.get("tp", ""))
            }
            message = "EXIT BUY"
        elif action == "exit_sell":
            payload = {
                "ticker": api_ticker,
                "action": "exit_sell",
                "price": str(price),
                "time": current_time,
                "size": str(kwargs.get("size", "")),
                "per": str(kwargs.get("per", "")),
                "sl": str(kwargs.get("sl", "")),
                "tp": str(kwargs.get("tp", ""))
            }
            message = "EXIT SELL"
        
        # Add additional contextual data if provided
        for key, value in kwargs.items():
            if key not in payload and key not in ["size", "per", "sl", "tp"]:
                payload[key] = str(value)
        
        success = False
        attempted_formats = [api_ticker]
        
        # Try sending with exponential backoff
        for attempt in range(self.max_retries):
            try:
                logging.info(f"Attempt {attempt+1} sending webhook for {symbol} ({api_ticker}): {message}\n{json.dumps(payload)}")
                logging.info(f"Using webhook URL: {webhook_url}")
                
                response = requests.post(webhook_url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    logging.info(f"Webhook sent successfully: {response.text}")
                    
                    # Save the successful format
                    if symbol not in SUCCESSFUL_FORMATS:
                        SUCCESSFUL_FORMATS[symbol] = api_ticker
                        self._update_config_file()
                    
                    success = True
                    break
                else:
                    logging.error(f"Failed to send webhook. Status code: {response.status_code}, Response: {response.text}")
                    
                    # If we get a "no matching pair" error, try alternative formats
                    if response.status_code == 404 and "No matching bot pair found" in response.text:
                        alternative_format = self._try_alternative_formats(symbol, action, price, webhook_url, payload, attempted_formats)
                        if alternative_format:
                            success = True
                            break
            
            except Exception as e:
                logging.error(f"Error sending webhook (attempt {attempt+1}): {str(e)}")
            
            # Wait before retrying with exponential backoff
            if attempt < self.max_retries - 1:
                sleep_time = self.backoff_factor ** attempt
                logging.info(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
        
        return success
    
    def _try_alternative_formats(self, symbol, action, price, webhook_url, payload, attempted_formats):
        """Try alternative ticker formats"""
        # Define fallback formats based on the symbol
        fallback_formats = []
        
        if symbol == 'BTC/USDT':
            fallback_formats = ['BTCUSDT', 'BTC-USD', 'XBTUSD', 'BTC']
        elif symbol == 'SOL/USDT':
            fallback_formats = ['SOLUSDT', 'SOL-USD', 'SOL']
        else:
            base, quote = symbol.split('/')
            fallback_formats = [f"{base}{quote}", f"{base}-{quote}", base]
        
        # Try each fallback format that hasn't been attempted
        for format_to_try in fallback_formats:
            if format_to_try in attempted_formats:
                continue
            
            attempted_formats.append(format_to_try)
            
            # Update the payload with the new ticker format
            payload["ticker"] = format_to_try
            
            try:
                logging.info(f"Trying fallback ticker format: {format_to_try}")
                fallback_response = requests.post(webhook_url, json=payload, timeout=10)
                
                if fallback_response.status_code == 200:
                    logging.info(f"Webhook sent successfully with fallback format: {fallback_response.text}")
                    
                    # Update the ticker map for future use
                    self.ticker_map[symbol] = format_to_try
                    SUCCESSFUL_FORMATS[symbol] = format_to_try
                    
                    # Update the config file
                    self._update_config_file()
                    
                    return format_to_try
            
            except Exception as e:
                logging.error(f"Error trying fallback format {format_to_try}: {str(e)}")
        
        return None
    
    def _update_config_file(self):
        """Update the configuration file with successful ticker formats"""
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_content = f.read()
            
            # Update the TICKER_MAP in the config
            import re
            ticker_pattern = r"TICKER_MAP = \{.*?\}"
            ticker_replacement = f"TICKER_MAP = {json.dumps(self.ticker_map, indent=4)}"
            config_content = re.sub(ticker_pattern, ticker_replacement, config_content, flags=re.DOTALL)
            
            # Update SUCCESSFUL_FORMATS
            formats_pattern = r"SUCCESSFUL_FORMATS = \{.*?\}"
            formats_replacement = f"SUCCESSFUL_FORMATS = {json.dumps(SUCCESSFUL_FORMATS, indent=4)}"
            config_content = re.sub(formats_pattern, formats_replacement, config_content, flags=re.DOTALL)
            
            with open(CONFIG_FILE, 'w') as f:
                f.write(config_content)
            
            logging.info(f"Updated config file with new ticker mappings")
        except Exception as e:
            logging.error(f"Error updating config file: {str(e)}")
    
    def register_trade(self, symbol, trade_id, action, entry_price, sl, tp, duration):
        """Register an active trade"""
        self.active_trades[symbol] = {
            'trade_id': trade_id,
            'action': action,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'entry_time': datetime.now(),
            'expected_duration': duration
        }
    
    def get_active_trade(self, symbol):
        """Get information about an active trade"""
        return self.active_trades.get(symbol)
    
    def remove_trade(self, symbol):
        """Remove a trade after it's closed"""
        if symbol in self.active_trades:
            del self.active_trades[symbol]

#################################
# 4. DATA HANDLING
#################################

def fetch_crypto_data(symbol, timeframe='1h', lookback_days=90):
    """
    Fetch historical crypto data from ccxt compatible exchange
    """
    try:
        # Use Binance by default (can be changed to any other exchange)
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        # Calculate date range
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        
        # Convert to milliseconds timestamp
        since = int(start.timestamp() * 1000)
        
        logging.info(f"Fetching data for {symbol} from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        logging.info(f"Successfully fetched {len(df)} data points for {symbol}")
        return df
    
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        # Try fallback exchange if the first one fails
        try:
            exchange = ccxt.kucoin({'enableRateLimit': True})
            since = int(start.timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            logging.info(f"Successfully fetched {len(df)} data points for {symbol} using fallback exchange")
            return df
        except Exception as fallback_error:
            logging.error(f"Fallback exchange also failed: {str(fallback_error)}")
            return None

def calculate_features(df):
    """
    Calculate advanced technical indicators
    """
    # Make a copy to avoid fragmentation warnings
    df = df.copy()
    
    # Convert all data to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any NaN values that might have appeared
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    features = pd.DataFrame(index=df.index)
    
    # Price data
    features['open'] = df['open']
    features['high'] = df['high']
    features['low'] = df['low']
    features['close'] = df['close']
    features['volume'] = df['volume']
    
    # Basic price features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features['range'] = df['high'] - df['low']
    features['range_pct'] = features['range'] / df['close']
    
    # Volatility features
    for window in [5, 7, 14]:  # Added shorter window for faster response
        features[f'volatility_{window}d'] = features['returns'].rolling(window=window).std() * np.sqrt(window)
    
    # Relative strength of recent moves
    features['bull_power'] = df['high'] - talib.EMA(df['close'], timeperiod=13)
    features['bear_power'] = df['low'] - talib.EMA(df['close'], timeperiod=13)
    
    # Moving Averages - Added shorter timeframes for quicker signals
    for period in [3, 5, 10, 20, 50, 100, 200]:
        features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Distance from price to MA
        features[f'dist_to_sma_{period}'] = (df['close'] - features[f'sma_{period}']) / df['close']
        features[f'dist_to_ema_{period}'] = (df['close'] - features[f'ema_{period}']) / df['close']
    
    # Moving average crossovers
    features['sma_3_10_cross'] = np.where(features['sma_3'] > features['sma_10'], 1, -1)  # Faster crossover
    features['sma_5_20_cross'] = np.where(features['sma_5'] > features['sma_20'], 1, -1)
    features['sma_20_50_cross'] = np.where(features['sma_20'] > features['sma_50'], 1, -1)
    features['ema_3_10_cross'] = np.where(features['ema_3'] > features['ema_10'], 1, -1)  # Faster crossover
    features['ema_5_20_cross'] = np.where(features['ema_5'] > features['ema_20'], 1, -1)
    
    # Bollinger Bands
    for period in [10, 20]:  # Added shorter timeframe for BB
        upper, middle, lower = talib.BBANDS(
            df['close'], timeperiod=period, nbdevup=2, nbdevdn=2
        )
        features[f'bb_upper_{period}'] = upper
        features[f'bb_lower_{period}'] = lower
        features[f'bb_width_{period}'] = (upper - lower) / middle
        features[f'bb_pos_{period}'] = (df['close'] - lower) / (upper - lower)
    
    # RSI and momentum - Added shorter timeframes for quicker response
    for period in [3, 7, 14]:
        features[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
    
    # RSI divergence detection
    features['price_higher'] = df['close'] > df['close'].shift(3)
    features['rsi_lower'] = features['rsi_14'] < features['rsi_14'].shift(3)
    features['bearish_div'] = (features['price_higher'] & features['rsi_lower']).astype(float)
    
    features['price_lower'] = df['close'] < df['close'].shift(3)
    features['rsi_higher'] = features['rsi_14'] > features['rsi_14'].shift(3)
    features['bullish_div'] = (features['price_lower'] & features['rsi_higher']).astype(float)
    
    # MACD - Adjusted for faster response
    macd, signal, hist = talib.MACD(
        df['close'], fastperiod=8, slowperiod=17, signalperiod=9  # Modified for faster signals
    )
    features['macd'] = macd
    features['macd_signal'] = signal
    features['macd_hist'] = hist
    features['macd_cross'] = np.where(macd > signal, 1, -1)
    
    # MACD histogram analysis
    features['macd_hist_change'] = features['macd_hist'].diff(1)
    features['macd_hist_slope'] = features['macd_hist'].diff(3)
    
    # Trend indicators
    features['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    features['adx_trend'] = np.where(features['adx'] > 25, 1, 0)
    
    # ATR (Average True Range)
    features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    features['atr_pct'] = features['atr'] / df['close']
    
    # Heikin Ashi indicators
    features['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    features['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    features['ha_high'] = df[['high', 'open', 'close']].max(axis=1)
    features['ha_low'] = df[['low', 'open', 'close']].min(axis=1)
    features['ha_trend'] = np.where(features['ha_close'] > features['ha_open'], 1, -1)
    
    # Momentum indicators
    features['mom'] = talib.MOM(df['close'], timeperiod=10)
    features['mom_pct'] = features['mom'] / df['close'].shift(10)
    
    # Stochastic oscillator - Adjusted for faster response
    features['slowk'], features['slowd'] = talib.STOCH(
        df['high'], df['low'], df['close'], 
        fastk_period=10, slowk_period=3, slowk_matype=0,  # Faster stochastic
        slowd_period=3, slowd_matype=0
    )
    features['stoch_cross'] = np.where(features['slowk'] > features['slowd'], 1, -1)
    
    # Directional Movement Index
    features['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    features['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    features['di_cross'] = np.where(features['plus_di'] > features['minus_di'], 1, -1)
    
    # Volume indicators
    features['volume_sma'] = df['volume'].rolling(window=20).mean()
    features['volume_ratio'] = df['volume'] / features['volume_sma']
    features['up_volume'] = df['volume'] * (df['close'] > df['close'].shift(1)).astype(float)
    features['down_volume'] = df['volume'] * (df['close'] < df['close'].shift(1)).astype(float)
    
    # Price patterns
    features['higher_high_3d'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    features['lower_low_3d'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    features['higher_high_3d'] = features['higher_high_3d'].astype(float)
    features['lower_low_3d'] = features['lower_low_3d'].astype(float)
    
    # Channel breakouts - Adjusted for shorter timeframe
    features['upper_channel'] = df['high'].rolling(15).max()  # Reduced from 20
    features['lower_channel'] = df['low'].rolling(15).min()   # Reduced from 20
    features['channel_pos'] = (df['close'] - features['lower_channel']) / (features['upper_channel'] - features['lower_channel'])
    
    # Short-term price dynamics - New for faster trading
    features['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'])
    features['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    features['price_accel'] = df['close'].diff(1).diff(1)
    
    # Ensure all columns are float type
    for col in features.columns:
        if features[col].dtype == 'object' or features[col].dtype == 'bool':
            features[col] = pd.to_numeric(features[col], errors='coerce').astype(float)
    
    # Fill any NaN values from calculations
    features = features.fillna(0).astype('float32')
    
    return features

def prepare_sequences(df, seq_length=SEQ_LENGTH, prediction_horizon=PREDICTION_HORIZON):
    """
    Prepare sequences for training with enhanced labeling
    """
    # Extract price data
    close_prices = df['close'].values
    
    # Scale features with RobustScaler
    scaler = RobustScaler()
    # Scale all columns except timestamps
    columns_to_scale = [col for col in df.columns if col != 'timestamp']
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[columns_to_scale]),
        columns=columns_to_scale,
        index=df.index
    )
    
    # Convert all to float32 for consistency
    df_scaled = df_scaled.astype('float32')
    
    # Create sequences
    X, y = [], []
    stop_losses, take_profits, durations = [], [], []
    sample_weights = []
    
    # For each possible sequence
    for i in range(len(df) - seq_length - prediction_horizon):
        # Input sequence
        seq = df_scaled.iloc[i:i+seq_length].values
        
        # Target: Future price movement
        current_price = close_prices[i+seq_length]
        future_price = close_prices[i+seq_length+prediction_horizon]
        price_change = (future_price / current_price) - 1
        
        # Get recent volatility for adaptive thresholds
        recent_atr_pct = df['atr_pct'].iloc[i+seq_length-10:i+seq_length].mean()
        
        # Enhanced adaptive thresholds based on volatility - Modified for shorter timeframe
        buy_threshold = max(0.005, recent_atr_pct * 1.0)  # More sensitive for shorter timeframe
        sell_threshold = min(-0.005, -recent_atr_pct * 1.0)
        
        # Determine target class with enhanced logic
        if price_change > buy_threshold:
            target = 0  # Buy
            # Dynamic take profit and stop loss based on volatility and price change magnitude
            # Reduced for shorter-term trades
            take_profit = min(max(price_change * 1.2, 0.01), 0.15)  # More conservative TP
            stop_loss = max(min(recent_atr_pct * 1.2, 0.08), 0.005)  # Tighter SL
            duration = min(24, max(4, prediction_horizon * 4))  # Hours until expected completion
            weight = 2.0  # Weight for class balance
        elif price_change < sell_threshold:
            target = 2  # Sell
            take_profit = min(max(abs(price_change) * 1.2, 0.01), 0.15)
            stop_loss = max(min(recent_atr_pct * 1.2, 0.08), 0.005)
            duration = min(24, max(4, prediction_horizon * 4))
            weight = 2.0  # Weight for class balance
        else:
            target = 1  # Hold
            take_profit = max(recent_atr_pct * 2.0, 0.02)
            stop_loss = max(recent_atr_pct * 1.0, 0.005)
            duration = 12  # Default duration for hold (not really used)
            weight = 1.0
        
        X.append(seq)
        y.append(target)
        stop_losses.append(stop_loss)
        take_profits.append(take_profit)
        durations.append(duration)
        sample_weights.append(weight)
    
    return np.array(X), np.array(y), np.array(stop_losses), np.array(take_profits), np.array(durations), np.array(sample_weights)

def prepare_sequence_for_prediction(df, seq_length=SEQ_LENGTH):
    """
    Prepare a single sequence for prediction with enhanced robustness
    """
    # Make a copy to avoid fragmentation issues
    df = df.copy()
    
    # Ensure all data is numeric
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace any remaining NaN values
    df = df.fillna(0)
    
    # Scale all features with RobustScaler
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    # Convert to float32
    df_scaled = df_scaled.astype('float32')
    
    # Get the last sequence
    sequence = df_scaled.iloc[-seq_length:].values
    
    # Ensure the sequence has the correct shape
    if len(sequence) < seq_length:
        padding = np.zeros((seq_length - len(sequence), sequence.shape[1]), dtype='float32')
        sequence = np.vstack((padding, sequence))
    
    return sequence

#################################
# 5. MODEL TRAINING WITH FEEDBACK
#################################

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss
        return loss.mean()

class CustomDataset(Dataset):
    """
    Dataset for trading data with sample weights
    """
    def __init__(self, sequences, targets, stop_losses, take_profits, durations, weights=None):
        self.sequences = sequences
        self.targets = targets
        self.stop_losses = stop_losses
        self.take_profits = take_profits
        self.durations = durations
        self.weights = weights
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.targets[idx]]),
            torch.FloatTensor([self.stop_losses[idx]]),
            torch.FloatTensor([self.take_profits[idx]]),
            torch.FloatTensor([self.durations[idx]]),
            torch.FloatTensor([self.weights[idx] if self.weights is not None else 1.0])
        )

def train_model(symbol, timeframe='1h', lookback_days=90, feedback_system=None):
    """
    Train a model for a specific symbol with enhanced training
    """
    logging.info(f"Training model for {symbol}...")
    
    # Fetch data
    df = fetch_crypto_data(symbol, timeframe, lookback_days)
    if df is None or len(df) < SEQ_LENGTH + 10:
        logging.error(f"Insufficient data for {symbol}")
        return None
    
    # Calculate features
    feature_df = calculate_features(df)
    
    # Prepare sequences with the new duration parameter
    X, y, stop_losses, take_profits, durations, sample_weights = prepare_sequences(feature_df)
    
    if len(X) == 0:
        logging.error(f"No sequences could be prepared for {symbol}")
        return None
    
    # Check class balance
    class_counts = np.bincount(y)
    logging.info(f"Class distribution - Buy: {class_counts[0]}, Hold: {class_counts[1]}, Sell: {class_counts[2]}")
    
    # Calculate class weights for focal loss
    if len(class_counts) == 3:  # Ensure all 3 classes exist
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * 3
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    else:
        class_weights = None
    
    # Incorporate feedback data if available
    if feedback_system:
        feedback_data = feedback_system.get_feedback_data(symbol)
        if feedback_data and len(feedback_data) > 0:
            logging.info(f"Incorporating {len(feedback_data)} feedback samples into training")
            
            # Extract feedback sequences and outcomes
            feedback_X = []
            feedback_y = []
            feedback_weights = []
            
            for feature_data, actual_outcome, confidence in feedback_data:
                feedback_X.append(feature_data)
                feedback_y.append(actual_outcome)
                feedback_weights.append(confidence * 3.0)  # Give higher weight to actual outcomes
            
            # Add feedback data to training set
            # Note: We only add target labels from feedback, not SL/TP values
            if len(feedback_X) > 0:
                # Generate dummy values for SL, TP, and duration for feedback data
                dummy_sl = np.mean(stop_losses) * np.ones(len(feedback_X))
                dummy_tp = np.mean(take_profits) * np.ones(len(feedback_X))
                dummy_duration = np.mean(durations) * np.ones(len(feedback_X))
                
                # Combine with regular training data
                X = np.vstack((X, np.array(feedback_X)))
                y = np.append(y, np.array(feedback_y))
                sample_weights = np.append(sample_weights, np.array(feedback_weights))
                stop_losses = np.append(stop_losses, dummy_sl)
                take_profits = np.append(take_profits, dummy_tp)
                durations = np.append(durations, dummy_duration)
                
                logging.info(f"Updated dataset size after feedback incorporation: {len(X)}")
                
                # Recalculate class distribution
                new_class_counts = np.bincount(y.astype(int))
                logging.info(f"Updated class distribution - Buy: {new_class_counts[0]}, "
                             f"Hold: {new_class_counts[1]}, Sell: {new_class_counts[2]}")
    
    # Split data
    X_train, X_val, y_train, y_val, sl_train, sl_val, tp_train, tp_val, dur_train, dur_val, w_train, w_val = train_test_split(
        X, y, stop_losses, take_profits, durations, sample_weights, test_size=0.2, shuffle=False
    )
    
    # Create datasets
    train_dataset = CustomDataset(X_train, y_train, sl_train, tp_train, dur_train, w_train)
    val_dataset = CustomDataset(X_val, y_val, sl_val, tp_val, dur_val, w_val)
    
    # Create weighted sampler for training to handle class imbalance
    train_weights = torch.FloatTensor(w_train)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    input_size = X_train.shape[2]
    logging.info(f"Input feature dimension: {input_size}")
    
    model = MainModel(input_size=input_size).to(DEVICE)
    
    # Adjust model parameters based on historical performance if feedback is available
    if feedback_system:
        adjustment_factors = feedback_system.get_model_adjustment_factors(symbol)
        if adjustment_factors and 'tp_adjustment' in adjustment_factors:
            logging.info(f"Applying model adjustments based on historical performance")
            
            # Apply take profit adjustment
            tp_adjustment = adjustment_factors['tp_adjustment']
            if hasattr(model, 'tp_head') and 0.5 <= tp_adjustment <= 2.0:
                logging.info(f"Adjusting TP prediction by factor: {tp_adjustment}")
                # This is a simplified approach - in practice you would want to modify
                # the final layer weights or adjust outputs during prediction
                for param in model.tp_head[-2].parameters():
                    if isinstance(param, nn.Parameter):
                        param.data = param.data * tp_adjustment
            
            # Apply SL adjustment
            sl_adjustment = adjustment_factors['sl_adjustment']
            if hasattr(model, 'sl_head') and 0.5 <= sl_adjustment <= 2.0:
                logging.info(f"Adjusting SL prediction by factor: {sl_adjustment}")
                for param in model.sl_head[-2].parameters():
                    if isinstance(param, nn.Parameter):
                        param.data = param.data * sl_adjustment
    
    # Loss functions
    direction_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    sl_criterion = nn.MSELoss()
    tp_criterion = nn.MSELoss()
    duration_criterion = nn.MSELoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=1e-4  # L2 regularization
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-6,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dir_correct = 0
        train_total = 0
        
        for batch in train_loader:
            # Get data
            sequences, dir_targets, sl_targets, tp_targets, dur_targets, _ = batch
            sequences = sequences.to(DEVICE)
            dir_targets = dir_targets.squeeze().to(DEVICE)
            sl_targets = sl_targets.to(DEVICE)
            tp_targets = tp_targets.to(DEVICE)
            dur_targets = dur_targets.to(DEVICE)
            
            # Forward pass
            dir_probs, sl_preds, tp_preds, dur_preds, _ = model(sequences)
            
            # Calculate losses
            dir_loss = direction_criterion(dir_probs, dir_targets)
            sl_loss = sl_criterion(sl_preds, sl_targets)
            tp_loss = tp_criterion(tp_preds, tp_targets)
            dur_loss = duration_criterion(dur_preds, dur_targets)
            
            # Combined loss (weighted)
            loss = dir_loss * 0.7 + sl_loss * 0.1 + tp_loss * 0.1 + dur_loss * 0.1
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Track accuracy
            _, predicted = torch.max(dir_probs, 1)
            train_total += dir_targets.size(0)
            train_dir_correct += (predicted == dir_targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dir_correct = 0
        val_total = 0
        val_class_correct = [0, 0, 0]  # Correct predictions for each class
        val_class_total = [0, 0, 0]    # Total predictions for each class
        
        with torch.no_grad():
            for batch in val_loader:
                sequences, dir_targets, sl_targets, tp_targets, dur_targets, _ = batch
                sequences = sequences.to(DEVICE)
                dir_targets = dir_targets.squeeze().to(DEVICE)
                sl_targets = sl_targets.to(DEVICE)
                tp_targets = tp_targets.to(DEVICE)
                dur_targets = dur_targets.to(DEVICE)
                
                # Forward pass
                dir_probs, sl_preds, tp_preds, dur_preds, _ = model(sequences)
                
                # Calculate losses
                dir_loss = direction_criterion(dir_probs, dir_targets)
                sl_loss = sl_criterion(sl_preds, sl_targets)
                tp_loss = tp_criterion(tp_preds, tp_targets)
                dur_loss = duration_criterion(dur_preds, dur_targets)
                
                # Combined loss
                loss = dir_loss * 0.7 + sl_loss * 0.1 + tp_loss * 0.1 + dur_loss * 0.1
                
                val_loss += loss.item()
                
                # Track accuracy for each class
                _, predicted = torch.max(dir_probs, 1)
                val_total += dir_targets.size(0)
                val_dir_correct += (predicted == dir_targets).sum().item()
                
                # Track per-class accuracy
                for i in range(3):  # 3 classes: Buy, Hold, Sell
                    class_mask = (dir_targets == i)
                    val_class_total[i] += class_mask.sum().item()
                    if val_class_total[i] > 0:
                        val_class_correct[i] += (predicted[class_mask] == i).sum().item()
        
        # Calculate average losses and accuracy metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_dir_correct / train_total if train_total > 0 else 0
        val_accuracy = 100 * val_dir_correct / val_total if val_total > 0 else 0
        
        # Calculate per-class accuracy
        class_accuracy = []
        for i in range(3):
            if val_class_total[i] > 0:
                class_accuracy.append(100 * val_class_correct[i] / val_class_total[i])
            else:
                class_accuracy.append(0)
        
        logging.info(f"Epoch {epoch+1}/{EPOCHS} | "
                   f"Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | "
                   f"Train Acc: {train_accuracy:.2f}% | "
                   f"Val Acc: {val_accuracy:.2f}% | "
                   f"Buy Acc: {class_accuracy[0]:.2f}% | "
                   f"Hold Acc: {class_accuracy[1]:.2f}% | "
                   f"Sell Acc: {class_accuracy[2]:.2f}%")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save model
            safe_symbol = symbol.replace('/', '_')
            model_path = f'models/{safe_symbol}_model.pth'
            torch.save(model.state_dict(), model_path)
            
            # Save model info
            model_info = {
                'symbol': symbol,
                'input_size': input_size,
                'accuracy': val_accuracy,
                'per_class_accuracy': class_accuracy,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_size': len(X),
                'epochs_trained': epoch + 1
            }
            
            with open(f'models/{safe_symbol}_model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logging.info(f"Model saved to {model_path}")
            counter = 0
        else:
            counter += 1
        
        # Early stopping
        if counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return model

#################################
# 6. ENHANCED SIGNAL GENERATION
#################################

class SignalGenerator:
    """
    Generates trading signals based on model predictions with feedback learning
    """
    def __init__(self, symbols):
        self.symbols = symbols
        self.models = {}
        self.positions = {symbol: None for symbol in symbols}  # None: no position, 'long': long, 'short': short
        self.entry_prices = {symbol: 0 for symbol in symbols}
        self.stop_losses = {symbol: 0 for symbol in symbols}
        self.take_profits = {symbol: 0 for symbol in symbols}
        self.position_times = {symbol: None for symbol in symbols}
        self.max_position_duration = {symbol: 24 for symbol in symbols}  # Default max duration in hours
        self.trade_ids = {symbol: None for symbol in symbols}
        self.last_signals = {symbol: {'action': None, 'time': None} for symbol in symbols}
        
        # Create signal dispatcher
        self.signal_dispatcher = SignalDispatcher()
        
        # Initialize trade feedback system
        self.feedback_system = TradeFeedbackSystem()
        
        # Load models for each symbol
        for symbol in symbols:
            self.load_model(symbol)
    
    def load_model(self, symbol):
        """
        Load a model for a specific symbol
        """
        try:
            safe_symbol = symbol.replace('/', '_')
            model_path = f'models/{safe_symbol}_model.pth'
            model_info_path = f'models/{safe_symbol}_model_info.json'
            
            # Check if model exists
            if not os.path.exists(model_path):
                logging.warning(f"No model found for {symbol}. The model needs to be trained first.")
                return False
            
            # Load model info
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            input_size = model_info.get('input_size', 100)
            
            # Initialize model
            model = MainModel(input_size=input_size).to(DEVICE)
            
            # Load state dict
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            # Store model
            self.models[symbol] = model
            
            logging.info(f"Model for {symbol} loaded successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to load model for {symbol}: {str(e)}")
            return False
    
    def should_signal(self, symbol, action, current_time):
        """
        Check if we should send a signal (avoid duplicate signals)
        """
        last_signal = self.last_signals[symbol]
        
        # If no previous signal, or different action, we should signal
        if last_signal['action'] is None or last_signal['action'] != action:
            return True
        
        # If same action, check time difference (minimum 2 hours between same signals)
        if last_signal['time'] is not None:
            time_diff = (current_time - last_signal['time']).total_seconds() / 3600
            return time_diff >= 2  # Reduced from 4h to 2h for shorter trading
        
        return True
    
    def check_position_age(self, symbol, current_time):
        """
        Check if a position should be closed based on time duration
        """
        if self.positions[symbol] is not None and self.position_times[symbol] is not None:
            time_diff = (current_time - self.position_times[symbol]).total_seconds() / 3600
            max_duration = self.max_position_duration[symbol]
            
            if time_diff > max_duration:
                logging.info(f"{symbol} position duration {time_diff:.1f}h exceeded maximum {max_duration:.1f}h")
                return True
        
        return False
    
    def get_current_price(self, symbol):
        """
        Get the current price for a symbol
        """
        try:
            # Use ccxt to get current price
            exchange = ccxt.binance({'enableRateLimit': True})
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logging.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def close_position(self, symbol, current_price, reason="manual"):
        """
        Close an existing position and record the outcome
        """
        position = self.positions[symbol]
        if position is None:
            logging.warning(f"No active position to close for {symbol}")
            return False
        
        try:
            # Calculate profit/loss
            entry_price = self.entry_prices[symbol]
            
            if position == "long":
                profit_loss = current_price - entry_price
                profit_percent = (profit_loss / entry_price) * 100
                action = "exit_buy"
            else:  # short
                profit_loss = entry_price - current_price
                profit_percent = (profit_loss / entry_price) * 100
                action = "exit_sell"
            
            # Modified: Use 0% for negative percentage when stop loss is hit
            if reason == "sl_hit" and profit_percent < 0:
                per_value = "0%"
            else:
                per_value = f"{profit_percent:.2f}%"
            
            # Send exit signal
            self.signal_dispatcher.send_webhook(
                symbol,
                action,
                current_price,
                size=1,
                per=per_value,
                sl=self.stop_losses[symbol],
                tp=self.take_profits[symbol],
                reason=reason
            )
            
            # Record trade outcome
            if self.trade_ids[symbol] is not None:
                self.feedback_system.record_trade_exit(
                    self.trade_ids[symbol], 
                    current_price,
                    exit_reason=reason
                )
            
            # Reset position tracking
            self.positions[symbol] = None
            self.position_times[symbol] = None
            self.trade_ids[symbol] = None
            self.signal_dispatcher.remove_trade(symbol)
            
            logging.info(f"{symbol} {position.upper()} position closed at {current_price} ({reason})")
            logging.info(f"Profit/Loss: {profit_percent:.2f}%")
            
            return True
            
        except Exception as e:
            logging.error(f"Error closing position for {symbol}: {str(e)}")
            return False
    
    def generate_signal(self, symbol, data):
        """
        Generate a trading signal for the given symbol and data
        """
        try:
            # Check if model exists
            if symbol not in self.models:
                logging.error(f"No model found for {symbol}")
                return {'symbol': symbol, 'action': 'error', 'error': 'No model loaded'}
            
            model = self.models[symbol]
            
            # Prepare the sequence for prediction
            sequence = prepare_sequence_for_prediction(data)
            
            # Convert to tensor
            x = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
            
            # Get prediction
            with torch.no_grad():
                direction_probs, stop_loss, take_profit, duration, last_features = model(x)
                
                # Get the predicted action
                action_idx = torch.argmax(direction_probs[0]).item()
                
                # Convert to action string (0: Buy, 1: Hold, 2: Sell)
                if action_idx == 0:
                    action = "buy"
                elif action_idx == 2:
                    action = "sell"
                else:
                    action = "hold"
                
                # Get stop loss and take profit values
                sl_value = float(stop_loss[0].cpu().numpy())
                tp_value = float(take_profit[0].cpu().numpy())
                duration_value = float(duration[0].cpu().numpy())
                
                # Get current price
                current_price = float(data['close'].iloc[-1])
                
                # Get current time
                current_time = datetime.now()
                
                # Get current position status from our tracker
                current_position = self.positions[symbol]
                
                # Check if existing position should be closed due to time
                if current_position is not None:
                    if self.check_position_age(symbol, current_time):
                        self.close_position(symbol, current_price, "time_exit")
                        current_position = None
                
                # Check if we should send a signal
                if action != "hold" and self.should_signal(symbol, action, current_time):
                    # Update last signal time
                    self.last_signals[symbol] = {'action': action, 'time': current_time}
                    
                    # Handle position entry
                    if action == "buy" and current_position != "long":
                        # If in a short position, exit first
                        if current_position == "short":
                            self.close_position(symbol, current_price, "reversal")
                        
                        # Calculate stop loss and take profit prices
                        stop_loss_price = current_price * (1 - sl_value)
                        take_profit_price = current_price * (1 + tp_value)
                        
                        # Send buy signal
                        self.signal_dispatcher.send_webhook(
                            symbol, 
                            "buy", 
                            current_price, 
                            sl=f"{stop_loss_price:.2f}",
                            tp=f"{take_profit_price:.2f}",
                            duration=f"{duration_value:.1f}h"
                        )
                        
                        # Record trade entry
                        trade_id = self.feedback_system.record_trade_entry(
                            symbol, 
                            "buy", 
                            current_price, 
                            sl_value, 
                            tp_value, 
                            duration_value
                        )
                        
                        # Update position tracking
                        self.positions[symbol] = "long"
                        self.entry_prices[symbol] = current_price
                        self.stop_losses[symbol] = stop_loss_price
                        self.take_profits[symbol] = take_profit_price
                        self.position_times[symbol] = current_time
                        self.max_position_duration[symbol] = duration_value
                        self.trade_ids[symbol] = trade_id
                        
                        # Register with signal dispatcher
                        self.signal_dispatcher.register_trade(
                            symbol, 
                            trade_id, 
                            "buy", 
                            current_price, 
                            stop_loss_price, 
                            take_profit_price, 
                            duration_value
                        )
                        
                        logging.info(f"{symbol} BUY signal at {current_price} | "
                                   f"SL: {stop_loss_price:.2f} | "
                                   f"TP: {take_profit_price:.2f} | "
                                   f"Duration: {duration_value:.1f}h")
                    
                    elif action == "sell" and current_position != "short":
                        # If in a long position, exit first
                        if current_position == "long":
                            self.close_position(symbol, current_price, "reversal")
                        
                        # Calculate stop loss and take profit prices
                        stop_loss_price = current_price * (1 + sl_value)
                        take_profit_price = current_price * (1 - tp_value)
                        
                        # Send sell signal
                        self.signal_dispatcher.send_webhook(
                            symbol, 
                            "sell", 
                            current_price, 
                            sl=f"{stop_loss_price:.2f}",
                            tp=f"{take_profit_price:.2f}",
                            duration=f"{duration_value:.1f}h"
                        )
                        
                        # Record trade entry
                        trade_id = self.feedback_system.record_trade_entry(
                            symbol, 
                            "sell", 
                            current_price, 
                            sl_value, 
                            tp_value, 
                            duration_value
                        )
                        
                        # Update position tracking
                        self.positions[symbol] = "short"
                        self.entry_prices[symbol] = current_price
                        self.stop_losses[symbol] = stop_loss_price
                        self.take_profits[symbol] = take_profit_price
                        self.position_times[symbol] = current_time
                        self.max_position_duration[symbol] = duration_value
                        self.trade_ids[symbol] = trade_id
                        
                        # Register with signal dispatcher
                        self.signal_dispatcher.register_trade(
                            symbol, 
                            trade_id, 
                            "sell", 
                            current_price, 
                            stop_loss_price, 
                            take_profit_price, 
                            duration_value
                        )
                        
                        logging.info(f"{symbol} SELL signal at {current_price} | "
                                   f"SL: {stop_loss_price:.2f} | "
                                   f"TP: {take_profit_price:.2f} | "
                                   f"Duration: {duration_value:.1f}h")
                
                # Check for stop loss or take profit if in a position
                if current_position == "long":
                    # Check stop loss
                    if current_price <= self.stop_losses[symbol]:
                        self.close_position(symbol, current_price, "sl_hit")
                    
                    # Check take profit
                    elif current_price >= self.take_profits[symbol]:
                        self.close_position(symbol, current_price, "tp_hit")
                        
                        # Record successful feature data for model improvement
                        self.feedback_system.save_model_feedback(
                            symbol, 
                            sequence, 
                            0,  # 0 = buy was successful
                            confidence=1.5  # Higher confidence for successful trades
                        )
                
                elif current_position == "short":
                    # Check stop loss
                    if current_price >= self.stop_losses[symbol]:
                        self.close_position(symbol, current_price, "sl_hit")
                    
                    # Check take profit
                    elif current_price <= self.take_profits[symbol]:
                        self.close_position(symbol, current_price, "tp_hit")
                        
                        # Record successful feature data for model improvement
                        self.feedback_system.save_model_feedback(
                            symbol, 
                            sequence, 
                            2,  # 2 = sell was successful
                            confidence=1.5  # Higher confidence for successful trades
                        )
                
                return {
                    'symbol': symbol,
                    'action': action,
                    'direction_probs': direction_probs[0].cpu().numpy().tolist(),
                    'stop_loss': sl_value,
                    'take_profit': tp_value,
                    'duration': duration_value,
                    'current_price': current_price,
                    'position': self.positions[symbol]
                }
                
        except Exception as e:
            logging.error(f"Error generating signal for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'action': 'error',
                'error': str(e)
            }
    
    def get_performance_report(self, symbol, days=30):
        """
        Get a performance report for a symbol
        """
        metrics = self.feedback_system.get_performance_metrics(symbol, days)
        if not metrics:
            return f"No trade data available for {symbol}"
        
        total_trades, winning_trades, avg_profit, avg_win, avg_loss, avg_duration = metrics
        
        if total_trades == 0:
            return f"No trades recorded for {symbol} in the last {days} days"
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        report = f"Performance Report for {symbol} (Last {days} days):\n"
        report += f"Total Trades: {total_trades}\n"
        report += f"Win Rate: {win_rate:.2f}%\n"
        report += f"Average Profit: {avg_profit:.2f}%\n"
        report += f"Average Win: {avg_win:.2f}%\n"
        report += f"Average Loss: {avg_loss:.2f}%\n"
        report += f"Average Trade Duration: {avg_duration:.1f} hours"
        
        return report

#################################
# 7. AUTOMATED DAILY RETRAINING
#################################

def scheduled_retraining(symbols, stop_event):
    """
    Function to handle periodic retraining of the models
    """
    logging.info("Starting scheduled retraining thread")
    
    while not stop_event.is_set():
        try:
            # Sleep until the next scheduled retraining time
            # Check every 10 minutes if it's time to retrain
            for _ in range(6 * RETRAINING_INTERVAL):  # 6 checks per hour * 24 hours
                if stop_event.is_set():
                    return
                time.sleep(600)  # 10 minutes
            
            # It's time to retrain
            logging.info("=== SCHEDULED RETRAINING STARTED ===")
            logging.info(f"Retraining models for symbols: {symbols}")
            
            # Initialize feedback system
            feedback_system = TradeFeedbackSystem()
            
            # Train models with feedback
            for symbol in symbols:
                try:
                    logging.info(f"Retraining model for {symbol}")
                    train_model(symbol, timeframe='1h', lookback_days=90, feedback_system=feedback_system)
                except Exception as e:
                    logging.error(f"Error retraining model for {symbol}: {str(e)}")
            
            # Close feedback system
            feedback_system.close()
            
            logging.info("=== SCHEDULED RETRAINING COMPLETED ===")
            
        except Exception as e:
            logging.error(f"Error in retraining thread: {str(e)}")

#################################
# 8. MAIN EXECUTION
#################################

def main():
    """
    Main function with automated daily retraining
    """
    # Define trading symbols
    symbols = ['BTC/USDT', 'SOL/USDT']
    
    logging.info("Starting enhanced trading bot with automated daily retraining")
    
    # Check if we need initial training
    initial_training_needed = False
    
    # Check if models already exist for all symbols
    for symbol in symbols:
        safe_symbol = symbol.replace('/', '_')
        model_path = f"models/{safe_symbol}_model.pth"
        if not os.path.exists(model_path):
            initial_training_needed = True
            logging.info(f"Model for {symbol} does not exist, initial training needed")
            break
    
    # If initial training is needed, train the models
    if initial_training_needed:
        logging.info("Starting initial model training")
        feedback_system = TradeFeedbackSystem()
        
        for symbol in symbols:
            try:
                train_model(symbol, feedback_system=feedback_system)
            except Exception as e:
                logging.error(f"Error training model for {symbol}: {str(e)}")
        
        feedback_system.close()
    
    # Create stop event for the retraining thread
    stop_event = threading.Event()
    
    # Start the retraining thread
    retraining_thread = threading.Thread(
        target=scheduled_retraining,
        args=(symbols, stop_event),
        daemon=True
    )
    retraining_thread.start()
    
    try:
        # Run signal generation continuously
        logging.info("Starting continuous signal generation")
        signal_generator = SignalGenerator(symbols)
        
        while True:
            for symbol in symbols:
                try:
                    # Skip symbols with no model
                    if symbol not in signal_generator.models:
                        logging.warning(f"Skipping {symbol} - no model available")
                        continue
                    
                    # Fetch latest data
                    df = fetch_crypto_data(symbol, timeframe='1h', lookback_days=30)
                    
                    if df is not None and len(df) > SEQ_LENGTH:
                        # Calculate features
                        feature_df = calculate_features(df)
                        
                        # Generate signal
                        result = signal_generator.generate_signal(symbol, feature_df)
                        
                        # Log result
                        if result.get('action') != 'error':
                            probs = result['direction_probs']
                            logging.info(f"Signal for {symbol}: {result['action'].upper()} | "
                                       f"Buy: {probs[0]:.4f}, Hold: {probs[1]:.4f}, Sell: {probs[2]:.4f} | "
                                       f"Price: {result['current_price']}")
                        else:
                            logging.error(f"Error generating signal for {symbol}: {result.get('error')}")
                    else:
                        logging.warning(f"Insufficient data for {symbol}")
                        
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {str(e)}")
            
            # Wait before the next check
            time.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        logging.info("Signal generation stopped by user")
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")
    finally:
        # Signal the retraining thread to stop
        stop_event.set()
        
        # Wait for the retraining thread to exit
        retraining_thread.join(timeout=5)
        
        # Clean up resources
        if hasattr(signal_generator, 'feedback_system'):
            signal_generator.feedback_system.close()
        
        logging.info("Trading bot shutdown complete")

if __name__ == "__main__":
    main()
