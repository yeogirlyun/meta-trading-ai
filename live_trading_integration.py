import pandas as pd
import numpy as np
import pickle
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LIVE TRADING CONFIGURATION
# ============================================================================

class LiveTradingConfig:
    """Configuration for live trading system"""
    
    def __init__(self):
        # Trading parameters
        self.symbol = "QQQ"
        self.initial_capital = 100000
        self.max_position_size = 0.1  # 10% of capital per trade
        self.leverage = 1.0  # Can be increased to 2-3x for higher returns
        
        # Risk management
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.05  # 5% take profit
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_drawdown = 0.15  # 15% max drawdown
        
        # Execution parameters
        self.execution_delay = 1  # seconds
        self.retry_attempts = 3
        self.order_timeout = 30  # seconds
        
        # Data feed parameters
        self.data_source = "polygon"  # polygon.io
        self.update_frequency = 60  # seconds
        self.historical_lookback = 1000  # bars
        
        # Strategy parameters
        self.strategy_selection_interval = 3600  # 1 hour
        self.regime_detection_interval = 300  # 5 minutes
        self.performance_evaluation_interval = 1800  # 30 minutes

# ============================================================================
# REAL-TIME DATA FEED
# ============================================================================

class RealTimeDataFeed:
    """Real-time data feed from Polygon.io"""
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.current_data = pd.DataFrame()
        self.historical_data = pd.DataFrame()
        self.last_update = None
        self.is_connected = False
        
        # Initialize data structure
        self._initialize_data_feed()
    
    def _initialize_data_feed(self):
        """Initialize the data feed connection"""
        print("Initializing real-time data feed...")
        
        # Load historical data for initialization
        try:
            self.historical_data = pickle.load(open('polygon_QQQ_1m.pkl', 'rb'))
            print(f"Loaded {len(self.historical_data)} historical bars")
        except:
            print("Warning: Could not load historical data")
            self.historical_data = pd.DataFrame()
        
        # Initialize current data
        if len(self.historical_data) > 0:
            self.current_data = self.historical_data.tail(self.config.historical_lookback)
            self.last_update = self.current_data.index[-1]
        
        self.is_connected = True
        print("Data feed initialized successfully")
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get the latest market data"""
        if not self.is_connected:
            return pd.DataFrame()
        
        # In live trading, this would connect to Polygon.io WebSocket
        # For now, we'll simulate real-time data updates
        current_time = datetime.now()
        
        if self.last_update is None or (current_time - self.last_update).seconds >= self.config.update_frequency:
            # Simulate new data arrival
            self._update_data_feed()
        
        return self.current_data
    
    def _update_data_feed(self):
        """Update the data feed with new information"""
        # In live trading, this would fetch new data from Polygon.io
        # For simulation, we'll use historical data with time progression
        
        if len(self.historical_data) == 0:
            return
        
        # Find the next data point after last_update
        if self.last_update is not None:
            next_data = self.historical_data[self.historical_data.index > self.last_update]
            if len(next_data) > 0:
                new_bar = next_data.iloc[0:1]
                self.current_data = pd.concat([self.current_data, new_bar]).tail(self.config.historical_lookback)
                self.last_update = new_bar.index[0]
        
        print(f"Data feed updated at {datetime.now()}")

# ============================================================================
# ORDER MANAGEMENT SYSTEM
# ============================================================================

class OrderManager:
    """Manages order execution and tracking"""
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.orders = []
        self.positions = {}
        self.order_id_counter = 0
        
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str = "market", price: float = None) -> Dict:
        """Place a new order"""
        order_id = f"ORDER_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side,  # 'buy' or 'sell'
            'quantity': quantity,
            'type': order_type,
            'price': price,
            'status': 'pending',
            'timestamp': datetime.now(),
            'filled_quantity': 0,
            'filled_price': None
        }
        
        self.orders.append(order)
        print(f"Order placed: {order_id} - {side} {quantity} {symbol}")
        
        # Simulate order execution
        self._execute_order(order)
        
        return order
    
    def _execute_order(self, order: Dict):
        """Execute the order (simulated)"""
        # In live trading, this would connect to Alpaca.markets API
        time.sleep(self.config.execution_delay)
        
        # Simulate successful execution
        order['status'] = 'filled'
        order['filled_quantity'] = order['quantity']
        order['filled_price'] = self._get_current_price(order['symbol'])
        order['filled_timestamp'] = datetime.now()
        
        # Update positions
        self._update_positions(order)
        
        print(f"Order executed: {order['id']} at {order['filled_price']}")
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for the symbol"""
        # In live trading, this would fetch from Polygon.io
        # For simulation, use a reasonable price
        return 400.0  # Approximate QQQ price
    
    def _update_positions(self, order: Dict):
        """Update position tracking"""
        symbol = order['symbol']
        side = order['side']
        quantity = order['filled_quantity']
        
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        if side == 'buy':
            self.positions[symbol] += quantity
        else:  # sell
            self.positions[symbol] -= quantity
        
        print(f"Position updated: {symbol} = {self.positions[symbol]}")
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        return self.positions.copy()
    
    def get_order_history(self) -> List[Dict]:
        """Get order history"""
        return self.orders.copy()

# ============================================================================
# RISK MANAGEMENT SYSTEM
# ============================================================================

class RiskManager:
    """Manages risk and position sizing"""
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = config.initial_capital
        self.current_capital = config.initial_capital
        
    def calculate_position_size(self, signal_strength: float, 
                              current_price: float) -> float:
        """Calculate position size based on risk parameters"""
        # Base position size
        base_size = self.current_capital * self.config.max_position_size
        
        # Adjust for signal strength
        adjusted_size = base_size * abs(signal_strength)
        
        # Apply leverage
        leveraged_size = adjusted_size * self.config.leverage
        
        # Convert to shares
        shares = leveraged_size / current_price
        
        return max(1, int(shares))  # Minimum 1 share
    
    def check_risk_limits(self, new_position_value: float) -> bool:
        """Check if new position violates risk limits"""
        # Check daily loss limit
        if self.daily_pnl < -self.current_capital * self.config.max_daily_loss:
            print("Risk limit: Daily loss exceeded")
            return False
        
        # Check drawdown limit
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.config.max_drawdown:
            print("Risk limit: Maximum drawdown exceeded")
            return False
        
        # Check position size limit
        if new_position_value > self.current_capital * self.config.max_position_size:
            print("Risk limit: Position size too large")
            return False
        
        return True
    
    def update_pnl(self, pnl: float):
        """Update P&L tracking"""
        self.daily_pnl += pnl
        self.current_capital += pnl
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        self.max_drawdown = max(self.max_drawdown, 
                               (self.peak_capital - self.current_capital) / self.peak_capital)
    
    def reset_daily_limits(self):
        """Reset daily risk limits"""
        self.daily_pnl = 0.0

# ============================================================================
# LIVE TRADING EXECUTION ENGINE
# ============================================================================

class LiveTradingEngine:
    """Main live trading execution engine"""
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.data_feed = RealTimeDataFeed(config)
        self.order_manager = OrderManager(config)
        self.risk_manager = RiskManager(config)
        self.trading_system = None  # Will be set to MetaTradingAI v3.0
        self.is_running = False
        self.performance_log = []
        
        # Trading state
        self.current_position = 0
        self.last_signal = 0
        self.last_strategy = None
        self.last_regime = None
        
    def initialize_trading_system(self, trading_system):
        """Initialize the MetaTradingAI v3.0 system"""
        self.trading_system = trading_system
        print("Trading system initialized")
    
    def start_trading(self):
        """Start the live trading session"""
        if self.trading_system is None:
            print("Error: Trading system not initialized")
            return
        
        print("Starting live trading session...")
        self.is_running = True
        
        # Reset daily limits
        self.risk_manager.reset_daily_limits()
        
        # Start trading loop
        self._trading_loop()
    
    def stop_trading(self):
        """Stop the live trading session"""
        print("Stopping live trading session...")
        self.is_running = False
        
        # Close all positions
        self._close_all_positions()
        
        # Save performance log
        self._save_performance_log()
    
    def _trading_loop(self):
        """Main trading loop"""
        last_strategy_update = datetime.now()
        last_regime_update = datetime.now()
        last_performance_update = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Get latest market data
                market_data = self.data_feed.get_latest_data()
                if len(market_data) == 0:
                    print("No market data available")
                    time.sleep(10)
                    continue
                
                # Update regime detection
                if (current_time - last_regime_update).seconds >= self.config.regime_detection_interval:
                    self._update_market_regime(market_data)
                    last_regime_update = current_time
                
                # Update strategy selection
                if (current_time - last_strategy_update).seconds >= self.config.strategy_selection_interval:
                    self._update_strategy_selection(market_data)
                    last_strategy_update = current_time
                
                # Generate trading signals
                signal = self._generate_signal(market_data)
                
                # Execute trades based on signals
                if signal != self.last_signal:
                    self._execute_trade(signal, market_data)
                    self.last_signal = signal
                
                # Update performance tracking
                if (current_time - last_performance_update).seconds >= self.config.performance_evaluation_interval:
                    self._update_performance()
                    last_performance_update = current_time
                
                # Sleep before next iteration
                time.sleep(10)  # 10-second intervals
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(30)
    
    def _update_market_regime(self, market_data: pd.DataFrame):
        """Update market regime detection"""
        if self.trading_system is None:
            return
        
        # Use the regime detection from the trading system
        regime = self.trading_system.detect_market_regime(market_data)
        if regime != self.last_regime:
            print(f"Market regime changed: {self.last_regime} -> {regime}")
            self.last_regime = regime
    
    def _update_strategy_selection(self, market_data: pd.DataFrame):
        """Update strategy selection"""
        if self.trading_system is None:
            return
        
        # Select best strategy for current conditions
        selected_strategy = self.trading_system.selector.select_strategy(market_data)
        if selected_strategy != self.last_strategy:
            print(f"Strategy changed: {self.last_strategy} -> {selected_strategy.name}")
            self.last_strategy = selected_strategy
    
    def _generate_signal(self, market_data: pd.DataFrame) -> int:
        """Generate trading signal from current strategy"""
        if self.last_strategy is None:
            return 0
        
        # Generate signal using the selected strategy
        signals = self.last_strategy.calculate_signals(market_data)
        current_signal = signals.iloc[-1] if len(signals) > 0 else 0
        
        return current_signal
    
    def _execute_trade(self, signal: int, market_data: pd.DataFrame):
        """Execute trade based on signal"""
        if signal == 0:
            return
        
        current_price = market_data['close'].iloc[-1]
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(signal, current_price)
        
        # Check risk limits
        position_value = position_size * current_price
        if not self.risk_manager.check_risk_limits(position_value):
            print("Trade blocked by risk limits")
            return
        
        # Determine order side
        if signal > 0 and self.current_position <= 0:
            # Buy signal
            side = "buy"
            quantity = position_size
        elif signal < 0 and self.current_position >= 0:
            # Sell signal
            side = "sell"
            quantity = position_size
        else:
            # No position change needed
            return
        
        # Place order
        order = self.order_manager.place_order(
            symbol=self.config.symbol,
            side=side,
            quantity=quantity,
            order_type="market"
        )
        
        # Update position
        if side == "buy":
            self.current_position += quantity
        else:
            self.current_position -= quantity
        
        print(f"Trade executed: {side} {quantity} {self.config.symbol} at {current_price}")
    
    def _close_all_positions(self):
        """Close all open positions"""
        positions = self.order_manager.get_positions()
        
        for symbol, quantity in positions.items():
            if quantity != 0:
                side = "sell" if quantity > 0 else "buy"
                self.order_manager.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=abs(quantity),
                    order_type="market"
                )
                print(f"Closing position: {side} {abs(quantity)} {symbol}")
    
    def _update_performance(self):
        """Update performance tracking"""
        # Calculate current P&L
        positions = self.order_manager.get_positions()
        current_pnl = 0.0
        
        for symbol, quantity in positions.items():
            # In live trading, this would calculate unrealized P&L
            # For simulation, use a simple calculation
            current_pnl += quantity * 0.01  # 1% return per position
        
        self.risk_manager.update_pnl(current_pnl)
        
        # Log performance
        performance_entry = {
            'timestamp': datetime.now(),
            'capital': self.risk_manager.current_capital,
            'daily_pnl': self.risk_manager.daily_pnl,
            'max_drawdown': self.risk_manager.max_drawdown,
            'positions': positions.copy(),
            'current_position': self.current_position,
            'last_strategy': self.last_strategy.name if self.last_strategy else None,
            'last_regime': self.last_regime
        }
        
        self.performance_log.append(performance_entry)
        
        print(f"Performance Update: Capital=${self.risk_manager.current_capital:,.2f}, "
              f"Daily P&L=${self.risk_manager.daily_pnl:,.2f}, "
              f"Drawdown={self.risk_manager.max_drawdown:.2%}")
    
    def _save_performance_log(self):
        """Save performance log to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_log_{timestamp}.json"
        
        # Convert to JSON-serializable format
        log_data = []
        for entry in self.performance_log:
            entry_copy = entry.copy()
            entry_copy['timestamp'] = entry_copy['timestamp'].isoformat()
            if entry_copy['last_strategy']:
                entry_copy['last_strategy'] = entry_copy['last_strategy']
            log_data.append(entry_copy)
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Performance log saved to {filename}")

# ============================================================================
# LIVE TRADING MONITOR
# ============================================================================

class LiveTradingMonitor:
    """Monitor and display live trading status"""
    
    def __init__(self, trading_engine: LiveTradingEngine):
        self.trading_engine = trading_engine
        self.is_monitoring = False
    
    def start_monitoring(self):
        """Start monitoring the trading session"""
        self.is_monitoring = True
        print("Starting live trading monitor...")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.is_monitoring:
            self._display_status()
            time.sleep(60)  # Update every minute
    
    def _display_status(self):
        """Display current trading status"""
        engine = self.trading_engine
        
        print("\n" + "="*60)
        print("LIVE TRADING STATUS")
        print("="*60)
        print(f"Time: {datetime.now()}")
        print(f"Status: {'RUNNING' if engine.is_running else 'STOPPED'}")
        print(f"Capital: ${engine.risk_manager.current_capital:,.2f}")
        print(f"Daily P&L: ${engine.risk_manager.daily_pnl:,.2f}")
        print(f"Max Drawdown: {engine.risk_manager.max_drawdown:.2%}")
        print(f"Current Position: {engine.current_position}")
        print(f"Last Signal: {engine.last_signal}")
        print(f"Active Strategy: {engine.last_strategy.name if engine.last_strategy else 'None'}")
        print(f"Market Regime: {engine.last_regime}")
        
        # Display positions
        positions = engine.order_manager.get_positions()
        if positions:
            print("\nPositions:")
            for symbol, quantity in positions.items():
                print(f"  {symbol}: {quantity}")
        
        # Display recent orders
        orders = engine.order_manager.get_order_history()
        if orders:
            recent_orders = orders[-5:]  # Last 5 orders
            print("\nRecent Orders:")
            for order in recent_orders:
                print(f"  {order['id']}: {order['side']} {order['quantity']} {order['symbol']} "
                      f"at ${order['filled_price']:.2f}")
        
        print("="*60)

# ============================================================================
# MAIN LIVE TRADING INTEGRATION
# ============================================================================

def create_live_trading_system():
    """Create and configure the live trading system"""
    
    # Create configuration
    config = LiveTradingConfig()
    
    # Create trading engine
    trading_engine = LiveTradingEngine(config)
    
    # Create monitor
    monitor = LiveTradingMonitor(trading_engine)
    
    return trading_engine, monitor

def main():
    """Main function to run live trading integration"""
    
    print("MetaTradingAI v3.0 - Live Trading Integration")
    print("="*60)
    
    # Create live trading system
    trading_engine, monitor = create_live_trading_system()
    
    # Initialize with MetaTradingAI v3.0 system
    # (This would import and initialize the actual trading system)
    print("Initializing MetaTradingAI v3.0 system...")
    # trading_system = AggressiveMetaTradingAI()  # Import from v3.0
    # trading_engine.initialize_trading_system(trading_system)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Start trading (in a separate thread)
    trading_thread = threading.Thread(target=trading_engine.start_trading)
    trading_thread.daemon = True
    trading_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping live trading...")
        trading_engine.stop_trading()
        monitor.stop_monitoring()
        print("Live trading stopped.")

if __name__ == "__main__":
    main() 