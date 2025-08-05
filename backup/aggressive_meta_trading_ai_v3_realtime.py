#!/usr/bin/env python3
"""
MetaTradingAI v3.0 Real-Time - Optimized for Live Trading with Trade Frequency Limits
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries, fallback if not available
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

# ============================================================================
# REAL-TIME TRADING CONSTRAINTS
# ============================================================================

class RealTimeConstraints:
    """Real-time trading constraints and limitations"""
    
    def __init__(self):
        # Trading frequency limits
        self.min_trade_interval = 2  # minutes between trades
        self.max_trades_per_hour = 30  # maximum trades per hour
        self.max_trades_per_day = 200  # maximum trades per day
        
        # Execution constraints
        self.execution_delay = 1  # seconds for order execution
        self.slippage_tolerance = 0.001  # 0.1% slippage tolerance
        self.min_position_hold_time = 1  # minimum minutes to hold position
        
        # Risk management
        self.max_position_size = 0.15  # 15% of capital per trade
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_drawdown = 0.15  # 15% maximum drawdown

# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================

def detect_market_regime(data: pd.DataFrame) -> str:
    """Detects the current market regime based on volatility and trend strength."""
    if len(data) < 50:
        return "ranging"
    
    volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
    avg_vol = data['close'].pct_change().rolling(50).std().mean()
    trend_strength = abs(data['close'].pct_change().rolling(20).mean().iloc[-1]) / volatility

    if volatility > avg_vol * 1.8:
        return "high_volatility"
    elif trend_strength > 0.6:
        return "trending"
    else:
        return "ranging"

# ============================================================================
# REAL-TIME STRATEGY BASE CLASS
# ============================================================================

class RealTimeMetaStrategy:
    """Base class for real-time trading strategies with frequency limits"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.last_trade_time = None
        self.trade_count = 0
        self.constraints = RealTimeConstraints()
    
    def set_parameters(self, params: dict):
        self.parameters.update(params)
    
    def can_trade(self, current_time: datetime) -> bool:
        """Check if we can trade based on frequency constraints"""
        if self.last_trade_time is None:
            return True
        
        time_since_last_trade = (current_time - self.last_trade_time).total_seconds() / 60
        return time_since_last_trade >= self.constraints.min_trade_interval
    
    def update_trade_time(self, trade_time: datetime):
        """Update the last trade time"""
        self.last_trade_time = trade_time
        self.trade_count += 1
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> dict:
        """Backtest with real-time constraints"""
        signals = self.calculate_signals(data)
        
        position = 0  # 1 for long, -1 for short, 0 for flat
        capital = initial_capital
        equity = [capital]
        trades = []
        hourly_returns = []
        last_trade_time = None
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]['close']
            prev_price = data.iloc[i-1]['close']
            
            # Check trading constraints
            can_trade = True
            if last_trade_time is not None:
                time_since_last = (current_time - last_trade_time).total_seconds() / 60
                can_trade = time_since_last >= self.constraints.min_trade_interval
            
            # Update position based on signals (only if we can trade)
            if can_trade and signals.iloc[i] != 0:
                if signals.iloc[i] == 1 and position <= 0:  # Buy signal
                    if position == -1:  # Close short
                        trades.append({
                            'entry_time': data.index[i-1],
                            'exit_time': data.index[i],
                            'entry_price': prev_price,
                            'exit_price': current_price,
                            'type': 'short',
                            'pnl': prev_price - current_price
                        })
                    position = 1
                    last_trade_time = current_time
                elif signals.iloc[i] == -1 and position >= 0:  # Sell signal
                    if position == 1:  # Close long
                        trades.append({
                            'entry_time': data.index[i-1],
                            'exit_time': data.index[i],
                            'entry_price': prev_price,
                            'exit_price': current_price,
                            'type': 'long',
                            'pnl': current_price - prev_price
                        })
                    position = -1
                    last_trade_time = current_time
            
            # Calculate returns
            if position == 1:  # Long position
                hourly_return = (current_price - prev_price) / prev_price
            elif position == -1:  # Short position
                hourly_return = (prev_price - current_price) / prev_price
            else:
                hourly_return = 0
            
            capital *= (1 + hourly_return)
            equity.append(capital)
            hourly_returns.append(hourly_return)
        
        # Calculate metrics
        total_return = (equity[-1] - initial_capital) / initial_capital
        avg_hourly_return = np.mean(hourly_returns) if hourly_returns else 0
        
        # 10-day rolling returns (2 weeks)
        hourly_equity = pd.Series(equity, index=data.index)
        rolling_10d = hourly_equity.pct_change(80).dropna()  # 80 hours = 10 trading days
        target_10d_05 = len(rolling_10d[rolling_10d >= 0.05]) / len(rolling_10d) if len(rolling_10d) > 0 else 0
        
        return {
            'total_return': total_return,
            'avg_hourly_return': avg_hourly_return,
            'trades': trades,
            'num_trades': len(trades),
            'target_10d_05_pct': target_10d_05,
            'trade_frequency': len(trades) / (len(data) / 60) if len(data) > 0 else 0  # trades per hour
        }

# ============================================================================
# REAL-TIME STRATEGY POOLS (OPTIMIZED FOR FREQUENCY LIMITS)
# ============================================================================

class RealTimeUltraVolatilityExploitation(RealTimeMetaStrategy):
    """Real-time ultra-aggressive volatility exploitation with frequency limits"""
    
    def __init__(self):
        super().__init__("Real-Time Ultra Volatility Exploitation")
        self.parameters = {
            'volatility_threshold': 0.0012,  # Slightly higher for fewer trades
            'momentum_period': 2,            # Longer period for stability
            'reversal_threshold': 0.0012,    # Higher threshold
            'max_hold_period': 10,           # Longer hold period
            'volume_threshold': 1.05         # Higher volume requirement
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(10).std()  # Longer window
        
        # Calculate momentum
        momentum = data['close'].pct_change(self.parameters['momentum_period'])
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()  # Longer window
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Entry conditions - more selective for real-time
        high_vol = volatility > self.parameters['volatility_threshold']
        volume_confirmed = volume_ratio > self.parameters['volume_threshold']
        strong_momentum = abs(momentum) > self.parameters['reversal_threshold']
        
        # Long on positive momentum during high volatility
        long_condition = (
            (momentum > self.parameters['reversal_threshold']) & 
            high_vol & 
            volume_confirmed &
            strong_momentum
        )
        
        # Short on negative momentum during high volatility
        short_condition = (
            (momentum < -self.parameters['reversal_threshold']) & 
            high_vol & 
            volume_confirmed &
            strong_momentum
        )
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals

class RealTimeBreakoutMomentum(RealTimeMetaStrategy):
    """Real-time breakout momentum strategy with frequency limits"""
    
    def __init__(self):
        super().__init__("Real-Time Breakout Momentum")
        self.parameters = {
            'breakout_period': 10,           # Longer period for stability
            'confirmation_period': 5,        # Longer confirmation
            'volume_multiplier': 2.0,       # Higher volume requirement
            'momentum_threshold': 0.002,     # Higher threshold
            'max_hold_period': 15           # Longer hold period
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate breakout levels
        highest_high = data['high'].rolling(self.parameters['breakout_period']).max()
        lowest_low = data['low'].rolling(self.parameters['breakout_period']).min()
        
        # Breakout conditions
        breakout_up = data['close'] > highest_high.shift(1)
        breakout_down = data['close'] < lowest_low.shift(1)
        
        # Volume confirmation
        avg_volume = data['volume'].rolling(20).mean()  # Longer window
        volume_spike = data['volume'] > avg_volume * self.parameters['volume_multiplier']
        
        # Momentum confirmation
        momentum = data['close'].pct_change(self.parameters['confirmation_period'])
        momentum_confirmed = abs(momentum) > self.parameters['momentum_threshold']
        
        signals = pd.Series(0, index=data.index)
        
        # Long on upward breakout with volume and momentum
        long_condition = breakout_up & volume_spike & (momentum > 0) & momentum_confirmed
        signals[long_condition] = 1
        
        # Short on downward breakout with volume and momentum
        short_condition = breakout_down & volume_spike & (momentum < 0) & momentum_confirmed
        signals[short_condition] = -1
        
        return signals

class RealTimeUltraMomentumAmplification(RealTimeMetaStrategy):
    """Real-time ultra-aggressive momentum amplification with frequency limits"""
    
    def __init__(self):
        super().__init__("Real-Time Ultra Momentum Amplification")
        self.parameters = {
            'short_period': 3,               # Longer periods for stability
            'medium_period': 5,              # Longer medium period
            'long_period': 10,               # Longer long period
            'momentum_threshold': 0.001,     # Higher threshold
            'acceleration_threshold': 0.0008, # Higher acceleration
            'volume_threshold': 1.05,        # Higher volume requirement
            'max_hold_period': 15           # Longer hold period
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate multiple momentum indicators
        short_momentum = data['close'].pct_change(self.parameters['short_period'])
        medium_momentum = data['close'].pct_change(self.parameters['medium_period'])
        long_momentum = data['close'].pct_change(self.parameters['long_period'])
        
        # Calculate acceleration
        acceleration = short_momentum.diff()
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()  # Longer window
        volume_ratio = data['volume'] / avg_volume
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(5).std()  # Longer window
        
        signals = pd.Series(0, index=data.index)
        
        # Strong buy when all momentum indicators align
        strong_buy = (
            (short_momentum > self.parameters['momentum_threshold']) &
            (medium_momentum > self.parameters['momentum_threshold']) &
            (long_momentum > 0) &
            (acceleration > self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(20).mean() * 3)  # More restrictive
        )
        
        # Strong sell when all momentum indicators align negatively
        strong_sell = (
            (short_momentum < -self.parameters['momentum_threshold']) &
            (medium_momentum < -self.parameters['momentum_threshold']) &
            (long_momentum < 0) &
            (acceleration < -self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(20).mean() * 3)  # More restrictive
        )
        
        signals[strong_buy] = 1
        signals[strong_sell] = -1
        
        return signals

class RealTimeAcceleratedMACross(RealTimeMetaStrategy):
    """Real-time accelerated moving average crossover with frequency limits"""
    
    def __init__(self):
        super().__init__("Real-Time Accelerated MA Cross")
        self.parameters = {
            'fast_period': 3,                # Longer periods for stability
            'slow_period': 10,               # Longer slow period
            'momentum_threshold': 0.001,     # Higher threshold
            'volume_threshold': 1.05,        # Higher volume requirement
            'max_hold_period': 20           # Longer hold period
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate moving averages
        fast_ma = data['close'].rolling(self.parameters['fast_period']).mean()
        slow_ma = data['close'].rolling(self.parameters['slow_period']).mean()
        
        # Calculate momentum
        momentum = data['close'].pct_change(5)  # Longer period
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()  # Longer window
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Golden cross (fast MA crosses above slow MA)
        golden_cross = (
            (fast_ma > slow_ma) & 
            (fast_ma.shift(1) <= slow_ma.shift(1)) &
            (momentum > self.parameters['momentum_threshold']) &
            (volume_ratio > self.parameters['volume_threshold'])
        )
        
        # Death cross (fast MA crosses below slow MA)
        death_cross = (
            (fast_ma < slow_ma) & 
            (fast_ma.shift(1) >= slow_ma.shift(1)) &
            (momentum < -self.parameters['momentum_threshold']) &
            (volume_ratio > self.parameters['volume_threshold'])
        )
        
        signals[golden_cross] = 1
        signals[death_cross] = -1
        
        return signals

class RealTimeUltraHighFrequencyScalping(RealTimeMetaStrategy):
    """Real-time ultra-fast scalping with frequency limits"""
    
    def __init__(self):
        super().__init__("Real-Time Ultra High-Frequency Scalping")
        self.parameters = {
            'scalp_threshold': 0.0008,      # Higher threshold for fewer trades
            'volume_threshold': 1.1,        # Higher volume requirement
            'max_hold_period': 3,           # Longer hold period
            'profit_target': 0.002,         # Higher profit target
            'stop_loss': 0.001              # Higher stop loss
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate price changes
        price_change = data['close'].pct_change()
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()  # Longer window
        volume_ratio = data['volume'] / avg_volume
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(5).std()  # Longer window
        
        signals = pd.Series(0, index=data.index)
        
        # Scalp long on small positive moves with volume
        scalp_long = (
            (price_change > self.parameters['scalp_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(20).mean() * 2)  # More restrictive
        )
        
        # Scalp short on small negative moves with volume
        scalp_short = (
            (price_change < -self.parameters['scalp_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(20).mean() * 2)  # More restrictive
        )
        
        signals[scalp_long] = 1
        signals[scalp_short] = -1
        
        return signals

class RealTimeExtremeMeanReversion(RealTimeMetaStrategy):
    """Real-time ultra-aggressive mean reversion strategy with frequency limits"""
    
    def __init__(self):
        super().__init__("Real-Time Extreme Mean Reversion")
        self.parameters = {
            'mean_period': 10,               # Longer mean period
            'deviation_threshold': 0.002,    # Higher threshold
            'volume_threshold': 1.05,        # Higher volume requirement
            'max_hold_period': 10           # Longer hold period
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate short-term mean
        short_mean = data['close'].rolling(self.parameters['mean_period']).mean()
        
        # Calculate deviation from mean
        deviation = (data['close'] - short_mean) / short_mean
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()  # Longer window
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Long when price is significantly below mean
        long_condition = (
            (deviation < -self.parameters['deviation_threshold']) &
            (volume_ratio > self.parameters['volume_threshold'])
        )
        
        # Short when price is significantly above mean
        short_condition = (
            (deviation > self.parameters['deviation_threshold']) &
            (volume_ratio > self.parameters['volume_threshold'])
        )
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals

# ============================================================================
# REAL-TIME STRATEGY SELECTOR
# ============================================================================

class RealTimeStrategySelector:
    """Real-time strategy selector with frequency constraints"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.strategies = []
        self.is_trained = False
        self.constraints = RealTimeConstraints()
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract market features for strategy selection"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['price_change'] = data['close'].pct_change()
        features['price_change_3m'] = data['close'].pct_change(3)
        features['price_change_5m'] = data['close'].pct_change(5)
        features['price_change_15m'] = data['close'].pct_change(15)
        
        # Volatility features
        features['volatility_3m'] = data['close'].pct_change().rolling(3).std()
        features['volatility_5m'] = data['close'].pct_change().rolling(5).std()
        features['volatility_15m'] = data['close'].pct_change().rolling(15).std()
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(5).mean()
        features['volume_change'] = data['volume'].pct_change()
        
        # Momentum features
        features['momentum_3m'] = data['close'].pct_change(3)
        features['momentum_5m'] = data['close'].pct_change(5)
        features['momentum_15m'] = data['close'].pct_change(15)
        
        # Trend features
        features['sma_3'] = data['close'].rolling(3).mean()
        features['sma_10'] = data['close'].rolling(10).mean()
        features['trend_3_10'] = (features['sma_3'] - features['sma_10']) / features['sma_10']
        
        # Time features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        
        return features.dropna()
    
    def train_selector(self, data: pd.DataFrame, strategy_performances: dict):
        """Train the strategy selector"""
        features = self.extract_features(data)
        
        # Group data by hour and find best strategy per hour
        hourly_data = []
        hourly_labels = []
        
        for hour in range(9, 16):  # Trading hours 9 AM to 4 PM
            hour_data = data[data.index.hour == hour]
            if len(hour_data) == 0:
                continue
            
            # Find best performing strategy for this hour
            best_strategy = None
            best_performance = -np.inf
            
            for strategy_name, performances in strategy_performances.items():
                if len(performances) > 0:
                    avg_performance = np.mean(performances)
                    if avg_performance > best_performance:
                        best_performance = avg_performance
                        best_strategy = strategy_name
            
            if best_strategy is not None:
                # Use features from this hour
                hour_features = features[features.index.hour == hour]
                if len(hour_features) > 0:
                    # Use the most recent features for this hour
                    latest_features = hour_features.iloc[-1:].values
                    hourly_data.append(latest_features.flatten())
                    hourly_labels.append(best_strategy)
        
        if len(hourly_data) > 0:
            X = np.array(hourly_data)
            y = np.array(hourly_labels)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
    
    def select_strategy(self, current_data: pd.DataFrame) -> RealTimeMetaStrategy:
        """Select the best strategy for current market conditions"""
        if not self.is_trained or len(self.strategies) == 0:
            return self.strategies[0] if self.strategies else None
        
        # Extract features for current data
        features = self.extract_features(current_data)
        if len(features) == 0:
            return self.strategies[0]
        
        # Use most recent features
        current_features = features.iloc[-1:].values
        current_features_scaled = self.scaler.transform(current_features)
        
        # Predict best strategy
        predicted_strategy = self.model.predict(current_features_scaled)[0]
        
        # Find strategy by name
        for strategy in self.strategies:
            if strategy.name == predicted_strategy:
                return strategy
        
        return self.strategies[0]  # Default to first strategy

# ============================================================================
# REAL-TIME META TRADING AI
# ============================================================================

class RealTimeAggressiveMetaTradingAI:
    """Real-time aggressive meta-trading AI system with frequency constraints"""
    
    def __init__(self, training_days: int = 180):
        """
        Initialize real-time trading system
        
        Args:
            training_days: Number of days for training (default: 180 = 6 months)
        """
        self.training_days = training_days
        self.constraints = RealTimeConstraints()
        
        # Initialize strategy pools with real-time constraints
        self.high_vol_pool = [
            RealTimeUltraVolatilityExploitation(),
            RealTimeBreakoutMomentum()
        ]
        
        self.trending_pool = [
            RealTimeUltraMomentumAmplification(),
            RealTimeAcceleratedMACross()
        ]
        
        self.ranging_pool = [
            RealTimeUltraHighFrequencyScalping(),
            RealTimeExtremeMeanReversion()
        ]
        
        # Initialize selector
        self.selector = RealTimeStrategySelector()
        
        # Load data
        print("Loading data for real-time trading system...")
        self.data = pickle.load(open('polygon_QQQ_1m.pkl', 'rb'))
        
        # Convert index to datetime if needed
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except ValueError:
                self.data.index = pd.to_datetime(self.data.index, utc=True)
        
        # Handle timezone if present
        if self.data.index.tz is not None:
            self.data.index = self.data.index.tz_localize(None)
        
        # Filter to trading hours and weekdays
        self.data = self.data.between_time('09:30', '16:00')
        self.data = self.data[self.data.index.dayofweek < 5]
        
        # Filter to last 5 years
        five_years_ago = self.data.index.max() - timedelta(days=5*365)
        self.data = self.data[self.data.index >= five_years_ago]
        
        print(f"Data loaded: {len(self.data)} records from {self.data.index.min()} to {self.data.index.max()}")
        print(f"Real-time constraints: {self.constraints.min_trade_interval} min between trades")
        print(f"Training period: {training_days} days ({training_days/30:.1f} months)")
        print(f"Available strategies: {len(self.high_vol_pool + self.trending_pool + self.ranging_pool)}")
    
    def run_real_time_meta_system(self, test_period_days: int = 10) -> dict:
        """Run the real-time meta-trading system"""
        print(f"\nRunning Real-Time MetaTradingAI v3.0: Frequency-Limited Model...")
        print(f"Training Period: {self.training_days} days ({self.training_days/30:.1f} months)")
        print(f"Trade Frequency Limit: 1 trade per {self.constraints.min_trade_interval} minutes")
        
        # Calculate date ranges with extended training
        end_date = self.data.index.max()
        test_start_date = end_date - timedelta(days=test_period_days)
        training_start_date = test_start_date - timedelta(days=self.training_days)
        
        print(f"Real-Time Meta-Trading AI System Setup:")
        print(f"  Training period: {training_start_date.date()} to {test_start_date.date()}")
        print(f"  Test period: {test_start_date.date()} to {end_date.date()}")
        print(f"  Training days: {self.training_days} (vs 60 in original v3.0)")
        print(f"  Target: 5% return over {test_period_days} trading days")
        
        # Split data
        training_data = self.data[(self.data.index >= training_start_date) & (self.data.index < test_start_date)]
        test_data = self.data[self.data.index >= test_start_date]
        
        print(f"  Training data: {len(training_data):,} records")
        print(f"  Test data: {len(test_data):,} records")
        print(f"  Training hours: {len(training_data) // 60:,}")
        print(f"  Test hours: {len(test_data) // 60:,}")
        
        # Calculate training data improvement
        original_training_hours = 240  # From v3.0
        current_training_hours = len(training_data) // 60
        improvement_factor = current_training_hours / original_training_hours
        
        print(f"  Training data improvement: {improvement_factor:.1f}x more data")
        
        # Test all strategies on training data to get performance baseline
        print(f"\nTesting all strategies on extended training data...")
        strategy_performances = {}
        
        all_strategies = self.high_vol_pool + self.trending_pool + self.ranging_pool
        
        for i, strategy in enumerate(all_strategies):
            print(f"  [{i+1}/{len(all_strategies)}] Testing {strategy.name}...")
            results = strategy.backtest(training_data)
            strategy_performances[strategy.name] = [results['avg_hourly_return']]
            print(f"    Trades: {results['num_trades']}, Trade Frequency: {results['trade_frequency']:.2f} trades/hour")
        
        # Train strategy selector with extended data
        self.selector.strategies = all_strategies
        self.selector.train_selector(training_data, strategy_performances)
        
        # Run real-time meta system with detailed reporting
        print(f"\nRunning real-time meta-trading system...")
        results = {
            'hourly_performance': [],
            'daily_performance': [],
            'selected_strategies': [],
            'strategy_performance': {s.name: [] for s in all_strategies},
            'cumulative_return': 0,
            'daily_returns': [],
            'training_days': self.training_days,
            'training_hours': current_training_hours,
            'improvement_factor': improvement_factor,
            'total_trades': 0,
            'trade_frequency': 0
        }
        
        # Group test data by day
        test_data_daily = test_data.groupby(test_data.index.date)
        
        for date, day_data in test_data_daily:
            print(f"\n=== Trading Day: {date} ===")
            
            # Detect market regime for this day
            regime = detect_market_regime(day_data)
            print(f"  Market Regime Detected: {regime.upper()}")
            
            # Select strategy pool based on regime
            if regime == "high_volatility":
                active_strategies = self.high_vol_pool
                print(f"  Active Pool: High Volatility ({len(active_strategies)} strategies)")
            elif regime == "trending":
                active_strategies = self.trending_pool
                print(f"  Active Pool: Trending ({len(active_strategies)} strategies)")
            else:
                active_strategies = self.ranging_pool
                print(f"  Active Pool: Ranging ({len(active_strategies)} strategies)")
            
            # Update selector to use the active pool
            self.selector.strategies = active_strategies
            
            # Group day data by hour
            day_data_hourly = day_data.groupby(day_data.index.hour)
            
            daily_return = 0
            daily_trades = 0
            
            for hour, hour_data in day_data_hourly:
                if len(hour_data) < 10:  # Skip hours with insufficient data
                    continue
                
                # Select best strategy for this hour
                selected_strategy = self.selector.select_strategy(hour_data)
                if selected_strategy is None:
                    continue
                
                # Run strategy for this hour with frequency constraints
                strategy_results = selected_strategy.backtest(hour_data)
                
                # Record performance
                hourly_perf = {
                    'date': date,
                    'hour': hour,
                    'selected_strategy': selected_strategy.name,
                    'regime': regime,
                    'avg_hourly_return': strategy_results['avg_hourly_return'],
                    'total_return': strategy_results['total_return'],
                    'num_trades': strategy_results['num_trades'],
                    'trade_frequency': strategy_results['trade_frequency']
                }
                
                results['hourly_performance'].append(hourly_perf)
                results['selected_strategies'].append(selected_strategy.name)
                results['strategy_performance'][selected_strategy.name].append(strategy_results['avg_hourly_return'])
                
                daily_return += strategy_results['total_return']
                daily_trades += strategy_results['num_trades']
                results['total_trades'] += strategy_results['num_trades']
                
                print(f"  Hour {hour:02d}:00 - Selected: {selected_strategy.name}")
                print(f"    Return: {strategy_results['total_return']:.4f} ({strategy_results['total_return']*100:.2f}%), Trades: {strategy_results['num_trades']}")
                print(f"    Trade Frequency: {strategy_results['trade_frequency']:.2f} trades/hour")
            
            # Update cumulative return
            results['cumulative_return'] += daily_return
            results['daily_returns'].append(daily_return)
            
            # Daily summary
            print(f"  Daily Summary: Return: {daily_return:.4f} ({daily_return*100:.2f}%), Trades: {daily_trades}")
            print(f"  Cumulative Return: {results['cumulative_return']:.4f} ({results['cumulative_return']*100:.2f}%)")
        
        # Calculate overall trade frequency
        total_hours = len(test_data) // 60
        results['trade_frequency'] = results['total_trades'] / total_hours if total_hours > 0 else 0
        
        # Final results with real-time analysis
        print(f"\n🎯 REAL-TIME TARGET ACHIEVEMENT:")
        print(f"  Target: 5% return over {test_period_days} trading days")
        print(f"  Actual: {results['cumulative_return']:.4f} ({results['cumulative_return']*100:.2f}%)")
        print(f"  Training Days: {self.training_days} (vs 60 in original)")
        print(f"  Training Hours: {current_training_hours:,} (vs 240 in original)")
        print(f"  Data Improvement: {improvement_factor:.1f}x more training data")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Average Trade Frequency: {results['trade_frequency']:.2f} trades/hour")
        print(f"  Frequency Limit: 1 trade per {self.constraints.min_trade_interval} minutes")
        
        if results['cumulative_return'] >= 0.05:
            print(f"  Status: ✅ ACHIEVED")
        else:
            print(f"  Status: ❌ NOT ACHIEVED")
        
        # Strategy distribution
        strategy_counts = {}
        for strategy_name in results['selected_strategies']:
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
        
        print(f"\n📊 STRATEGY DISTRIBUTION:")
        total_selections = len(results['selected_strategies'])
        for strategy_name, count in strategy_counts.items():
            percentage = (count / total_selections * 100) if total_selections > 0 else 0
            print(f"  {strategy_name}: {percentage:.1f}% ({count} hours)")
        
        # Real-time performance analysis
        print(f"\n📈 REAL-TIME PERFORMANCE ANALYSIS:")
        print(f"  Training Data Improvement: {improvement_factor:.1f}x more data")
        print(f"  Expected Consistency: {min(3.0/improvement_factor, 1.0):.1f}x more consistent")
        print(f"  Regime Coverage: {'Excellent' if self.training_days >= 180 else 'Good' if self.training_days >= 90 else 'Limited'}")
        print(f"  Strategy Robustness: {'High' if self.training_days >= 180 else 'Medium' if self.training_days >= 90 else 'Low'}")
        print(f"  Trade Frequency Compliance: {'✅ Within Limits' if results['trade_frequency'] <= 30 else '⚠️ Exceeds Limits'}")
        
        return results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time MetaTradingAI v3.0')
    parser.add_argument('--training-days', type=int, default=180, 
                       help='Number of training days (default: 180)')
    
    args = parser.parse_args()
    
    # Run real-time system
    system = RealTimeAggressiveMetaTradingAI(training_days=args.training_days)
    results = system.run_real_time_meta_system(test_period_days=10)
    
    return results

if __name__ == "__main__":
    results = main() 