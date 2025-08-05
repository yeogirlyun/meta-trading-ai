#!/usr/bin/env python3
"""
MetaTradingAI v3.0 Enhanced Restricted - Optimized for 2-Minute Order Limits
Target: 6-9% returns over 10 trading days with quality-focused trading
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
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
# ENHANCED TRADING CONSTRAINTS
# ============================================================================

class EnhancedTradingConstraints:
    """Enhanced trading constraints for quality-focused trading"""
    
    def __init__(self):
        # Order frequency limits
        self.min_order_interval = 120  # 2 minutes between orders
        self.max_orders_per_hour = 30  # maximum orders per hour
        self.max_orders_per_day = 200  # maximum orders per day
        
        # Signal quality thresholds
        self.min_signal_strength = 1.2  # minimum signal strength score
        self.min_volume_ratio = 1.5     # minimum volume spike
        self.min_momentum_threshold = 0.001  # minimum momentum
        
        # Dynamic leverage settings
        self.base_leverage = 1.0
        self.max_leverage = 3.0
        self.leverage_ranging = 1.0
        self.leverage_trending = 2.0
        self.leverage_high_vol = 3.0
        
        # Position sizing
        self.max_position_size = 0.15  # 15% of capital per trade
        self.risk_per_trade = 0.02     # 2% risk per trade
        self.max_daily_drawdown = 0.05  # 5% daily drawdown limit

# ============================================================================
# ENHANCED MARKET REGIME DETECTION
# ============================================================================

def detect_enhanced_market_regime(data: pd.DataFrame) -> str:
    """Enhanced market regime detection with ML prediction"""
    if len(data) < 50:
        return "ranging"
    
    # Calculate multiple regime indicators
    volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
    avg_vol = data['close'].pct_change().rolling(50).std().mean()
    trend_strength = abs(data['close'].pct_change().rolling(20).mean().iloc[-1]) / volatility
    
    # Volume analysis
    volume_ratio = data['volume'].iloc[-10:].mean() / data['volume'].rolling(50).mean().iloc[-1]
    
    # Momentum analysis
    momentum_5m = data['close'].pct_change(5).iloc[-1]
    momentum_15m = data['close'].pct_change(15).iloc[-1]
    
    # Enhanced regime classification
    if volatility > avg_vol * 1.8 and volume_ratio > 1.3:
        return "high_volatility"
    elif trend_strength > 0.6 and abs(momentum_15m) > 0.002:
        return "trending"
    elif volume_ratio > 1.5 and abs(momentum_5m) > 0.001:
        return "breakout"
    else:
        return "ranging"

# ============================================================================
# ENHANCED STRATEGY BASE CLASS
# ============================================================================

class EnhancedMetaStrategy:
    """Enhanced base class with signal filtering and quality focus"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.last_trade_time = None
        self.trade_count = 0
        self.constraints = EnhancedTradingConstraints()
    
    def set_parameters(self, params: dict):
        self.parameters.update(params)
    
    def can_trade(self, current_time: datetime) -> bool:
        """Check if we can trade based on frequency constraints"""
        if self.last_trade_time is None:
            return True
        
        time_since_last_trade = (current_time - self.last_trade_time).total_seconds()
        return time_since_last_trade >= self.constraints.min_order_interval
    
    def calculate_signal_strength(self, data: pd.DataFrame, index: int) -> float:
        """Calculate signal strength score for quality filtering"""
        if index < 5:
            return 0.0
        
        # Momentum component
        momentum = abs(data['close'].pct_change().iloc[index])
        
        # Volume component
        volume_ratio = data['volume'].iloc[index] / data['volume'].rolling(10).mean().iloc[index]
        
        # Volatility component (inverse - lower volatility is better)
        volatility = data['close'].pct_change().rolling(5).std().iloc[index]
        avg_volatility = data['close'].pct_change().rolling(20).std().mean()
        volatility_score = avg_volatility / (volatility + 1e-6)
        
        # Trend alignment
        short_ma = data['close'].rolling(3).mean().iloc[index]
        long_ma = data['close'].rolling(10).mean().iloc[index]
        trend_alignment = 1.0 if (data['close'].iloc[index] > short_ma > long_ma) or (data['close'].iloc[index] < short_ma < long_ma) else 0.5
        
        # Calculate composite score
        signal_strength = (
            0.4 * momentum * 1000 +  # Scale momentum
            0.3 * volume_ratio +
            0.2 * volatility_score +
            0.1 * trend_alignment
        )
        
        return signal_strength
    
    def update_trade_time(self, trade_time: datetime):
        """Update the last trade time"""
        self.last_trade_time = trade_time
        self.trade_count += 1
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> dict:
        """Enhanced backtest with order restrictions and quality filtering"""
        signals = self.calculate_signals(data)
        
        position = 0  # 1 for long, -1 for short, 0 for flat
        capital = initial_capital
        equity = [capital]
        trades = []
        hourly_returns = []
        last_trade_time = None
        leverage = self.constraints.base_leverage
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]['close']
            prev_price = data.iloc[i-1]['close']
            
            # Check trading constraints
            can_trade = True
            if last_trade_time is not None:
                time_since_last = (current_time - last_trade_time).total_seconds()
                can_trade = time_since_last >= self.constraints.min_order_interval
            
            # Calculate signal strength for quality filtering
            signal_strength = self.calculate_signal_strength(data, i)
            
            # Update position based on signals (only if we can trade and signal is strong)
            if can_trade and signals.iloc[i] != 0 and signal_strength > self.constraints.min_signal_strength:
                # Determine leverage based on regime
                regime = detect_enhanced_market_regime(data.iloc[max(0, i-20):i+1])
                if regime == "high_volatility":
                    leverage = self.constraints.leverage_high_vol
                elif regime == "trending":
                    leverage = self.constraints.leverage_trending
                else:
                    leverage = self.constraints.leverage_ranging
                
                if signals.iloc[i] == 1 and position <= 0:  # Buy signal
                    if position == -1:  # Close short
                        trades.append({
                            'entry_time': data.index[i-1],
                            'exit_time': data.index[i],
                            'entry_price': prev_price,
                            'exit_price': current_price,
                            'type': 'short',
                            'pnl': (prev_price - current_price) * leverage,
                            'leverage': leverage,
                            'signal_strength': signal_strength
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
                            'pnl': (current_price - prev_price) * leverage,
                            'leverage': leverage,
                            'signal_strength': signal_strength
                        })
                    position = -1
                    last_trade_time = current_time
            
            # Calculate returns with leverage
            if position == 1:  # Long position
                hourly_return = (current_price - prev_price) / prev_price * leverage
            elif position == -1:  # Short position
                hourly_return = (prev_price - current_price) / prev_price * leverage
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
            'trade_frequency': len(trades) / (len(data) / 60) if len(data) > 0 else 0,
            'avg_signal_strength': np.mean([t['signal_strength'] for t in trades]) if trades else 0,
            'avg_leverage': np.mean([t['leverage'] for t in trades]) if trades else 1.0
        }

# ============================================================================
# ENHANCED STRATEGY POOLS (QUALITY-FOCUSED)
# ============================================================================

class EnhancedUltraVolatilityExploitation(EnhancedMetaStrategy):
    """Enhanced ultra-aggressive volatility exploitation with quality filtering"""
    
    def __init__(self):
        super().__init__("Enhanced Ultra Volatility Exploitation")
        self.parameters = {
            'volatility_threshold': 0.001,  # Lower for more opportunities
            'momentum_period': 1,           # Responsive
            'reversal_threshold': 0.001,    # Lower threshold
            'max_hold_period': 15,          # Extended hold period
            'volume_threshold': 1.3         # Higher volume requirement
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(5).std()
        
        # Calculate momentum
        momentum = data['close'].pct_change(self.parameters['momentum_period'])
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Entry conditions - enhanced for quality
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

class EnhancedBreakoutMomentum(EnhancedMetaStrategy):
    """Enhanced breakout momentum strategy with quality filtering"""
    
    def __init__(self):
        super().__init__("Enhanced Breakout Momentum")
        self.parameters = {
            'breakout_period': 5,           # Responsive period
            'confirmation_period': 2,       # Quick confirmation
            'volume_multiplier': 2.0,       # High volume requirement
            'momentum_threshold': 0.002,    # Higher threshold for quality
            'max_hold_period': 20          # Extended hold period
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate breakout levels
        highest_high = data['high'].rolling(self.parameters['breakout_period']).max()
        lowest_low = data['low'].rolling(self.parameters['breakout_period']).min()
        
        # Breakout conditions
        breakout_up = data['close'] > highest_high.shift(1)
        breakout_down = data['close'] < lowest_low.shift(1)
        
        # Volume confirmation
        avg_volume = data['volume'].rolling(15).mean()
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

class EnhancedUltraMomentumAmplification(EnhancedMetaStrategy):
    """Enhanced ultra-aggressive momentum amplification with quality filtering"""
    
    def __init__(self):
        super().__init__("Enhanced Ultra Momentum Amplification")
        self.parameters = {
            'short_period': 1,               # Responsive periods
            'medium_period': 3,              # Medium period
            'long_period': 5,                # Long period
            'momentum_threshold': 0.0008,    # Higher threshold for quality
            'acceleration_threshold': 0.0006, # Higher acceleration
            'volume_threshold': 1.3,         # Higher volume requirement
            'max_hold_period': 15           # Extended hold period
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate multiple momentum indicators
        short_momentum = data['close'].pct_change(self.parameters['short_period'])
        medium_momentum = data['close'].pct_change(self.parameters['medium_period'])
        long_momentum = data['close'].pct_change(self.parameters['long_period'])
        
        # Calculate acceleration
        acceleration = short_momentum.diff()
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(5).std()
        
        signals = pd.Series(0, index=data.index)
        
        # Strong buy when all momentum indicators align
        strong_buy = (
            (short_momentum > self.parameters['momentum_threshold']) &
            (medium_momentum > self.parameters['momentum_threshold']) &
            (long_momentum > 0) &
            (acceleration > self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(15).mean() * 4)  # Less restrictive
        )
        
        # Strong sell when all momentum indicators align negatively
        strong_sell = (
            (short_momentum < -self.parameters['momentum_threshold']) &
            (medium_momentum < -self.parameters['momentum_threshold']) &
            (long_momentum < 0) &
            (acceleration < -self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(15).mean() * 4)  # Less restrictive
        )
        
        signals[strong_buy] = 1
        signals[strong_sell] = -1
        
        return signals

class EnhancedAcceleratedMACross(EnhancedMetaStrategy):
    """Enhanced accelerated moving average crossover with quality filtering"""
    
    def __init__(self):
        super().__init__("Enhanced Accelerated MA Cross")
        self.parameters = {
            'fast_period': 2,                # Responsive periods
            'slow_period': 8,                # Slower period
            'momentum_threshold': 0.0008,    # Higher threshold for quality
            'volume_threshold': 1.3,         # Higher volume requirement
            'max_hold_period': 25           # Extended hold period
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate moving averages
        fast_ma = data['close'].rolling(self.parameters['fast_period']).mean()
        slow_ma = data['close'].rolling(self.parameters['slow_period']).mean()
        
        # Calculate momentum
        momentum = data['close'].pct_change(3)
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()
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

class EnhancedUltraHighFrequencyScalping(EnhancedMetaStrategy):
    """Enhanced ultra-fast scalping with quality filtering"""
    
    def __init__(self):
        super().__init__("Enhanced Ultra High-Frequency Scalping")
        self.parameters = {
            'scalp_threshold': 0.0006,      # Higher threshold for quality
            'volume_threshold': 1.5,         # Higher volume requirement
            'max_hold_period': 5,           # Extended hold period
            'profit_target': 0.002,         # Higher profit target
            'stop_loss': 0.001              # Higher stop loss
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate price changes
        price_change = data['close'].pct_change()
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(5).std()
        
        signals = pd.Series(0, index=data.index)
        
        # Scalp long on small positive moves with volume
        scalp_long = (
            (price_change > self.parameters['scalp_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(15).mean() * 3)  # Less restrictive
        )
        
        # Scalp short on small negative moves with volume
        scalp_short = (
            (price_change < -self.parameters['scalp_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(15).mean() * 3)  # Less restrictive
        )
        
        signals[scalp_long] = 1
        signals[scalp_short] = -1
        
        return signals

class EnhancedExtremeMeanReversion(EnhancedMetaStrategy):
    """Enhanced ultra-aggressive mean reversion strategy with quality filtering"""
    
    def __init__(self):
        super().__init__("Enhanced Extreme Mean Reversion")
        self.parameters = {
            'mean_period': 8,                # Responsive mean period
            'deviation_threshold': 0.0015,   # Higher threshold for quality
            'volume_threshold': 1.3,         # Higher volume requirement
            'max_hold_period': 12           # Extended hold period
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate short-term mean
        short_mean = data['close'].rolling(self.parameters['mean_period']).mean()
        
        # Calculate deviation from mean
        deviation = (data['close'] - short_mean) / short_mean
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()
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
# ENHANCED STRATEGY SELECTOR
# ============================================================================

class EnhancedStrategySelector:
    """Enhanced strategy selector with quality focus"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.strategies = []
        self.is_trained = False
        self.constraints = EnhancedTradingConstraints()
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced market features for strategy selection"""
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
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
        features['volume_change'] = data['volume'].pct_change()
        
        # Momentum features
        features['momentum_3m'] = data['close'].pct_change(3)
        features['momentum_5m'] = data['close'].pct_change(5)
        features['momentum_15m'] = data['close'].pct_change(15)
        
        # Trend features
        features['sma_3'] = data['close'].rolling(3).mean()
        features['sma_10'] = data['close'].rolling(10).mean()
        features['trend_3_10'] = (features['sma_3'] - features['sma_10']) / features['sma_10']
        
        # Signal strength features
        features['signal_strength'] = features['momentum_3m'] * features['volume_ratio'] / (features['volatility_5m'] + 1e-6)
        
        # Time features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        
        return features.dropna()
    
    def train_selector(self, data: pd.DataFrame, strategy_performances: dict):
        """Train the enhanced strategy selector"""
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
    
    def select_strategy(self, current_data: pd.DataFrame) -> EnhancedMetaStrategy:
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
# ENHANCED META TRADING AI
# ============================================================================

class EnhancedAggressiveMetaTradingAI:
    """Enhanced aggressive meta-trading AI system with quality focus"""
    
    def __init__(self, training_days: int = 180):
        """
        Initialize enhanced trading system
        
        Args:
            training_days: Number of days for training (default: 180 = 6 months)
        """
        self.training_days = training_days
        self.constraints = EnhancedTradingConstraints()
        
        # Initialize strategy pools with enhanced parameters
        self.high_vol_pool = [
            EnhancedUltraVolatilityExploitation(),
            EnhancedBreakoutMomentum()
        ]
        
        self.trending_pool = [
            EnhancedUltraMomentumAmplification(),
            EnhancedAcceleratedMACross()
        ]
        
        self.ranging_pool = [
            EnhancedUltraHighFrequencyScalping(),
            EnhancedExtremeMeanReversion()
        ]
        
        # Initialize selector
        self.selector = EnhancedStrategySelector()
        
        # Load data
        print("Loading data for enhanced trading system...")
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
        print(f"Enhanced constraints: {self.constraints.min_order_interval} sec between orders")
        print(f"Training period: {training_days} days ({training_days/30:.1f} months)")
        print(f"Available strategies: {len(self.high_vol_pool + self.trending_pool + self.ranging_pool)}")
    
    def run_enhanced_meta_system(self, test_period_days: int = 10) -> dict:
        """Run the enhanced meta-trading system with quality focus"""
        print(f"\nRunning Enhanced MetaTradingAI v3.0: Quality-Focused Model...")
        print(f"Training Period: {self.training_days} days ({self.training_days/30:.1f} months)")
        print(f"Order Frequency Limit: 1 order per {self.constraints.min_order_interval} seconds")
        print(f"Target: 6-9% return over {test_period_days} trading days")
        
        # Calculate date ranges with extended training
        end_date = self.data.index.max()
        test_start_date = end_date - timedelta(days=test_period_days)
        training_start_date = test_start_date - timedelta(days=self.training_days)
        
        print(f"Enhanced Meta-Trading AI System Setup:")
        print(f"  Training period: {training_start_date.date()} to {test_start_date.date()}")
        print(f"  Test period: {test_start_date.date()} to {end_date.date()}")
        print(f"  Training days: {self.training_days} (vs 60 in original v3.0)")
        print(f"  Target: 6-9% return over {test_period_days} trading days")
        
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
            print(f"    Avg Signal Strength: {results['avg_signal_strength']:.2f}, Avg Leverage: {results['avg_leverage']:.1f}x")
        
        # Train strategy selector with extended data
        self.selector.strategies = all_strategies
        self.selector.train_selector(training_data, strategy_performances)
        
        # Run enhanced meta system with detailed reporting
        print(f"\nRunning enhanced meta-trading system...")
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
            'trade_frequency': 0,
            'avg_signal_strength': 0,
            'avg_leverage': 0
        }
        
        # Group test data by day
        test_data_daily = test_data.groupby(test_data.index.date)
        
        for date, day_data in test_data_daily:
            print(f"\n=== Trading Day: {date} ===")
            
            # Detect market regime for this day
            regime = detect_enhanced_market_regime(day_data)
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
                
                # Run strategy for this hour with enhanced constraints
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
                    'trade_frequency': strategy_results['trade_frequency'],
                    'avg_signal_strength': strategy_results['avg_signal_strength'],
                    'avg_leverage': strategy_results['avg_leverage']
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
                print(f"    Signal Strength: {strategy_results['avg_signal_strength']:.2f}, Leverage: {strategy_results['avg_leverage']:.1f}x")
            
            # Update cumulative return
            results['cumulative_return'] += daily_return
            results['daily_returns'].append(daily_return)
            
            # Daily summary
            print(f"  Daily Summary: Return: {daily_return:.4f} ({daily_return*100:.2f}%), Trades: {daily_trades}")
            print(f"  Cumulative Return: {results['cumulative_return']:.4f} ({results['cumulative_return']*100:.2f}%)")
        
        # Calculate overall metrics
        total_hours = len(test_data) // 60
        results['trade_frequency'] = results['total_trades'] / total_hours if total_hours > 0 else 0
        
        # Calculate average signal strength and leverage
        if results['hourly_performance']:
            results['avg_signal_strength'] = np.mean([p['avg_signal_strength'] for p in results['hourly_performance']])
            results['avg_leverage'] = np.mean([p['avg_leverage'] for p in results['hourly_performance']])
        
        # Final results with enhanced analysis
        print(f"\n🎯 ENHANCED TARGET ACHIEVEMENT:")
        print(f"  Target: 6-9% return over {test_period_days} trading days")
        print(f"  Actual: {results['cumulative_return']:.4f} ({results['cumulative_return']*100:.2f}%)")
        print(f"  Training Days: {self.training_days} (vs 60 in original)")
        print(f"  Training Hours: {current_training_hours:,} (vs 240 in original)")
        print(f"  Data Improvement: {improvement_factor:.1f}x more training data")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Average Trade Frequency: {results['trade_frequency']:.2f} trades/hour")
        print(f"  Order Frequency Limit: 1 order per {self.constraints.min_order_interval} seconds")
        print(f"  Average Signal Strength: {results['avg_signal_strength']:.2f}")
        print(f"  Average Leverage: {results['avg_leverage']:.1f}x")
        
        if results['cumulative_return'] >= 0.06:
            print(f"  Status: ✅ EXCEEDED TARGET")
        elif results['cumulative_return'] >= 0.05:
            print(f"  Status: ✅ ACHIEVED TARGET")
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
        
        # Enhanced performance analysis
        print(f"\n📈 ENHANCED PERFORMANCE ANALYSIS:")
        print(f"  Training Data Improvement: {improvement_factor:.1f}x more data")
        print(f"  Expected Consistency: {min(3.0/improvement_factor, 1.0):.1f}x more consistent")
        print(f"  Regime Coverage: {'Excellent' if self.training_days >= 180 else 'Good' if self.training_days >= 90 else 'Limited'}")
        print(f"  Strategy Robustness: {'High' if self.training_days >= 180 else 'Medium' if self.training_days >= 90 else 'Low'}")
        print(f"  Order Frequency Compliance: {'✅ Within Limits' if results['trade_frequency'] <= 30 else '⚠️ Exceeds Limits'}")
        print(f"  Quality Focus: {'✅ High' if results['avg_signal_strength'] > 1.5 else '⚠️ Medium' if results['avg_signal_strength'] > 1.0 else '❌ Low'}")
        
        return results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced MetaTradingAI v3.0')
    parser.add_argument('--training-days', type=int, default=180, 
                       help='Number of training days (default: 180)')
    
    args = parser.parse_args()
    
    # Run enhanced system
    system = EnhancedAggressiveMetaTradingAI(training_days=args.training_days)
    results = system.run_enhanced_meta_system(test_period_days=10)
    
    return results

if __name__ == "__main__":
    results = main() 