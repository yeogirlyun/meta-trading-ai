import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================

def detect_market_regime(data: pd.DataFrame) -> str:
    """
    Detects the current market regime based on volatility and trend strength.
    """
    if len(data) < 50:
        return "ranging"  # Default for insufficient data
    
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
# HYPER-AGGRESSIVE STRATEGY POOLS
# ============================================================================

class AggressiveMetaStrategy:
    """Base class for aggressive meta strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
    
    def set_parameters(self, params: dict):
        self.parameters.update(params)
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> dict:
        signals = self.calculate_signals(data)
        
        position = 0  # 1 for long, -1 for short, 0 for flat
        capital = initial_capital
        equity = [capital]
        trades = []
        hourly_returns = []
        
        for i in range(1, len(data)):
            current_price = data.iloc[i]['close']
            prev_price = data.iloc[i-1]['close']
            
            # Update position based on signals
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
            'target_10d_05_pct': target_10d_05
        }

# ============================================================================
# HIGH-VOLATILITY POOL STRATEGIES
# ============================================================================

class UltraVolatilityExploitation(AggressiveMetaStrategy):
    """Ultra-aggressive volatility exploitation with even lower thresholds"""
    
    def __init__(self):
        super().__init__("Ultra Volatility Exploitation")
        self.parameters = {
            'volatility_threshold': 0.0015,  # Ultra-low threshold
            'momentum_period': 2,            # Very short period
            'reversal_threshold': 0.0015,    # Ultra-low threshold
            'max_hold_period': 10,           # Shorter hold period
            'volume_threshold': 1.05         # Lower volume requirement
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(10).std()
        
        # Calculate momentum
        momentum = data['close'].pct_change(self.parameters['momentum_period'])
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Entry conditions - ultra aggressive
        high_vol = volatility > self.parameters['volatility_threshold']
        volume_confirmed = volume_ratio > self.parameters['volume_threshold']
        
        # Long on positive momentum during high volatility
        long_condition = (
            (momentum > self.parameters['reversal_threshold']) & 
            high_vol & 
            volume_confirmed
        )
        
        # Short on negative momentum during high volatility
        short_condition = (
            (momentum < -self.parameters['reversal_threshold']) & 
            high_vol & 
            volume_confirmed
        )
        
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        return signals

class BreakoutMomentum(AggressiveMetaStrategy):
    """New strategy for trading sharp, sudden price movements"""
    
    def __init__(self):
        super().__init__("Breakout Momentum")
        self.parameters = {
            'breakout_period': 10,
            'confirmation_period': 3,
            'volume_multiplier': 2.0,
            'momentum_threshold': 0.002,
            'max_hold_period': 15
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate breakout levels
        highest_high = data['high'].rolling(self.parameters['breakout_period']).max()
        lowest_low = data['low'].rolling(self.parameters['breakout_period']).min()
        
        # Breakout conditions
        breakout_up = data['close'] > highest_high.shift(1)
        breakout_down = data['close'] < lowest_low.shift(1)
        
        # Volume confirmation
        avg_volume = data['volume'].rolling(20).mean()
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

# ============================================================================
# TRENDING POOL STRATEGIES
# ============================================================================

class UltraMomentumAmplification(AggressiveMetaStrategy):
    """Ultra-aggressive momentum amplification for trending markets"""
    
    def __init__(self):
        super().__init__("Ultra Momentum Amplification")
        self.parameters = {
            'short_period': 1,              # Ultra-short period
            'medium_period': 3,             # Shorter medium period
            'long_period': 10,              # Shorter long period
            'momentum_threshold': 0.0008,   # Ultra-low threshold
            'acceleration_threshold': 0.0005, # Ultra-low acceleration
            'volume_threshold': 1.05,       # Lower volume requirement
            'max_hold_period': 15           # Shorter hold period
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
            (volatility < volatility.rolling(20).mean() * 3)
        )
        
        # Strong sell when all momentum indicators align negatively
        strong_sell = (
            (short_momentum < -self.parameters['momentum_threshold']) &
            (medium_momentum < -self.parameters['momentum_threshold']) &
            (long_momentum < 0) &
            (acceleration < -self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(20).mean() * 3)
        )
        
        signals[strong_buy] = 1
        signals[strong_sell] = -1
        
        return signals

class AcceleratedMACross(AggressiveMetaStrategy):
    """Faster version of moving average crossover to catch trends earlier"""
    
    def __init__(self):
        super().__init__("Accelerated MA Cross")
        self.parameters = {
            'fast_period': 2,               # Ultra-fast period
            'slow_period': 8,               # Shorter slow period
            'momentum_threshold': 0.001,    # Low momentum threshold
            'volume_threshold': 1.05,       # Lower volume requirement
            'max_hold_period': 20           # Shorter hold period
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

# ============================================================================
# RANGING POOL STRATEGIES
# ============================================================================

class UltraHighFrequencyScalping(AggressiveMetaStrategy):
    """Ultra-fast scalping with very tight profit targets and stop-losses"""
    
    def __init__(self):
        super().__init__("Ultra High-Frequency Scalping")
        self.parameters = {
            'scalp_threshold': 0.0005,      # Ultra-tight entry threshold
            'volume_threshold': 1.0,        # Lower volume requirement
            'max_hold_period': 1,           # 1-minute max hold
            'profit_target': 0.001,         # 0.1% profit target
            'stop_loss': 0.0005             # 0.05% stop loss
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
            (volatility < volatility.rolling(20).mean() * 2)
        )
        
        # Scalp short on small negative moves with volume
        scalp_short = (
            (price_change < -self.parameters['scalp_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(20).mean() * 2)
        )
        
        signals[scalp_long] = 1
        signals[scalp_short] = -1
        
        return signals

class ExtremeMeanReversion(AggressiveMetaStrategy):
    """Strategy that trades on extreme deviations from the short-term mean"""
    
    def __init__(self):
        super().__init__("Extreme Mean Reversion")
        self.parameters = {
            'mean_period': 5,               # Short-term mean
            'deviation_threshold': 0.002,   # Low deviation threshold
            'volume_threshold': 1.05,       # Lower volume requirement
            'max_hold_period': 10           # Shorter hold period
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
# STRATEGY SELECTOR
# ============================================================================

class AggressiveStrategySelector:
    """Strategy selector using Random Forest for regime-specific strategy pools"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.strategies = []
        self.is_trained = False
    
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
        
        # Gap features
        daily_data = data.resample('D').agg({'open': 'first', 'close': 'last'}).dropna()
        gaps = (daily_data['open'] - daily_data['close'].shift(1)) / daily_data['close'].shift(1)
        gap_signals = gaps.reindex(data.index, method='ffill')
        features['gap_size'] = gap_signals
        
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
    
    def select_strategy(self, current_data: pd.DataFrame) -> AggressiveMetaStrategy:
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
# MAIN SYSTEM
# ============================================================================

class AggressiveMetaTradingAI:
    """Aggressive meta-trading AI system with regime detection"""
    
    def __init__(self):
        # Initialize strategy pools
        self.high_vol_pool = [
            UltraVolatilityExploitation(),
            BreakoutMomentum()
        ]
        
        self.trending_pool = [
            UltraMomentumAmplification(),
            AcceleratedMACross()
        ]
        
        self.ranging_pool = [
            UltraHighFrequencyScalping(),
            ExtremeMeanReversion()
        ]
        
        # Initialize selector
        self.selector = AggressiveStrategySelector()
        
        # Load data
        print("Loading data...")
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
    
    def run_aggressive_meta_system(self, test_period_days: int = 10) -> dict:
        """Run the aggressive meta-trading system with regime detection"""
        print(f"\nRunning MetaTradingAI v2.2: The Ultra-Adaptive Aggression Model...")
        
        # Calculate date ranges
        end_date = self.data.index.max()
        test_start_date = end_date - timedelta(days=test_period_days)
        training_start_date = test_start_date - timedelta(days=60)
        
        print(f"Aggressive Meta-Trading AI System Setup:")
        print(f"  Training period: {training_start_date.date()} to {test_start_date.date()}")
        print(f"  Test period: {test_start_date.date()} to {end_date.date()}")
        print(f"  Target: 5% return over {test_period_days} trading days")
        
        # Split data
        training_data = self.data[(self.data.index >= training_start_date) & (self.data.index < test_start_date)]
        test_data = self.data[self.data.index >= test_start_date]
        
        print(f"  Training data: {len(training_data)} records")
        print(f"  Test data: {len(test_data)} records")
        
        # Test all strategies on training data to get performance baseline
        print(f"\nTesting all strategies on training data...")
        strategy_performances = {}
        
        all_strategies = self.high_vol_pool + self.trending_pool + self.ranging_pool
        
        for i, strategy in enumerate(all_strategies):
            print(f"  [{i+1}/{len(all_strategies)}] Testing {strategy.name}...")
            results = strategy.backtest(training_data)
            strategy_performances[strategy.name] = [results['avg_hourly_return']]
        
        # Train strategy selector
        self.selector.strategies = all_strategies
        self.selector.train_selector(training_data, strategy_performances)
        
        # Run aggressive meta system with detailed reporting
        print(f"\nRunning aggressive meta-trading system...")
        results = {
            'hourly_performance': [],
            'daily_performance': [],
            'selected_strategies': [],
            'strategy_performance': {s.name: [] for s in all_strategies},
            'cumulative_return': 0,
            'daily_returns': []
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
                
                # Run strategy for this hour
                strategy_results = selected_strategy.backtest(hour_data)
                
                # Record performance
                hourly_perf = {
                    'date': date,
                    'hour': hour,
                    'selected_strategy': selected_strategy.name,
                    'regime': regime,
                    'avg_hourly_return': strategy_results['avg_hourly_return'],
                    'total_return': strategy_results['total_return'],
                    'num_trades': strategy_results['num_trades']
                }
                
                results['hourly_performance'].append(hourly_perf)
                results['selected_strategies'].append(selected_strategy.name)
                results['strategy_performance'][selected_strategy.name].append(strategy_results['avg_hourly_return'])
                
                daily_return += strategy_results['total_return']
                daily_trades += strategy_results['num_trades']
                
                print(f"  Hour {hour:02d}:00 - Selected: {selected_strategy.name}")
                print(f"    Return: {strategy_results['total_return']:.4f} ({strategy_results['total_return']*100:.2f}%), Trades: {strategy_results['num_trades']}")
            
            # Update cumulative return
            results['cumulative_return'] += daily_return
            results['daily_returns'].append(daily_return)
            
            # Daily summary
            print(f"  Daily Summary: Return: {daily_return:.4f} ({daily_return*100:.2f}%), Trades: {daily_trades}")
            print(f"  Cumulative Return: {results['cumulative_return']:.4f} ({results['cumulative_return']*100:.2f}%)")
        
        # Final results
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        print(f"  Target: 5% return over {test_period_days} trading days")
        print(f"  Actual: {results['cumulative_return']:.4f} ({results['cumulative_return']*100:.2f}%)")
        
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
        
        return results

def main():
    """Main execution function"""
    start_time = datetime.now()
    
    # Initialize and run the system
    system = AggressiveMetaTradingAI()
    results = system.run_aggressive_meta_system(test_period_days=10)
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    print(f"\n⏱️ EXECUTION TIME: {execution_time:.1f} seconds")
    
    return results

if __name__ == "__main__":
    results = main() 