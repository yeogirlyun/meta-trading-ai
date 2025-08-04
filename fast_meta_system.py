import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


class FastMetaStrategy:
    """Fast meta strategy with pre-optimized parameters"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        
    def set_parameters(self, params: Dict):
        """Set strategy parameters"""
        self.parameters.update(params)
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trading signals (1 for buy, -1 for sell, 0 for hold)"""
        raise NotImplementedError
        
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Run backtest and return performance metrics"""
        signals = self.calculate_signals(data)
        
        # Initialize tracking variables
        position = 0  # 1 for long, -1 for short, 0 for flat
        capital = initial_capital
        equity = [initial_capital]
        trades = []
        
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
                daily_return = (current_price - prev_price) / prev_price
            elif position == -1:  # Short position
                daily_return = (prev_price - current_price) / prev_price
            else:
                daily_return = 0
                
            capital *= (1 + daily_return)
            equity.append(capital)
            
        # Calculate performance metrics
        equity_series = pd.Series(equity, index=data.index)
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity[-1] - initial_capital) / initial_capital
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        # Calculate hourly returns for target analysis
        hourly_data = data.resample('H').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        hourly_equity = equity_series.resample('H').last()
        hourly_returns = hourly_equity.pct_change().dropna()
        
        # Target metrics
        avg_hourly_return = hourly_returns.mean() if len(hourly_returns) > 0 else 0
        target_hourly_01 = len(hourly_returns[hourly_returns >= 0.001]) / len(hourly_returns) if len(hourly_returns) > 0 else 0
        
        # 4-hour rolling returns
        rolling_4h = hourly_equity.pct_change(4).dropna()
        target_4h_01 = len(rolling_4h[rolling_4h >= 0.01]) / len(rolling_4h) if len(rolling_4h) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': equity_series,
            'trades': trades,
            'avg_hourly_return': avg_hourly_return,
            'target_hourly_01_pct': target_hourly_01,
            'target_4h_01_pct': target_4h_01,
            'num_trades': len(trades)
        }


class FastVolatilityExploitation(FastMetaStrategy):
    """Fast volatility exploitation strategy with pre-optimized parameters"""
    
    def __init__(self):
        super().__init__("Fast Volatility Exploitation")
        self.parameters = {
            'volatility_threshold': 0.008,
            'momentum_period': 5,
            'reversal_threshold': 0.008,
            'max_hold_period': 30,
            'volume_threshold': 1.3
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(15).std()
        
        # Calculate momentum
        momentum = data['close'].pct_change(self.parameters['momentum_period'])
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(15).mean()
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Entry conditions
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
        
        # Exit after max hold period
        for i in range(self.parameters['max_hold_period'], len(signals)):
            if signals.iloc[i] != 0:
                if i - self.parameters['max_hold_period'] >= 0:
                    last_signal = signals.iloc[i - self.parameters['max_hold_period']]
                    if last_signal != 0:
                        signals.iloc[i] = 0  # Exit position
        
        return signals


class FastMomentumAmplification(FastMetaStrategy):
    """Fast momentum amplification strategy with pre-optimized parameters"""
    
    def __init__(self):
        super().__init__("Fast Momentum Amplification")
        self.parameters = {
            'short_period': 3,
            'medium_period': 8,
            'long_period': 20,
            'momentum_threshold': 0.006,
            'volume_threshold': 1.3,
            'acceleration_threshold': 0.003,
            'max_hold_period': 45
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate multiple momentum indicators
        short_momentum = data['close'].pct_change(self.parameters['short_period'])
        medium_momentum = data['close'].pct_change(self.parameters['medium_period'])
        long_momentum = data['close'].pct_change(self.parameters['long_period'])
        
        # Calculate acceleration
        acceleration = short_momentum.diff()
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(15).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(10).std()
        
        signals = pd.Series(0, index=data.index)
        
        # Strong buy when all momentum indicators align
        strong_buy = (
            (short_momentum > self.parameters['momentum_threshold']) &
            (medium_momentum > self.parameters['momentum_threshold']) &
            (long_momentum > 0) &
            (acceleration > self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(25).mean() * 2.5)
        )
        
        # Strong sell when all momentum indicators align negatively
        strong_sell = (
            (short_momentum < -self.parameters['momentum_threshold']) &
            (medium_momentum < -self.parameters['momentum_threshold']) &
            (long_momentum < 0) &
            (acceleration < -self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(25).mean() * 2.5)
        )
        
        signals[strong_buy] = 1
        signals[strong_sell] = -1
        
        # Exit after max hold period
        for i in range(self.parameters['max_hold_period'], len(signals)):
            if signals.iloc[i] != 0:
                if i - self.parameters['max_hold_period'] >= 0:
                    last_signal = signals.iloc[i - self.parameters['max_hold_period']]
                    if last_signal != 0:
                        signals.iloc[i] = 0  # Exit position
        
        return signals


class FastGapExploitation(FastMetaStrategy):
    """Fast gap exploitation strategy with pre-optimized parameters"""
    
    def __init__(self):
        super().__init__("Fast Gap Exploitation")
        self.parameters = {
            'gap_threshold': 0.004,
            'fade_threshold': 0.006,
            'momentum_threshold': 0.002,
            'volume_threshold': 1.3,
            'max_hold_period': 45
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Resample to daily data for gap detection
        daily_data = data.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        # Calculate gaps
        gaps = (daily_data['open'] - daily_data['close'].shift(1)) / daily_data['close'].shift(1)
        
        # Resample back to minute data
        gap_signals = gaps.reindex(data.index, method='ffill')
        
        # Calculate intraday momentum
        momentum = data['close'].pct_change(5)
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(15).mean()
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Gap fade strategy
        large_gap_up = gap_signals > self.parameters['gap_threshold']
        large_gap_down = gap_signals < -self.parameters['gap_threshold']
        
        # Fade gap up with negative momentum
        fade_up = (
            large_gap_up &
            (momentum < -self.parameters['momentum_threshold']) &
            (volume_ratio > self.parameters['volume_threshold'])
        )
        
        # Fade gap down with positive momentum
        fade_down = (
            large_gap_down &
            (momentum > self.parameters['momentum_threshold']) &
            (volume_ratio > self.parameters['volume_threshold'])
        )
        
        signals[fade_up] = -1  # Short gap up
        signals[fade_down] = 1  # Long gap down
        
        # Exit after max hold period
        for i in range(self.parameters['max_hold_period'], len(signals)):
            if signals.iloc[i] != 0:
                if i - self.parameters['max_hold_period'] >= 0:
                    last_signal = signals.iloc[i - self.parameters['max_hold_period']]
                    if last_signal != 0:
                        signals.iloc[i] = 0  # Exit position
        
        return signals


class FastMeanReversion(FastMetaStrategy):
    """Fast mean reversion strategy with pre-optimized parameters"""
    
    def __init__(self):
        super().__init__("Fast Mean Reversion")
        self.parameters = {
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.2,
            'max_hold_period': 60
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate Bollinger Bands
        bb_middle = data['close'].rolling(self.parameters['bollinger_period']).mean()
        bb_std = data['close'].rolling(self.parameters['bollinger_period']).std()
        bb_upper = bb_middle + (bb_std * self.parameters['bollinger_std'])
        bb_lower = bb_middle - (bb_std * self.parameters['bollinger_std'])
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.parameters['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(15).mean()
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Buy signals: price below lower band and RSI oversold
        buy_condition = (
            (data['close'] < bb_lower) &
            (rsi < self.parameters['rsi_oversold']) &
            (volume_ratio > self.parameters['volume_threshold'])
        )
        
        # Sell signals: price above upper band and RSI overbought
        sell_condition = (
            (data['close'] > bb_upper) &
            (rsi > self.parameters['rsi_overbought']) &
            (volume_ratio > self.parameters['volume_threshold'])
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        # Exit after max hold period
        for i in range(self.parameters['max_hold_period'], len(signals)):
            if signals.iloc[i] != 0:
                if i - self.parameters['max_hold_period'] >= 0:
                    last_signal = signals.iloc[i - self.parameters['max_hold_period']]
                    if last_signal != 0:
                        signals.iloc[i] = 0  # Exit position
        
        return signals


class FastStrategySelector:
    """Fast meta-learning system to select the best strategy for each hour"""
    
    def __init__(self, strategies: List[FastMetaStrategy]):
        self.strategies = strategies
        self.selector_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def extract_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract market features for strategy selection"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['price_change'] = data['close'].pct_change()
        features['price_change_5m'] = data['close'].pct_change(5)
        features['price_change_15m'] = data['close'].pct_change(15)
        features['price_change_60m'] = data['close'].pct_change(60)
        
        # Volatility features
        features['volatility_5m'] = data['close'].pct_change().rolling(5).std()
        features['volatility_15m'] = data['close'].pct_change().rolling(15).std()
        features['volatility_60m'] = data['close'].pct_change().rolling(60).std()
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_change'] = data['volume'].pct_change()
        
        # Momentum features
        features['momentum_5m'] = data['close'].pct_change(5)
        features['momentum_15m'] = data['close'].pct_change(15)
        features['momentum_60m'] = data['close'].pct_change(60)
        
        # Trend features
        features['sma_5'] = data['close'].rolling(5).mean()
        features['sma_20'] = data['close'].rolling(20).mean()
        features['trend_5_20'] = (features['sma_5'] - features['sma_20']) / features['sma_20']
        
        # Gap features
        daily_data = data.resample('D').agg({
            'open': 'first', 'close': 'last'
        }).dropna()
        gaps = (daily_data['open'] - daily_data['close'].shift(1)) / daily_data['close'].shift(1)
        gap_signals = gaps.reindex(data.index, method='ffill')
        features['gap_size'] = gap_signals
        
        # Time features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def train_selector(self, training_data: pd.DataFrame, 
                      strategy_performances: Dict[str, List[float]]) -> None:
        """Train the strategy selector model"""
        print("Training fast strategy selector...")
        
        # Extract market features
        features = self.extract_market_features(training_data)
        
        # Prepare labels (strategy with best performance for each hour)
        labels = []
        feature_samples = []
        
        # Group by hour and find best performing strategy
        for hour in range(24):
            hour_data = features[features['hour'] == hour]
            if len(hour_data) == 0:
                continue
                
            # Get performance for this hour for each strategy
            hour_performances = {}
            for strategy_name, performances in strategy_performances.items():
                if len(performances) > hour:
                    hour_performances[strategy_name] = performances[hour]
            
            if hour_performances:
                # Find best performing strategy
                best_strategy = max(hour_performances.keys(), 
                                  key=lambda x: hour_performances[x])
                
                # Create labels for this hour
                for strategy_name in self.strategies:
                    if strategy_name.name == best_strategy:
                        labels.append(1)
                    else:
                        labels.append(0)
                    
                    # Use average features for this hour
                    hour_features = hour_data.mean()
                    feature_samples.append(hour_features.values)
        
        if len(feature_samples) > 0:
            # Convert to numpy arrays
            X = np.array(feature_samples)
            y = np.array(labels)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.selector_model.fit(X_scaled, y)
            self.feature_columns = features.columns.tolist()
            self.is_trained = True
            
            print(f"Fast strategy selector trained with {len(X)} samples")
        else:
            print("Warning: No training data available for strategy selector")
    
    def select_best_strategy(self, current_data: pd.DataFrame) -> FastMetaStrategy:
        """Select the best strategy for the current market conditions"""
        if not self.is_trained:
            # If not trained, return random strategy
            return np.random.choice(self.strategies)
        
        # Extract features for current data
        features = self.extract_market_features(current_data)
        if len(features) == 0:
            return np.random.choice(self.strategies)
        
        # Use most recent features
        current_features = features.iloc[-1:].values
        current_features_scaled = self.scaler.transform(current_features)
        
        # Predict probabilities for each strategy
        probabilities = self.selector_model.predict_proba(current_features_scaled)
        
        # Find strategy with highest probability
        best_strategy_idx = np.argmax(probabilities[0])
        best_strategy = self.strategies[best_strategy_idx]
        
        print(f"Selected strategy: {best_strategy.name} (confidence: {probabilities[0][best_strategy_idx]:.3f})")
        
        return best_strategy


class FastMetaAdaptiveSystem:
    """Fast meta-adaptive system with strategy selection"""
    
    def __init__(self, data: pd.DataFrame, strategies: List[FastMetaStrategy]):
        self.data = data
        self.strategies = strategies
        self.selector = FastStrategySelector(strategies)
        self.start_time = None
        
    def run_hourly_adaptive_system(self, test_period_days: int = 7) -> Dict:
        """Run the fast meta-adaptive system with hourly strategy selection"""
        
        # Get the most recent data for testing
        test_end_date = self.data.index.max()
        test_start_date = test_end_date - timedelta(days=test_period_days)
        training_end_date = test_start_date - timedelta(days=1)
        training_start_date = training_end_date - timedelta(days=30)  # 30 days training
        
        print(f"Fast meta-adaptive system setup:")
        print(f"  Training period: {training_start_date} to {training_end_date}")
        print(f"  Test period: {test_start_date} to {test_end_date}")
        
        # Get training and test data
        training_data = self.data[(self.data.index >= training_start_date) & 
                                (self.data.index <= training_end_date)]
        test_data = self.data[(self.data.index >= test_start_date) & 
                             (self.data.index <= test_end_date)]
        
        print(f"  Training data: {len(training_data)} records")
        print(f"  Test data: {len(test_data)} records")
        
        # Start timing
        self.start_time = time.time()
        
        # Test all strategies on training data to get performance baseline
        print(f"\nTesting all strategies on training data...")
        strategy_performances = {}
        
        for i, strategy in enumerate(self.strategies):
            print(f"  [{i+1}/{len(self.strategies)}] Testing {strategy.name}...")
            results = strategy.backtest(training_data)
            strategy_performances[strategy.name] = [results['avg_hourly_return']]
            
            print(f"    Avg hourly return: {results['avg_hourly_return']:.4f}")
            print(f"    Total return: {results['total_return']:.4f}")
            print(f"    Sharpe ratio: {results['sharpe_ratio']:.3f}")
        
        # Train strategy selector
        self.selector.train_selector(training_data, strategy_performances)
        
        # Run hourly adaptive system
        print(f"\nRunning hourly adaptive system...")
        results = {
            'hourly_performance': [],
            'selected_strategies': [],
            'strategy_performance': {s.name: [] for s in self.strategies}
        }
        
        # Group test data by hour
        test_data_hourly = test_data.groupby(test_data.index.hour)
        
        for hour, hour_data in test_data_hourly:
            if len(hour_data) == 0:
                continue
                
            print(f"  Hour {hour:02d}:00 - Selecting best strategy...")
            
            # Select best strategy for this hour
            selected_strategy = self.selector.select_best_strategy(hour_data)
            
            # Run selected strategy for this hour
            strategy_results = selected_strategy.backtest(hour_data)
            
            # Store results
            results['hourly_performance'].append({
                'hour': hour,
                'selected_strategy': selected_strategy.name,
                'avg_hourly_return': strategy_results['avg_hourly_return'],
                'total_return': strategy_results['total_return'],
                'num_trades': strategy_results['num_trades']
            })
            
            results['selected_strategies'].append(selected_strategy.name)
            
            # Update strategy performance tracking
            for strategy in self.strategies:
                if strategy.name == selected_strategy.name:
                    results['strategy_performance'][strategy.name].append(
                        strategy_results['avg_hourly_return']
                    )
                else:
                    results['strategy_performance'][strategy.name].append(0)
            
            print(f"    Selected: {selected_strategy.name}")
            print(f"    Hourly return: {strategy_results['avg_hourly_return']:.4f}")
            print(f"    Trades: {strategy_results['num_trades']}")
        
        # Final timing
        total_time = time.time() - self.start_time
        print(f"\nFast meta-adaptive system completed in {total_time:.1f} seconds")
        
        return results


def analyze_fast_meta_results(results: Dict) -> None:
    """Analyze and display fast meta-adaptive system results"""
    
    print("\n" + "="*80)
    print("FAST META-ADAPTIVE SYSTEM RESULTS")
    print("="*80)
    
    # Calculate overall performance
    total_return = 0
    total_trades = 0
    strategy_counts = {}
    
    for perf in results['hourly_performance']:
        total_return += perf['total_return']
        total_trades += perf['num_trades']
        strategy = perf['selected_strategy']
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f"\nOverall Performance:")
    print(f"  Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"  Total Trades: {total_trades}")
    if len(results['hourly_performance']) > 0:
        avg_return_per_hour = total_return / len(results['hourly_performance'])
        print(f"  Average Return per Hour: {avg_return_per_hour:.4f}")
    
    print(f"\nStrategy Selection Distribution:")
    for strategy, count in strategy_counts.items():
        percentage = (count / len(results['hourly_performance'])) * 100
        print(f"  {strategy}: {count} hours ({percentage:.1f}%)")
    
    print(f"\nStrategy Performance Summary:")
    for strategy_name, performances in results['strategy_performance'].items():
        active_performances = [p for p in performances if p > 0]
        if active_performances:
            avg_performance = np.mean(active_performances)
            print(f"  {strategy_name}: {avg_performance:.4f} avg return when selected")
    
    # Show hourly breakdown
    print(f"\nHourly Performance Breakdown:")
    for perf in results['hourly_performance'][:10]:  # Show first 10 hours
        print(f"  Hour {perf['hour']:02d}:00 - {perf['selected_strategy']} - "
              f"Return: {perf['avg_hourly_return']:.4f} ({perf['avg_hourly_return']*100:.2f}%)")


def main():
    """Main execution function"""
    
    # Load data
    print("Loading data...")
    data = pickle.load(open('polygon_QQQ_1m.pkl', 'rb'))
    
    # Convert index to datetime if needed
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except ValueError:
            data.index = pd.to_datetime(data.index, utc=True)
    
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    # Filter to trading hours and weekdays
    data = data.between_time('09:30', '16:00')
    data = data[data.index.dayofweek < 5]
    
    # Filter to last 5 years of data
    five_years_ago = data.index.max() - timedelta(days=5*365)
    data = data[data.index >= five_years_ago]
    
    print(f"Data shape after filtering: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Using last 5 years of data for faster processing")
    
    # Define fast meta-adaptive strategies
    strategies = [
        FastVolatilityExploitation(),
        FastMomentumAmplification(),
        FastGapExploitation(),
        FastMeanReversion()
    ]
    
    # Create fast meta-adaptive optimization system
    fast_system = FastMetaAdaptiveSystem(data, strategies)
    
    # Run fast meta-adaptive system
    print("\nRunning fast meta-adaptive system...")
    results = fast_system.run_hourly_adaptive_system(test_period_days=7)
    
    # Analyze results
    analyze_fast_meta_results(results)
    
    # Save results
    with open('fast_meta_adaptive_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to 'fast_meta_adaptive_results.pkl'")
    
    return results


if __name__ == "__main__":
    results = main() 