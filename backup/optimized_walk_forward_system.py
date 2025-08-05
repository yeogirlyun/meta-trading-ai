#!/usr/bin/env python3
"""
Optimized Walk-Forward Testing System - MetaTradingAI v4.0
Implements speed optimizations for fast multi-period testing
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Try to import TA-Lib for faster technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using pandas for technical indicators.")

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

class OptimizedDataLoader:
    """Optimized data loading with Feather format support"""
    
    def __init__(self):
        self.data = None
        self.feather_file = 'polygon_QQQ_1m.feather'
        self.pickle_file = 'polygon_QQQ_1m.pkl'
    
    def convert_to_feather(self):
        """Convert pickle data to Feather format for faster loading"""
        print("Converting data to Feather format for faster loading...")
        
        # Load pickle data
        with open(self.pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # Convert index to datetime if needed
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except ValueError:
                data.index = pd.to_datetime(data.index, utc=True)
        
        # Handle timezone if present
        try:
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
        except:
            pass
        
        # Filter to trading hours and weekdays
        data = data.between_time('09:30', '16:00')
        data = data[data.index.dayofweek < 5]
        
        # Filter to last 5 years
        five_years_ago = data.index.max() - timedelta(days=5*365)
        data = data[data.index >= five_years_ago]
        
        # Save as Feather
        data.to_feather(self.feather_file)
        print(f"Data converted and saved to {self.feather_file}")
        print(f"Records: {len(data):,}")
        
        return data
    
    def load_data(self):
        """Load data from Feather format (fast) or pickle (fallback)"""
        try:
            # Try to load from Feather first
            self.data = pd.read_feather(self.feather_file)
            print(f"Data loaded from Feather: {len(self.data):,} records")
        except FileNotFoundError:
            # Fallback to pickle and convert
            print("Feather file not found. Converting from pickle...")
            self.data = self.convert_to_feather()
        
        return self.data

class OptimizedFeatureEngineer:
    """Optimized feature engineering using TA-Lib when available"""
    
    def __init__(self):
        self.use_talib = TALIB_AVAILABLE
    
    def calculate_sma(self, data, period):
        """Calculate SMA using TA-Lib or pandas"""
        if self.use_talib:
            return talib.SMA(data, timeperiod=period)
        else:
            return data.rolling(period).mean()
    
    def calculate_rsi(self, data, period):
        """Calculate RSI using TA-Lib or pandas"""
        if self.use_talib:
            return talib.RSI(data, timeperiod=period)
        else:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, data, period, std_dev=2):
        """Calculate Bollinger Bands using TA-Lib or pandas"""
        if self.use_talib:
            upper, middle, lower = talib.BBANDS(data, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return upper, middle, lower
        else:
            middle = data.rolling(period).mean()
            std = data.rolling(period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper, middle, lower
    
    def calculate_atr(self, high, low, close, period):
        """Calculate ATR using TA-Lib or pandas"""
        if self.use_talib:
            return talib.ATR(high, low, close, timeperiod=period)
        else:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(period).mean()
    
    def extract_features(self, data):
        """Extract features using optimized methods"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features (vectorized)
        features['price_change'] = data['close'].pct_change()
        features['price_change_3m'] = data['close'].pct_change(3)
        features['price_change_5m'] = data['close'].pct_change(5)
        features['price_change_15m'] = data['close'].pct_change(15)
        
        # Volatility features (vectorized)
        features['volatility_3m'] = data['close'].pct_change().rolling(3).std()
        features['volatility_5m'] = data['close'].pct_change().rolling(5).std()
        features['volatility_15m'] = data['close'].pct_change().rolling(15).std()
        features['vol_of_vol'] = features['volatility_5m'].rolling(10).std()
        
        # Volume features (vectorized)
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
        features['volume_imbalance'] = (data['volume'] - data['volume'].rolling(20).mean()) / (data['volume'].rolling(20).mean() + 1e-6)
        
        # Momentum features (vectorized)
        features['momentum_3m'] = data['close'].pct_change(3)
        features['momentum_5m'] = data['close'].pct_change(5)
        features['momentum_15m'] = data['close'].pct_change(15)
        features['momentum_acceleration'] = features['momentum_5m'].diff()
        
        # Technical indicators (optimized)
        features['sma_3'] = self.calculate_sma(data['close'], 3)
        features['sma_10'] = self.calculate_sma(data['close'], 10)
        features['rsi_14'] = self.calculate_rsi(data['close'], 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['close'], 20)
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR for volatility
        features['atr_14'] = self.calculate_atr(data['high'], data['low'], data['close'], 14)
        
        # Trend features (vectorized)
        features['trend_3_10'] = (features['sma_3'] - features['sma_10']) / features['sma_10']
        
        # Signal strength features (vectorized)
        features['signal_strength'] = features['momentum_3m'] * features['volume_ratio'] / (features['volatility_5m'] + 1e-6)
        
        # Time features (vectorized)
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        
        return features.dropna()

class OptimizedStrategy:
    """Optimized strategy with vectorized backtesting"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.feature_engineer = OptimizedFeatureEngineer()
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signals using vectorized operations"""
        # Simple momentum-based signal (vectorized)
        momentum = data['close'].pct_change(5)
        volume_ratio = data['volume'] / data['volume'].rolling(10).mean()
        volatility = data['close'].pct_change().rolling(5).std()
        
        # Vectorized signal generation
        signals = pd.Series(0, index=data.index)
        
        # Long signals (vectorized)
        long_condition = (momentum > 0.0008) & (volume_ratio > 1.3) & (volatility < 0.01)
        signals[long_condition] = 1
        
        # Short signals (vectorized)
        short_condition = (momentum < -0.0008) & (volume_ratio > 1.3) & (volatility < 0.01)
        signals[short_condition] = -1
        
        return signals
    
    def vectorized_backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> dict:
        """Vectorized backtesting for maximum speed"""
        signals = self.calculate_signals(data)
        
        # Create positions series (vectorized)
        positions = signals.replace(0, np.nan).ffill().fillna(0).shift(1)
        
        # Calculate returns (vectorized)
        returns = data['close'].pct_change() * positions
        
        # Calculate equity curve (vectorized)
        equity_curve = initial_capital * (1 + returns).cumprod()
        
        # Calculate metrics (vectorized)
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        
        # Calculate hourly returns
        hourly_returns = equity_curve.resample('H').last().pct_change().dropna()
        avg_hourly_return = hourly_returns.mean() if not hourly_returns.empty else 0
        
        # Count trades (vectorized)
        trade_signals = signals[signals != 0]
        num_trades = len(trade_signals)
        
        # Calculate trade frequency
        total_hours = len(data) // 60
        trade_frequency = num_trades / total_hours if total_hours > 0 else 0
        
        # Calculate signal strength (vectorized)
        features = self.feature_engineer.extract_features(data)
        avg_signal_strength = features['signal_strength'].mean() if 'signal_strength' in features else 0
        
        return {
            'total_return': total_return,
            'avg_hourly_return': avg_hourly_return,
            'num_trades': num_trades,
            'trade_frequency': trade_frequency,
            'avg_signal_strength': avg_signal_strength,
            'avg_leverage': 1.0,  # Default for now
            'avg_position_size': 0.15,  # Default for now
            'win_rate': 0.6,  # Default for now
            'avg_win_loss_ratio': 1.5  # Default for now
        }

class OptimizedStrategySelector:
    """Optimized strategy selector with parallel processing"""
    
    def __init__(self):
        # Use all available CPU cores for parallel processing
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.strategies = []
        self.is_trained = False
        self.feature_engineer = OptimizedFeatureEngineer()
    
    def extract_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract market features using optimized methods"""
        return self.feature_engineer.extract_features(data)
    
    def train_selector(self, training_data: pd.DataFrame, strategy_performances: dict) -> None:
        """Train the strategy selector model"""
        print("Training optimized strategy selector...")
        
        features = self.extract_market_features(training_data)
        
        # Prepare labels for training
        labels = []
        for hour in range(24):
            hour_performances = {}
            for strategy_name, performances in strategy_performances.items():
                if len(performances) > hour:
                    hour_performances[strategy_name] = performances[hour]
            
            if hour_performances:
                best_strategy = max(hour_performances.keys(), key=lambda x: hour_performances[x])
                labels.append(best_strategy)
            else:
                labels.append('default')
        
        # Train model
        if len(features) > 0 and len(labels) > 0:
            X = self.scaler.fit_transform(features.iloc[:len(labels)])
            self.model.fit(X, labels)
            self.is_trained = True
            print("Strategy selector trained successfully")

class OptimizedWalkForwardSystem:
    """Optimized walk-forward testing system"""
    
    def __init__(self, training_days: int = 180):
        self.training_days = training_days
        self.data_loader = OptimizedDataLoader()
        self.strategy_selector = OptimizedStrategySelector()
        
        # Load data
        print("Loading optimized data...")
        self.data = self.data_loader.load_data()
        print(f"Data loaded: {len(self.data)} records")
    
    def get_test_periods(self, start_date: str = "2020-05-01", test_days: int = 10) -> list:
        """Generate test periods efficiently"""
        start_dt = pd.to_datetime(start_date)
        end_dt = self.data.index.max()
        
        periods = []
        current_start = start_dt
        
        while current_start + timedelta(days=test_days) <= end_dt:
            test_start = current_start
            test_end = test_start + timedelta(days=test_days)
            training_start = test_start - timedelta(days=self.training_days)
            
            periods.append({
                'training_start': training_start,
                'training_end': test_start,
                'test_start': test_start,
                'test_end': test_end
            })
            
            # Advance by 1 week
            current_start += timedelta(days=7)
        
        return periods
    
    def test_model_period(self, period: dict, model_name: str) -> dict:
        """Test a single model period efficiently"""
        # Extract data for this period
        training_data = self.data[
            (self.data.index >= period['training_start']) & 
            (self.data.index < period['training_end'])
        ]
        test_data = self.data[
            (self.data.index >= period['test_start']) & 
            (self.data.index < period['test_end'])
        ]
        
        # Create strategy and run backtest
        strategy = OptimizedStrategy(model_name)
        results = strategy.vectorized_backtest(test_data)
        
        return {
            'total_return': results['total_return'],
            'num_trades': results['num_trades'],
            'trade_frequency': results['trade_frequency'],
            'avg_signal_strength': results['avg_signal_strength'],
            'avg_leverage': results['avg_leverage']
        }
    
    def run_parallel_testing(self, periods: list, models: list) -> dict:
        """Run testing in parallel for maximum speed"""
        print(f"Running parallel testing for {len(periods)} periods and {len(models)} models...")
        
        results = {}
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            
            # Submit all tasks
            for period in periods:
                for model_name in models:
                    future = executor.submit(self.test_model_period, period, model_name)
                    futures.append((future, period, model_name))
            
            # Collect results
            for i, (future, period, model_name) in enumerate(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    
                    if model_name not in results:
                        results[model_name] = []
                    
                    results[model_name].append({
                        'period': period,
                        'results': result
                    })
                    
                    # Progress update
                    if (i + 1) % 10 == 0:
                        print(f"Completed {i + 1}/{len(futures)} tests...")
                        
                except Exception as e:
                    print(f"Error testing {model_name} for period {period}: {e}")
        
        return results
    
    def run_optimized_walk_forward(self, start_date: str = "2020-05-01", 
                                  test_days: int = 10) -> dict:
        """Run optimized walk-forward testing"""
        print(f"\n🚀 Running Optimized Walk-Forward Testing System...")
        print(f"Training Period: {self.training_days} days")
        print(f"Test Period: {test_days} days")
        print(f"Start Date: {start_date}")
        
        # Get test periods
        periods = self.get_test_periods(start_date, test_days)
        print(f"Generated {len(periods)} test periods")
        
        # Define models to test
        models = [
            "v3.0 Ultra-Aggressive",
            "v3.0 Optimized RT", 
            "v3.0 Enhanced"
        ]
        
        # Run parallel testing
        results = self.run_parallel_testing(periods, models)
        
        # Calculate summary statistics
        summary = self.calculate_summary_statistics(results)
        
        return summary
    
    def calculate_summary_statistics(self, results: dict) -> dict:
        """Calculate summary statistics for all models"""
        summary = {}
        
        for model_name, model_results in results.items():
            returns = [r['results']['total_return'] for r in model_results]
            
            if returns:
                summary[model_name] = {
                    'avg_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'min_return': np.min(returns),
                    'max_return': np.max(returns),
                    'median_return': np.median(returns),
                    'target_achievement_rate': sum(1 for r in returns if r >= 0.05) / len(returns),
                    'num_periods': len(returns),
                    'avg_trades': np.mean([r['results']['num_trades'] for r in model_results]),
                    'avg_frequency': np.mean([r['results']['trade_frequency'] for r in model_results]),
                    'avg_signal_strength': np.mean([r['results']['avg_signal_strength'] for r in model_results]),
                    'avg_leverage': np.mean([r['results']['avg_leverage'] for r in model_results])
                }
        
        return summary

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Walk-Forward Testing System')
    parser.add_argument('--start-date', type=str, default='2020-05-01',
                       help='Start date for testing (YYYY-MM-DD)')
    parser.add_argument('--test-days', type=int, default=10,
                       help='Number of test days per period')
    parser.add_argument('--training-days', type=int, default=180,
                       help='Number of training days')
    
    args = parser.parse_args()
    
    # Run optimized system
    system = OptimizedWalkForwardSystem(training_days=args.training_days)
    results = system.run_optimized_walk_forward(
        start_date=args.start_date,
        test_days=args.test_days
    )
    
    # Print results
    print(f"\n📊 OPTIMIZED WALK-FORWARD TESTING RESULTS:")
    print(f"Testing Periods: {len(results)} models tested")
    
    for model_name, stats in results.items():
        print(f"\n{model_name}:")
        print(f"  Average Return: {stats['avg_return']:.4f} ({stats['avg_return']*100:.2f}%)")
        print(f"  Standard Deviation: {stats['std_return']:.4f} ({stats['std_return']*100:.2f}%)")
        print(f"  Min Return: {stats['min_return']:.4f} ({stats['min_return']*100:.2f}%)")
        print(f"  Max Return: {stats['max_return']:.4f} ({stats['max_return']*100:.2f}%)")
        print(f"  Target Achievement Rate: {stats['target_achievement_rate']:.2%}")
        print(f"  Number of Periods: {stats['num_periods']}")
        print(f"  Average Trades: {stats['avg_trades']:.1f}")
        print(f"  Average Frequency: {stats['avg_frequency']:.2f} trades/hour")
        print(f"  Average Signal Strength: {stats['avg_signal_strength']:.2f}")
        print(f"  Average Leverage: {stats['avg_leverage']:.1f}x")
    
    return results

if __name__ == "__main__":
    results = main() 