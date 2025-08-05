#!/usr/bin/env python3
"""
OPTIMIZED_RT_WALK_TEST_FIXED - Corrected walk-through test
Properly trains the model before each testing period
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


class OptimizedRTWalkTestFixed:
    """Corrected walk-through test for v3.0 Optimized RT model"""
    
    def __init__(self):
        self.data = None
        self.results = {}
        
    def load_data(self):
        """Load QQQ data from Feather format"""
        try:
            self.data = pd.read_feather('polygon_QQQ_1m.feather')
            print(f"Data loaded: {len(self.data):,} records")
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            return True
        except FileNotFoundError:
            print("Feather file not found. Converting from pickle...")
            self.convert_to_feather()
            self.data = pd.read_feather('polygon_QQQ_1m.feather')
            print(f"Data loaded: {len(self.data):,} records")
            return True
    
    def convert_to_feather(self):
        """Convert pickle data to Feather format"""
        import pickle
        print("Converting pickle data to Feather format...")
        with open('polygon_QQQ_1m.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Clean and filter data
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Filter to trading hours and weekdays
        data = data.between_time('09:30', '16:00')
        data = data[data.index.dayofweek < 5]
        
        # Filter to last 5 years
        five_years_ago = data.index.max() - timedelta(days=5*365)
        data = data[data.index >= five_years_ago]
        
        data.to_feather('polygon_QQQ_1m.feather')
        print("Data converted and saved to polygon_QQQ_1m.feather")
    
    def generate_test_periods(self):
        """Generate 10 test periods of 2 weeks each"""
        end_date = self.data.index.max()
        
        periods = []
        for i in range(10):
            # Each period is 2 weeks
            period_end = end_date - timedelta(weeks=i*2)
            period_start = period_end - timedelta(weeks=2)
            training_end = period_start - timedelta(days=1)
            training_start = training_end - timedelta(weeks=12)  # 3 months training
            
            periods.append({
                'training_start': training_start,
                'training_end': training_end,
                'test_start': period_start,
                'test_end': period_end,
                'period_number': i + 1
            })
        
        return periods
    
    def train_model_on_data(self, training_data):
        """Train the model on training data and return optimized parameters"""
        # Calculate market characteristics from training data
        returns = training_data['close'].pct_change()
        volatility = returns.std()
        
        # Adjust parameters based on training data characteristics
        base_params = {
            'momentum_threshold': 0.0010,  # Higher for quality
            'volume_threshold': 1.5,       # Higher for quality
            'volatility_threshold': 0.012, # Higher for quality
            'max_hold_period': 5,          # Longer holds
            'position_size': 0.30,         # Larger positions
            'stop_loss': 0.004,           # Wider stops
            'profit_target': 0.008,        # Higher targets
            'leverage': 1.0,              # Conservative leverage
            'min_trade_interval': 2        # 2-minute constraint
        }
        
        # Adjust thresholds based on training data volatility
        if volatility > 0.02:  # High volatility period
            base_params['momentum_threshold'] *= 1.2
            base_params['volume_threshold'] *= 1.1
        elif volatility < 0.01:  # Low volatility period
            base_params['momentum_threshold'] *= 0.8
            base_params['volume_threshold'] *= 0.9
        
        # Adjust position size based on volatility
        if volatility > 0.02:
            base_params['position_size'] = min(0.35, base_params['position_size'] * 1.1)
        elif volatility < 0.01:
            base_params['position_size'] = max(0.25, base_params['position_size'] * 0.9)
        
        print(f"  Training Data: {len(training_data):,} records")
        print(f"  Volatility: {volatility:.4f}")
        print(f"  Adjusted Parameters: momentum_threshold={base_params['momentum_threshold']:.6f}")
        
        return base_params
    
    def create_optimized_rt_strategy(self, name, trained_params):
        """Create the v3.0 Optimized RT strategy with trained parameters"""
        class OptimizedRTStrategy:
            def __init__(self, name, params):
                self.name = name
                self.parameters = params
                self.daily_performance = []
                self.adaptive_updates = 0
                self.last_trade_time = None
                self.strategy_pool = self.create_strategy_pool()
            
            def create_strategy_pool(self):
                """Create the 8-strategy pool from successful model"""
                return {
                    'ultra_volatility': self.ultra_volatility_strategy,
                    'breakout_momentum': self.breakout_momentum_strategy,
                    'ultra_momentum': self.ultra_momentum_strategy,
                    'accelerated_ma': self.accelerated_ma_strategy,
                    'ultra_scalping': self.ultra_scalping_strategy,
                    'extreme_mean_reversion': self.extreme_mean_reversion_strategy,
                    'garch_forecasting': self.garch_forecasting_strategy,
                    'kalman_adaptive': self.kalman_adaptive_strategy
                }
            
            def can_trade(self, timestamp):
                """Check if enough time has passed since last trade"""
                if self.last_trade_time is None:
                    return True
                
                time_diff = (timestamp - self.last_trade_time).total_seconds() / 60
                return time_diff >= self.parameters['min_trade_interval']
            
            def ultra_volatility_strategy(self, data):
                """Ultra volatility exploitation strategy"""
                signals = pd.Series(0, index=data.index)
                
                # Calculate volatility
                returns = data['close'].pct_change()
                volatility = returns.rolling(5).std()
                
                # Use trained volatility threshold
                vol_threshold = self.parameters['volatility_threshold']
                
                # Generate signals
                high_vol = volatility > vol_threshold
                momentum = data['close'].pct_change()
                
                # Long signals in high volatility with positive momentum
                long_condition = (high_vol & (momentum > self.parameters['momentum_threshold'])).fillna(False)
                short_condition = (high_vol & (momentum < -self.parameters['momentum_threshold'])).fillna(False)
                
                signals[long_condition] = 1
                signals[short_condition] = -1
                
                return signals
            
            def breakout_momentum_strategy(self, data):
                """Breakout momentum strategy"""
                signals = pd.Series(0, index=data.index)
                
                # Calculate breakout levels
                high_20 = data['high'].rolling(20).max()
                low_20 = data['low'].rolling(20).min()
                
                # Breakout conditions
                breakout_up = data['close'] > high_20.shift(1)
                breakout_down = data['close'] < low_20.shift(1)
                
                # Volume confirmation
                volume_spike = data['volume'] > data['volume'].rolling(20).mean() * self.parameters['volume_threshold']
                
                signals[breakout_up & volume_spike] = 1
                signals[breakout_down & volume_spike] = -1
                
                return signals
            
            def ultra_momentum_strategy(self, data):
                """Ultra momentum amplification strategy"""
                signals = pd.Series(0, index=data.index)
                
                # Calculate momentum
                momentum = data['close'].pct_change()
                momentum_ma = momentum.rolling(10).mean()
                
                # Use trained parameters
                long_condition = (
                    (momentum > self.parameters['momentum_threshold']) & 
                    (momentum_ma > self.parameters['momentum_threshold'] * 0.5) &
                    (data['volume'] > data['volume'].rolling(15).mean() * self.parameters['volume_threshold'])
                ).fillna(False)
                
                short_condition = (
                    (momentum < -self.parameters['momentum_threshold']) & 
                    (momentum_ma < -self.parameters['momentum_threshold'] * 0.5) &
                    (data['volume'] > data['volume'].rolling(15).mean() * self.parameters['volume_threshold'])
                ).fillna(False)
                
                signals[long_condition] = 1
                signals[short_condition] = -1
                
                return signals
            
            def accelerated_ma_strategy(self, data):
                """Accelerated moving average crossover"""
                signals = pd.Series(0, index=data.index)
                
                # Fast and slow moving averages
                fast_ma = data['close'].rolling(3).mean()
                slow_ma = data['close'].rolling(8).mean()
                
                # Crossover conditions
                crossover_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
                crossover_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
                
                signals[crossover_up] = 1
                signals[crossover_down] = -1
                
                return signals
            
            def ultra_scalping_strategy(self, data):
                """Ultra high-frequency scalping strategy"""
                signals = pd.Series(0, index=data.index)
                
                # Use trained parameters
                momentum = data['close'].pct_change()
                volume_ratio = data['volume'] / data['volume'].rolling(10).mean()
                
                # Quality scalping conditions
                long_condition = (
                    (momentum > self.parameters['momentum_threshold'] * 0.8) &  # Higher threshold
                    (volume_ratio > self.parameters['volume_threshold'] * 0.9) &  # Higher volume requirement
                    (momentum.rolling(3).std() < self.parameters['volatility_threshold'] * 0.5)  # Low volatility
                ).fillna(False)
                
                short_condition = (
                    (momentum < -self.parameters['momentum_threshold'] * 0.8) &
                    (volume_ratio > self.parameters['volume_threshold'] * 0.9) &
                    (momentum.rolling(3).std() < self.parameters['volatility_threshold'] * 0.5)
                ).fillna(False)
                
                signals[long_condition] = 1
                signals[short_condition] = -1
                
                return signals
            
            def extreme_mean_reversion_strategy(self, data):
                """Extreme mean reversion strategy"""
                signals = pd.Series(0, index=data.index)
                
                # Calculate mean reversion indicators
                ma_20 = data['close'].rolling(20).mean()
                ma_50 = data['close'].rolling(50).mean()
                
                # Extreme deviations
                deviation_20 = (data['close'] - ma_20) / ma_20
                deviation_50 = (data['close'] - ma_50) / ma_50
                
                # Mean reversion conditions
                long_condition = (
                    (deviation_20 < -0.02) &  # 2% below 20MA
                    (deviation_50 < -0.03) &  # 3% below 50MA
                    (data['volume'] > data['volume'].rolling(20).mean() * self.parameters['volume_threshold'] * 1.1)
                ).fillna(False)
                
                short_condition = (
                    (deviation_20 > 0.02) &   # 2% above 20MA
                    (deviation_50 > 0.03) &   # 3% above 50MA
                    (data['volume'] > data['volume'].rolling(20).mean() * self.parameters['volume_threshold'] * 1.1)
                ).fillna(False)
                
                signals[long_condition] = 1
                signals[short_condition] = -1
                
                return signals
            
            def garch_forecasting_strategy(self, data):
                """GARCH volatility forecasting strategy"""
                signals = pd.Series(0, index=data.index)
                
                # Simplified GARCH-like approach
                returns = data['close'].pct_change()
                volatility = returns.rolling(10).std()
                
                # Volatility forecasting
                vol_forecast = volatility.rolling(5).mean()
                
                # Trade when volatility is in optimal range
                optimal_vol = (vol_forecast > self.parameters['volatility_threshold'] * 0.5) & (vol_forecast < self.parameters['volatility_threshold'] * 1.5)
                momentum = data['close'].pct_change()
                
                signals[optimal_vol & (momentum > self.parameters['momentum_threshold'])] = 1
                signals[optimal_vol & (momentum < -self.parameters['momentum_threshold'])] = -1
                
                return signals
            
            def kalman_adaptive_strategy(self, data):
                """Kalman filter adaptive moving average"""
                signals = pd.Series(0, index=data.index)
                
                # Simplified Kalman-like adaptive MA
                short_ma = data['close'].rolling(3).mean()
                long_ma = data['close'].rolling(10).mean()
                
                # Adaptive crossover
                crossover_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
                crossover_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
                
                signals[crossover_up] = 1
                signals[crossover_down] = -1
                
                return signals
            
            def calculate_signals(self, data):
                """Calculate signals using all 8 strategies"""
                # Get signals from all strategies
                ultra_signals = self.ultra_volatility_strategy(data)
                breakout_signals = self.breakout_momentum_strategy(data)
                momentum_signals = self.ultra_momentum_strategy(data)
                ma_signals = self.accelerated_ma_strategy(data)
                scalping_signals = self.ultra_scalping_strategy(data)
                mean_reversion_signals = self.extreme_mean_reversion_strategy(data)
                garch_signals = self.garch_forecasting_strategy(data)
                kalman_signals = self.kalman_adaptive_strategy(data)
                
                # Combine signals with priority weighting
                combined_signals = pd.Series(0, index=data.index)
                
                # Priority 1: GARCH and Kalman (advanced algorithms)
                combined_signals[garch_signals != 0] = garch_signals[garch_signals != 0]
                combined_signals[kalman_signals != 0] = kalman_signals[kalman_signals != 0]
                
                # Priority 2: Ultra volatility and breakout
                combined_signals[ultra_signals != 0] = ultra_signals[ultra_signals != 0]
                combined_signals[breakout_signals != 0] = breakout_signals[breakout_signals != 0]
                
                # Priority 3: Momentum and MA
                combined_signals[momentum_signals != 0] = momentum_signals[momentum_signals != 0]
                combined_signals[ma_signals != 0] = ma_signals[ma_signals != 0]
                
                # Priority 4: Scalping and mean reversion
                combined_signals[scalping_signals != 0] = scalping_signals[scalping_signals != 0]
                combined_signals[mean_reversion_signals != 0] = mean_reversion_signals[mean_reversion_signals != 0]
                
                return combined_signals
            
            def backtest(self, data, initial_capital=100000):
                """Backtest with the trained model parameters"""
                signals = self.calculate_signals(data)
                
                # Initialize tracking variables
                position = 0
                capital = initial_capital
                equity_curve = [initial_capital]
                trades = []
                daily_returns = []
                
                # Group data by day for daily updates
                daily_groups = data.groupby(data.index.date)
                
                for date, day_data in daily_groups:
                    day_signals = signals.loc[day_data.index]
                    
                    # Daily adaptive update
                    if len(self.daily_performance) > 0:
                        self.adaptive_update()
                    
                    # Process day's signals
                    for timestamp, signal in day_signals.items():
                        if signal != 0 and position == 0:
                            # Check real-time constraints
                            if self.can_trade(timestamp):
                                # Open position
                                position = signal
                                entry_price = data.loc[timestamp, 'close']
                                entry_time = timestamp
                                position_size = self.parameters['position_size']
                                self.last_trade_time = timestamp
                        elif position != 0:
                            # Check exit conditions
                            current_price = data.loc[timestamp, 'close']
                            hold_period = (timestamp - entry_time).total_seconds() / 60
                            
                            # Enhanced exit conditions
                            exit_signal = False
                            if hold_period >= self.parameters['max_hold_period']:
                                exit_signal = True
                            elif position == 1:
                                # Long position exit conditions
                                if current_price < entry_price * (1 - self.parameters['stop_loss']):
                                    exit_signal = True  # Stop loss
                                elif current_price > entry_price * (1 + self.parameters['profit_target']):
                                    exit_signal = True  # Profit target
                            elif position == -1:
                                # Short position exit conditions
                                if current_price > entry_price * (1 + self.parameters['stop_loss']):
                                    exit_signal = True  # Stop loss
                                elif current_price < entry_price * (1 - self.parameters['profit_target']):
                                    exit_signal = True  # Profit target
                            
                            if exit_signal:
                                # Close position
                                pnl = (current_price - entry_price) / entry_price * position
                                capital *= (1 + pnl * position_size)
                                
                                trades.append({
                                    'entry_time': entry_time,
                                    'exit_time': timestamp,
                                    'position': position,
                                    'entry_price': entry_price,
                                    'exit_price': current_price,
                                    'pnl': pnl,
                                    'capital': capital
                                })
                                position = 0
                    
                    # Record daily performance
                    daily_return = (capital - equity_curve[-1]) / equity_curve[-1]
                    daily_returns.append(daily_return)
                    self.daily_performance.append(daily_return)
                    equity_curve.append(capital)
                
                # Calculate metrics
                total_return = (capital - initial_capital) / initial_capital
                num_trades = len(trades)
                win_rate = len([t for t in trades if t['pnl'] > 0]) / num_trades if num_trades > 0 else 0
                
                return {
                    'total_return': total_return,
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'trades': trades,
                    'daily_returns': daily_returns,
                    'equity_curve': equity_curve,
                    'adaptive_updates': self.adaptive_updates,
                    'leverage': self.parameters['leverage'],
                    'min_trade_interval': self.parameters['min_trade_interval']
                }
            
            def adaptive_update(self):
                """Adaptive update based on recent performance"""
                if len(self.daily_performance) < 3:
                    return
                
                # Calculate recent performance
                recent_performance = np.mean(self.daily_performance[-3:])
                
                # Adaptive parameter adjustments
                if recent_performance > 0.015:  # Very good performance
                    # Increase aggressiveness
                    self.parameters['momentum_threshold'] *= 0.90
                    self.parameters['volume_threshold'] *= 0.90
                    self.parameters['position_size'] = min(0.40, self.parameters['position_size'] * 1.10)
                elif recent_performance < -0.015:  # Poor performance
                    # Decrease aggressiveness
                    self.parameters['momentum_threshold'] *= 1.10
                    self.parameters['volume_threshold'] *= 1.10
                    self.parameters['position_size'] = max(0.25, self.parameters['position_size'] * 0.90)
                
                self.adaptive_updates += 1
        
        return OptimizedRTStrategy(name, trained_params)
    
    def run_optimized_rt_walk_test(self):
        """Run the corrected Optimized RT walk-through test"""
        print("🚀 Starting Corrected Optimized RT Walk-Through Test")
        print("=" * 70)
        print("📊 Testing v3.0 Optimized RT with proper training")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            print("❌ Failed to load data")
            return
        
        # Generate test periods
        periods = self.generate_test_periods()
        print(f"📅 Generated {len(periods)} test periods")
        
        # Initialize results
        self.results = {
            'v3.0_Optimized_RT': []
        }
        
        # Test each period
        for i, period in enumerate(periods):
            print(f"\n📊 Testing Period {period['period_number']}/10")
            print(f"Training: {period['training_start'].strftime('%Y-%m-%d')} to {period['training_end'].strftime('%Y-%m-%d')}")
            print(f"Testing: {period['test_start'].strftime('%Y-%m-%d')} to {period['test_end'].strftime('%Y-%m-%d')}")
            
            # Get training data
            training_data = self.data[
                (self.data.index >= period['training_start']) &
                (self.data.index <= period['training_end'])
            ]
            
            # Get test data
            test_data = self.data[
                (self.data.index >= period['test_start']) &
                (self.data.index <= period['test_end'])
            ]
            
            if len(training_data) < 1000 or len(test_data) < 100:
                print(f"⚠️ Insufficient data: Training={len(training_data)}, Test={len(test_data)}")
                continue
            
            print(f"Training data: {len(training_data):,} records")
            print(f"Test data: {len(test_data):,} records")
            
            # Train the model on training data
            print("🔧 Training model on training data...")
            trained_params = self.train_model_on_data(training_data)
            
            # Test the model
            try:
                strategy = self.create_optimized_rt_strategy('v3.0_Optimized_RT', trained_params)
                result = strategy.backtest(test_data)
                
                result['period'] = period['period_number']
                result['test_start'] = period['test_start']
                result['test_end'] = period['test_end']
                result['model'] = 'v3.0_Optimized_RT'
                result['trained_params'] = trained_params
                
                self.results['v3.0_Optimized_RT'].append(result)
                
                print(f"✅ v3.0 Optimized RT: {result['total_return']:.4f} ({result['total_return']*100:.2f}%)")
                print(f"   Trades: {result['num_trades']}, Win Rate: {result['win_rate']:.2f}")
                print(f"   Leverage: {result['leverage']:.1f}x, Min Interval: {result['min_trade_interval']}min")
                print(f"   Adaptive Updates: {result['adaptive_updates']}")
                
            except Exception as e:
                print(f"❌ Error testing v3.0 Optimized RT: {str(e)}")
            
            print(f"Completed {i+1}/10 periods...")
        
        # Calculate summary statistics
        self.calculate_summary_statistics()
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics"""
        print("\n" + "=" * 70)
        print("📊 CORRECTED OPTIMIZED RT WALK-THROUGH TEST RESULTS")
        print("=" * 70)
        
        for model_name, results in self.results.items():
            if not results:
                continue
            
            returns = [r['total_return'] for r in results]
            num_trades = [r['num_trades'] for r in results]
            win_rates = [r['win_rate'] for r in results]
            adaptive_updates = [r['adaptive_updates'] for r in results]
            
            # Calculate statistics
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            min_return = np.min(returns)
            max_return = np.max(returns)
            
            # Target achievement
            target_achievements = [1 for r in returns if r >= 0.05]
            target_achievement_rate = len(target_achievements) / len(returns) * 100
            
            # Average metrics
            avg_trades = np.mean(num_trades)
            avg_win_rate = np.mean(win_rates)
            avg_adaptive_updates = np.mean(adaptive_updates)
            
            print(f"\n{model_name}:")
            print(f"  Average Return: {avg_return:.4f} ({avg_return*100:.2f}%)")
            print(f"  Standard Deviation: {std_return:.4f} ({std_return*100:.2f}%)")
            print(f"  Min Return: {min_return:.4f} ({min_return*100:.2f}%)")
            print(f"  Max Return: {max_return:.4f} ({max_return*100:.2f}%)")
            print(f"  Target Achievement Rate: {target_achievement_rate:.2f}%")
            print(f"  Number of Periods: {len(results)}")
            print(f"  Average Trades: {avg_trades:.1f}")
            print(f"  Average Win Rate: {avg_win_rate:.2f}")
            print(f"  Average Adaptive Updates: {avg_adaptive_updates:.1f}")
            
            # Most recent performance
            if results:
                most_recent = results[-1]
                print(f"  Most Recent Return: {most_recent['total_return']:.4f} ({most_recent['total_return']*100:.2f}%)")
        
        print(f"\nTest data: {len(self.data):,} records")


def main():
    """Main function to run the corrected Optimized RT walk-through test"""
    system = OptimizedRTWalkTestFixed()
    system.run_optimized_rt_walk_test()


if __name__ == "__main__":
    main() 