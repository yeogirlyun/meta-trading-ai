#!/usr/bin/env python3
"""
ADAPTIVE_ULTRA_LV - Advanced Adaptive Trading System with Leverage
Most sophisticated strategy with 2.5x leverage
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


class ADAPTIVE_ULTRA_LV:
    """Advanced adaptive trading system with leverage"""
    
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
            training_start = training_end - timedelta(weeks=26)  # 6 months training
            
            periods.append({
                'training_start': training_start,
                'training_end': training_end,
                'test_start': period_start,
                'test_end': period_end,
                'period_number': i + 1
            })
        
        return periods
    
    def create_adaptive_ultra_lv_strategy(self, name):
        """Create the most advanced adaptive strategy with leverage"""
        class AdaptiveUltraLVStrategy:
            def __init__(self, name):
                self.name = name
                # Optimized parameters for maximum performance with leverage
                self.parameters = {
                    'momentum_threshold': 0.0006,  # More sensitive
                    'volume_threshold': 1.2,  # Lower threshold
                    'volatility_threshold': 0.008,  # Lower for more signals
                    'max_hold_period': 2,  # Shorter for more frequent trading
                    'position_size': 0.25,  # Larger position size
                    'stop_loss': 0.002,  # Tighter stop loss
                    'profit_target': 0.004,  # Tighter profit target
                    'leverage': 2.5  # 2.5x leverage
                }
                self.daily_performance = []
                self.adaptive_updates = 0
                self.strategy_pool = self.create_strategy_pool()
            
            def create_strategy_pool(self):
                """Create pool of ultra-advanced strategies"""
                return {
                    'ultra_volatility': self.ultra_volatility_strategy,
                    'breakout_momentum': self.breakout_momentum_strategy,
                    'high_frequency_scalping': self.high_frequency_scalping_strategy,
                    'garch_forecasting': self.garch_forecasting_strategy,
                    'kalman_adaptive': self.kalman_adaptive_strategy,
                    'multi_timeframe': self.multi_timeframe_strategy
                }
            
            def ultra_volatility_strategy(self, data):
                """Ultra-aggressive volatility exploitation"""
                signals = pd.Series(0, index=data.index)
                
                # Calculate volatility
                returns = data['close'].pct_change()
                volatility = returns.rolling(3).std()  # Shorter window
                
                # Ultra-aggressive volatility thresholds
                vol_threshold = 0.006  # Lower threshold for more signals
                
                # Generate signals
                high_vol = volatility > vol_threshold
                momentum = data['close'].pct_change()
                
                # Long signals in high volatility with positive momentum
                long_condition = (high_vol & (momentum > 0.0008)).fillna(False)
                short_condition = (high_vol & (momentum < -0.0008)).fillna(False)
                
                signals[long_condition] = 1
                signals[short_condition] = -1
                
                return signals
            
            def breakout_momentum_strategy(self, data):
                """Aggressive breakout momentum strategy"""
                signals = pd.Series(0, index=data.index)
                
                # Calculate breakout levels with shorter windows
                high_10 = data['high'].rolling(10).max()  # Shorter window
                low_10 = data['low'].rolling(10).min()
                
                # Breakout conditions
                breakout_up = data['close'] > high_10.shift(1)
                breakout_down = data['close'] < low_10.shift(1)
                
                # Volume confirmation
                volume_spike = data['volume'] > data['volume'].rolling(15).mean() * 1.3
                
                signals[breakout_up & volume_spike] = 1
                signals[breakout_down & volume_spike] = -1
                
                return signals
            
            def high_frequency_scalping_strategy(self, data):
                """Ultra-high-frequency scalping strategy"""
                signals = pd.Series(0, index=data.index)
                
                # Very tight thresholds for scalping
                momentum = data['close'].pct_change()
                volume_ratio = data['volume'] / data['volume'].rolling(8).mean()
                
                # Scalping conditions
                long_condition = (
                    (momentum > 0.0004) &  # Very low threshold
                    (volume_ratio > 1.1) &  # Lower volume requirement
                    (momentum.rolling(2).std() < 0.004)  # Lower volatility
                ).fillna(False)
                
                short_condition = (
                    (momentum < -0.0004) &
                    (volume_ratio > 1.1) &
                    (momentum.rolling(2).std() < 0.004)
                ).fillna(False)
                
                signals[long_condition] = 1
                signals[short_condition] = -1
                
                return signals
            
            def garch_forecasting_strategy(self, data):
                """Advanced GARCH volatility forecasting"""
                signals = pd.Series(0, index=data.index)
                
                # Simplified GARCH-like approach
                returns = data['close'].pct_change()
                volatility = returns.rolling(8).std()  # Shorter window
                
                # Volatility forecasting
                vol_forecast = volatility.rolling(3).mean()
                
                # Trade when volatility is in optimal range
                optimal_vol = (vol_forecast > 0.004) & (vol_forecast < 0.012)
                momentum = data['close'].pct_change()
                
                signals[optimal_vol & (momentum > 0.0008)] = 1
                signals[optimal_vol & (momentum < -0.0008)] = -1
                
                return signals
            
            def kalman_adaptive_strategy(self, data):
                """Advanced Kalman filter adaptive moving average"""
                signals = pd.Series(0, index=data.index)
                
                # Simplified Kalman-like adaptive MA
                short_ma = data['close'].rolling(2).mean()  # Very short
                long_ma = data['close'].rolling(8).mean()  # Shorter
                
                # Adaptive crossover
                crossover_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
                crossover_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
                
                signals[crossover_up] = 1
                signals[crossover_down] = -1
                
                return signals
            
            def multi_timeframe_strategy(self, data):
                """Multi-timeframe confirmation strategy"""
                signals = pd.Series(0, index=data.index)
                
                # 1-minute signals
                momentum_1m = data['close'].pct_change()
                
                # 5-minute signals (resampled)
                data_5m = data.resample('5T').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 
                    'close': 'last', 'volume': 'sum'
                }).dropna()
                momentum_5m = data_5m['close'].pct_change()
                
                # Align 5-minute signals with 1-minute data
                momentum_5m_aligned = momentum_5m.reindex(data.index, method='ffill')
                
                # Multi-timeframe confirmation
                long_condition = (
                    (momentum_1m > 0.0006) & 
                    (momentum_5m_aligned > 0.001) &
                    (data['volume'] > data['volume'].rolling(10).mean() * 1.2)
                ).fillna(False)
                
                short_condition = (
                    (momentum_1m < -0.0006) & 
                    (momentum_5m_aligned < -0.001) &
                    (data['volume'] > data['volume'].rolling(10).mean() * 1.2)
                ).fillna(False)
                
                signals[long_condition] = 1
                signals[short_condition] = -1
                
                return signals
            
            def calculate_signals(self, data):
                """Calculate signals using advanced strategy combination"""
                # Get signals from all strategies
                ultra_signals = self.ultra_volatility_strategy(data)
                breakout_signals = self.breakout_momentum_strategy(data)
                scalping_signals = self.high_frequency_scalping_strategy(data)
                garch_signals = self.garch_forecasting_strategy(data)
                kalman_signals = self.kalman_adaptive_strategy(data)
                multi_signals = self.multi_timeframe_strategy(data)
                
                # Combine signals with priority weighting
                combined_signals = pd.Series(0, index=data.index)
                
                # Priority 1: Multi-timeframe (highest confidence)
                combined_signals[multi_signals != 0] = multi_signals[multi_signals != 0]
                
                # Priority 2: Ultra volatility
                combined_signals[ultra_signals != 0] = ultra_signals[ultra_signals != 0]
                
                # Priority 3: Breakout momentum
                combined_signals[breakout_signals != 0] = breakout_signals[breakout_signals != 0]
                
                # Priority 4: High-frequency scalping
                combined_signals[scalping_signals != 0] = scalping_signals[scalping_signals != 0]
                
                # Priority 5: GARCH and Kalman
                combined_signals[garch_signals != 0] = garch_signals[garch_signals != 0]
                combined_signals[kalman_signals != 0] = kalman_signals[kalman_signals != 0]
                
                return combined_signals
            
            def backtest(self, data, initial_capital=100000):
                """Advanced backtest with leverage"""
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
                            # Open position with leverage
                            position = signal
                            entry_price = data.loc[timestamp, 'close']
                            entry_time = timestamp
                            position_size = self.parameters['position_size']
                            leverage = self.parameters['leverage']
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
                                # Close position with leverage
                                pnl = (current_price - entry_price) / entry_price * position
                                capital *= (1 + pnl * position_size * leverage)
                                
                                trades.append({
                                    'entry_time': entry_time,
                                    'exit_time': timestamp,
                                    'position': position,
                                    'entry_price': entry_price,
                                    'exit_price': current_price,
                                    'pnl': pnl,
                                    'capital': capital,
                                    'leverage': leverage
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
                    'leverage': self.parameters['leverage']
                }
            
            def adaptive_update(self):
                """Ultra-adaptive update based on recent performance"""
                if len(self.daily_performance) < 3:
                    return
                
                # Calculate recent performance
                recent_performance = np.mean(self.daily_performance[-3:])
                
                # Ultra-adaptive parameter adjustments
                if recent_performance > 0.015:  # Very good performance
                    # Increase aggressiveness
                    self.parameters['momentum_threshold'] *= 0.85
                    self.parameters['volume_threshold'] *= 0.85
                    self.parameters['position_size'] = min(0.35, self.parameters['position_size'] * 1.15)
                elif recent_performance < -0.015:  # Poor performance
                    # Decrease aggressiveness
                    self.parameters['momentum_threshold'] *= 1.15
                    self.parameters['volume_threshold'] *= 1.15
                    self.parameters['position_size'] = max(0.20, self.parameters['position_size'] * 0.85)
                
                self.adaptive_updates += 1
        
        return AdaptiveUltraLVStrategy(name)
    
    def run_adaptive_ultra_lv_test(self):
        """Run the ADAPTIVE_ULTRA_LV test"""
        print("🚀 Starting ADAPTIVE_ULTRA_LV Testing System")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            print("❌ Failed to load data")
            return
        
        # Generate test periods
        periods = self.generate_test_periods()
        print(f"📅 Generated {len(periods)} test periods")
        
        # Initialize results
        self.results = {
            'ADAPTIVE_ULTRA_LV': []
        }
        
        # Test each period
        for i, period in enumerate(periods):
            print(f"\n📊 Testing Period {period['period_number']}/10")
            print(f"Training: {period['training_start'].strftime('%Y-%m-%d')} to {period['training_end'].strftime('%Y-%m-%d')}")
            print(f"Testing: {period['test_start'].strftime('%Y-%m-%d')} to {period['test_end'].strftime('%Y-%m-%d')}")
            
            # Get test data
            test_data = self.data[
                (self.data.index >= period['test_start']) &
                (self.data.index <= period['test_end'])
            ]
            
            if len(test_data) < 100:
                print(f"⚠️ Insufficient test data: {len(test_data)} records")
                continue
            
            print(f"Test data: {len(test_data):,} records")
            
            # Test the model
            try:
                strategy = self.create_adaptive_ultra_lv_strategy('ADAPTIVE_ULTRA_LV')
                result = strategy.backtest(test_data)
                
                result['period'] = period['period_number']
                result['test_start'] = period['test_start']
                result['test_end'] = period['test_end']
                result['model'] = 'ADAPTIVE_ULTRA_LV'
                
                self.results['ADAPTIVE_ULTRA_LV'].append(result)
                
                print(f"✅ ADAPTIVE_ULTRA_LV: {result['total_return']:.4f} ({result['total_return']*100:.2f}%)")
                print(f"   Trades: {result['num_trades']}, Win Rate: {result['win_rate']:.2f}")
                print(f"   Leverage: {result['leverage']:.1f}x, Adaptive Updates: {result['adaptive_updates']}")
                
            except Exception as e:
                print(f"❌ Error testing ADAPTIVE_ULTRA_LV: {str(e)}")
            
            print(f"Completed {i+1}/10 periods...")
        
        # Calculate summary statistics
        self.calculate_summary_statistics()
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics"""
        print("\n" + "=" * 60)
        print("📊 ADAPTIVE_ULTRA_LV TESTING RESULTS")
        print("=" * 60)
        
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
    """Main function to run the ADAPTIVE_ULTRA_LV test"""
    system = ADAPTIVE_ULTRA_LV()
    system.run_adaptive_ultra_lv_test()


if __name__ == "__main__":
    main() 