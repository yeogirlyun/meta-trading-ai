#!/usr/bin/env python3
"""
Adaptive Walk-Forward Testing System v2.0
Performs 10 test periods of 2 weeks each, covering the most recent 11 weeks
Each test period includes daily adaptive updates
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


class AdaptiveWalkForwardSystemV2:
    """Improved adaptive walk-forward testing system with daily updates"""
    
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
        """Generate 10 test periods of 2 weeks each, covering most recent 11 weeks"""
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
    
    def create_adaptive_strategy(self, name):
        """Create an adaptive strategy with daily updates"""
        class AdaptiveStrategy:
            def __init__(self, name):
                self.name = name
                self.parameters = {
                    'momentum_threshold': 0.0008,
                    'volume_threshold': 1.3,
                    'volatility_threshold': 0.01,
                    'max_hold_period': 5,
                    'position_size': 0.15
                }
                self.daily_performance = []
                self.adaptive_updates = 0
            
            def calculate_signals(self, data):
                """Calculate trading signals with proper Series handling"""
                signals = pd.Series(0, index=data.index)
                
                # Calculate features
                momentum = data['close'].pct_change()
                volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
                volatility = momentum.rolling(5).std()
                
                # Generate signals with proper boolean handling
                long_condition = (
                    (momentum > self.parameters['momentum_threshold']) &
                    (volume_ratio > self.parameters['volume_threshold']) &
                    (volatility < self.parameters['volatility_threshold'])
                ).fillna(False)
                
                short_condition = (
                    (momentum < -self.parameters['momentum_threshold']) &
                    (volume_ratio > self.parameters['volume_threshold']) &
                    (volatility < self.parameters['volatility_threshold'])
                ).fillna(False)
                
                signals[long_condition] = 1
                signals[short_condition] = -1
                
                return signals
            
            def backtest(self, data, initial_capital=100000):
                """Backtest with daily adaptive updates"""
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
                            # Open position
                            position = signal
                            entry_price = data.loc[timestamp, 'close']
                            entry_time = timestamp
                        elif position != 0:
                            # Check exit conditions
                            current_price = data.loc[timestamp, 'close']
                            hold_period = (timestamp - entry_time).total_seconds() / 60
                            
                            # Exit conditions
                            exit_signal = False
                            if hold_period >= self.parameters['max_hold_period']:
                                exit_signal = True
                            elif position == 1 and current_price < entry_price * 0.995:
                                exit_signal = True
                            elif position == -1 and current_price > entry_price * 1.005:
                                exit_signal = True
                            
                            if exit_signal:
                                # Close position
                                pnl = (current_price - entry_price) / entry_price * position
                                capital *= (1 + pnl * self.parameters['position_size'])
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
                    'adaptive_updates': self.adaptive_updates
                }
            
            def adaptive_update(self):
                """Daily adaptive update based on recent performance"""
                if len(self.daily_performance) < 3:
                    return
                
                # Calculate recent performance
                recent_performance = np.mean(self.daily_performance[-3:])
                
                # Adaptive parameter adjustments
                if recent_performance > 0.01:  # Good performance
                    # Increase aggressiveness
                    self.parameters['momentum_threshold'] *= 0.95
                    self.parameters['volume_threshold'] *= 0.95
                    self.parameters['position_size'] = min(0.20, self.parameters['position_size'] * 1.05)
                elif recent_performance < -0.01:  # Poor performance
                    # Decrease aggressiveness
                    self.parameters['momentum_threshold'] *= 1.05
                    self.parameters['volume_threshold'] *= 1.05
                    self.parameters['position_size'] = max(0.10, self.parameters['position_size'] * 0.95)
                
                self.adaptive_updates += 1
        
        return AdaptiveStrategy(name)
    
    def run_adaptive_walk_forward_test(self):
        """Run the adaptive walk-forward test"""
        print("🚀 Starting Adaptive Walk-Forward Testing System v2.0")
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
            'v3.0_Ultra_Aggressive': [],
            'v3.0_Optimized_RT': [],
            'v3.0_Enhanced': []
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
            
            # Test each model
            for model_name in self.results.keys():
                try:
                    strategy = self.create_adaptive_strategy(model_name)
                    result = strategy.backtest(test_data)
                    
                    result['period'] = period['period_number']
                    result['test_start'] = period['test_start']
                    result['test_end'] = period['test_end']
                    result['model'] = model_name
                    
                    self.results[model_name].append(result)
                    
                    print(f"✅ {model_name}: {result['total_return']:.4f} ({result['total_return']*100:.2f}%)")
                    print(f"   Trades: {result['num_trades']}, Win Rate: {result['win_rate']:.2f}")
                    print(f"   Adaptive Updates: {result['adaptive_updates']}")
                    
                except Exception as e:
                    print(f"❌ Error testing {model_name}: {str(e)}")
            
            print(f"Completed {i+1}/10 periods...")
        
        # Calculate summary statistics
        self.calculate_summary_statistics()
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics for all models"""
        print("\n" + "=" * 60)
        print("📊 ADAPTIVE WALK-FORWARD TESTING RESULTS v2.0")
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
        
        print(f"\nTest data: {len(self.data):,} records")


def main():
    """Main function to run the adaptive walk-forward test"""
    system = AdaptiveWalkForwardSystemV2()
    system.run_adaptive_walk_forward_test()


if __name__ == "__main__":
    main() 