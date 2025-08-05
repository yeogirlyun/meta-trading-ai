#!/usr/bin/env python3
"""
MetaTradingAI v1.0 - Walk-Forward Testing System
Comprehensive multi-period testing to ensure model consistency across different market conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List
warnings.filterwarnings('ignore')


class WalkForwardTester:
    def __init__(self):
        self.data = None
        self.results = {
            'window_returns': [],
            'window_trades': [],
            'window_drawdowns': [],
            'window_sharpe': [],
            'window_details': []
        }
        
    def load_data(self):
        """Load QQQ data for walk-forward testing"""
        print("Loading data for walk-forward testing...")
        self.data = pd.read_feather('polygon_QQQ_1m.feather')
        print(f"Data loaded: {len(self.data):,} records from "
              f"{self.data.index.min()} to {self.data.index.max()}")
        
    def calculate_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from a list of returns"""
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    def calculate_sharpe_ratio(self, returns: List[float], 
                              risk_free_rate: float = 0.04) -> float:
        """Calculate Sharpe ratio (annualized)"""
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming 10-day periods, 252 trading days per year)
        annualized_return = avg_return * (252 / 10)
        annualized_std = std_return * np.sqrt(252 / 10)
        
        sharpe = (annualized_return - risk_free_rate) / annualized_std
        return sharpe
    
    def run_single_window_test(self, train_start: datetime, train_end: datetime, 
                              test_start: datetime, test_end: datetime, 
                              model_type: str = 'ultra_aggressive',
                              initial_capital: float = 100000) -> Dict:
        """Run a single window test with specified model"""
        
        # Subset data
        train_data = self.data[
            (self.data.index >= train_start) & 
            (self.data.index < train_end)
        ]
        
        test_data = self.data[
            (self.data.index >= test_start) & 
            (self.data.index <= test_end)
        ]
        
        if len(train_data) < 1000 or len(test_data) < 100:
            return None
        
        print(f"  Training: {len(train_data):,} records, "
              f"Testing: {len(test_data):,} records")
        
        # Import and run appropriate model
        if model_type == 'ultra_aggressive':
            try:
                from aggressive_meta_trading_ai_v3 import AggressiveMetaTradingAI
                model = AggressiveMetaTradingAI()
                model.data = self.data
                
                # Set up training and test periods (ensure model uses these; update model if needed)
                model.training_start = train_start
                model.training_end = train_end
                model.test_start = test_start
                model.test_end = test_end
                
                # Run the model
                result = model.run_aggressive_meta_system(test_period_days=10, initial_capital=initial_capital)
                total_return = result.get('cumulative_return', 0.0)  # Fixed key
                total_trades = result.get('total_trades', sum(perf.get('num_trades', 0) for perf in result.get('hourly_performance', [])))  # Aggregate if missing
                target_achieved = total_return >= 0.05  # Compute here if missing
                return {
                    'return': total_return,
                    'trades': total_trades,
                    'target_achieved': target_achieved,
                    'model_type': model_type
                }
            except Exception as e:
                print(f"    Error running ultra-aggressive model: {e}")
                return None
                
        elif model_type == 'optimized_rt':
            try:
                from aggressive_meta_trading_ai_v3_optimized import OptimizedAggressiveMetaTradingAI  # Fixed import name
                model = OptimizedAggressiveMetaTradingAI()
                model.data = self.data
                
                # Set up training and test periods
                model.training_start = train_start
                model.training_end = train_end
                model.test_start = test_start
                model.test_end = test_end
                
                # Run the model
                result = model.run_optimized_meta_system(test_period_days=10, initial_capital=initial_capital)
                total_return = result.get('cumulative_return', 0.0)  # Fixed key
                total_trades = result.get('total_trades', sum(perf.get('num_trades', 0) for perf in result.get('hourly_performance', [])))  # Aggregate if missing
                target_achieved = total_return >= 0.05  # Compute here if missing
                return {
                    'return': total_return,
                    'trades': total_trades,
                    'target_achieved': target_achieved,
                    'model_type': model_type
                }
            except Exception as e:
                print(f"    Error running optimized RT model: {e}")
                return None
                
        elif model_type == 'enhanced_optimized':
            try:
                from aggressive_meta_trading_ai_v3_enhanced_optimized import EnhancedOptimizedAggressiveMetaTradingAI
                model = EnhancedOptimizedAggressiveMetaTradingAI()
                model.data = self.data
                
                # Set up training and test periods
                model.training_start = train_start
                model.training_end = train_end
                model.test_start = test_start
                model.test_end = test_end
                
                # Run the enhanced optimized model
                result = model.run_enhanced_optimized_meta_system(test_period_days=10, initial_capital=initial_capital)
                total_return = result.get('cumulative_return', 0.0)
                total_trades = result.get('total_trades', sum(perf.get('num_trades', 0) for perf in result.get('hourly_performance', [])))
                target_achieved = total_return >= 0.05
                return {
                    'return': total_return,
                    'trades': total_trades,
                    'target_achieved': target_achieved,
                    'model_type': model_type
                }
            except Exception as e:
                print(f"    Error running enhanced optimized model: {e}")
                return None
        
        return None
    
    def run_walk_forward_tests(self, num_windows: int = 20, test_days: int = 10, 
                              training_days: int = 180, rolling_step: int = 5,
                              model_type: str = 'ultra_aggressive') -> Dict:
        """Run comprehensive walk-forward testing with cumulative returns"""
        
        print(f"\n🚀 Starting Walk-Forward Testing")
        print(f"  Model: {model_type}")
        print(f"  Windows: {num_windows}")
        print(f"  Training Days: {training_days}")
        print(f"  Test Days: {test_days}")
        print(f"  Rolling Step: {rolling_step} days")
        
        # Initialize results
        self.results = {
            'window_returns': [],
            'window_trades': [],
            'window_drawdowns': [],
            'window_sharpe': [],
            'window_details': [],
            'model_type': model_type,
            'parameters': {
                'num_windows': num_windows,
                'test_days': test_days,
                'training_days': training_days,
                'rolling_step': rolling_step
            }
        }
        
        end_date = self.data.index.max()
        current_end = end_date
        
        successful_windows = 0
        
        for i in range(num_windows):
            test_start = current_end - timedelta(days=test_days)
            train_start = test_start - timedelta(days=training_days)
            
            # Ensure valid dates
            if train_start < self.data.index.min():
                print(f"  Window {i+1}: Insufficient historical data, stopping")
                break
            
            print(f"\n📊 Window {i+1}/{num_windows}")
            print(f"  Train: {train_start.strftime('%Y-%m-%d')} to "
                  f"{test_start.strftime('%Y-%m-%d')}")
            print(f"  Test:  {test_start.strftime('%Y-%m-%d')} to "
                  f"{current_end.strftime('%Y-%m-%d')}")
            
            # Run single window test with fixed initial capital (independent test)
            window_result = self.run_single_window_test(
                train_start, test_start, test_start, current_end, model_type,
                initial_capital=100000  # Fixed initial capital for each window
            )
            
            if window_result:
                window_return = window_result['return']
                window_trades = window_result['trades']
                
                # Store individual window results (no cumulative build-up)
                self.results['window_returns'].append(window_return)
                self.results['window_trades'].append(window_trades)
                
                self.results['window_details'].append({
                    'window': i+1,
                    'train_start': train_start.isoformat(),
                    'train_end': test_start.isoformat(),
                    'test_start': test_start.isoformat(),
                    'test_end': current_end.isoformat(),
                    'window_return': window_return,
                    'initial_capital': 100000,
                    'final_capital': 100000 * (1 + window_return),
                    'trades': window_trades,
                    'target_achieved': window_result['target_achieved']
                })
                
                successful_windows += 1
                print(f"  ✅ Window Return: {window_return:.4f} "
                      f"({window_return*100:.2f}%)")
                print(f"  💰 Final Capital: ${100000 * (1 + window_return):,.0f}")
                print(f"  📈 Trades: {window_trades}")
                print(f"  🎯 Target: {'✅' if window_result['target_achieved'] else '❌'}")
            else:
                print(f"  ❌ Window failed")
            
            # Step back for next window
            current_end -= timedelta(days=rolling_step)
        
        # Calculate aggregate statistics
        if self.results['window_returns']:
            self.calculate_aggregate_statistics()
        
        return self.results
    
    def calculate_aggregate_statistics(self):
        """Calculate comprehensive statistics from walk-forward results"""
        
        returns = np.array(self.results['window_returns'], dtype=float)  # Ensure float
        trades = np.array(self.results['window_trades'], dtype=int)  # Ensure int
        
        # Basic statistics
        self.results['stats'] = {
            'num_windows': int(len(returns)),
            'avg_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
            'median_return': float(np.median(returns)),
            
            # Win rate metrics
            'win_rate_5pct': float(np.mean(returns >= 0.05)),
            'win_rate_3pct': float(np.mean(returns >= 0.03)),
            'win_rate_positive': float(np.mean(returns > 0)),
            
            # Risk metrics
            'sharpe_ratio': float(self.calculate_sharpe_ratio(returns.tolist())),
            'max_drawdown': float(self.calculate_drawdown(returns.tolist())),
            
            # Trading metrics
            'avg_trades': float(np.mean(trades)),
            'std_trades': float(np.std(trades)),
            'total_trades': int(np.sum(trades)),
            
            # Consistency metrics
            'consistency_score': float((1 - (np.std(returns) / np.abs(np.mean(returns)))) 
                                if np.mean(returns) != 0 else 0)
        }
    
    def print_results_summary(self):
        """Print comprehensive results summary"""
        
        if not self.results.get('stats'):
            print("❌ No results to summarize")
            return
        
        stats = self.results['stats']
        
        print(f"\n🎯 WALK-FORWARD TESTING RESULTS")
        print(f"=" * 60)
        print(f"Model: {self.results['model_type']}")
        print(f"Windows Tested: {stats['num_windows']}")
        
        print(f"\n📊 PERFORMANCE METRICS")
        print(f"  Average Return: {stats['avg_return']:.4f} "
              f"({stats['avg_return']*100:.2f}%)")
        print(f"  Standard Deviation: {stats['std_return']:.4f} "
              f"({stats['std_return']*100:.2f}%)")
        print(f"  Min Return: {stats['min_return']:.4f} "
              f"({stats['min_return']*100:.2f}%)")
        print(f"  Max Return: {stats['max_return']:.4f} "
              f"({stats['max_return']*100:.2f}%)")
        print(f"  Median Return: {stats['median_return']:.4f} "
              f"({stats['median_return']*100:.2f}%)")
        
        print(f"\n🎯 TARGET ACHIEVEMENT")
        print(f"  Win Rate (≥5%): {stats['win_rate_5pct']:.2%}")
        print(f"  Win Rate (≥3%): {stats['win_rate_3pct']:.2%}")
        print(f"  Win Rate (>0%): {stats['win_rate_positive']:.2%}")
        
        print(f"\n📈 RISK METRICS")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.4f} "
              f"({stats['max_drawdown']*100:.2f}%)")
        print(f"  Consistency Score: {stats['consistency_score']:.2f}")
        
        print(f"\n🔄 TRADING METRICS")
        print(f"  Average Trades/Window: {stats['avg_trades']:.1f}")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Trade Std Dev: {stats['std_trades']:.1f}")
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT")
        if stats['avg_return'] >= 0.05:
            print(f"  ✅ Target Achieved: Average return ≥5%")
        else:
            print(f"  ❌ Target Not Met: Average return <5%")
            
        if stats['win_rate_5pct'] >= 0.6:
            print(f"  ✅ Consistent Performance: ≥60% windows achieve 5%")
        else:
            print(f"  ⚠️  Inconsistent Performance: <60% windows achieve 5%")
            
        if stats['sharpe_ratio'] >= 1.0:
            print(f"  ✅ Good Risk-Adjusted Returns: Sharpe ≥1.0")
        else:
            print(f"  ⚠️  Poor Risk-Adjusted Returns: Sharpe <1.0")
    
    def save_results(self, filename: str = None):
        """Save results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"walk_forward_results_{self.results['model_type']}_{timestamp}.json"
        
        import json
        
        # Convert datetime objects to strings and numpy types to Python natives
        serializable_results = self.results.copy()
        for detail in serializable_results.get('window_details', []):
            for key, value in detail.items():
                if isinstance(value, datetime):
                    detail[key] = value.isoformat()
        
        # Convert lists to serializable
        serializable_results['window_returns'] = [float(r) for r in serializable_results['window_returns']]
        serializable_results['window_trades'] = [int(t) for t in serializable_results['window_trades']]
        
        # Convert window details
        for detail in serializable_results.get('window_details', []):
            for key, value in detail.items():
                if isinstance(value, bool):
                    detail[key] = bool(value)
                elif isinstance(value, float):
                    detail[key] = float(value)
                elif isinstance(value, int):
                    detail[key] = int(value)
                elif isinstance(value, np.bool_):
                    detail[key] = bool(value)
                elif isinstance(value, np.float64):
                    detail[key] = float(value)
                elif isinstance(value, np.int64):
                    detail[key] = int(value)
        
        # Convert stats
        if 'stats' in serializable_results:
            for k, v in serializable_results['stats'].items():
                if isinstance(v, np.float64) or isinstance(v, float):
                    serializable_results['stats'][k] = float(v)
                elif isinstance(v, np.int64) or isinstance(v, int):
                    serializable_results['stats'][k] = int(v)
                elif isinstance(v, bool) or isinstance(v, np.bool_):
                    serializable_results['stats'][k] = bool(v)
                # Handle other types if needed
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename


def main():
    """Main execution function"""
    print("🚀 MetaTradingAI v1.0 - Walk-Forward Testing System")
    print("=" * 60)
    
    # Initialize tester
    tester = WalkForwardTester()
    
    # Load data
    tester.load_data()
    
    # Test models (can be customized)
    models = ['ultra_aggressive', 'optimized_rt', 'enhanced_optimized']
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper().replace('_', ' ')} Model")
        print(f"{'='*60}")
        
        # Run walk-forward tests
        results = tester.run_walk_forward_tests(
            num_windows=20,
            test_days=10,
            training_days=180,
            rolling_step=5,
            model_type=model_type
        )
        
        # Print summary
        tester.print_results_summary()
        
        # Save results
        tester.save_results()


if __name__ == "__main__":
    main() 