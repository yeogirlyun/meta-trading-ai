#!/usr/bin/env python3
"""
Walk-Forward Testing System for MetaTradingAI Models
Tests all models over multiple 2-week periods with training stopping before each test period
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import all model classes
from aggressive_meta_trading_ai_v3 import AggressiveMetaTradingAI
from aggressive_meta_trading_ai_v3_optimized import OptimizedAggressiveMetaTradingAI
from enhanced_restricted_trading import EnhancedAggressiveMetaTradingAI

class WalkForwardTestingSystem:
    """Comprehensive walk-forward testing system for all MetaTradingAI models"""
    
    def __init__(self):
        """Initialize the walk-forward testing system"""
        self.results = {}
        self.load_data()
        
    def load_data(self):
        """Load and prepare data for walk-forward testing"""
        print("Loading data for walk-forward testing...")
        self.data = pickle.load(open('polygon_QQQ_1m.pkl', 'rb'))
        
        # Convert index to datetime if needed
        if not isinstance(self.data.index, pd.DatetimeIndex):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except ValueError:
                self.data.index = pd.to_datetime(self.data.index, utc=True)
        
        # Handle timezone if present
        try:
            if hasattr(self.data.index, 'tz') and self.data.index.tz is not None:
                self.data.index = self.data.index.tz_localize(None)
        except:
            pass
        
        # Filter to trading hours and weekdays
        self.data = self.data.between_time('09:30', '16:00')
        self.data = self.data[self.data.index.dayofweek < 5]
        
        # Filter to last 5 years
        five_years_ago = self.data.index.max() - timedelta(days=5*365)
        self.data = self.data[self.data.index >= five_years_ago]
        
        print(f"Data loaded: {len(self.data)} records")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
    
    def get_test_periods(self, start_date=None, test_days=10, training_days=180):
        """Generate test periods for walk-forward testing"""
        if start_date is None:
            # Start from May 1st of the first year in data
            first_year = self.data.index.min().year
            start_date = datetime(first_year, 5, 1)
        
        end_date = self.data.index.max()
        test_periods = []
        
        current_date = start_date
        while current_date + timedelta(days=test_days) <= end_date:
            test_start = current_date
            test_end = current_date + timedelta(days=test_days)
            training_start = test_start - timedelta(days=training_days)
            
            # Ensure we have enough data
            if training_start >= self.data.index.min():
                test_periods.append({
                    'training_start': training_start,
                    'training_end': test_start,
                    'test_start': test_start,
                    'test_end': test_end,
                    'period_name': f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"
                })
            
            # Move forward by 1 week (7 days)
            current_date += timedelta(days=7)
        
        return test_periods
    
    def test_model_period(self, model_class, model_name, test_period, training_days=180):
        """Test a specific model on a specific period"""
        try:
            # Initialize model
            if model_name == "v3.0 Ultra-Aggressive":
                model = AggressiveMetaTradingAI()
            elif model_name == "v3.0 Optimized RT":
                model = OptimizedAggressiveMetaTradingAI(training_days=training_days)
            elif model_name == "v3.0 Enhanced":
                model = EnhancedAggressiveMetaTradingAI(training_days=training_days)
            else:
                return None
            
            # Get training and test data
            training_data = self.data[
                (self.data.index >= test_period['training_start']) & 
                (self.data.index < test_period['training_end'])
            ]
            test_data = self.data[
                (self.data.index >= test_period['test_start']) & 
                (self.data.index < test_period['test_end'])
            ]
            
            if len(training_data) < 1000 or len(test_data) < 100:
                return None
            
            # Run model on test period
            if model_name == "v3.0 Ultra-Aggressive":
                results = model.run_aggressive_meta_system(test_period_days=10)
                return {
                    'total_return': results['cumulative_return'],
                    'num_trades': results['total_trades'],
                    'trade_frequency': results['trade_frequency'],
                    'avg_signal_strength': results.get('avg_signal_strength', 0),
                    'avg_leverage': results.get('avg_leverage', 1.0)
                }
            elif model_name == "v3.0 Optimized RT":
                results = model.run_optimized_meta_system(test_period_days=10)
                return {
                    'total_return': results['cumulative_return'],
                    'num_trades': results['total_trades'],
                    'trade_frequency': results['trade_frequency'],
                    'avg_signal_strength': results.get('avg_signal_strength', 0),
                    'avg_leverage': results.get('avg_leverage', 1.0)
                }
            elif model_name == "v3.0 Enhanced":
                results = model.run_enhanced_meta_system(test_period_days=10)
                return {
                    'total_return': results['total_return'],
                    'num_trades': results['num_trades'],
                    'trade_frequency': results.get('trade_frequency', 0),
                    'avg_signal_strength': results.get('avg_signal_strength', 0),
                    'avg_leverage': results.get('avg_leverage', 1.0)
                }
            
        except Exception as e:
            print(f"Error testing {model_name} on {test_period['period_name']}: {e}")
            return None
    
    def run_walk_forward_testing(self, start_date=None, test_days=10, training_days=180):
        """Run comprehensive walk-forward testing for all models"""
        print(f"\n🚀 Starting Walk-Forward Testing System")
        print(f"Test Periods: {test_days} days each")
        print(f"Training Periods: {training_days} days each")
        print(f"Advancement: 1 week between tests")
        
        # Get test periods
        test_periods = self.get_test_periods(start_date, test_days, training_days)
        print(f"Total Test Periods: {len(test_periods)}")
        
        # Define models to test
        models = [
            "v3.0 Ultra-Aggressive",
            "v3.0 Optimized RT", 
            "v3.0 Enhanced"
        ]
        
        # Initialize results structure
        self.results = {model: [] for model in models}
        
        # Run tests for each period
        for i, period in enumerate(test_periods):
            print(f"\n📊 Testing Period {i+1}/{len(test_periods)}: {period['period_name']}")
            print(f"  Training: {period['training_start'].strftime('%Y-%m-%d')} to {period['training_end'].strftime('%Y-%m-%d')}")
            print(f"  Testing: {period['test_start'].strftime('%Y-%m-%d')} to {period['test_end'].strftime('%Y-%m-%d')}")
            
            for model_name in models:
                print(f"  Testing {model_name}...")
                result = self.test_model_period(model_name, model_name, period, training_days)
                
                if result:
                    result['period'] = period['period_name']
                    result['test_start'] = period['test_start']
                    result['test_end'] = period['test_end']
                    self.results[model_name].append(result)
                    print(f"    Return: {result['total_return']:.4f} ({result['total_return']*100:.2f}%)")
                    print(f"    Trades: {result['num_trades']}, Frequency: {result['trade_frequency']:.2f}/hour")
                else:
                    print(f"    ❌ Failed or insufficient data")
        
        # Calculate summary statistics
        self.calculate_summary_statistics()
        
        return self.results
    
    def calculate_summary_statistics(self):
        """Calculate comprehensive summary statistics for all models"""
        print(f"\n📈 WALK-FORWARD TESTING SUMMARY STATISTICS")
        print("=" * 80)
        
        summary_stats = {}
        
        for model_name, results in self.results.items():
            if not results:
                continue
            
            returns = [r['total_return'] for r in results]
            trades = [r['num_trades'] for r in results]
            frequencies = [r['trade_frequency'] for r in results]
            signal_strengths = [r['avg_signal_strength'] for r in results]
            leverages = [r['avg_leverage'] for r in results]
            
            # Calculate statistics
            summary_stats[model_name] = {
                'num_periods': len(results),
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'min_return': np.min(returns),
                'max_return': np.max(returns),
                'median_return': np.median(returns),
                'target_achievement_rate': len([r for r in returns if r >= 0.05]) / len(returns),
                'avg_trades': np.mean(trades),
                'avg_frequency': np.mean(frequencies),
                'avg_signal_strength': np.mean(signal_strengths),
                'avg_leverage': np.mean(leverages),
                'consistency_score': 1 - (np.std(returns) / (abs(np.mean(returns)) + 1e-6))
            }
        
        # Print detailed statistics
        for model_name, stats in summary_stats.items():
            print(f"\n🎯 {model_name}")
            print(f"  Test Periods: {stats['num_periods']}")
            print(f"  Average Return: {stats['avg_return']:.4f} ({stats['avg_return']*100:.2f}%)")
            print(f"  Return Std Dev: {stats['std_return']:.4f} ({stats['std_return']*100:.2f}%)")
            print(f"  Min Return: {stats['min_return']:.4f} ({stats['min_return']*100:.2f}%)")
            print(f"  Max Return: {stats['max_return']:.4f} ({stats['max_return']*100:.2f}%)")
            print(f"  Median Return: {stats['median_return']:.4f} ({stats['median_return']*100:.2f}%)")
            print(f"  Target Achievement Rate: {stats['target_achievement_rate']:.2%}")
            print(f"  Average Trades: {stats['avg_trades']:.1f}")
            print(f"  Average Trade Frequency: {stats['avg_frequency']:.2f} trades/hour")
            print(f"  Average Signal Strength: {stats['avg_signal_strength']:.2f}")
            print(f"  Average Leverage: {stats['avg_leverage']:.1f}x")
            print(f"  Consistency Score: {stats['consistency_score']:.3f}")
        
        # Model comparison
        print(f"\n🏆 MODEL COMPARISON")
        print("=" * 80)
        
        comparison_data = []
        for model_name, stats in summary_stats.items():
            comparison_data.append({
                'Model': model_name,
                'Avg Return (%)': stats['avg_return'] * 100,
                'Target Achievement (%)': stats['target_achievement_rate'] * 100,
                'Consistency': stats['consistency_score'],
                'Avg Trades': stats['avg_trades'],
                'Avg Frequency': stats['avg_frequency']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        
        # Find best performing model
        best_model = max(summary_stats.items(), key=lambda x: x[1]['avg_return'])
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0]}")
        print(f"  Average Return: {best_model[1]['avg_return']:.4f} ({best_model[1]['avg_return']*100:.2f}%)")
        print(f"  Target Achievement Rate: {best_model[1]['target_achievement_rate']:.2%}")
        print(f"  Consistency Score: {best_model[1]['consistency_score']:.3f}")
        
        return summary_stats
    
    def generate_period_analysis(self):
        """Generate detailed analysis of performance across different periods"""
        print(f"\n📊 PERIOD-BY-PERIOD ANALYSIS")
        print("=" * 80)
        
        # Create period comparison
        all_periods = set()
        for model_results in self.results.values():
            for result in model_results:
                all_periods.add(result['period'])
        
        all_periods = sorted(all_periods)
        
        # Create comparison table
        period_data = []
        for period in all_periods:
            period_row = {'Period': period}
            for model_name in self.results.keys():
                model_results = [r for r in self.results[model_name] if r['period'] == period]
                if model_results:
                    period_row[f'{model_name} Return (%)'] = model_results[0]['total_return'] * 100
                    period_row[f'{model_name} Trades'] = model_results[0]['num_trades']
                else:
                    period_row[f'{model_name} Return (%)'] = None
                    period_row[f'{model_name} Trades'] = None
            period_data.append(period_row)
        
        period_df = pd.DataFrame(period_data)
        print(period_df.to_string(index=False, float_format='%.2f'))
        
        return period_df

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Walk-Forward Testing System')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for testing (YYYY-MM-DD)')
    parser.add_argument('--test-days', type=int, default=10,
                       help='Number of days per test period (default: 10)')
    parser.add_argument('--training-days', type=int, default=180,
                       help='Number of training days (default: 180)')
    
    args = parser.parse_args()
    
    # Parse start date if provided
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    
    # Run walk-forward testing
    system = WalkForwardTestingSystem()
    results = system.run_walk_forward_testing(
        start_date=start_date,
        test_days=args.test_days,
        training_days=args.training_days
    )
    
    # Generate period analysis
    period_df = system.generate_period_analysis()
    
    return results, period_df

if __name__ == "__main__":
    results, period_df = main() 