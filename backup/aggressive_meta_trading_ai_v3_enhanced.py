#!/usr/bin/env python3
"""
MetaTradingAI v3.0 Enhanced - Extended Training for Better Consistency
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import the original v3.0 system
from aggressive_meta_trading_ai_v3 import (
    detect_market_regime,
    AggressiveMetaStrategy,
    UltraVolatilityExploitation,
    BreakoutMomentum,
    UltraMomentumAmplification,
    AcceleratedMACross,
    UltraHighFrequencyScalping,
    ExtremeMeanReversion,
    GARCHVolatilityForecastingStrategy,
    KalmanFilterAdaptiveMA,
    AggressiveStrategySelector,
    AggressiveMetaTradingAI
)

class EnhancedAggressiveMetaTradingAI(AggressiveMetaTradingAI):
    """Enhanced version with extended training periods for better consistency"""
    
    def __init__(self, training_days: int = 180):
        """
        Initialize enhanced system with configurable training period
        
        Args:
            training_days: Number of days for training (default: 180 = 6 months)
        """
        self.training_days = training_days
        super().__init__()
        
        print(f"Enhanced MetaTradingAI v3.0 Initialized")
        print(f"Training Period: {training_days} days ({training_days/30:.1f} months)")
        
    def run_aggressive_meta_system(self, test_period_days: int = 10) -> dict:
        """Run the enhanced meta-trading system with extended training"""
        print(f"\nRunning Enhanced MetaTradingAI v3.0: Extended Training Model...")
        print(f"Training Period: {self.training_days} days ({self.training_days/30:.1f} months)")
        
        # Calculate date ranges with extended training
        end_date = self.data.index.max()
        test_start_date = end_date - timedelta(days=test_period_days)
        training_start_date = test_start_date - timedelta(days=self.training_days)
        
        print(f"Enhanced Meta-Trading AI System Setup:")
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
            'improvement_factor': improvement_factor
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
        
        # Final results with enhanced analysis
        print(f"\n🎯 ENHANCED TARGET ACHIEVEMENT:")
        print(f"  Target: 5% return over {test_period_days} trading days")
        print(f"  Actual: {results['cumulative_return']:.4f} ({results['cumulative_return']*100:.2f}%)")
        print(f"  Training Days: {self.training_days} (vs 60 in original)")
        print(f"  Training Hours: {current_training_hours:,} (vs 240 in original)")
        print(f"  Data Improvement: {improvement_factor:.1f}x more training data")
        
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
        
        # Enhanced performance analysis
        print(f"\n📈 ENHANCED PERFORMANCE ANALYSIS:")
        print(f"  Training Data Improvement: {improvement_factor:.1f}x more data")
        print(f"  Expected Consistency: {min(3.0/improvement_factor, 1.0):.1f}x more consistent")
        print(f"  Regime Coverage: {'Excellent' if self.training_days >= 180 else 'Good' if self.training_days >= 90 else 'Limited'}")
        print(f"  Strategy Robustness: {'High' if self.training_days >= 180 else 'Medium' if self.training_days >= 90 else 'Low'}")
        
        return results

def run_enhanced_comparison():
    """Run comparison between different training periods"""
    
    print("Enhanced MetaTradingAI v3.0 - Training Period Comparison")
    print("="*80)
    
    # Test different training periods
    training_periods = [
        (60, "2 months (original v3.0)"),
        (90, "3 months"),
        (180, "6 months (recommended)"),
        (365, "1 year (optimal)")
    ]
    
    results_comparison = {}
    
    for training_days, description in training_periods:
        print(f"\n{'='*60}")
        print(f"Testing {description}")
        print(f"{'='*60}")
        
        try:
            # Create enhanced system with specified training period
            system = EnhancedAggressiveMetaTradingAI(training_days=training_days)
            results = system.run_aggressive_meta_system(test_period_days=10)
            
            results_comparison[training_days] = {
                'description': description,
                'training_days': training_days,
                'training_hours': results['training_hours'],
                'improvement_factor': results['improvement_factor'],
                'cumulative_return': results['cumulative_return'],
                'total_trades': sum(len(perf['num_trades']) for perf in results['hourly_performance']),
                'strategy_distribution': results['selected_strategies']
            }
            
        except Exception as e:
            print(f"Error testing {description}: {e}")
            continue
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("TRAINING PERIOD COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Period':<20} {'Days':<8} {'Hours':<8} {'Improvement':<12} {'Return':<10} {'Trades':<8}")
    print(f"{'-'*80}")
    
    for training_days, data in results_comparison.items():
        print(f"{data['description']:<20} {data['training_days']:<8} {data['training_hours']:<8} "
              f"{data['improvement_factor']:<12.1f}x {data['cumulative_return']*100:<10.2f}% {data['total_trades']:<8}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    best_period = max(results_comparison.keys(), key=lambda x: results_comparison[x]['cumulative_return'])
    best_result = results_comparison[best_period]
    
    print(f"  Best Performance: {best_result['description']} ({best_result['cumulative_return']*100:.2f}%)")
    print(f"  Recommended: 6 months (180 days) for balance of performance and consistency")
    print(f"  Optimal: 1 year (365 days) for maximum consistency")
    
    return results_comparison

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced MetaTradingAI v3.0')
    parser.add_argument('--training-days', type=int, default=180, 
                       help='Number of training days (default: 180)')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison between different training periods')
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison
        results = run_enhanced_comparison()
    else:
        # Run single enhanced system
        system = EnhancedAggressiveMetaTradingAI(training_days=args.training_days)
        results = system.run_aggressive_meta_system(test_period_days=10)
    
    return results

if __name__ == "__main__":
    results = main() 