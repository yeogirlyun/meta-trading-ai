#!/usr/bin/env python3
"""
MetaTradingAI v1.0 - Simple Walk-Forward Testing
Focused testing to evaluate model consistency across multiple periods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')


def load_data():
    """Load QQQ data for testing"""
    print("Loading data for walk-forward testing...")
    data = pd.read_feather('polygon_QQQ_1m.feather')
    print(f"Data loaded: {len(data):,} records from "
          f"{data.index.min()} to {data.index.max()}")
    return data


def run_ultra_aggressive_test(data, train_start, train_end, test_start, test_end):
    """Run ultra-aggressive model test"""
    try:
        from aggressive_meta_trading_ai_v3 import AggressiveMetaTradingAI
        
        # Create model instance
        model = AggressiveMetaTradingAI()
        model.data = data
        
        # Set training and test periods
        model.training_start = train_start
        model.training_end = train_end
        model.test_start = test_start
        model.test_end = test_end
        
        # Run the model
        result = model.run_aggressive_meta_system(test_period_days=10)
        
        return {
            'return': float(result.get('cumulative_return', 0.0)),
            'trades': int(result.get('total_trades', 0)),
            'target_achieved': bool(result.get('target_achieved', False))
        }
    except Exception as e:
        print(f"    Error running ultra-aggressive model: {e}")
        return None


def run_optimized_rt_test(data, train_start, train_end, test_start, test_end):
    """Run optimized RT model test"""
    try:
        from aggressive_meta_trading_ai_v3_optimized import OptimizedAggressiveMetaTradingAI
        
        # Create model instance
        model = OptimizedAggressiveMetaTradingAI()
        model.data = data
        
        # Set training and test periods
        model.training_start = train_start
        model.training_end = train_end
        model.test_start = test_start
        model.test_end = test_end
        
        # Run the model
        result = model.run_optimized_meta_system(test_period_days=10)
        
        return {
            'return': float(result.get('cumulative_return', 0.0)),
            'trades': int(result.get('total_trades', 0)),
            'target_achieved': bool(result.get('target_achieved', False))
        }
    except Exception as e:
        print(f"    Error running optimized RT model: {e}")
        return None


def run_walk_forward_tests(data, num_windows=10, test_days=10, training_days=180, 
                          rolling_step=5, model_type='ultra_aggressive'):
    """Run walk-forward testing for a specific model"""
    
    print(f"\n🚀 Starting Walk-Forward Testing")
    print(f"  Model: {model_type}")
    print(f"  Windows: {num_windows}")
    print(f"  Training Days: {training_days}")
    print(f"  Test Days: {test_days}")
    print(f"  Rolling Step: {rolling_step} days")
    
    results = {
        'model_type': model_type,
        'parameters': {
            'num_windows': num_windows,
            'test_days': test_days,
            'training_days': training_days,
            'rolling_step': rolling_step
        },
        'windows': []
    }
    
    end_date = data.index.max()
    current_end = end_date
    
    successful_windows = 0
    
    for i in range(num_windows):
        test_start = current_end - timedelta(days=test_days)
        train_start = test_start - timedelta(days=training_days)
        
        # Ensure valid dates
        if train_start < data.index.min():
            print(f"  Window {i+1}: Insufficient historical data, stopping")
            break
        
        print(f"\n📊 Window {i+1}/{num_windows}")
        print(f"  Train: {train_start.strftime('%Y-%m-%d')} to "
              f"{test_start.strftime('%Y-%m-%d')}")
        print(f"  Test:  {test_start.strftime('%Y-%m-%d')} to "
              f"{current_end.strftime('%Y-%m-%d')}")
        
        # Run test based on model type
        if model_type == 'ultra_aggressive':
            window_result = run_ultra_aggressive_test(
                data, train_start, test_start, test_start, current_end
            )
        elif model_type == 'optimized_rt':
            window_result = run_optimized_rt_test(
                data, train_start, test_start, test_start, current_end
            )
        else:
            print(f"  ❌ Unknown model type: {model_type}")
            continue
        
        if window_result:
            window_data = {
                'window': i+1,
                'train_start': train_start.isoformat(),
                'train_end': test_start.isoformat(),
                'test_start': test_start.isoformat(),
                'test_end': current_end.isoformat(),
                'return': window_result['return'],
                'trades': window_result['trades'],
                'target_achieved': window_result['target_achieved']
            }
            
            results['windows'].append(window_data)
            successful_windows += 1
            
            print(f"  ✅ Return: {window_result['return']:.4f} "
                  f"({window_result['return']*100:.2f}%)")
            print(f"  📈 Trades: {window_result['trades']}")
            print(f"  🎯 Target: {'✅' if window_result['target_achieved'] else '❌'}")
        else:
            print(f"  ❌ Window failed")
        
        # Step back for next window
        current_end -= timedelta(days=rolling_step)
    
    # Calculate statistics
    if results['windows']:
        returns = [w['return'] for w in results['windows']]
        trades = [w['trades'] for w in results['windows']]
        
        results['statistics'] = {
            'num_windows': len(returns),
            'avg_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
            'median_return': float(np.median(returns)),
            'win_rate_5pct': float(np.mean(np.array(returns) >= 0.05)),
            'win_rate_3pct': float(np.mean(np.array(returns) >= 0.03)),
            'win_rate_positive': float(np.mean(np.array(returns) > 0)),
            'avg_trades': float(np.mean(trades)),
            'total_trades': int(np.sum(trades))
        }
    
    return results


def print_results_summary(results):
    """Print comprehensive results summary"""
    
    if not results.get('statistics'):
        print("❌ No results to summarize")
        return
    
    stats = results['statistics']
    
    print(f"\n🎯 WALK-FORWARD TESTING RESULTS")
    print(f"=" * 60)
    print(f"Model: {results['model_type']}")
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
    
    print(f"\n🔄 TRADING METRICS")
    print(f"  Average Trades/Window: {stats['avg_trades']:.1f}")
    print(f"  Total Trades: {stats['total_trades']}")
    
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


def save_results(results, filename=None):
    """Save results to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"walk_forward_results_{results['model_type']}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    return filename


def main():
    """Main execution function"""
    print("🚀 MetaTradingAI v1.0 - Simple Walk-Forward Testing System")
    print("=" * 60)
    
    # Load data
    data = load_data()
    
    # Test both models
    models = ['ultra_aggressive', 'optimized_rt']
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper().replace('_', ' ')} Model")
        print(f"{'='*60}")
        
        # Run walk-forward tests
        results = run_walk_forward_tests(
            data=data,
            num_windows=10,
            test_days=10,
            training_days=180,
            rolling_step=5,
            model_type=model_type
        )
        
        # Print summary
        print_results_summary(results)
        
        # Save results
        save_results(results)


if __name__ == "__main__":
    main() 