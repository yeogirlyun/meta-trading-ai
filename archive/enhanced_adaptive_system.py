import pickle
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


class EnhancedAdaptiveStrategy:
    """Enhanced adaptive strategy with more aggressive parameters"""
    
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
            
        # Calculate max drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate daily returns for target analysis
        daily_data = data.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        daily_equity = equity_series.resample('D').last()
        daily_returns = daily_equity.pct_change().dropna()
        
        # Target metrics
        avg_daily_return = daily_returns.mean() if len(daily_returns) > 0 else 0
        target_daily_05 = len(daily_returns[daily_returns >= 0.005]) / len(daily_returns) if len(daily_returns) > 0 else 0
        
        # 10-day rolling returns
        rolling_10d = daily_equity.pct_change(10).dropna()
        target_10d_05 = len(rolling_10d[rolling_10d >= 0.05]) / len(rolling_10d) if len(rolling_10d) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_series,
            'trades': trades,
            'avg_daily_return': avg_daily_return,
            'target_daily_05_pct': target_daily_05,
            'target_10d_05_pct': target_10d_05,
            'num_trades': len(trades)
        }


class EnhancedVolatilityExploitation(EnhancedAdaptiveStrategy):
    """Enhanced volatility exploitation with more aggressive parameters"""
    
    def __init__(self):
        super().__init__("Enhanced Volatility Exploitation")
        self.parameters = {
            'volatility_threshold': 0.01,  # Lower threshold for more signals
            'momentum_period': 3,          # Shorter period for faster response
            'reversal_threshold': 0.01,    # Lower threshold for more signals
            'max_hold_period': 20,         # Shorter hold period
            'volume_threshold': 1.2        # Lower volume requirement
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(10).std()  # Shorter window
        
        # Calculate momentum
        momentum = data['close'].pct_change(self.parameters['momentum_period'])
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()  # Shorter window
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Entry conditions - more aggressive
        high_vol = volatility > self.parameters['volatility_threshold']
        strong_momentum = abs(momentum) > self.parameters['reversal_threshold']
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


class EnhancedMomentumAmplification(EnhancedAdaptiveStrategy):
    """Enhanced momentum amplification with more aggressive parameters"""
    
    def __init__(self):
        super().__init__("Enhanced Momentum Amplification")
        self.parameters = {
            'short_period': 2,            # Very short period
            'medium_period': 5,           # Shorter medium period
            'long_period': 15,            # Shorter long period
            'momentum_threshold': 0.005,  # Lower threshold
            'volume_threshold': 1.2,      # Lower volume requirement
            'acceleration_threshold': 0.002,  # Lower acceleration threshold
            'max_hold_period': 30         # Shorter hold period
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate multiple momentum indicators
        short_momentum = data['close'].pct_change(self.parameters['short_period'])
        medium_momentum = data['close'].pct_change(self.parameters['medium_period'])
        long_momentum = data['close'].pct_change(self.parameters['long_period'])
        
        # Calculate acceleration
        acceleration = short_momentum.diff()
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()  # Shorter window
        volume_ratio = data['volume'] / avg_volume
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(5).std()  # Shorter window
        
        signals = pd.Series(0, index=data.index)
        
        # Strong buy when all momentum indicators align - more aggressive
        strong_buy = (
            (short_momentum > self.parameters['momentum_threshold']) &
            (medium_momentum > self.parameters['momentum_threshold']) &
            (long_momentum > 0) &
            (acceleration > self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(20).mean() * 2)  # More permissive
        )
        
        # Strong sell when all momentum indicators align negatively
        strong_sell = (
            (short_momentum < -self.parameters['momentum_threshold']) &
            (medium_momentum < -self.parameters['momentum_threshold']) &
            (long_momentum < 0) &
            (acceleration < -self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(20).mean() * 2)
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


class EnhancedGapExploitation(EnhancedAdaptiveStrategy):
    """Enhanced gap exploitation with more aggressive parameters"""
    
    def __init__(self):
        super().__init__("Enhanced Gap Exploitation")
        self.parameters = {
            'gap_threshold': 0.003,       # Lower gap threshold
            'fade_threshold': 0.005,      # Lower fade threshold
            'momentum_threshold': 0.001,  # Lower momentum threshold
            'volume_threshold': 1.2,      # Lower volume requirement
            'max_hold_period': 30         # Shorter hold period
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
        momentum = data['close'].pct_change(3)  # Shorter period
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(10).mean()  # Shorter window
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Gap fade strategy - more aggressive
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


class EnhancedAdaptiveOptimizer:
    """Enhanced adaptive optimizer with progress tracking"""
    
    def __init__(self, data: pd.DataFrame, strategies: List[EnhancedAdaptiveStrategy]):
        self.data = data
        self.strategies = strategies
        self.start_time = None
        
    def optimize_strategy(self, strategy: EnhancedAdaptiveStrategy, 
                         training_data: pd.DataFrame) -> Dict:
        """Optimize a single strategy's parameters with progress tracking"""
        
        def objective_function(params):
            """Objective function to maximize target metrics"""
            # Round integer parameters
            param_names = list(strategy.parameters.keys())
            rounded_params = []
            for i, param_name in enumerate(param_names):
                if 'period' in param_name.lower() or 'hold' in param_name.lower() or 'minutes' in param_name.lower():
                    rounded_params.append(int(round(params[i])))
                else:
                    rounded_params.append(params[i])
            
            strategy.set_parameters(dict(zip(param_names, rounded_params)))
            results = strategy.backtest(training_data)
            
            # More aggressive objective function
            target_score = (
                results['avg_daily_return'] * 200 +  # Higher weight on daily returns
                results['target_daily_05_pct'] * 100 +  # Higher weight on target achievement
                results['target_10d_05_pct'] * 100 +  # Higher weight on 10-day target
                min(results['sharpe_ratio'], 3.0) * 5 +  # Lower weight on Sharpe
                results['num_trades'] * 0.1  # Bonus for more trades
            )
            
            return -target_score  # Minimize negative score
        
        # More aggressive parameter bounds
        param_bounds = self._get_parameter_bounds(strategy)
        
        # Use differential evolution for better global optimization
        result = differential_evolution(
            objective_function,
            param_bounds,
            maxiter=50,
            popsize=10,
            seed=42
        )
        
        # Set optimal parameters
        param_names = list(strategy.parameters.keys())
        optimal_params = []
        for i, param_name in enumerate(param_names):
            if 'period' in param_name.lower() or 'hold' in param_name.lower() or 'minutes' in param_name.lower():
                optimal_params.append(int(round(result.x[i])))
            else:
                optimal_params.append(result.x[i])
        
        strategy.set_parameters(dict(zip(param_names, optimal_params)))
        
        return {
            'optimal_parameters': dict(zip(param_names, optimal_params)),
            'optimization_success': result.success,
            'optimization_score': -result.fun
        }
    
    def _get_parameter_bounds(self, strategy: EnhancedAdaptiveStrategy) -> List[Tuple]:
        """Get more aggressive parameter bounds"""
        bounds_map = {
            'Enhanced Volatility Exploitation': [
                (0.005, 0.02),  # volatility_threshold - lower range
                (2, 8),         # momentum_period - shorter range
                (0.005, 0.02),  # reversal_threshold - lower range
                (10, 40),       # max_hold_period - shorter range
                (1.0, 2.0)      # volume_threshold - lower range
            ],
            'Enhanced Momentum Amplification': [
                (1, 5),         # short_period - shorter range
                (3, 10),        # medium_period - shorter range
                (10, 25),       # long_period - shorter range
                (0.002, 0.01),  # momentum_threshold - lower range
                (1.0, 2.0),     # volume_threshold - lower range
                (0.001, 0.005), # acceleration_threshold - lower range
                (15, 60)        # max_hold_period - shorter range
            ],
            'Enhanced Gap Exploitation': [
                (0.002, 0.008), # gap_threshold - lower range
                (0.003, 0.01),  # fade_threshold - lower range
                (0.0005, 0.003), # momentum_threshold - lower range
                (1.0, 2.0),     # volume_threshold - lower range
                (15, 60)        # max_hold_period - shorter range
            ]
        }
        
        return bounds_map.get(strategy.name, [(0, 100)] * len(strategy.parameters))
    
    def walk_forward_test(self, test_period_days: int = 30, 
                         training_period_days: int = 30,
                         reoptimization_frequency: str = 'daily') -> Dict:
        """Walk-forward testing with enhanced optimization and progress tracking"""
        
        # Get the most recent data for testing
        test_end_date = self.data.index.max()
        test_start_date = test_end_date - timedelta(days=test_period_days)
        training_end_date = test_start_date - timedelta(days=1)
        training_start_date = training_end_date - timedelta(days=training_period_days)
        
        print(f"Enhanced walk-forward test setup:")
        print(f"  Training period: {training_start_date} to {training_end_date}")
        print(f"  Test period: {test_start_date} to {test_end_date}")
        print(f"  Reoptimization frequency: {reoptimization_frequency}")
        
        # Get training and test data
        training_data = self.data[(self.data.index >= training_start_date) & 
                                (self.data.index <= training_end_date)]
        test_data = self.data[(self.data.index >= test_start_date) & 
                             (self.data.index <= test_end_date)]
        
        print(f"  Training data: {len(training_data)} records")
        print(f"  Test data: {len(test_data)} records")
        
        # Calculate total iterations for progress tracking
        total_iterations = len(self.strategies) + (test_period_days * len(self.strategies))
        current_iteration = 0
        
        # Start timing
        self.start_time = time.time()
        
        # Initial optimization on training data
        print(f"\nInitial optimization on training data...")
        initial_results = {}
        for i, strategy in enumerate(self.strategies):
            current_iteration += 1
            progress = (current_iteration / total_iterations) * 100
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                estimated_total_time = (elapsed_time / current_iteration) * total_iterations
                remaining_time = estimated_total_time - elapsed_time
                print(f"  [{progress:.1f}%] Optimizing {strategy.name}... (Est. {remaining_time:.0f}s remaining)")
            else:
                print(f"  [{progress:.1f}%] Optimizing {strategy.name}...")
            
            initial_results[strategy.name] = self.optimize_strategy(strategy, training_data)
        
        # Walk-forward testing
        results = {
            'daily_performance': [],
            'strategy_performance': {},
            'reoptimization_history': {}
        }
        
        # Initialize strategy performance tracking
        for strategy in self.strategies:
            results['strategy_performance'][strategy.name] = {
                'equity_curve': [],
                'daily_returns': [],
                'trades': [],
                'parameters_history': []
            }
        
        # Daily walk-forward testing
        current_date = test_start_date
        day_count = 0
        while current_date <= test_end_date:
            next_date = current_date + timedelta(days=1)
            
            # Get current day's data
            day_data = test_data[(test_data.index >= current_date) & 
                               (test_data.index < next_date)]
            
            if len(day_data) > 0:
                day_count += 1
                current_iteration += len(self.strategies)
                progress = (current_iteration / total_iterations) * 100
                elapsed_time = time.time() - self.start_time
                
                if elapsed_time > 0:
                    estimated_total_time = (elapsed_time / current_iteration) * total_iterations
                    remaining_time = estimated_total_time - elapsed_time
                    print(f"\n[{progress:.1f}%] Testing {current_date.strftime('%Y-%m-%d')}... (Est. {remaining_time:.0f}s remaining)")
                else:
                    print(f"\n[{progress:.1f}%] Testing {current_date.strftime('%Y-%m-%d')}...")
                
                # Test each strategy with current parameters
                for strategy in self.strategies:
                    strategy_results = strategy.backtest(day_data)
                    
                    # Store results
                    results['strategy_performance'][strategy.name]['equity_curve'].append(
                        strategy_results['equity_curve'].iloc[-1] if len(strategy_results['equity_curve']) > 0 else 100000
                    )
                    results['strategy_performance'][strategy.name]['daily_returns'].append(
                        strategy_results['avg_daily_return']
                    )
                    results['strategy_performance'][strategy.name]['trades'].extend(
                        strategy_results['trades']
                    )
                    results['strategy_performance'][strategy.name]['parameters_history'].append(
                        strategy.parameters.copy()
                    )
                
                # Reoptimize if needed (daily)
                if reoptimization_frequency == 'daily':
                    # Add current day to training data
                    updated_training_data = pd.concat([training_data, day_data])
                    
                    # Reoptimize strategies
                    print(f"  Reoptimizing strategies...")
                    for strategy in self.strategies:
                        reopt_result = self.optimize_strategy(strategy, updated_training_data)
                        results['reoptimization_history'][f"{strategy.name}_{current_date.strftime('%Y-%m-%d')}"] = reopt_result
                
                # Store daily performance
                daily_perf = {
                    'date': current_date,
                    'strategies': {}
                }
                
                for strategy in self.strategies:
                    daily_perf['strategies'][strategy.name] = {
                        'daily_return': strategy_results['avg_daily_return'],
                        'total_return': strategy_results['total_return'],
                        'sharpe_ratio': strategy_results['sharpe_ratio'],
                        'num_trades': strategy_results['num_trades']
                    }
                
                results['daily_performance'].append(daily_perf)
            
            current_date = next_date
        
        # Final progress update
        total_time = time.time() - self.start_time
        print(f"\n[100.0%] Walk-forward testing completed in {total_time:.1f} seconds")
        
        return results


def analyze_enhanced_results(results: Dict) -> None:
    """Analyze and display enhanced walk-forward testing results"""
    
    print("\n" + "="*80)
    print("ENHANCED WALK-FORWARD TESTING RESULTS")
    print("="*80)
    
    # Calculate overall performance for each strategy
    for strategy_name, performance in results['strategy_performance'].items():
        print(f"\n{strategy_name}:")
        
        # Calculate cumulative performance
        if len(performance['equity_curve']) > 0:
            initial_capital = 100000
            final_capital = performance['equity_curve'][-1]
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Calculate daily returns
            daily_returns = performance['daily_returns']
            avg_daily_return = np.mean(daily_returns) if daily_returns else 0
            target_achievement = len([r for r in daily_returns if r >= 0.005]) / len(daily_returns) if daily_returns else 0
            
            # Calculate Sharpe ratio
            if len(daily_returns) > 1:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            print(f"  Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
            print(f"  Average Daily Return: {avg_daily_return:.4f} ({avg_daily_return*100:.2f}%)")
            print(f"  Target 0.5% Achievement: {target_achievement:.2f}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"  Total Trades: {len(performance['trades'])}")
            
            # Show parameter evolution
            if len(performance['parameters_history']) > 0:
                print(f"  Parameter Evolution:")
                for i, params in enumerate(performance['parameters_history'][:3]):  # Show first 3
                    print(f"    Day {i+1}: {params}")
    
    # Show daily performance summary
    print(f"\nDaily Performance Summary:")
    for daily_perf in results['daily_performance']:
        date = daily_perf['date']
        print(f"  {date.strftime('%Y-%m-%d')}:")
        for strategy_name, perf in daily_perf['strategies'].items():
            print(f"    {strategy_name}: {perf['daily_return']:.4f} ({perf['daily_return']*100:.2f}%)")


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
    
    # Define enhanced adaptive strategies
    strategies = [
        EnhancedVolatilityExploitation(),
        EnhancedMomentumAmplification(),
        EnhancedGapExploitation()
    ]
    
    # Create enhanced adaptive optimization system
    enhanced_system = EnhancedAdaptiveOptimizer(data, strategies)
    
    # Run enhanced walk-forward testing
    print("\nRunning enhanced walk-forward testing...")
    results = enhanced_system.walk_forward_test(
        test_period_days=30,  # Last 30 days
        training_period_days=30,  # 30 days before test
        reoptimization_frequency='daily'  # Reoptimize daily
    )
    
    # Analyze results
    analyze_enhanced_results(results)
    
    # Save results
    with open('enhanced_adaptive_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to 'enhanced_adaptive_results.pkl'")
    
    return results


if __name__ == "__main__":
    results = main() 