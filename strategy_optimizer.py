import pickle
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TradingStrategy:
    """Base class for algorithmic trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.positions = []
        self.equity_curve = []
        
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
                
            # Calculate daily returns
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
        max_drawdown = self._calculate_max_drawdown(equity_series)
        if trades:
            win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades)
        else:
            win_rate = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'equity_curve': equity_series,
            'trades': trades
        }
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return drawdown.min()


class MovingAverageCrossover(TradingStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self):
        super().__init__("Moving Average Crossover")
        self.parameters = {
            'fast_period': 10,
            'slow_period': 50,
            'min_hold_period': 5
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_ma = data['close'].rolling(self.parameters['fast_period']).mean()
        slow_ma = data['close'].rolling(self.parameters['slow_period']).mean()
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals
        signals[fast_ma > slow_ma] = 1  # Buy when fast MA > slow MA
        signals[fast_ma < slow_ma] = -1  # Sell when fast MA < slow MA
        
        # Apply minimum hold period
        for i in range(self.parameters['min_hold_period'], len(signals)):
            if signals.iloc[i] != 0:
                # Check if we should hold the position
                if signals.iloc[i] == signals.iloc[i-1]:
                    continue
                else:
                    # Check minimum hold period
                    last_signal_change = i - 1
                    while last_signal_change >= 0 and signals.iloc[last_signal_change] == signals.iloc[i-1]:
                        last_signal_change -= 1
                    
                    if i - last_signal_change < self.parameters['min_hold_period']:
                        signals.iloc[i] = signals.iloc[i-1]
        
        return signals


class RSIStrategy(TradingStrategy):
    """RSI-based Mean Reversion Strategy"""
    
    def __init__(self):
        super().__init__("RSI Strategy")
        self.parameters = {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'exit_threshold': 50
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals
        signals[rsi < self.parameters['oversold']] = 1  # Buy when oversold
        signals[rsi > self.parameters['overbought']] = -1  # Sell when overbought
        signals[(rsi >= self.parameters['exit_threshold']) & (rsi <= self.parameters['exit_threshold'])] = 0  # Exit
        
        return signals


class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands Mean Reversion Strategy"""
    
    def __init__(self):
        super().__init__("Bollinger Bands Strategy")
        self.parameters = {
            'period': 20,
            'std_dev': 2.0,
            'exit_threshold': 0.5
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate Bollinger Bands
        sma = data['close'].rolling(window=self.parameters['period']).mean()
        std = data['close'].rolling(window=self.parameters['period']).std()
        upper_band = sma + (self.parameters['std_dev'] * std)
        lower_band = sma - (self.parameters['std_dev'] * std)
        
        # Calculate position within bands (0 = at lower band, 1 = at upper band)
        position_in_bands = (data['close'] - lower_band) / (upper_band - lower_band)
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals
        signals[position_in_bands < 0.1] = 1  # Buy when near lower band
        signals[position_in_bands > 0.9] = -1  # Sell when near upper band
        signals[(position_in_bands >= self.parameters['exit_threshold']) & 
               (position_in_bands <= (1 - self.parameters['exit_threshold']))] = 0  # Exit
        
        return signals


class MomentumStrategy(TradingStrategy):
    """Momentum-based Strategy"""
    
    def __init__(self):
        super().__init__("Momentum Strategy")
        self.parameters = {
            'lookback_period': 20,
            'momentum_threshold': 0.02,
            'volatility_threshold': 0.015
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate momentum
        momentum = data['close'].pct_change(self.parameters['lookback_period'])
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(self.parameters['lookback_period']).std()
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals
        signals[(momentum > self.parameters['momentum_threshold']) & 
               (volatility < self.parameters['volatility_threshold'])] = 1  # Buy on positive momentum
        
        signals[(momentum < -self.parameters['momentum_threshold']) & 
               (volatility < self.parameters['volatility_threshold'])] = -1  # Sell on negative momentum
        
        return signals


class StrategyOptimizer:
    """Optimize strategy parameters using recent market data"""
    
    def __init__(self, data: pd.DataFrame, strategies: List[TradingStrategy]):
        self.data = data
        self.strategies = strategies
        self.optimization_results = {}
        
    def optimize_strategy(self, strategy: TradingStrategy, 
                         optimization_window: int = 252,  # 1 year of trading days
                         optimization_method: str = 'differential_evolution') -> Dict:
        """Optimize a single strategy's parameters"""
        
        # Use recent data for optimization
        recent_data = self.data.tail(optimization_window)
        
        def objective_function(params):
            """Objective function to minimize (negative Sharpe ratio)"""
            # Round integer parameters
            rounded_params = self._round_integer_parameters(strategy, params)
            strategy.set_parameters(dict(zip(strategy.parameters.keys(), rounded_params)))
            results = strategy.backtest(recent_data)
            return -results['sharpe_ratio']  # Minimize negative Sharpe ratio
        
        # Parameter bounds
        param_bounds = self._get_parameter_bounds(strategy)
        
        if optimization_method == 'differential_evolution':
            result = differential_evolution(
                objective_function, 
                param_bounds,
                maxiter=100,
                popsize=15,
                seed=42
            )
        else:
            result = minimize(
                objective_function,
                x0=[(b[0] + b[1])/2 for b in param_bounds],
                bounds=param_bounds,
                method='L-BFGS-B'
            )
        
        # Set optimal parameters and run final backtest
        optimal_params = self._round_integer_parameters(strategy, result.x)
        strategy.set_parameters(dict(zip(strategy.parameters.keys(), optimal_params)))
        final_results = strategy.backtest(recent_data)
        
        return {
            'optimal_parameters': dict(zip(strategy.parameters.keys(), result.x)),
            'optimization_success': result.success,
            'final_metrics': final_results
        }
    
    def _get_parameter_bounds(self, strategy: TradingStrategy) -> List[Tuple]:
        """Get parameter bounds for optimization"""
        bounds_map = {
            'Moving Average Crossover': [
                (5, 50),    # fast_period
                (20, 200),  # slow_period
                (1, 20)     # min_hold_period
            ],
            'RSI Strategy': [
                (5, 30),    # rsi_period
                (20, 40),   # oversold
                (60, 80),   # overbought
                (40, 60)    # exit_threshold
            ],
            'Bollinger Bands Strategy': [
                (10, 50),   # period
                (1.5, 3.0), # std_dev
                (0.1, 0.9)  # exit_threshold
            ],
            'Momentum Strategy': [
                (5, 50),    # lookback_period
                (0.01, 0.05), # momentum_threshold
                (0.005, 0.03) # volatility_threshold
            ]
        }
        
        return bounds_map.get(strategy.name, [(0, 100)] * len(strategy.parameters))
    
    def _round_integer_parameters(self, strategy: TradingStrategy, params: List[float]) -> List[float]:
        """Round parameters that should be integers"""
        param_names = list(strategy.parameters.keys())
        rounded_params = []
        
        for i, param_name in enumerate(param_names):
            if 'period' in param_name.lower() or 'hold' in param_name.lower():
                rounded_params.append(int(round(params[i])))
            else:
                rounded_params.append(params[i])
        
        return rounded_params
    
    def optimize_all_strategies(self, optimization_window: int = 252) -> Dict:
        """Optimize all strategies and return results"""
        results = {}
        
        for strategy in self.strategies:
            print(f"Optimizing {strategy.name}...")
            results[strategy.name] = self.optimize_strategy(strategy, optimization_window)
            
        return results
    
    def compare_strategies(self, optimization_results: Dict, 
                          test_window: int = 63) -> pd.DataFrame:
        """Compare optimized strategies on out-of-sample data"""
        
        # Use data before the optimization window for testing
        test_data = self.data.iloc[:-test_window]
        
        comparison_results = []
        
        for strategy_name, opt_result in optimization_results.items():
            strategy = next(s for s in self.strategies if s.name == strategy_name)
            
            # Run backtest with optimized parameters
            test_results = strategy.backtest(test_data)
            
            comparison_results.append({
                'Strategy': strategy_name,
                'Total Return': test_results['total_return'],
                'Sharpe Ratio': test_results['sharpe_ratio'],
                'Max Drawdown': test_results['max_drawdown'],
                'Win Rate': test_results['win_rate'],
                'Num Trades': test_results['num_trades']
            })
        
        return pd.DataFrame(comparison_results)


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load and prepare the QQQ data"""
    print("Loading data...")
    data = pickle.load(open(file_path, 'rb'))
    
    # Convert index to datetime if needed, handling timezone issues
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except ValueError:
            # Handle timezone-aware datetime conversion
            data.index = pd.to_datetime(data.index, utc=True)
    
    # Convert to timezone-naive if needed
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    
    # Filter to trading hours (9:30 AM - 4:00 PM ET)
    data = data.between_time('09:30', '16:00')
    
    # Remove weekends
    data = data[data.index.dayofweek < 5]
    
    print(f"Data shape after filtering: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data


def main():
    """Main execution function"""
    
    # Load data
    data = load_and_prepare_data('polygon_QQQ_1m.pkl')
    
    # Define strategies
    strategies = [
        MovingAverageCrossover(),
        RSIStrategy(),
        BollingerBandsStrategy(),
        MomentumStrategy()
    ]
    
    # Create optimizer
    optimizer = StrategyOptimizer(data, strategies)
    
    # Optimize all strategies
    print("\nOptimizing strategies...")
    optimization_results = optimizer.optimize_all_strategies(optimization_window=252)
    
    # Compare strategies
    print("\nComparing strategies on out-of-sample data...")
    comparison_df = optimizer.compare_strategies(optimization_results, test_window=63)
    
    # Display results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    for strategy_name, result in optimization_results.items():
        print(f"\n{strategy_name}:")
        print(f"  Optimal Parameters: {result['optimal_parameters']}")
        print(f"  Sharpe Ratio: {result['final_metrics']['sharpe_ratio']:.3f}")
        print(f"  Total Return: {result['final_metrics']['total_return']:.3f}")
        print(f"  Max Drawdown: {result['final_metrics']['max_drawdown']:.3f}")
    
    print("\n" + "="*80)
    print("OUT-OF-SAMPLE COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save results
    results = {
        'optimization_results': optimization_results,
        'comparison_df': comparison_df,
        'data_info': {
            'shape': data.shape,
            'date_range': (data.index.min(), data.index.max())
        }
    }
    
    with open('strategy_optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to 'strategy_optimization_results.pkl'")
    
    return results


if __name__ == "__main__":
    results = main() 