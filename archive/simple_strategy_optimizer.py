import pickle
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleTradingStrategy:
    """Simplified trading strategy base class"""
    
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
        
        for i in range(1, len(data)):
            current_price = data.iloc[i]['close']
            prev_price = data.iloc[i-1]['close']
            
            # Update position based on signals
            if signals.iloc[i] == 1 and position <= 0:  # Buy signal
                position = 1
            elif signals.iloc[i] == -1 and position >= 0:  # Sell signal
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
            
        # Calculate max drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_series
        }


class SimpleMACrossover(SimpleTradingStrategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self):
        super().__init__("Simple MA Crossover")
        self.parameters = {
            'fast_period': 10,
            'slow_period': 50
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_ma = data['close'].rolling(self.parameters['fast_period']).mean()
        slow_ma = data['close'].rolling(self.parameters['slow_period']).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1  # Buy when fast MA > slow MA
        signals[fast_ma < slow_ma] = -1  # Sell when fast MA < slow MA
        
        return signals


class SimpleRSI(SimpleTradingStrategy):
    """Simple RSI Strategy"""
    
    def __init__(self):
        super().__init__("Simple RSI")
        self.parameters = {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.parameters['oversold']] = 1  # Buy when oversold
        signals[rsi > self.parameters['overbought']] = -1  # Sell when overbought
        
        return signals


class SimpleBollingerBands(SimpleTradingStrategy):
    """Simple Bollinger Bands Strategy"""
    
    def __init__(self):
        super().__init__("Simple Bollinger Bands")
        self.parameters = {
            'period': 20,
            'std_dev': 2.0
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate Bollinger Bands
        sma = data['close'].rolling(window=self.parameters['period']).mean()
        std = data['close'].rolling(window=self.parameters['period']).std()
        upper_band = sma + (self.parameters['std_dev'] * std)
        lower_band = sma - (self.parameters['std_dev'] * std)
        
        signals = pd.Series(0, index=data.index)
        signals[data['close'] < lower_band] = 1  # Buy when below lower band
        signals[data['close'] > upper_band] = -1  # Sell when above upper band
        
        return signals


class SimpleOptimizer:
    """Simple strategy optimizer"""
    
    def __init__(self, data: pd.DataFrame, strategies: List[SimpleTradingStrategy]):
        self.data = data
        self.strategies = strategies
        
    def optimize_strategy(self, strategy: SimpleTradingStrategy, 
                         optimization_window: int = 1000) -> Dict:
        """Optimize a single strategy's parameters"""
        
        # Use recent data for optimization
        recent_data = self.data.tail(optimization_window)
        
        def objective_function(params):
            """Objective function to minimize (negative Sharpe ratio)"""
            # Round integer parameters
            param_names = list(strategy.parameters.keys())
            rounded_params = []
            for i, param_name in enumerate(param_names):
                if 'period' in param_name.lower():
                    rounded_params.append(int(round(params[i])))
                else:
                    rounded_params.append(params[i])
            
            strategy.set_parameters(dict(zip(param_names, rounded_params)))
            results = strategy.backtest(recent_data)
            return -results['sharpe_ratio']  # Minimize negative Sharpe ratio
        
        # Parameter bounds
        param_bounds = self._get_parameter_bounds(strategy)
        
        # Use L-BFGS-B for faster optimization
        result = minimize(
            objective_function,
            x0=[(b[0] + b[1])/2 for b in param_bounds],
            bounds=param_bounds,
            method='L-BFGS-B',
            options={'maxiter': 50}
        )
        
        # Set optimal parameters and run final backtest
        param_names = list(strategy.parameters.keys())
        optimal_params = []
        for i, param_name in enumerate(param_names):
            if 'period' in param_name.lower():
                optimal_params.append(int(round(result.x[i])))
            else:
                optimal_params.append(result.x[i])
        
        strategy.set_parameters(dict(zip(param_names, optimal_params)))
        final_results = strategy.backtest(recent_data)
        
        return {
            'optimal_parameters': dict(zip(param_names, optimal_params)),
            'optimization_success': result.success,
            'final_metrics': final_results
        }
    
    def _get_parameter_bounds(self, strategy: SimpleTradingStrategy) -> List[Tuple]:
        """Get parameter bounds for optimization"""
        bounds_map = {
            'Simple MA Crossover': [
                (5, 30),    # fast_period
                (20, 100)   # slow_period
            ],
            'Simple RSI': [
                (5, 30),    # rsi_period
                (20, 40),   # oversold
                (60, 80)    # overbought
            ],
            'Simple Bollinger Bands': [
                (10, 50),   # period
                (1.5, 3.0)  # std_dev
            ]
        }
        
        return bounds_map.get(strategy.name, [(0, 100)] * len(strategy.parameters))
    
    def optimize_all_strategies(self, optimization_window: int = 1000) -> Dict:
        """Optimize all strategies and return results"""
        results = {}
        
        for strategy in self.strategies:
            print(f"Optimizing {strategy.name}...")
            results[strategy.name] = self.optimize_strategy(strategy, optimization_window)
            
        return results


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
        SimpleMACrossover(),
        SimpleRSI(),
        SimpleBollingerBands()
    ]
    
    # Create optimizer
    optimizer = SimpleOptimizer(data, strategies)
    
    # Optimize all strategies
    print("\nOptimizing strategies...")
    optimization_results = optimizer.optimize_all_strategies(optimization_window=1000)
    
    # Display results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    for strategy_name, result in optimization_results.items():
        print(f"\n{strategy_name}:")
        print(f"  Optimal Parameters: {result['optimal_parameters']}")
        print(f"  Optimization Success: {result['optimization_success']}")
        metrics = result['final_metrics']
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Total Return: {metrics['total_return']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.3f}")
    
    # Save results
    results = {
        'optimization_results': optimization_results,
        'data_info': {
            'shape': data.shape,
            'date_range': (data.index.min(), data.index.max())
        }
    }
    
    with open('simple_strategy_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to 'simple_strategy_results.pkl'")
    
    return results


if __name__ == "__main__":
    results = main() 