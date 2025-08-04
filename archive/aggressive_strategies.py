import pickle
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AggressiveTradingStrategy:
    """Aggressive trading strategy targeting high returns"""
    
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
        avg_daily_return = daily_returns.mean()
        if len(daily_returns) > 0:
            target_daily_05 = len(daily_returns[daily_returns >= 0.005]) / len(daily_returns)
        else:
            target_daily_05 = 0
        
        # 10-day rolling returns
        rolling_10d = daily_equity.pct_change(10).dropna()
        if len(rolling_10d) > 0:
            target_10d_05 = len(rolling_10d[rolling_10d >= 0.05]) / len(rolling_10d)
        else:
            target_10d_05 = 0
        
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


class VolatilityExploitationStrategy(AggressiveTradingStrategy):
    """Strategy that exploits volatility spikes for high returns"""
    
    def __init__(self):
        super().__init__("Volatility Exploitation")
        self.parameters = {
            'volatility_threshold': 0.015,
            'momentum_period': 5,
            'reversal_threshold': 0.02,
            'max_hold_period': 30
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(20).std()
        
        # Calculate momentum
        momentum = data['close'].pct_change(self.parameters['momentum_period'])
        
        # Calculate price acceleration
        acceleration = momentum.diff()
        
        # Calculate reversal signals
        high_vol = volatility > self.parameters['volatility_threshold']
        strong_momentum = abs(momentum) > self.parameters['reversal_threshold']
        
        signals = pd.Series(0, index=data.index)
        
        # Enter long on strong positive momentum during high volatility
        long_condition = (momentum > self.parameters['reversal_threshold']) & high_vol
        signals[long_condition] = 1
        
        # Enter short on strong negative momentum during high volatility
        short_condition = (momentum < -self.parameters['reversal_threshold']) & high_vol
        signals[short_condition] = -1
        
        # Exit after max hold period
        for i in range(self.parameters['max_hold_period'], len(signals)):
            if signals.iloc[i] != 0:
                # Check if we should exit
                if i - self.parameters['max_hold_period'] >= 0:
                    last_signal = signals.iloc[i - self.parameters['max_hold_period']]
                    if last_signal != 0:
                        signals.iloc[i] = 0  # Exit position
        
        return signals


class MomentumAmplificationStrategy(AggressiveTradingStrategy):
    """Strategy that amplifies momentum moves for maximum gains"""
    
    def __init__(self):
        super().__init__("Momentum Amplification")
        self.parameters = {
            'short_period': 3,
            'medium_period': 10,
            'long_period': 30,
            'momentum_threshold': 0.01,
            'volume_threshold': 1.5,
            'acceleration_threshold': 0.005
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate multiple momentum indicators
        short_momentum = data['close'].pct_change(self.parameters['short_period'])
        medium_momentum = data['close'].pct_change(self.parameters['medium_period'])
        long_momentum = data['close'].pct_change(self.parameters['long_period'])
        
        # Calculate acceleration
        acceleration = short_momentum.diff()
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(20).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(10).std()
        
        signals = pd.Series(0, index=data.index)
        
        # Strong buy when all momentum indicators align and volume confirms
        strong_buy = (
            (short_momentum > self.parameters['momentum_threshold']) &
            (medium_momentum > self.parameters['momentum_threshold']) &
            (long_momentum > 0) &
            (acceleration > self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(50).mean() * 1.5)  # Not too volatile
        )
        
        # Strong sell when all momentum indicators align negatively
        strong_sell = (
            (short_momentum < -self.parameters['momentum_threshold']) &
            (medium_momentum < -self.parameters['momentum_threshold']) &
            (long_momentum < 0) &
            (acceleration < -self.parameters['acceleration_threshold']) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(50).mean() * 1.5)
        )
        
        signals[strong_buy] = 1
        signals[strong_sell] = -1
        
        return signals


class HighFrequencyScalpingStrategy(AggressiveTradingStrategy):
    """High-frequency scalping strategy for capturing small but frequent gains"""
    
    def __init__(self):
        super().__init__("High-Frequency Scalping")
        self.parameters = {
            'entry_threshold': 0.001,
            'exit_threshold': 0.002,
            'stop_loss': 0.003,
            'max_hold_minutes': 15,
            'volume_threshold': 2.0
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate 1-minute returns
        returns = data['close'].pct_change()
        
        # Calculate volume spikes
        avg_volume = data['volume'].rolling(20).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Calculate price acceleration
        acceleration = returns.diff()
        
        # Calculate volatility
        volatility = returns.rolling(10).std()
        
        signals = pd.Series(0, index=data.index)
        
        # Entry conditions for scalping
        entry_long = (
            (returns > self.parameters['entry_threshold']) &
            (acceleration > 0) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(50).mean() * 2)  # Not extreme volatility
        )
        
        entry_short = (
            (returns < -self.parameters['entry_threshold']) &
            (acceleration < 0) &
            (volume_ratio > self.parameters['volume_threshold']) &
            (volatility < volatility.rolling(50).mean() * 2)
        )
        
        signals[entry_long] = 1
        signals[entry_short] = -1
        
        # Exit logic - implement trailing stops and time-based exits
        for i in range(self.parameters['max_hold_minutes'], len(signals)):
            if signals.iloc[i] != 0:
                # Check if we should exit based on time or profit target
                entry_price = data.iloc[i - self.parameters['max_hold_minutes']]['close']
                current_price = data.iloc[i]['close']
                
                if signals.iloc[i] == 1:  # Long position
                    profit_pct = (current_price - entry_price) / entry_price
                    if profit_pct >= self.parameters['exit_threshold'] or profit_pct <= -self.parameters['stop_loss']:
                        signals.iloc[i] = 0  # Exit
                elif signals.iloc[i] == -1:  # Short position
                    profit_pct = (entry_price - current_price) / entry_price
                    if profit_pct >= self.parameters['exit_threshold'] or profit_pct <= -self.parameters['stop_loss']:
                        signals.iloc[i] = 0  # Exit
        
        return signals


class GapExploitationStrategy(AggressiveTradingStrategy):
    """Strategy that exploits gaps and overnight moves"""
    
    def __init__(self):
        super().__init__("Gap Exploitation")
        self.parameters = {
            'gap_threshold': 0.005,
            'fade_threshold': 0.01,
            'momentum_threshold': 0.002,
            'volume_threshold': 1.5,
            'max_hold_period': 60
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
        momentum = data['close'].pct_change(5)
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(20).mean()
        volume_ratio = data['volume'] / avg_volume
        
        signals = pd.Series(0, index=data.index)
        
        # Gap fade strategy - fade large gaps
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


class AggressiveOptimizer:
    """Optimizer for aggressive strategies targeting high returns"""
    
    def __init__(self, data: pd.DataFrame, strategies: List[AggressiveTradingStrategy]):
        self.data = data
        self.strategies = strategies
        
    def optimize_strategy(self, strategy: AggressiveTradingStrategy, 
                         optimization_window: int = 1000) -> Dict:
        """Optimize a single strategy's parameters"""
        
        # Use recent data for optimization
        recent_data = self.data.tail(optimization_window)
        
        def objective_function(params):
            """Objective function to maximize target metrics"""
            # Round integer parameters
            param_names = list(strategy.parameters.keys())
            rounded_params = []
            for i, param_name in enumerate(param_names):
                if 'period' in param_name.lower() or 'hold' in param_name.lower():
                    rounded_params.append(int(round(params[i])))
                else:
                    rounded_params.append(params[i])
            
            strategy.set_parameters(dict(zip(param_names, rounded_params)))
            results = strategy.backtest(recent_data)
            
            # Multi-objective optimization
            # Target: high daily returns, high target achievement, reasonable Sharpe
            target_score = (
                results['avg_daily_return'] * 100 +  # Daily return (scaled)
                results['target_daily_05_pct'] * 50 +  # Target achievement
                results['target_10d_05_pct'] * 50 +  # 10-day target achievement
                min(results['sharpe_ratio'], 2.0) * 10  # Sharpe ratio (capped)
            )
            
            return -target_score  # Minimize negative score
        
        # Parameter bounds
        param_bounds = self._get_parameter_bounds(strategy)
        
        # Use L-BFGS-B for optimization
        result = minimize(
            objective_function,
            x0=[(b[0] + b[1])/2 for b in param_bounds],
            bounds=param_bounds,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        
        # Set optimal parameters and run final backtest
        param_names = list(strategy.parameters.keys())
        optimal_params = []
        for i, param_name in enumerate(param_names):
            if 'period' in param_name.lower() or 'hold' in param_name.lower():
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
    
    def _get_parameter_bounds(self, strategy: AggressiveTradingStrategy) -> List[Tuple]:
        """Get parameter bounds for optimization"""
        bounds_map = {
            'Volatility Exploitation': [
                (0.01, 0.03),  # volatility_threshold
                (3, 15),       # momentum_period
                (0.01, 0.05),  # reversal_threshold
                (10, 60)       # max_hold_period
            ],
            'Momentum Amplification': [
                (2, 10),       # short_period
                (5, 20),       # medium_period
                (15, 50),      # long_period
                (0.005, 0.02), # momentum_threshold
                (1.0, 3.0),    # volume_threshold
                (0.002, 0.01)  # acceleration_threshold
            ],
            'High-Frequency Scalping': [
                (0.0005, 0.002), # entry_threshold
                (0.001, 0.005),  # exit_threshold
                (0.001, 0.005),  # stop_loss
                (5, 30),         # max_hold_minutes
                (1.5, 4.0)       # volume_threshold
            ],
            'Gap Exploitation': [
                (0.003, 0.01),   # gap_threshold
                (0.005, 0.02),   # fade_threshold
                (0.001, 0.005),  # momentum_threshold
                (1.0, 3.0),      # volume_threshold
                (30, 120)        # max_hold_period
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
    
    print(f"Data shape after filtering: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data


def main():
    """Main execution function"""
    
    # Load data
    data = load_and_prepare_data('polygon_QQQ_1m.pkl')
    
    # Define aggressive strategies
    strategies = [
        VolatilityExploitationStrategy(),
        MomentumAmplificationStrategy(),
        HighFrequencyScalpingStrategy(),
        GapExploitationStrategy()
    ]
    
    # Create optimizer
    optimizer = AggressiveOptimizer(data, strategies)
    
    # Optimize all strategies
    print("\nOptimizing aggressive strategies...")
    optimization_results = optimizer.optimize_all_strategies(optimization_window=1000)
    
    # Display results
    print("\n" + "="*80)
    print("AGGRESSIVE STRATEGY OPTIMIZATION RESULTS")
    print("="*80)
    
    for strategy_name, result in optimization_results.items():
        print(f"\n{strategy_name}:")
        print(f"  Optimal Parameters: {result['optimal_parameters']}")
        print(f"  Optimization Success: {result['optimization_success']}")
        metrics = result['final_metrics']
        print(f"  Average Daily Return: {metrics['avg_daily_return']:.4f} ({metrics['avg_daily_return']*100:.2f}%)")
        print(f"  Target Daily 0.5% Achievement: {metrics['target_daily_05_pct']:.2f}")
        print(f"  Target 10-day 5% Achievement: {metrics['target_10d_05_pct']:.2f}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Total Return: {metrics['total_return']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.3f}")
        print(f"  Number of Trades: {metrics['num_trades']}")
    
    # Save results
    results = {
        'optimization_results': optimization_results,
        'data_info': {
            'shape': data.shape,
            'date_range': (data.index.min(), data.index.max())
        }
    }
    
    with open('aggressive_strategy_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to 'aggressive_strategy_results.pkl'")
    
    return results


if __name__ == "__main__":
    results = main() 