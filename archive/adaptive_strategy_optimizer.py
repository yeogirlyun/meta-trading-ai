import pickle
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AdaptiveTradingStrategy:
    """Adaptive trading strategy that can handle different market regimes"""
    
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


class AdaptiveMACrossover(AdaptiveTradingStrategy):
    """Adaptive Moving Average Crossover with volatility adjustment"""
    
    def __init__(self):
        super().__init__("Adaptive MA Crossover")
        self.parameters = {
            'fast_period': 10,
            'slow_period': 50,
            'volatility_threshold': 0.02,
            'trend_strength_threshold': 0.01
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_ma = data['close'].rolling(self.parameters['fast_period']).mean()
        slow_ma = data['close'].rolling(self.parameters['slow_period']).mean()
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(20).std()
        
        # Calculate trend strength
        trend_strength = abs(fast_ma - slow_ma) / slow_ma
        
        signals = pd.Series(0, index=data.index)
        
        # Only trade when volatility is low and trend is strong
        low_vol_condition = volatility < self.parameters['volatility_threshold']
        strong_trend_condition = trend_strength > self.parameters['trend_strength_threshold']
        
        # Generate signals with conditions
        buy_condition = (fast_ma > slow_ma) & low_vol_condition & strong_trend_condition
        sell_condition = (fast_ma < slow_ma) & low_vol_condition & strong_trend_condition
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals


class AdaptiveRSI(AdaptiveTradingStrategy):
    """Adaptive RSI with dynamic thresholds"""
    
    def __init__(self):
        super().__init__("Adaptive RSI")
        self.parameters = {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'volatility_multiplier': 1.5,
            'momentum_threshold': 0.01
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate volatility-adjusted thresholds
        volatility = data['close'].pct_change().rolling(20).std()
        vol_adjustment = volatility * self.parameters['volatility_multiplier']
        
        # Dynamic thresholds
        dynamic_oversold = self.parameters['oversold'] - (vol_adjustment * 100)
        dynamic_overbought = self.parameters['overbought'] + (vol_adjustment * 100)
        
        # Calculate momentum
        momentum = data['close'].pct_change(self.parameters['rsi_period'])
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals with momentum confirmation
        oversold_condition = (rsi < dynamic_oversold) & (momentum > self.parameters['momentum_threshold'])
        overbought_condition = (rsi > dynamic_overbought) & (momentum < -self.parameters['momentum_threshold'])
        
        signals[oversold_condition] = 1  # Buy when oversold with positive momentum
        signals[overbought_condition] = -1  # Sell when overbought with negative momentum
        
        return signals


class AdaptiveBollingerBands(AdaptiveTradingStrategy):
    """Adaptive Bollinger Bands with regime detection"""
    
    def __init__(self):
        super().__init__("Adaptive Bollinger Bands")
        self.parameters = {
            'period': 20,
            'std_dev': 2.0,
            'regime_threshold': 0.5,
            'volume_threshold': 1.5
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate Bollinger Bands
        sma = data['close'].rolling(window=self.parameters['period']).mean()
        std = data['close'].rolling(window=self.parameters['period']).std()
        upper_band = sma + (self.parameters['std_dev'] * std)
        lower_band = sma - (self.parameters['std_dev'] * std)
        
        # Calculate market regime (trending vs ranging)
        price_position = (data['close'] - lower_band) / (upper_band - lower_band)
        regime = abs(price_position - 0.5)  # Distance from middle of bands
        
        # Calculate volume confirmation
        avg_volume = data['volume'].rolling(20).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(20).std()
        
        signals = pd.Series(0, index=data.index)
        
        # Only trade in ranging markets (low regime value)
        ranging_market = regime < self.parameters['regime_threshold']
        high_volume = volume_ratio > self.parameters['volume_threshold']
        low_volatility = volatility < volatility.rolling(50).mean()
        
        # Generate signals
        buy_condition = (data['close'] < lower_band) & ranging_market & high_volume & low_volatility
        sell_condition = (data['close'] > upper_band) & ranging_market & high_volume & low_volatility
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals


class MultiTimeframeStrategy(AdaptiveTradingStrategy):
    """Multi-timeframe strategy combining different signals"""
    
    def __init__(self):
        super().__init__("Multi-Timeframe Strategy")
        self.parameters = {
            'short_period': 5,
            'medium_period': 20,
            'long_period': 50,
            'signal_threshold': 2,
            'volatility_threshold': 0.015
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate moving averages for different timeframes
        short_ma = data['close'].rolling(self.parameters['short_period']).mean()
        medium_ma = data['close'].rolling(self.parameters['medium_period']).mean()
        long_ma = data['close'].rolling(self.parameters['long_period']).mean()
        
        # Calculate signals for each timeframe
        short_signal = np.where(short_ma > medium_ma, 1, -1)
        medium_signal = np.where(medium_ma > long_ma, 1, -1)
        long_signal = np.where(data['close'] > long_ma, 1, -1)
        
        # Combine signals
        combined_signal = short_signal + medium_signal + long_signal
        
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(20).std()
        
        signals = pd.Series(0, index=data.index)
        
        # Generate signals based on consensus and volatility
        low_vol_condition = volatility < self.parameters['volatility_threshold']
        
        # Strong buy when all timeframes agree and volatility is low
        strong_buy = (combined_signal >= self.parameters['signal_threshold']) & low_vol_condition
        strong_sell = (combined_signal <= -self.parameters['signal_threshold']) & low_vol_condition
        
        signals[strong_buy] = 1
        signals[strong_sell] = -1
        
        return signals


class AdaptiveOptimizer:
    """Adaptive strategy optimizer with regime detection"""
    
    def __init__(self, data: pd.DataFrame, strategies: List[AdaptiveTradingStrategy]):
        self.data = data
        self.strategies = strategies
        
    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""
        # Calculate volatility
        volatility = data['close'].pct_change().rolling(20).std()
        current_vol = volatility.iloc[-1]
        avg_vol = volatility.rolling(50).mean().iloc[-1]
        
        # Calculate trend strength
        returns = data['close'].pct_change()
        trend_strength = abs(returns.rolling(20).mean()) / returns.rolling(20).std()
        current_trend = trend_strength.iloc[-1]
        
        if current_vol > avg_vol * 1.5:
            return "high_volatility"
        elif current_trend > 0.5:
            return "trending"
        else:
            return "ranging"
    
    def optimize_strategy(self, strategy: AdaptiveTradingStrategy, 
                         optimization_window: int = 1000) -> Dict:
        """Optimize a single strategy's parameters"""
        
        # Use recent data for optimization
        recent_data = self.data.tail(optimization_window)
        
        # Detect market regime
        regime = self.detect_market_regime(recent_data)
        print(f"  Market regime: {regime}")
        
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
            'final_metrics': final_results,
            'market_regime': regime
        }
    
    def _get_parameter_bounds(self, strategy: AdaptiveTradingStrategy) -> List[Tuple]:
        """Get parameter bounds for optimization"""
        bounds_map = {
            'Adaptive MA Crossover': [
                (5, 30),    # fast_period
                (20, 100),  # slow_period
                (0.01, 0.05), # volatility_threshold
                (0.005, 0.02) # trend_strength_threshold
            ],
            'Adaptive RSI': [
                (5, 30),    # rsi_period
                (20, 40),   # oversold
                (60, 80),   # overbought
                (0.5, 3.0), # volatility_multiplier
                (0.005, 0.02) # momentum_threshold
            ],
            'Adaptive Bollinger Bands': [
                (10, 50),   # period
                (1.5, 3.0), # std_dev
                (0.3, 0.7), # regime_threshold
                (1.0, 3.0)  # volume_threshold
            ],
            'Multi-Timeframe Strategy': [
                (3, 15),    # short_period
                (10, 40),   # medium_period
                (30, 100),  # long_period
                (1, 3),     # signal_threshold
                (0.01, 0.03) # volatility_threshold
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
        AdaptiveMACrossover(),
        AdaptiveRSI(),
        AdaptiveBollingerBands(),
        MultiTimeframeStrategy()
    ]
    
    # Create optimizer
    optimizer = AdaptiveOptimizer(data, strategies)
    
    # Optimize all strategies
    print("\nOptimizing strategies...")
    optimization_results = optimizer.optimize_all_strategies(optimization_window=1000)
    
    # Display results
    print("\n" + "="*80)
    print("ADAPTIVE STRATEGY OPTIMIZATION RESULTS")
    print("="*80)
    
    for strategy_name, result in optimization_results.items():
        print(f"\n{strategy_name}:")
        print(f"  Market Regime: {result['market_regime']}")
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
    
    with open('adaptive_strategy_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to 'adaptive_strategy_results.pkl'")
    
    return results


if __name__ == "__main__":
    results = main() 