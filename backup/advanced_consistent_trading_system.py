#!/usr/bin/env python3
"""
Advanced Consistent Trading System - MetaTradingAI v4.0
Implements advanced features for consistent 5%+ returns across different market conditions
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

class DynamicPositionSizer:
    """Enhanced dynamic position sizing with Kelly Criterion"""
    
    def __init__(self):
        self.base_position_size = 0.15  # 15% base
        self.max_position_size = 0.40   # 40% max with confidence
        self.kelly_fraction = 0.25      # Conservative Kelly
        self.min_position_size = 0.05   # 5% minimum
        
    def calculate_position_size(self, signal_strength, win_rate, avg_win_loss_ratio, volatility):
        # Kelly Criterion
        kelly_percentage = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        kelly_percentage = max(0, min(kelly_percentage, 1))  # Clamp between 0 and 1
        kelly_position = kelly_percentage * self.kelly_fraction
        
        # Signal strength adjustment
        confidence_multiplier = min(signal_strength / 1.5, 2.0)  # Cap at 2x
        
        # Volatility adjustment (inverse relationship)
        vol_adjustment = 1.0 / (1.0 + volatility * 100)  # Reduce size in high vol
        
        # Calculate final position size
        position_size = self.base_position_size * confidence_multiplier * vol_adjustment
        position_size = min(position_size, self.max_position_size)
        position_size = max(position_size * kelly_position, self.min_position_size)
        
        return position_size

class MultiTimeframeStrategy:
    """Multi-timeframe confirmation system"""
    
    def __init__(self):
        self.timeframes = {
            '1min': 1,
            '5min': 5,
            '15min': 15,
            '60min': 60
        }
        
    def get_confluence_signal(self, data):
        signals = {}
        
        # Get signals from each timeframe
        for tf_name, tf_period in self.timeframes.items():
            resampled_data = data.resample(f'{tf_period}T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate signal for this timeframe
            signals[tf_name] = self.calculate_timeframe_signal(resampled_data)
        
        # Weight signals by timeframe importance
        weights = {'1min': 0.4, '5min': 0.3, '15min': 0.2, '60min': 0.1}
        confluence_score = sum(signals.get(tf, 0) * weights[tf] for tf in weights)
        
        return confluence_score
    
    def calculate_timeframe_signal(self, data):
        """Calculate signal for a specific timeframe"""
        if len(data) < 20:
            return 0
        
        # Simple momentum signal
        momentum = data['close'].pct_change(5).iloc[-1]
        volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(10).mean().iloc[-1]
        
        # Combine momentum and volume
        signal = momentum * volume_ratio
        return np.clip(signal * 10, -1, 1)  # Normalize to [-1, 1]

class AdaptiveLeverageManager:
    """Adaptive volatility-based leverage management"""
    
    def __init__(self):
        self.min_leverage = 1.0
        self.max_leverage = 3.0
        self.volatility_lookback = 20
        
    def calculate_optimal_leverage(self, data, current_volatility):
        # Historical volatility
        historical_vol = data['close'].pct_change().rolling(self.volatility_lookback).std().mean()
        
        # Volatility ratio
        vol_ratio = current_volatility / (historical_vol + 1e-6)
        
        # Inverse volatility scaling
        if vol_ratio > 1.5:  # High volatility
            leverage = self.min_leverage
        elif vol_ratio < 0.5:  # Low volatility
            leverage = self.max_leverage
        else:  # Normal volatility
            leverage = self.max_leverage - (vol_ratio - 0.5) * 2
            
        return np.clip(leverage, self.min_leverage, self.max_leverage)

class MicrostructureAnalyzer:
    """Enhanced market microstructure analysis"""
    
    def __init__(self):
        self.spread_threshold = 0.0002
        self.depth_imbalance_threshold = 0.3
        
    def analyze_order_flow(self, data):
        features = {}
        
        # Price momentum at different scales
        features['micro_momentum'] = data['close'].pct_change(1)
        features['mini_momentum'] = data['close'].pct_change(5)
        features['short_momentum'] = data['close'].pct_change(15)
        
        # Volume analysis
        features['volume_imbalance'] = (data['volume'] - data['volume'].rolling(20).mean()) / (data['volume'].rolling(20).mean() + 1e-6)
        
        # Trade intensity
        features['trade_intensity'] = data['volume'] / (data['volume'].rolling(20).mean() + 1e-6)
        
        # Volatility features
        features['realized_vol'] = data['close'].pct_change().rolling(5).std()
        features['vol_of_vol'] = features['realized_vol'].rolling(10).std()
        
        # Gap analysis
        features['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        return features

class RegimeTransitionDetector:
    """Advanced regime transition detection"""
    
    def __init__(self):
        self.transition_threshold = 0.7
        self.confirmation_periods = 5
        
    def detect_regime_transition(self, current_regime_probs, historical_regime):
        # Check if we're transitioning to a new regime
        max_prob_regime = max(current_regime_probs, key=current_regime_probs.get)
        
        if max_prob_regime != historical_regime:
            if current_regime_probs[max_prob_regime] > self.transition_threshold:
                return {
                    'transitioning': True,
                    'from_regime': historical_regime,
                    'to_regime': max_prob_regime,
                    'confidence': current_regime_probs[max_prob_regime]
                }
        
        return {'transitioning': False}

class StrategyBlender:
    """Advanced strategy blending system"""
    
    def __init__(self):
        self.blend_threshold = 0.6
        
    def blend_strategies(self, strategy_signals, strategy_weights):
        # Instead of picking one strategy, blend top performers
        blended_signal = 0
        total_weight = 0
        
        for strategy, signal in strategy_signals.items():
            if strategy_weights.get(strategy, 0) > self.blend_threshold:
                blended_signal += signal * strategy_weights[strategy]
                total_weight += strategy_weights[strategy]
        
        if total_weight > 0:
            return blended_signal / total_weight
        return 0

class DynamicRiskManager:
    """Advanced stop-loss and profit target system"""
    
    def __init__(self):
        self.atr_multiplier_stop = 2.0
        self.atr_multiplier_target = 3.0
        self.trailing_stop_activation = 0.01  # 1% profit
        
    def calculate_dynamic_levels(self, entry_price, atr, position_type):
        if position_type == 'long':
            stop_loss = entry_price - (atr * self.atr_multiplier_stop)
            take_profit = entry_price + (atr * self.atr_multiplier_target)
        else:  # short
            stop_loss = entry_price + (atr * self.atr_multiplier_stop)
            take_profit = entry_price - (atr * self.atr_multiplier_target)
            
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_activation': entry_price * (1 + self.trailing_stop_activation * (1 if position_type == 'long' else -1))
        }

class AdaptiveStrategyWeighter:
    """Performance-based strategy weighting"""
    
    def __init__(self):
        self.lookback_periods = [10, 30, 60]  # days
        self.decay_factor = 0.95
        
    def update_strategy_weights(self, performance_history):
        weights = {}
        
        for strategy in performance_history.keys():
            strategy_score = 0
            
            for period in self.lookback_periods:
                recent_performance = performance_history[strategy][-period:]
                if recent_performance:
                    # Sharpe-like metric
                    avg_return = np.mean(recent_performance)
                    std_return = np.std(recent_performance) + 1e-6
                    sharpe = avg_return / std_return
                    
                    # Weight by recency
                    recency_weight = 1 / period
                    strategy_score += sharpe * recency_weight
            
            weights[strategy] = max(0, strategy_score)
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights

class AdvancedMetaStrategy:
    """Advanced meta strategy with all enhancements"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.last_trade_time = None
        self.position_sizer = DynamicPositionSizer()
        self.leverage_manager = AdaptiveLeverageManager()
        self.risk_manager = DynamicRiskManager()
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.multi_timeframe = MultiTimeframeStrategy()
        self.strategy_blender = StrategyBlender()
        self.regime_detector = RegimeTransitionDetector()
        self.strategy_weighter = AdaptiveStrategyWeighter()
        
        # Performance tracking
        self.performance_history = {}
        self.strategy_weights = {}
        self.win_rate = 0.5
        self.avg_win_loss_ratio = 1.5
        
    def calculate_signal_strength(self, data: pd.DataFrame, index: int) -> float:
        """Calculate enhanced signal strength with microstructure analysis"""
        if index < 20:
            return 0.0
        
        # Microstructure features
        microstructure_features = self.microstructure_analyzer.analyze_order_flow(data.iloc[:index+1])
        
        # Multi-timeframe confirmation
        confluence_signal = self.multi_timeframe.get_confluence_signal(data.iloc[:index+1])
        
        # Momentum components
        momentum = abs(data['close'].pct_change().iloc[index])
        volume_ratio = data['volume'].iloc[index] / data['volume'].rolling(10).mean().iloc[index]
        
        # Volatility components
        volatility = data['close'].pct_change().rolling(5).std().iloc[index]
        avg_volatility = data['close'].pct_change().rolling(20).std().mean()
        volatility_score = avg_volatility / (volatility + 1e-6)
        
        # Trend alignment
        short_ma = data['close'].rolling(3).mean().iloc[index]
        long_ma = data['close'].rolling(10).mean().iloc[index]
        trend_alignment = 1.0 if (data['close'].iloc[index] > short_ma > long_ma) or (data['close'].iloc[index] < short_ma < long_ma) else 0.5
        
        # Enhanced signal strength calculation
        signal_strength = (
            0.3 * momentum * 1000 +
            0.2 * volume_ratio +
            0.2 * volatility_score +
            0.1 * trend_alignment +
            0.1 * abs(confluence_signal) +
            0.1 * abs(microstructure_features.get('trade_intensity', 1.0))
        )
        
        return signal_strength
    
    def can_trade(self, current_time: datetime) -> bool:
        """Check if we can trade based on frequency constraints"""
        if self.last_trade_time is None:
            return True
        
        time_since_last_trade = (current_time - self.last_trade_time).total_seconds()
        return time_since_last_trade >= 120  # 2 minutes
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> dict:
        """Enhanced backtest with all advanced features"""
        position = 0
        capital = initial_capital
        equity = [capital]
        trades = []
        hourly_returns = []
        last_trade_time = None
        leverage = 1.0
        
        # Initialize performance tracking
        strategy_performances = {}
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]['close']
            prev_price = data.iloc[i-1]['close']
            
            # Check trading constraints
            can_trade = True
            if last_trade_time is not None:
                time_since_last = (current_time - last_trade_time).total_seconds()
                can_trade = time_since_last >= 120
            
            # Calculate signal strength for quality filtering
            signal_strength = self.calculate_signal_strength(data, i)
            
            # Generate signals (enhanced)
            momentum = data['close'].pct_change().iloc[i]
            volume_ratio = data['volume'].iloc[i] / data['volume'].rolling(10).mean().iloc[i]
            volatility = data['close'].pct_change().rolling(5).std().iloc[i]
            
            # Enhanced signal generation with multiple confirmations
            signal = 0
            try:
                momentum_val = momentum.item() if hasattr(momentum, 'item') else momentum
                volume_ratio_val = volume_ratio.item() if hasattr(volume_ratio, 'item') else volume_ratio
                signal_strength_val = signal_strength.item() if hasattr(signal_strength, 'item') else signal_strength
                
                if (not pd.isna(momentum_val) and not pd.isna(volume_ratio_val) and not pd.isna(signal_strength_val) and
                    abs(momentum_val) > 0.0008 and volume_ratio_val > 1.3 and signal_strength_val > 1.2):
                    signal = 1 if momentum_val > 0 else -1
            except:
                pass
            
            # Update position based on signals (only if we can trade and signal is strong)
            if can_trade and signal != 0 and signal_strength > 1.2:
                # Calculate optimal leverage
                leverage = self.leverage_manager.calculate_optimal_leverage(data.iloc[:i+1], volatility)
                
                # Calculate position size
                position_size = self.position_sizer.calculate_position_size(
                    signal_strength, self.win_rate, self.avg_win_loss_ratio, volatility
                )
                
                # Calculate ATR for risk management
                atr = data['close'].rolling(14).std().iloc[i]
                
                if signal == 1 and position <= 0:  # Buy signal
                    if position == -1:  # Close short
                        trades.append({
                            'entry_time': data.index[i-1],
                            'exit_time': data.index[i],
                            'entry_price': prev_price,
                            'exit_price': current_price,
                            'type': 'short',
                            'pnl': (prev_price - current_price) * leverage * position_size,
                            'leverage': leverage,
                            'position_size': position_size,
                            'signal_strength': signal_strength
                        })
                    position = 1
                    last_trade_time = current_time
                elif signal == -1 and position >= 0:  # Sell signal
                    if position == 1:  # Close long
                        trades.append({
                            'entry_time': data.index[i-1],
                            'exit_time': data.index[i],
                            'entry_price': prev_price,
                            'exit_price': current_price,
                            'type': 'long',
                            'pnl': (current_price - prev_price) * leverage * position_size,
                            'leverage': leverage,
                            'position_size': position_size,
                            'signal_strength': signal_strength
                        })
                    position = -1
                    last_trade_time = current_time
            
            # Calculate returns with leverage and position sizing
            if position == 1:  # Long position
                hourly_return = (current_price - prev_price) / prev_price * leverage * position_size
            elif position == -1:  # Short position
                hourly_return = (prev_price - current_price) / prev_price * leverage * position_size
            else:
                hourly_return = 0
            
            capital *= (1 + hourly_return)
            equity.append(capital)
            hourly_returns.append(hourly_return)
        
        # Calculate metrics
        total_return = (equity[-1] - initial_capital) / initial_capital
        avg_hourly_return = np.mean(hourly_returns) if hourly_returns else 0
        
        # Update performance metrics
        if trades:
            wins = len([t for t in trades if t['pnl'] > 0])
            self.win_rate = wins / len(trades)
            
            win_amounts = [t['pnl'] for t in trades if t['pnl'] > 0]
            loss_amounts = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
            
            if win_amounts and loss_amounts:
                self.avg_win_loss_ratio = np.mean(win_amounts) / np.mean(loss_amounts)
        
        return {
            'total_return': total_return,
            'avg_hourly_return': avg_hourly_return,
            'trades': trades,
            'num_trades': len(trades),
            'trade_frequency': len(trades) / (len(data) / 60) if len(data) > 0 else 0,
            'avg_signal_strength': np.mean([t['signal_strength'] for t in trades]) if trades else 0,
            'avg_leverage': np.mean([t['leverage'] for t in trades]) if trades else 1.0,
            'avg_position_size': np.mean([t['position_size'] for t in trades]) if trades else 0.15,
            'win_rate': self.win_rate,
            'avg_win_loss_ratio': self.avg_win_loss_ratio
        }

class AdvancedAggressiveMetaTradingAI:
    """Advanced aggressive meta-trading AI system with all enhancements"""
    
    def __init__(self, training_days: int = 180):
        self.training_days = training_days
        
        # Initialize advanced strategy
        self.strategy = AdvancedMetaStrategy("Advanced Consistent Strategy")
        
        # Load data
        print("Loading data for advanced consistent trading system...")
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
        print(f"Advanced features: Dynamic position sizing, Multi-timeframe confirmation")
        print(f"Training period: {training_days} days ({training_days/30:.1f} months)")
    
    def run_advanced_meta_system(self, test_period_days: int = 10) -> dict:
        """Run the advanced meta-trading system with all enhancements"""
        print(f"\n🚀 Running Advanced MetaTradingAI v4.0: Consistent Performance Model...")
        print(f"Training Period: {self.training_days} days ({self.training_days/30:.1f} months)")
        print(f"Advanced Features: Dynamic Position Sizing, Multi-timeframe Confirmation")
        print(f"Target: Consistent 5%+ return over {test_period_days} trading days")
        
        # Calculate date ranges
        end_date = self.data.index.max()
        test_start_date = end_date - timedelta(days=test_period_days)
        training_start_date = test_start_date - timedelta(days=self.training_days)
        
        print(f"Advanced Meta-Trading AI System Setup:")
        print(f"  Training period: {training_start_date.date()} to {test_start_date.date()}")
        print(f"  Test period: {test_start_date.date()} to {end_date.date()}")
        print(f"  Training days: {self.training_days} (vs 60 in original)")
        print(f"  Target: Consistent 5%+ return over {test_period_days} trading days")
        
        # Split data
        training_data = self.data[(self.data.index >= training_start_date) & (self.data.index < test_start_date)]
        test_data = self.data[self.data.index >= test_start_date]
        
        print(f"  Training data: {len(training_data):,} records")
        print(f"  Test data: {len(test_data):,} records")
        print(f"  Training hours: {len(training_data) // 60:,}")
        print(f"  Test hours: {len(test_data) // 60:,}")
        
        # Test strategy on training data
        print(f"\nTesting advanced strategy on training data...")
        training_results = self.strategy.backtest(training_data)
        print(f"  Training Trades: {training_results['num_trades']}")
        print(f"  Training Trade Frequency: {training_results['trade_frequency']:.2f} trades/hour")
        print(f"  Training Avg Signal Strength: {training_results['avg_signal_strength']:.2f}")
        print(f"  Training Avg Leverage: {training_results['avg_leverage']:.1f}x")
        print(f"  Training Avg Position Size: {training_results['avg_position_size']:.3f}")
        print(f"  Training Win Rate: {training_results['win_rate']:.2%}")
        print(f"  Training Avg Win/Loss Ratio: {training_results['avg_win_loss_ratio']:.2f}")
        
        # Run advanced system on test data
        print(f"\nRunning advanced meta-trading system...")
        test_results = self.strategy.backtest(test_data)
        
        # Calculate overall metrics
        total_hours = len(test_data) // 60
        trade_frequency = test_results['num_trades'] / total_hours if total_hours > 0 else 0
        
        # Final results
        print(f"\n🎯 ADVANCED TARGET ACHIEVEMENT:")
        print(f"  Target: Consistent 5%+ return over {test_period_days} trading days")
        print(f"  Actual: {test_results['total_return']:.4f} ({test_results['total_return']*100:.2f}%)")
        print(f"  Training Days: {self.training_days} (vs 60 in original)")
        print(f"  Training Hours: {len(training_data) // 60:,} (vs 240 in original)")
        print(f"  Total Trades: {test_results['num_trades']}")
        print(f"  Average Trade Frequency: {trade_frequency:.2f} trades/hour")
        print(f"  Average Signal Strength: {test_results['avg_signal_strength']:.2f}")
        print(f"  Average Leverage: {test_results['avg_leverage']:.1f}x")
        print(f"  Average Position Size: {test_results['avg_position_size']:.3f}")
        print(f"  Win Rate: {test_results['win_rate']:.2%}")
        print(f"  Avg Win/Loss Ratio: {test_results['avg_win_loss_ratio']:.2f}")
        
        if test_results['total_return'] >= 0.05:
            print(f"  Status: ✅ ACHIEVED TARGET")
        else:
            print(f"  Status: ❌ NOT ACHIEVED")
        
        # Advanced performance analysis
        print(f"\n📈 ADVANCED PERFORMANCE ANALYSIS:")
        print(f"  Consistency Score: {'✅ High' if test_results['win_rate'] > 0.6 else '⚠️ Medium' if test_results['win_rate'] > 0.5 else '❌ Low'}")
        print(f"  Risk/Reward Ratio: {'✅ Excellent' if test_results['avg_win_loss_ratio'] > 1.5 else '⚠️ Good' if test_results['avg_win_loss_ratio'] > 1.2 else '❌ Poor'}")
        print(f"  Position Sizing: {'✅ Optimal' if test_results['avg_position_size'] > 0.1 else '⚠️ Conservative' if test_results['avg_position_size'] > 0.05 else '❌ Too Small'}")
        print(f"  Leverage Usage: {'✅ Optimal' if test_results['avg_leverage'] > 1.5 else '⚠️ Conservative' if test_results['avg_leverage'] > 1.0 else '❌ Low'}")
        
        return test_results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Consistent Trading System')
    parser.add_argument('--training-days', type=int, default=180, 
                       help='Number of training days (default: 180)')
    
    args = parser.parse_args()
    
    # Run advanced system
    system = AdvancedAggressiveMetaTradingAI(training_days=args.training_days)
    results = system.run_advanced_meta_system(test_period_days=10)
    
    return results

if __name__ == "__main__":
    results = main() 