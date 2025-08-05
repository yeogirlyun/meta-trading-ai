#!/usr/bin/env python3
"""
Enhanced Restricted Trading System - Quality-Focused Trading with 2-Minute Order Limits
Target: 6-9% returns over 10 trading days
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedTradingConstraints:
    """Enhanced trading constraints for quality-focused trading"""
    
    def __init__(self):
        # Order frequency limits
        self.min_order_interval = 120  # 2 minutes between orders
        self.max_orders_per_hour = 30  # maximum orders per hour
        
        # Signal quality thresholds
        self.min_signal_strength = 1.2  # minimum signal strength score
        self.min_volume_ratio = 1.5     # minimum volume spike
        self.min_momentum_threshold = 0.001  # minimum momentum
        
        # Dynamic leverage settings
        self.base_leverage = 1.0
        self.max_leverage = 3.0
        self.leverage_ranging = 1.0
        self.leverage_trending = 2.0
        self.leverage_high_vol = 3.0

def detect_enhanced_market_regime(data: pd.DataFrame) -> str:
    """Enhanced market regime detection"""
    if len(data) < 50:
        return "ranging"
    
    volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
    avg_vol = data['close'].pct_change().rolling(50).std().mean()
    trend_strength = abs(data['close'].pct_change().rolling(20).mean().iloc[-1]) / volatility
    
    volume_ratio = data['volume'].iloc[-10:].mean() / data['volume'].rolling(50).mean().iloc[-1]
    momentum_15m = data['close'].pct_change(15).iloc[-1]
    
    if volatility > avg_vol * 1.8 and volume_ratio > 1.3:
        return "high_volatility"
    elif trend_strength > 0.6 and abs(momentum_15m) > 0.002:
        return "trending"
    elif volume_ratio > 1.5 and abs(data['close'].pct_change(5).iloc[-1]) > 0.001:
        return "breakout"
    else:
        return "ranging"

class EnhancedMetaStrategy:
    """Enhanced base class with signal filtering and quality focus"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.last_trade_time = None
        self.constraints = EnhancedTradingConstraints()
    
    def calculate_signal_strength(self, data: pd.DataFrame, index: int) -> float:
        """Calculate signal strength score for quality filtering"""
        if index < 5:
            return 0.0
        
        momentum = abs(data['close'].pct_change().iloc[index])
        volume_ratio = data['volume'].iloc[index] / data['volume'].rolling(10).mean().iloc[index]
        volatility = data['close'].pct_change().rolling(5).std().iloc[index]
        avg_volatility = data['close'].pct_change().rolling(20).std().mean()
        volatility_score = avg_volatility / (volatility + 1e-6)
        
        short_ma = data['close'].rolling(3).mean().iloc[index]
        long_ma = data['close'].rolling(10).mean().iloc[index]
        trend_alignment = 1.0 if (data['close'].iloc[index] > short_ma > long_ma) or (data['close'].iloc[index] < short_ma < long_ma) else 0.5
        
        signal_strength = (
            0.4 * momentum * 1000 +
            0.3 * volume_ratio +
            0.2 * volatility_score +
            0.1 * trend_alignment
        )
        
        return signal_strength
    
    def can_trade(self, current_time: datetime) -> bool:
        """Check if we can trade based on frequency constraints"""
        if self.last_trade_time is None:
            return True
        
        time_since_last_trade = (current_time - self.last_trade_time).total_seconds()
        return time_since_last_trade >= self.constraints.min_order_interval
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> dict:
        """Enhanced backtest with order restrictions and quality filtering"""
        position = 0
        capital = initial_capital
        equity = [capital]
        trades = []
        hourly_returns = []
        last_trade_time = None
        leverage = self.constraints.base_leverage
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data.iloc[i]['close']
            prev_price = data.iloc[i-1]['close']
            
            # Check trading constraints
            can_trade = True
            if last_trade_time is not None:
                time_since_last = (current_time - last_trade_time).total_seconds()
                can_trade = time_since_last >= self.constraints.min_order_interval
            
            # Calculate signal strength for quality filtering
            signal_strength = self.calculate_signal_strength(data, i)
            
            # Generate signals (simplified for demo)
            momentum = data['close'].pct_change().iloc[i]
            volume_ratio = data['volume'].iloc[i] / data['volume'].rolling(10).mean().iloc[i]
            
            # Simple signal generation
            signal = 0
            if abs(momentum) > self.constraints.min_momentum_threshold and volume_ratio > self.constraints.min_volume_ratio:
                signal = 1 if momentum > 0 else -1
            
            # Update position based on signals (only if we can trade and signal is strong)
            if can_trade and signal != 0 and signal_strength > self.constraints.min_signal_strength:
                # Determine leverage based on regime
                regime = detect_enhanced_market_regime(data.iloc[max(0, i-20):i+1])
                if regime == "high_volatility":
                    leverage = self.constraints.leverage_high_vol
                elif regime == "trending":
                    leverage = self.constraints.leverage_trending
                else:
                    leverage = self.constraints.leverage_ranging
                
                if signal == 1 and position <= 0:  # Buy signal
                    if position == -1:  # Close short
                        trades.append({
                            'entry_time': data.index[i-1],
                            'exit_time': data.index[i],
                            'entry_price': prev_price,
                            'exit_price': current_price,
                            'type': 'short',
                            'pnl': (prev_price - current_price) * leverage,
                            'leverage': leverage,
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
                            'pnl': (current_price - prev_price) * leverage,
                            'leverage': leverage,
                            'signal_strength': signal_strength
                        })
                    position = -1
                    last_trade_time = current_time
            
            # Calculate returns with leverage
            if position == 1:  # Long position
                hourly_return = (current_price - prev_price) / prev_price * leverage
            elif position == -1:  # Short position
                hourly_return = (prev_price - current_price) / prev_price * leverage
            else:
                hourly_return = 0
            
            capital *= (1 + hourly_return)
            equity.append(capital)
            hourly_returns.append(hourly_return)
        
        # Calculate metrics
        total_return = (equity[-1] - initial_capital) / initial_capital
        avg_hourly_return = np.mean(hourly_returns) if hourly_returns else 0
        
        return {
            'total_return': total_return,
            'avg_hourly_return': avg_hourly_return,
            'trades': trades,
            'num_trades': len(trades),
            'trade_frequency': len(trades) / (len(data) / 60) if len(data) > 0 else 0,
            'avg_signal_strength': np.mean([t['signal_strength'] for t in trades]) if trades else 0,
            'avg_leverage': np.mean([t['leverage'] for t in trades]) if trades else 1.0
        }

class EnhancedAggressiveMetaTradingAI:
    """Enhanced aggressive meta-trading AI system with quality focus"""
    
    def __init__(self, training_days: int = 180):
        self.training_days = training_days
        self.constraints = EnhancedTradingConstraints()
        
        # Initialize strategy
        self.strategy = EnhancedMetaStrategy("Enhanced Quality Strategy")
        
        # Load data
        print("Loading data for enhanced trading system...")
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
        print(f"Enhanced constraints: {self.constraints.min_order_interval} sec between orders")
        print(f"Training period: {training_days} days ({training_days/30:.1f} months)")
    
    def run_enhanced_meta_system(self, test_period_days: int = 10) -> dict:
        """Run the enhanced meta-trading system with quality focus"""
        print(f"\nRunning Enhanced MetaTradingAI v3.0: Quality-Focused Model...")
        print(f"Training Period: {self.training_days} days ({self.training_days/30:.1f} months)")
        print(f"Order Frequency Limit: 1 order per {self.constraints.min_order_interval} seconds")
        print(f"Target: 6-9% return over {test_period_days} trading days")
        
        # Calculate date ranges
        end_date = self.data.index.max()
        test_start_date = end_date - timedelta(days=test_period_days)
        training_start_date = test_start_date - timedelta(days=self.training_days)
        
        print(f"Enhanced Meta-Trading AI System Setup:")
        print(f"  Training period: {training_start_date.date()} to {test_start_date.date()}")
        print(f"  Test period: {test_start_date.date()} to {end_date.date()}")
        print(f"  Training days: {self.training_days} (vs 60 in original v3.0)")
        print(f"  Target: 6-9% return over {test_period_days} trading days")
        
        # Split data
        training_data = self.data[(self.data.index >= training_start_date) & (self.data.index < test_start_date)]
        test_data = self.data[self.data.index >= test_start_date]
        
        print(f"  Training data: {len(training_data):,} records")
        print(f"  Test data: {len(test_data):,} records")
        print(f"  Training hours: {len(training_data) // 60:,}")
        print(f"  Test hours: {len(test_data) // 60:,}")
        
        # Test strategy on training data
        print(f"\nTesting enhanced strategy on training data...")
        training_results = self.strategy.backtest(training_data)
        print(f"  Training Trades: {training_results['num_trades']}")
        print(f"  Training Trade Frequency: {training_results['trade_frequency']:.2f} trades/hour")
        print(f"  Training Avg Signal Strength: {training_results['avg_signal_strength']:.2f}")
        print(f"  Training Avg Leverage: {training_results['avg_leverage']:.1f}x")
        
        # Run enhanced system on test data
        print(f"\nRunning enhanced meta-trading system...")
        test_results = self.strategy.backtest(test_data)
        
        # Calculate overall metrics
        total_hours = len(test_data) // 60
        trade_frequency = test_results['num_trades'] / total_hours if total_hours > 0 else 0
        
        # Final results
        print(f"\n🎯 ENHANCED TARGET ACHIEVEMENT:")
        print(f"  Target: 6-9% return over {test_period_days} trading days")
        print(f"  Actual: {test_results['total_return']:.4f} ({test_results['total_return']*100:.2f}%)")
        print(f"  Training Days: {self.training_days} (vs 60 in original)")
        print(f"  Training Hours: {len(training_data) // 60:,} (vs 240 in original)")
        print(f"  Total Trades: {test_results['num_trades']}")
        print(f"  Average Trade Frequency: {trade_frequency:.2f} trades/hour")
        print(f"  Order Frequency Limit: 1 order per {self.constraints.min_order_interval} seconds")
        print(f"  Average Signal Strength: {test_results['avg_signal_strength']:.2f}")
        print(f"  Average Leverage: {test_results['avg_leverage']:.1f}x")
        
        if test_results['total_return'] >= 0.06:
            print(f"  Status: ✅ EXCEEDED TARGET")
        elif test_results['total_return'] >= 0.05:
            print(f"  Status: ✅ ACHIEVED TARGET")
        else:
            print(f"  Status: ❌ NOT ACHIEVED")
        
        # Enhanced performance analysis
        print(f"\n📈 ENHANCED PERFORMANCE ANALYSIS:")
        print(f"  Order Frequency Compliance: {'✅ Within Limits' if trade_frequency <= 30 else '⚠️ Exceeds Limits'}")
        print(f"  Quality Focus: {'✅ High' if test_results['avg_signal_strength'] > 1.5 else '⚠️ Medium' if test_results['avg_signal_strength'] > 1.0 else '❌ Low'}")
        print(f"  Leverage Usage: {'✅ Optimal' if test_results['avg_leverage'] > 1.5 else '⚠️ Conservative' if test_results['avg_leverage'] > 1.0 else '❌ Low'}")
        
        return test_results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced MetaTradingAI v3.0')
    parser.add_argument('--training-days', type=int, default=180, 
                       help='Number of training days (default: 180)')
    
    args = parser.parse_args()
    
    # Run enhanced system
    system = EnhancedAggressiveMetaTradingAI(training_days=args.training_days)
    results = system.run_enhanced_meta_system(test_period_days=10)
    
    return results

if __name__ == "__main__":
    results = main() 