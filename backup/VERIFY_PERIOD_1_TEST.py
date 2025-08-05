#!/usr/bin/env python3
"""
VERIFY_PERIOD_1_TEST - Test to verify Period 1 matches original 5%+ performance
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


def test_period_1_original_parameters():
    """Test Period 1 with original parameters that achieved 5%+"""
    
    # Load data
    data = pd.read_feather('polygon_QQQ_1m.feather')
    print(f"Data loaded: {len(data):,} records")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Define Period 1 (most recent 2 weeks)
    end_date = data.index.max()
    period_end = end_date - timedelta(weeks=0*2)  # Most recent
    period_start = period_end - timedelta(weeks=2)
    
    print(f"\nPeriod 1 (Most Recent):")
    print(f"Test Period: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
    
    # Get test data
    test_data = data[
        (data.index >= period_start) &
        (data.index <= period_end)
    ]
    
    print(f"Test data: {len(test_data):,} records")
    
    # Use original successful parameters
    original_params = {
        'momentum_threshold': 0.0008,
        'volume_threshold': 1.3,
        'volatility_threshold': 0.01,
        'max_hold_period': 3,
        'position_size': 0.20,
        'stop_loss': 0.003,
        'profit_target': 0.005,
        'leverage': 1.0
    }
    
    print(f"\nOriginal Parameters:")
    for key, value in original_params.items():
        print(f"  {key}: {value}")
    
    # Simple backtest with original parameters
    result = simple_backtest(test_data, original_params)
    
    print(f"\nPeriod 1 Results:")
    print(f"  Total Return: {result['total_return']:.4f} ({result['total_return']*100:.2f}%)")
    print(f"  Trades: {result['num_trades']}")
    print(f"  Win Rate: {result['win_rate']:.2f}")
    print(f"  Target Achievement: {'✅' if result['total_return'] >= 0.05 else '❌'}")
    
    return result


def simple_backtest(data, params):
    """Simple backtest with given parameters"""
    
    # Calculate basic signals
    momentum = data['close'].pct_change()
    volume_ratio = data['volume'] / data['volume'].rolling(10).mean()
    
    # Generate signals
    long_condition = (
        (momentum > params['momentum_threshold']) &
        (volume_ratio > params['volume_threshold'])
    ).fillna(False)
    
    short_condition = (
        (momentum < -params['momentum_threshold']) &
        (volume_ratio > params['volume_threshold'])
    ).fillna(False)
    
    # Combine signals
    signals = pd.Series(0, index=data.index)
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    # Backtest
    position = 0
    capital = 100000
    trades = []
    
    for timestamp, signal in signals.items():
        if signal != 0 and position == 0:
            # Open position
            position = signal
            # Handle duplicate timestamps by taking first occurrence
            price_data = data.loc[timestamp, 'close']
            entry_price = float(price_data.iloc[0] if hasattr(price_data, 'iloc') else price_data)
            entry_time = timestamp
        elif position != 0:
            # Check exit conditions
            # Handle duplicate timestamps by taking first occurrence
            price_data = data.loc[timestamp, 'close']
            current_price = float(price_data.iloc[0] if hasattr(price_data, 'iloc') else price_data)
            hold_period = (timestamp - entry_time).total_seconds() / 60
            
            # Exit conditions
            exit_signal = False
            if hold_period >= params['max_hold_period']:
                exit_signal = True
            elif position == 1:
                if current_price < entry_price * (1 - params['stop_loss']):
                    exit_signal = True
                elif current_price > entry_price * (1 + params['profit_target']):
                    exit_signal = True
            elif position == -1:
                if current_price > entry_price * (1 + params['stop_loss']):
                    exit_signal = True
                elif current_price < entry_price * (1 - params['profit_target']):
                    exit_signal = True
            
            if exit_signal:
                # Close position
                pnl = (current_price - entry_price) / entry_price * position
                capital *= (1 + pnl * params['position_size'])
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl
                })
                position = 0
    
    # Calculate metrics
    total_return = (capital - 100000) / 100000
    num_trades = len(trades)
    win_rate = len([t for t in trades if t['pnl'] > 0]) / num_trades if num_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'trades': trades
    }


if __name__ == "__main__":
    test_period_1_original_parameters() 