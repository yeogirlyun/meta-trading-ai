import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def analyze_recent_performance():
    """Analyze recent QQQ performance to understand realistic targets"""
    
    # Load data
    print("Loading QQQ data...")
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
    
    # Get last 3 months of data
    three_months_ago = data.index.max() - timedelta(days=90)
    recent_data = data[data.index >= three_months_ago]
    
    print(f"Recent data shape: {recent_data.shape}")
    print(f"Date range: {recent_data.index.min()} to {recent_data.index.max()}")
    
    # Calculate daily returns
    daily_data = recent_data.resample('D').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    daily_returns = daily_data['close'].pct_change().dropna()
    
    # Calculate statistics
    print("\n" + "="*60)
    print("RECENT QQQ PERFORMANCE ANALYSIS (Last 3 Months)")
    print("="*60)
    
    print(f"Total trading days: {len(daily_returns)}")
    print(f"Average daily return: {daily_returns.mean():.4f} ({daily_returns.mean()*100:.2f}%)")
    print(f"Daily return std dev: {daily_returns.std():.4f} ({daily_returns.std()*100:.2f}%)")
    print(f"Best daily return: {daily_returns.max():.4f} ({daily_returns.max()*100:.2f}%)")
    print(f"Worst daily return: {daily_returns.min():.4f} ({daily_returns.min()*100:.2f}%)")
    print(f"Positive days: {len(daily_returns[daily_returns > 0])} ({len(daily_returns[daily_returns > 0])/len(daily_returns)*100:.1f}%)")
    print(f"Negative days: {len(daily_returns[daily_returns < 0])} ({len(daily_returns[daily_returns < 0])/len(daily_returns)*100:.1f}%)")
    
    # Calculate rolling 2-week returns
    rolling_10d_returns = daily_data['close'].pct_change(10).dropna()
    
    print(f"\nRolling 10-day returns:")
    print(f"Average 10-day return: {rolling_10d_returns.mean():.4f} ({rolling_10d_returns.mean()*100:.2f}%)")
    print(f"Best 10-day return: {rolling_10d_returns.max():.4f} ({rolling_10d_returns.max()*100:.2f}%)")
    print(f"Worst 10-day return: {rolling_10d_returns.min():.4f} ({rolling_10d_returns.min()*100:.2f}%)")
    
    # Find periods with high volatility
    volatility = daily_returns.rolling(5).std()
    high_vol_periods = volatility[volatility > volatility.quantile(0.8)]
    
    print(f"\nHigh volatility periods (top 20%):")
    print(f"Average daily return during high vol: {daily_returns[high_vol_periods.index].mean():.4f}")
    print(f"Number of high vol days: {len(high_vol_periods)}")
    
    # Calculate intraday ranges
    intraday_ranges = (daily_data['high'] - daily_data['low']) / daily_data['open']
    print(f"\nIntraday analysis:")
    print(f"Average intraday range: {intraday_ranges.mean():.4f} ({intraday_ranges.mean()*100:.2f}%)")
    print(f"Max intraday range: {intraday_ranges.max():.4f} ({intraday_ranges.max()*100:.2f}%)")
    
    # Find best performing periods
    best_10d_periods = rolling_10d_returns.nlargest(5)
    print(f"\nBest 10-day periods:")
    for i, (date, ret) in enumerate(best_10d_periods.items()):
        print(f"  {i+1}. {date.strftime('%Y-%m-%d')}: {ret:.4f} ({ret*100:.2f}%)")
    
    return {
        'daily_returns': daily_returns,
        'rolling_10d_returns': rolling_10d_returns,
        'daily_data': daily_data,
        'volatility': volatility,
        'intraday_ranges': intraday_ranges
    }

def analyze_high_frequency_opportunities():
    """Analyze 1-minute data for high-frequency opportunities"""
    
    # Load data
    data = pickle.load(open('polygon_QQQ_1m.pkl', 'rb'))
    
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
    
    # Get last 3 months
    three_months_ago = data.index.max() - timedelta(days=90)
    recent_data = data[data.index >= three_months_ago]
    
    # Calculate 1-minute returns
    minute_returns = recent_data['close'].pct_change().dropna()
    
    print("\n" + "="*60)
    print("HIGH-FREQUENCY OPPORTUNITY ANALYSIS")
    print("="*60)
    
    print(f"Total 1-minute periods: {len(minute_returns)}")
    print(f"Average 1-minute return: {minute_returns.mean():.6f}")
    print(f"1-minute return std dev: {minute_returns.std():.6f}")
    print(f"Best 1-minute return: {minute_returns.max():.6f}")
    print(f"Worst 1-minute return: {minute_returns.min():.6f}")
    
    # Find large moves
    large_moves = minute_returns[abs(minute_returns) > minute_returns.std() * 3]
    print(f"\nLarge moves (>3 std dev): {len(large_moves)}")
    print(f"Average large move: {large_moves.mean():.6f}")
    
    # Analyze volume patterns
    volume_analysis = recent_data.groupby(recent_data.index.time)['volume'].mean()
    high_volume_times = volume_analysis.sort_values(ascending=False).head(5)
    print(f"\nHighest volume times:")
    for time, vol in high_volume_times.items():
        print(f"  {time}: {vol:,.0f}")
    
    return {
        'minute_returns': minute_returns,
        'large_moves': large_moves,
        'volume_analysis': volume_analysis
    }

if __name__ == "__main__":
    # Analyze recent performance
    daily_stats = analyze_recent_performance()
    
    # Analyze high-frequency opportunities
    hf_stats = analyze_high_frequency_opportunities()
    
    print("\n" + "="*60)
    print("REALISTIC TARGET ASSESSMENT")
    print("="*60)
    
    print("Your targets:")
    print("  - 0.5% daily gain")
    print("  - 5% over 2 weeks")
    
    print("\nRecent QQQ performance:")
    print(f"  - Average daily return: {daily_stats['daily_returns'].mean()*100:.2f}%")
    print(f"  - Average 10-day return: {daily_stats['rolling_10d_returns'].mean()*100:.2f}%")
    
    print("\nAssessment:")
    print("  - Your daily target is {:.1f}x the average daily return".format(
        0.005 / daily_stats['daily_returns'].mean() if daily_stats['daily_returns'].mean() > 0 else float('inf')
    ))
    print("  - Your 10-day target is {:.1f}x the average 10-day return".format(
        0.05 / daily_stats['rolling_10d_returns'].mean() if daily_stats['rolling_10d_returns'].mean() > 0 else float('inf')
    ))
    
    print("\nRecommendation:")
    print("  - These targets are extremely aggressive")
    print("  - Consider more realistic targets or high-risk strategies")
    print("  - Focus on volatility capture rather than consistent gains") 