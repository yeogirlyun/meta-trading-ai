#!/usr/bin/env python3
"""
Analyze training data usage in MetaTradingAI v3.0
"""

import pickle
import pandas as pd
from datetime import datetime, timedelta

def analyze_training_data():
    """Analyze the training data usage in v3.0"""
    
    print("MetaTradingAI v3.0 - Training Data Analysis")
    print("="*60)
    
    # Load data
    data = pickle.load(open('polygon_QQQ_1m.pkl', 'rb'))
    
    # Handle timezone issues
    try:
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
    except:
        pass
    
    print(f"Total data available: {len(data)} records")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Calculate years of data
    total_days = (data.index.max() - data.index.min()).days
    total_years = total_days / 365
    print(f"Total years of data: {total_years:.1f} years")
    
    # Analyze v3.0 training period
    test_period_days = 10
    training_days = 60
    
    end_date = data.index.max()
    test_start_date = end_date - timedelta(days=test_period_days)
    training_start_date = test_start_date - timedelta(days=training_days)
    
    print(f"\nCurrent v3.0 Training Configuration:")
    print(f"  Training period: {training_days} days ({training_days/30:.1f} months)")
    print(f"  Test period: {test_period_days} days")
    print(f"  Training start: {training_start_date.date()}")
    print(f"  Training end: {test_start_date.date()}")
    print(f"  Test start: {test_start_date.date()}")
    print(f"  Test end: {end_date.date()}")
    
    # Calculate training data size
    training_data = data[(data.index >= training_start_date) & (data.index < test_start_date)]
    test_data = data[data.index >= test_start_date]
    
    print(f"\nData Usage:")
    print(f"  Training records: {len(training_data):,}")
    print(f"  Test records: {len(test_data):,}")
    print(f"  Training hours: {len(training_data) // 60}")
    print(f"  Test hours: {len(test_data) // 60}")
    
    # Analyze different training periods
    print(f"\nTraining Period Analysis:")
    periods = [
        (30, "1 month"),
        (60, "2 months (current)"),
        (90, "3 months"),
        (180, "6 months"),
        (365, "1 year"),
        (730, "2 years"),
        (1095, "3 years")
    ]
    
    for days, description in periods:
        if days <= total_days:
            start_date = end_date - timedelta(days=test_period_days + days)
            period_data = data[(data.index >= start_date) & (data.index < test_start_date)]
            records = len(period_data)
            hours = records // 60
            
            print(f"  {description:12} ({days:3d} days): {records:8,} records, {hours:6,} hours")
    
    # Recommendations
    print(f"\nRecommendations for Better Consistency:")
    print(f"1. Current training period: {training_days} days ({training_days/30:.1f} months)")
    print(f"2. Recommended minimum: 180 days (6 months) for better consistency")
    print(f"3. Optimal training period: 365-730 days (1-2 years)")
    print(f"4. Available for training: {total_days - test_period_days} days")
    
    # Calculate potential improvements
    current_hours = len(training_data) // 60
    recommended_hours = (180 * 6.5 * 60) // 60  # 6 months of trading hours
    
    print(f"\nPotential Improvements:")
    print(f"  Current training hours: {current_hours:,}")
    print(f"  Recommended training hours: {recommended_hours:,}")
    print(f"  Improvement factor: {recommended_hours / current_hours:.1f}x more data")
    
    # Market regime analysis
    print(f"\nMarket Regime Considerations:")
    print(f"  - 60 days may not capture all market regimes")
    print(f"  - 6+ months provides better regime coverage")
    print(f"  - 1+ year captures full market cycles")
    print(f"  - More data = more robust strategy selection")
    
    return {
        'total_data': len(data),
        'total_years': total_years,
        'current_training_days': training_days,
        'current_training_hours': current_hours,
        'recommended_training_days': 180,
        'recommended_training_hours': recommended_hours,
        'improvement_factor': recommended_hours / current_hours
    }

if __name__ == "__main__":
    results = analyze_training_data() 