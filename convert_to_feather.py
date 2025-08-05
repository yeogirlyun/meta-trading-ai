#!/usr/bin/env python3
"""
Convert pickle data to Feather format for faster loading
"""

import pandas as pd
import pickle
from datetime import timedelta

def convert_pickle_to_feather():
    """Convert polygon_QQQ_1m.pkl to Feather format"""
    print("Converting pickle data to Feather format...")
    
    # Load pickle data
    with open('polygon_QQQ_1m.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"Original data: {len(data):,} records")
    
    # Convert index to datetime if needed
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except ValueError:
            data.index = pd.to_datetime(data.index, utc=True)
    
    # Handle timezone if present
    try:
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
    except Exception:
        pass
    
    # Filter to trading hours and weekdays
    data = data.between_time('09:30', '16:00')
    data = data[data.index.dayofweek < 5]
    
    # Filter to last 5 years
    five_years_ago = data.index.max() - timedelta(days=5*365)
    data = data[data.index >= five_years_ago]
    
    print(f"Filtered data: {len(data):,} records")
    
    # Save as Feather
    data.to_feather('polygon_QQQ_1m.feather')
    print("Data converted and saved to polygon_QQQ_1m.feather")
    
    # Verify the file
    test_data = pd.read_feather('polygon_QQQ_1m.feather')
    print(f"Verification: {len(test_data):,} records loaded from Feather")
    
    return data

if __name__ == "__main__":
    convert_pickle_to_feather() 