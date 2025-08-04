# MetaTradingAI Architecture Documentation

## 🏗️ **System Overview**

MetaTradingAI is a sophisticated **aggressive meta-adaptive trading system** designed to achieve **5% returns over 2-week periods**. The system uses machine learning to dynamically select the optimal trading strategy for each hour based on current market conditions.

## 📊 **Core Architecture**

### **1. Data Layer**
```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  • QQQ 1-minute OHLCV data (Polygon.io)                  │
│  • Historical data: 5 years (2020-2025)                   │
│  • Trading hours: 9:30 AM - 4:00 PM ET                   │
│  • Weekdays only (Monday-Friday)                          │
└─────────────────────────────────────────────────────────────┘
```

### **2. Feature Engineering Layer**
```
┌─────────────────────────────────────────────────────────────┐
│                FEATURE ENGINEERING                         │
├─────────────────────────────────────────────────────────────┤
│  Price Features:                                           │
│  • price_change (1-min, 3-min, 5-min, 15-min)           │
│  • momentum_3m, momentum_5m, momentum_15m                │
│                                                           │
│  Volatility Features:                                     │
│  • volatility_3m, volatility_5m, volatility_15m          │
│                                                           │
│  Volume Features:                                         │
│  • volume_ratio, volume_change                           │
│                                                           │
│  Trend Features:                                          │
│  • sma_3, sma_10, trend_3_10                            │
│                                                           │
│  Gap Features:                                            │
│  • gap_size (daily gaps)                                 │
│                                                           │
│  Time Features:                                           │
│  • hour, day_of_week                                     │
└─────────────────────────────────────────────────────────────┘
```

### **3. Strategy Selection Layer**
```
┌─────────────────────────────────────────────────────────────┐
│                STRATEGY SELECTOR                           │
├─────────────────────────────────────────────────────────────┤
│  Model: Random Forest Classifier                          │
│  • n_estimators: 100                                     │
│  • random_state: 42                                      │
│  • Features: 15+ market features                          │
│  • Output: Strategy probabilities                         │
│                                                           │
│  Training Process:                                        │
│  • Group data by hour                                     │
│  • Calculate best strategy per hour                       │
│  • Train on historical performance                        │
│  • Scale features with StandardScaler                     │
└─────────────────────────────────────────────────────────────┘
```

### **4. Strategy Pool Layer**
```
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGY POOL                          │
├─────────────────────────────────────────────────────────────┤
│  1. Aggressive Volatility Exploitation                    │
│     • Volatility threshold: 0.005                         │
│     • Momentum period: 3                                 │
│     • Max hold: 15 minutes                               │
│                                                           │
│  2. Aggressive Momentum Amplification                     │
│     • Short period: 2, Medium: 5, Long: 15              │
│     • Momentum threshold: 0.003                          │
│     • Max hold: 20 minutes                               │
│                                                           │
│  3. Aggressive Gap Exploitation                          │
│     • Gap threshold: 0.002                               │
│     • Momentum threshold: 0.001                          │
│     • Max hold: 20 minutes                               │
│                                                           │
│  4. Aggressive High-Frequency Scalping                   │
│     • Scalp threshold: 0.001                             │
│     • Volume threshold: 1.05                             │
│     • Max hold: 5 minutes                                │
└─────────────────────────────────────────────────────────────┘
```

### **5. Execution Layer**
```
┌─────────────────────────────────────────────────────────────┐
│                   EXECUTION LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  • Strategy execution per hour                            │
│  • Position tracking (Long/Short/Flat)                   │
│  • Trade recording and PnL calculation                   │
│  • Performance metrics calculation                        │
│  • Risk management (max hold periods)                     │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 **Data Flow Process**

### **Step 1: Data Loading**
```python
# Load 1-minute QQQ data
data = pickle.load(open('polygon_QQQ_1m.pkl', 'rb'))

# Filter to trading hours and weekdays
data = data.between_time('09:30', '16:00')
data = data[data.index.dayofweek < 5]

# Filter to last 5 years
five_years_ago = data.index.max() - timedelta(days=5*365)
data = data[data.index >= five_years_ago]
```

### **Step 2: Feature Extraction**
```python
def extract_market_features(data):
    features = pd.DataFrame(index=data.index)
    
    # Price-based features
    features['price_change'] = data['close'].pct_change()
    features['price_change_3m'] = data['close'].pct_change(3)
    features['price_change_5m'] = data['close'].pct_change(5)
    features['price_change_15m'] = data['close'].pct_change(15)
    
    # Volatility features
    features['volatility_3m'] = data['close'].pct_change().rolling(3).std()
    features['volatility_5m'] = data['close'].pct_change().rolling(5).std()
    features['volatility_15m'] = data['close'].pct_change().rolling(15).std()
    
    # Volume features
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
    features['volume_change'] = data['volume'].pct_change()
    
    # Momentum features
    features['momentum_3m'] = data['close'].pct_change(3)
    features['momentum_5m'] = data['close'].pct_change(5)
    features['momentum_15m'] = data['close'].pct_change(15)
    
    # Trend features
    features['sma_3'] = data['close'].rolling(3).mean()
    features['sma_10'] = data['close'].rolling(10).mean()
    features['trend_3_10'] = (features['sma_3'] - features['sma_10']) / features['sma_10']
    
    # Gap features
    daily_data = data.resample('D').agg({'open': 'first', 'close': 'last'}).dropna()
    gaps = (daily_data['open'] - daily_data['close'].shift(1)) / daily_data['close'].shift(1)
    gap_signals = gaps.reindex(data.index, method='ffill')
    features['gap_size'] = gap_signals
    
    # Time features
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    
    return features.dropna()
```

### **Step 3: Strategy Selection**
```python
def select_best_strategy(current_data):
    # Extract features for current data
    features = extract_market_features(current_data)
    
    # Use most recent features
    current_features = features.iloc[-1:].values
    current_features_scaled = scaler.transform(current_features)
    
    # Predict probabilities for each strategy
    probabilities = selector_model.predict_proba(current_features_scaled)
    
    # Find strategy with highest probability
    best_strategy_idx = np.argmax(probabilities[0])
    return strategies[best_strategy_idx]
```

### **Step 4: Strategy Execution**
```python
def backtest_strategy(strategy, data):
    signals = strategy.calculate_signals(data)
    
    position = 0  # 1 for long, -1 for short, 0 for flat
    capital = 100000
    equity = [capital]
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
    
    return {
        'total_return': (equity[-1] - 100000) / 100000,
        'trades': trades,
        'num_trades': len(trades)
    }
```

## 🎯 **Strategy Details**

### **Strategy 1: Aggressive Volatility Exploitation**
```python
def calculate_signals(self, data):
    # Calculate volatility
    volatility = data['close'].pct_change().rolling(10).std()
    
    # Calculate momentum
    momentum = data['close'].pct_change(self.parameters['momentum_period'])
    
    # Calculate volume confirmation
    avg_volume = data['volume'].rolling(10).mean()
    volume_ratio = data['volume'] / avg_volume
    
    signals = pd.Series(0, index=data.index)
    
    # Entry conditions - more aggressive
    high_vol = volatility > self.parameters['volatility_threshold']
    volume_confirmed = volume_ratio > self.parameters['volume_threshold']
    
    # Long on positive momentum during high volatility
    long_condition = (
        (momentum > self.parameters['reversal_threshold']) & 
        high_vol & 
        volume_confirmed
    )
    
    # Short on negative momentum during high volatility
    short_condition = (
        (momentum < -self.parameters['reversal_threshold']) & 
        high_vol & 
        volume_confirmed
    )
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals
```

### **Strategy 2: Aggressive Momentum Amplification**
```python
def calculate_signals(self, data):
    # Calculate multiple momentum indicators
    short_momentum = data['close'].pct_change(self.parameters['short_period'])
    medium_momentum = data['close'].pct_change(self.parameters['medium_period'])
    long_momentum = data['close'].pct_change(self.parameters['long_period'])
    
    # Calculate acceleration
    acceleration = short_momentum.diff()
    
    # Calculate volume confirmation
    avg_volume = data['volume'].rolling(10).mean()
    volume_ratio = data['volume'] / avg_volume
    
    # Calculate volatility
    volatility = data['close'].pct_change().rolling(5).std()
    
    signals = pd.Series(0, index=data.index)
    
    # Strong buy when all momentum indicators align
    strong_buy = (
        (short_momentum > self.parameters['momentum_threshold']) &
        (medium_momentum > self.parameters['momentum_threshold']) &
        (long_momentum > 0) &
        (acceleration > self.parameters['acceleration_threshold']) &
        (volume_ratio > self.parameters['volume_threshold']) &
        (volatility < volatility.rolling(20).mean() * 3)
    )
    
    # Strong sell when all momentum indicators align negatively
    strong_sell = (
        (short_momentum < -self.parameters['momentum_threshold']) &
        (medium_momentum < -self.parameters['momentum_threshold']) &
        (long_momentum < 0) &
        (acceleration < -self.parameters['acceleration_threshold']) &
        (volume_ratio > self.parameters['volume_threshold']) &
        (volatility < volatility.rolling(20).mean() * 3)
    )
    
    signals[strong_buy] = 1
    signals[strong_sell] = -1
    
    return signals
```

## 📊 **Performance Tracking**

### **Hourly Performance**
```python
hourly_perf = {
    'date': date,
    'hour': hour,
    'selected_strategy': selected_strategy.name,
    'avg_hourly_return': strategy_results['avg_hourly_return'],
    'total_return': strategy_results['total_return'],
    'num_trades': strategy_results['num_trades']
}
```

### **Daily Performance**
```python
daily_perf = {
    'date': date,
    'daily_return': daily_return,
    'daily_trades': daily_trades,
    'cumulative_return': cumulative_return
}
```

### **Overall Metrics**
- **Total Return**: Overall portfolio performance
- **Total Trades**: Number of trades executed
- **Strategy Distribution**: Which strategies were selected most often
- **Execution Time**: System performance speed
- **Target Achievement**: Progress toward 5% goal

## 🔧 **Configuration Parameters**

### **System Configuration**
```python
# Training period (days)
training_period_days = 60

# Test period (days) 
test_period_days = 10

# Target return
target_return = 0.05  # 5%
```

### **Strategy Parameters**
```python
# Aggressive Volatility Exploitation
volatility_threshold = 0.005  # Lower for more signals
momentum_period = 3          # Shorter for faster response
reversal_threshold = 0.005   # Lower for more signals
max_hold_period = 15         # Shorter hold period
volume_threshold = 1.1       # Lower volume requirement

# Aggressive Momentum Amplification
short_period = 2             # Very short period
medium_period = 5            # Shorter medium period
long_period = 15             # Shorter long period
momentum_threshold = 0.003   # Lower threshold
acceleration_threshold = 0.001  # Lower acceleration threshold
max_hold_period = 20         # Shorter hold period

# Aggressive Gap Exploitation
gap_threshold = 0.002        # Lower gap threshold
fade_threshold = 0.003       # Lower fade threshold
momentum_threshold = 0.001   # Lower momentum threshold
max_hold_period = 20         # Shorter hold period

# Aggressive High-Frequency Scalping
scalp_threshold = 0.001      # Very low threshold
volume_threshold = 1.05      # Very low volume requirement
max_hold_period = 5          # Very short hold period
```

## 🚀 **Real-Time Implementation**

### **Live Trading Setup**
1. **Data Feed**: Connect to Polygon.io or Alpaca for real-time data
2. **Strategy Selection**: Run every hour during market hours
3. **Order Execution**: Execute trades through broker API
4. **Risk Management**: Implement position sizing and stop losses
5. **Monitoring**: Track performance and send alerts

### **Hourly Execution Loop**
```python
while market_open:
    # Get current market data
    current_data = get_live_data()
    
    # Select best strategy for this hour
    selected_strategy = selector.select_best_strategy(current_data)
    
    # Execute strategy for this hour
    results = selected_strategy.execute(current_data)
    
    # Record performance
    record_performance(results)
    
    # Wait for next hour
    time.sleep(3600)  # 1 hour
```

## 📈 **Performance Optimization**

### **Current Performance**
- **Execution Speed**: 2.1 seconds for full system
- **Memory Usage**: ~500MB
- **Accuracy**: Strategy selection based on historical performance
- **Scalability**: Easy to add new strategies

### **Optimization Strategies**
1. **More Aggressive Parameters**: Lower thresholds for more signals
2. **Leverage Implementation**: Use 2-3x leverage to amplify returns
3. **Market Regime Detection**: Adapt to different market conditions
4. **Additional Strategies**: Add trend-following and breakout strategies

## 🔮 **Future Enhancements**

### **Phase 2: Ultra-Aggressive System**
- Lower all thresholds by 50%
- Implement leverage simulation
- Add market regime detection
- Include more strategies

### **Phase 3: Real-Time System**
- Live data integration
- Order execution
- Risk management
- Performance monitoring

### **Phase 4: Advanced Features**
- Web dashboard
- Mobile alerts
- Portfolio optimization
- Multi-asset support

---

**MetaTradingAI Architecture** - Designed for 5% Returns Over 2-Week Periods 