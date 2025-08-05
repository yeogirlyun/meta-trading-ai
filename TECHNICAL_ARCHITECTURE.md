# MetaTradingAI Technical Architecture

## 🏗️ **System Overview**

MetaTradingAI is a sophisticated algorithmic trading system designed for consistent 5%+ returns over 10-day periods. The system employs a multi-layered architecture combining machine learning, regime detection, and adaptive strategy selection.

### **Core Architecture Components**

```
┌─────────────────────────────────────────────────────────────┐
│                    MetaTradingAI v4.0                     │
├─────────────────────────────────────────────────────────────┤
│  Data Layer: 1-minute OHLCV QQQ data (5 years)          │
├─────────────────────────────────────────────────────────────┤
│  Feature Engineering: 85+ technical indicators            │
├─────────────────────────────────────────────────────────────┤
│  Regime Detection: Dynamic market state classification    │
├─────────────────────────────────────────────────────────────┤
│  Strategy Pool: 8 adaptive strategies                    │
├─────────────────────────────────────────────────────────────┤
│  Meta-Selector: Random Forest with future performance    │
├─────────────────────────────────────────────────────────────┤
│  Risk Management: Dynamic position sizing & leverage      │
├─────────────────────────────────────────────────────────────┤
│  Execution Engine: Real-time order management            │
└─────────────────────────────────────────────────────────────┘
```

## 📊 **Data Processing Pipeline**

### **Data Sources & Preprocessing**
- **Primary Data**: QQQ 1-minute OHLCV from Polygon.io
- **Time Range**: Last 5 years (2020-2025)
- **Trading Hours**: 9:30 AM - 4:00 PM EST
- **Trading Days**: Monday-Friday only
- **Data Points**: ~440,000 records

### **Feature Engineering Pipeline**
```python
def create_advanced_features(data):
    features = {}
    
    # Price-based features (15 features)
    features['price_change'] = data['close'].pct_change()
    features['price_change_3m'] = data['close'].pct_change(3)
    features['price_change_5m'] = data['close'].pct_change(5)
    features['price_change_15m'] = data['close'].pct_change(15)
    
    # Volatility features (8 features)
    features['volatility_3m'] = data['close'].pct_change().rolling(3).std()
    features['volatility_5m'] = data['close'].pct_change().rolling(5).std()
    features['volatility_15m'] = data['close'].pct_change().rolling(15).std()
    features['vol_of_vol'] = features['volatility_5m'].rolling(10).std()
    
    # Volume features (6 features)
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
    features['volume_imbalance'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).mean()
    
    # Momentum features (12 features)
    features['momentum_3m'] = data['close'].pct_change(3)
    features['momentum_5m'] = data['close'].pct_change(5)
    features['momentum_15m'] = data['close'].pct_change(15)
    features['momentum_acceleration'] = features['momentum_5m'].diff()
    
    # Trend features (8 features)
    features['sma_3'] = data['close'].rolling(3).mean()
    features['sma_10'] = data['close'].rolling(10).mean()
    features['trend_3_10'] = (features['sma_3'] - features['sma_10']) / features['sma_10']
    
    # Signal strength features (4 features)
    features['signal_strength'] = features['momentum_3m'] * features['volume_ratio'] / (features['volatility_5m'] + 1e-6)
    
    # Time features (4 features)
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    
    # Microstructure features (8 features)
    features['kyle_lambda'] = data['close'].diff().abs() / data['volume']
    features['amihud_illiquidity'] = features['kyle_lambda'].rolling(20).mean()
    features['net_buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    return features
```

## 🎯 **Regime Detection System**

### **Market Regime Classification**
The system identifies three distinct market regimes:

1. **High Volatility Regime**
   - Volatility > 1.8x historical average
   - Volume spike > 1.3x average
   - Strategy Pool: Ultra Volatility Exploitation, Breakout Momentum

2. **Trending Regime**
   - Trend strength > 0.6
   - Momentum > 0.002 (15-minute)
   - Strategy Pool: Ultra Momentum Amplification, Accelerated MA Cross

3. **Ranging Regime**
   - Low volatility, sideways movement
   - Strategy Pool: Ultra High-Frequency Scalping, Extreme Mean Reversion

### **Regime Detection Algorithm**
```python
def detect_market_regime(data):
    volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
    avg_vol = data['close'].pct_change().rolling(50).std().mean()
    trend_strength = abs(data['close'].pct_change().rolling(20).mean().iloc[-1]) / volatility
    
    volume_ratio = data['volume'].iloc[-10:].mean() / data['volume'].rolling(50).mean().iloc[-1]
    momentum_15m = data['close'].pct_change(15).iloc[-1]
    
    if volatility > avg_vol * 1.8 and volume_ratio > 1.3:
        return "high_volatility"
    elif trend_strength > 0.6 and abs(momentum_15m) > 0.002:
        return "trending"
    else:
        return "ranging"
```

## 🧠 **Strategy Pool Architecture**

### **Strategy Categories**

#### **1. Volatility-Based Strategies**
- **Ultra Volatility Exploitation**
  - Parameters: `volatility_threshold=0.001`, `momentum_period=1`
  - Logic: Trade momentum during high volatility periods
  - Risk: High frequency, rapid reversals

- **Breakout Momentum**
  - Parameters: `breakout_period=5`, `volume_multiplier=2.0`
  - Logic: Trade breakouts with volume confirmation
  - Risk: False breakouts, whipsaws

#### **2. Momentum-Based Strategies**
- **Ultra Momentum Amplification**
  - Parameters: `short_period=1`, `medium_period=3`, `long_period=5`
  - Logic: Multi-timeframe momentum alignment
  - Risk: Momentum exhaustion

- **Accelerated MA Cross**
  - Parameters: `fast_period=2`, `slow_period=8`
  - Logic: Fast moving average crossovers
  - Risk: Lag in trend changes

#### **3. Mean Reversion Strategies**
- **Ultra High-Frequency Scalping**
  - Parameters: `scalp_threshold=0.0006`, `max_hold_period=5`
  - Logic: Quick scalps on small price movements
  - Risk: High transaction costs

- **Extreme Mean Reversion**
  - Parameters: `mean_period=8`, `deviation_threshold=0.0015`
  - Logic: Trade extreme deviations from short-term mean
  - Risk: Trend continuation

#### **4. Advanced ML Strategies**
- **GARCH Volatility Forecasting**
  - Parameters: `forecast_horizon=5`, `optimal_vol_min=0.0001`
  - Logic: Forecast volatility and trade optimal ranges
  - Risk: Model instability

- **Kalman Filter Adaptive MA**
  - Parameters: `observation_covariance=0.01`
  - Logic: Dynamically adjusting moving average
  - Risk: Overfitting to noise

## 🤖 **Meta-Learning System**

### **Strategy Selector Architecture**
```python
class AggressiveStrategySelector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.strategies = []
        self.is_trained = False
    
    def extract_market_features(self, data):
        features = pd.DataFrame(index=data.index)
        
        # Price features
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
        
        # Signal strength features
        features['signal_strength'] = features['momentum_3m'] * features['volume_ratio'] / (features['volatility_5m'] + 1e-6)
        
        # Advanced features
        features['vol_of_vol'] = features['volatility_5m'].rolling(10).std()
        features['momentum_acceleration'] = features['momentum_5m'].diff()
        
        # Time features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        
        return features.dropna()
```

### **Training Methodology**
The selector is trained on **future performance** rather than past performance:

1. **Feature Extraction**: Extract market features for each hour
2. **Performance Calculation**: Calculate each strategy's performance for the next hour
3. **Label Generation**: Label each hour with the best-performing strategy
4. **Model Training**: Train Random Forest to predict the best strategy

## 💰 **Risk Management System**

### **Dynamic Position Sizing**
```python
class DynamicPositionSizer:
    def __init__(self):
        self.base_position_size = 0.15  # 15% base
        self.max_position_size = 0.40   # 40% max with confidence
        self.kelly_fraction = 0.25      # Conservative Kelly
        self.min_position_size = 0.05   # 5% minimum
    
    def calculate_position_size(self, signal_strength, win_rate, avg_win_loss_ratio, volatility):
        # Kelly Criterion
        kelly_percentage = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        kelly_percentage = max(0, min(kelly_percentage, 1))
        kelly_position = kelly_percentage * self.kelly_fraction
        
        # Signal strength adjustment
        confidence_multiplier = min(signal_strength / 1.5, 2.0)
        
        # Volatility adjustment (inverse relationship)
        vol_adjustment = 1.0 / (1.0 + volatility * 100)
        
        # Calculate final position size
        position_size = self.base_position_size * confidence_multiplier * vol_adjustment
        position_size = min(position_size, self.max_position_size)
        position_size = max(position_size * kelly_position, self.min_position_size)
        
        return position_size
```

### **Adaptive Leverage Management**
```python
class AdaptiveLeverageManager:
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
```

## 🔄 **Multi-Timeframe Confirmation System**

### **Timeframe Hierarchy**
```python
class MultiTimeframeStrategy:
    def __init__(self):
        self.timeframes = {
            '1min': 1,    # Weight: 0.4 (40%)
            '5min': 5,    # Weight: 0.3 (30%)
            '15min': 15,  # Weight: 0.2 (20%)
            '60min': 60   # Weight: 0.1 (10%)
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
```

## 📈 **Performance Optimization**

### **Walk-Forward Testing**
The system uses walk-forward testing to ensure robustness:

1. **Training Period**: 180 days (6 months)
2. **Test Period**: 10 days (2 weeks)
3. **Advancement**: 1 week between tests
4. **Validation**: Multiple 2-week periods across different market conditions

### **Consistency Metrics**
- **Target Achievement Rate**: Percentage of periods achieving 5%+ returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss Ratio**: Average profit per winning trade / Average loss per losing trade

## 🔧 **System Configuration**

### **Key Parameters**
```python
# Training Configuration
training_days = 180          # 6 months of training data
test_period_days = 10       # 2 weeks of testing
order_frequency_limit = 120  # 2 minutes between orders

# Strategy Parameters
min_signal_strength = 1.2   # Minimum signal quality
min_volume_ratio = 1.3      # Minimum volume confirmation
min_momentum_threshold = 0.0008  # Minimum momentum

# Risk Management
base_position_size = 0.15   # 15% base position size
max_position_size = 0.40    # 40% maximum position size
min_leverage = 1.0          # Minimum leverage
max_leverage = 3.0          # Maximum leverage
```

### **Advanced Features**
- **Dynamic Position Sizing**: Kelly Criterion with volatility adjustment
- **Multi-Timeframe Confirmation**: 4-timeframe signal confluence
- **Regime-Based Strategy Selection**: Adaptive strategy pools
- **Microstructure Analysis**: Order flow and liquidity features
- **Performance-Based Weighting**: Adaptive strategy weights
- **Real-Time Constraints**: 2-minute order frequency limits

## 🚀 **Deployment Architecture**

### **Live Trading Components**
1. **Real-Time Data Feed**: Polygon.io WebSocket connection
2. **Order Management**: Alpaca.markets API integration
3. **Risk Management**: Real-time position monitoring
4. **Performance Tracking**: Live P&L and metrics
5. **Alert System**: Email/SMS notifications

### **System Requirements**
- **Data Storage**: 149MB pickle file (5 years of 1-minute data)
- **Processing**: Multi-core CPU for parallel strategy execution
- **Memory**: 8GB RAM for real-time feature calculation
- **Network**: Low-latency internet for live trading
- **Backup**: Redundant data feeds and execution systems

This technical architecture provides the foundation for consistent 5%+ returns while maintaining robust risk management and adaptability to changing market conditions. 