# MetaTradingAI

A sophisticated **aggressive meta-adaptive trading system** designed to achieve **5% returns over 2-week periods** using machine learning to select the optimal trading strategy for different market conditions. The system implements a "pick-and-choose" approach where multiple aggressive strategies are evaluated and the best performing one is selected for each hour.

## 🎯 **Project Mission**

**Target**: Achieve 5% returns over any 10-day trading period using aggressive meta-trading strategies.

**Current Performance**: 1.21% return over 10 trading days (July 22 - August 1, 2025)

**Status**: ✅ **System Operational** | ❌ **Target Not Yet Achieved**

## 🚀 **Key Features**

### **Meta-Adaptive Strategy Selection**
- **Random Forest Classifier** for strategy selection
- **Real-time market feature analysis**
- **Hourly strategy switching** based on market conditions
- **Aggressive parameter optimization** for higher returns

### **Four Core Aggressive Strategies**
1. **Aggressive Volatility Exploitation** - Exploits high volatility with momentum confirmation
2. **Aggressive Momentum Amplification** - Multi-timeframe momentum alignment
3. **Aggressive Gap Exploitation** - Fades gaps with momentum confirmation
4. **Aggressive High-Frequency Scalping** - Ultra-fast scalping with low thresholds

### **Advanced Features**
- **Hourly performance tracking** with detailed reporting
- **Daily gain summaries** with cumulative returns
- **Strategy selection distribution** analysis
- **Fast execution** (2.1 seconds for full system)
- **Comprehensive backtesting** framework

## 📊 **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Market Data   │───▶│  Feature Engine  │───▶│ Strategy Selector│
│   (1-min QQQ)   │    │                  │    │ (Random Forest) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Strategy Pool   │◀───│  Selected       │
                       │                  │    │  Strategy       │
                       │ • Volatility     │    └─────────────────┘
                       │ • Momentum       │              │
                       │ • Gap Exploit    │              ▼
                       │ • HF Scalping    │    ┌─────────────────┐
                       └──────────────────┘    │   Execution     │
                                               │   Engine        │
                                               └─────────────────┘
```

### **Data Flow**
1. **Market Data Processing**: 1-minute QQQ OHLCV data
2. **Feature Extraction**: 15+ market features (price, volatility, volume, momentum, trends, gaps)
3. **Strategy Selection**: ML-based selection for each hour
4. **Strategy Execution**: Selected strategy runs for the hour
5. **Performance Tracking**: Hourly and daily returns calculation

## 🛠️ **Installation**

1. **Clone the repository:**
```bash
git clone https://github.com/yeogirlyun/meta-trading-ai.git
cd meta-trading-ai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 **Quick Start**

### **Run the Aggressive Meta-Trading System**
```bash
python3 aggressive_meta_trading_ai.py
```

### **Expected Output**
```
Loading data...
Running aggressive meta-trading AI system...
Aggressive Meta-Trading AI System Setup:
  Training period: 2025-05-22 to 2025-07-21
  Test period: 2025-07-22 to 2025-08-01
  Target: 5% return over 10 trading days

=== Trading Day: 2025-07-22 ===
  Hour 16:00 - Selected: Aggressive High-Frequency Scalping
  Daily Summary: Return: 0.00% - Cumulative: 0.00%

=== Trading Day: 2025-07-31 ===
  Daily Summary: Return: 0.45% - Cumulative: 0.45%

=== Trading Day: 2025-08-01 ===
  Daily Summary: Return: 0.76% - Cumulative: 1.21%

🎯 TARGET ACHIEVEMENT:
  Target: 5% return over 10 trading days
  Actual: 1.21% (1.21%)
  Status: ❌ NOT ACHIEVED
```

## 📈 **Performance Metrics**

### **Latest Results (July 22 - August 1, 2025)**
- **Total Return**: 1.21%
- **Total Trades**: 1
- **Execution Time**: 2.1 seconds
- **Strategy Distribution**: Well balanced across all 4 strategies

### **Daily Performance Breakdown**
| Date | Daily Return | Cumulative | Trades |
|------|-------------|------------|--------|
| 2025-07-22 | 0.00% | 0.00% | 0 |
| 2025-07-23 | 0.00% | 0.00% | 0 |
| 2025-07-24 | 0.00% | 0.00% | 0 |
| 2025-07-25 | 0.00% | 0.00% | 0 |
| 2025-07-28 | 0.00% | 0.00% | 0 |
| 2025-07-29 | 0.00% | 0.00% | 0 |
| 2025-07-30 | 0.00% | 0.00% | 0 |
| 2025-07-31 | 0.45% | 0.45% | 0 |
| 2025-08-01 | 0.76% | 1.21% | 1 |

### **Strategy Selection Distribution**
- **Aggressive Momentum Amplification**: 30.8% (20 hours)
- **Aggressive High-Frequency Scalping**: 24.6% (16 hours)
- **Aggressive Volatility Exploitation**: 24.6% (16 hours)
- **Aggressive Gap Exploitation**: 20.0% (13 hours)

## 🔧 **Strategy Details**

### **1. Aggressive Volatility Exploitation**
- **Purpose**: Exploit high volatility periods with momentum confirmation
- **Parameters**: 
  - Volatility threshold: 0.005 (lower for more signals)
  - Momentum period: 3 (shorter for faster response)
  - Max hold period: 15 minutes
- **Logic**: Long on positive momentum during high volatility, short on negative momentum

### **2. Aggressive Momentum Amplification**
- **Purpose**: Multi-timeframe momentum alignment
- **Parameters**:
  - Short period: 2, Medium period: 5, Long period: 15
  - Momentum threshold: 0.003 (lower threshold)
  - Max hold period: 20 minutes
- **Logic**: Strong buy/sell when all momentum indicators align

### **3. Aggressive Gap Exploitation**
- **Purpose**: Fade gaps with momentum confirmation
- **Parameters**:
  - Gap threshold: 0.002 (lower for more signals)
  - Momentum threshold: 0.001
  - Max hold period: 20 minutes
- **Logic**: Short gap ups, long gap downs with momentum confirmation

### **4. Aggressive High-Frequency Scalping**
- **Purpose**: Ultra-fast scalping with very low thresholds
- **Parameters**:
  - Scalp threshold: 0.001 (very low)
  - Volume threshold: 1.05 (very low)
  - Max hold period: 5 minutes
- **Logic**: Quick in-and-out trades on small price movements

## 🎯 **Target Achievement Analysis**

### **Current Status**
- **Target**: 5% return over 10 trading days
- **Actual**: 1.21% return
- **Gap**: 3.79% remaining

### **Challenges Identified**
1. **Low Market Volatility**: Most strategies showing 0.0000 returns
2. **Conservative Behavior**: Only 1 trade in 10 days
3. **Late Performance**: All gains came in the last 2 days

### **Improvement Strategies**
1. **More Aggressive Parameters**: Lower thresholds further
2. **Leverage Implementation**: Use 2-3x leverage to amplify returns
3. **Market Regime Detection**: Adapt to different market conditions
4. **Additional Strategies**: Add trend-following and breakout strategies

## 📁 **Project Structure**

```
MetaTradingAI/
├── aggressive_meta_trading_ai.py    # Main aggressive system
├── requirements.txt                 # Python dependencies
├── README.md                       # This documentation
├── .gitignore                      # Git ignore rules
└── archive/                        # Previous system versions
    ├── fast_meta_system.py
    ├── enhanced_adaptive_system.py
    ├── meta_adaptive_system.py
    └── [other previous versions]
```

## 🔄 **Real-Time Implementation**

### **For Live Trading**
1. **Data Feed Integration**: Connect to Polygon.io or Alpaca
2. **Execution Layer**: Add order execution through broker
3. **Risk Management**: Implement position sizing and stop losses
4. **Monitoring**: Set up alerts and performance tracking
5. **Retraining**: Schedule periodic model retraining

### **Hourly Execution Flow**
```
Hour Start → Market Analysis → Strategy Selection → 
Strategy Execution → Performance Tracking → Hour End
```

## 📊 **Technical Specifications**

### **Data Requirements**
- **Symbol**: QQQ ETF
- **Timeframe**: 1-minute OHLCV
- **Trading Hours**: 9:30 AM - 4:00 PM ET
- **Data Source**: Polygon.io (historical), Alpaca (live)

### **System Requirements**
- **Python**: 3.8+
- **Memory**: 4GB+ RAM
- **Processing**: Multi-core CPU recommended
- **Storage**: 1GB+ for data and models

### **Performance Metrics**
- **Execution Speed**: 2.1 seconds for full system
- **Memory Usage**: ~500MB
- **Accuracy**: Strategy selection based on historical performance
- **Scalability**: Easy to add new strategies

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 **Development Roadmap**

### **Phase 1: Current Status** ✅
- [x] Basic meta-adaptive system
- [x] Four aggressive strategies
- [x] Hourly strategy selection
- [x] Performance tracking

### **Phase 2: Enhancements** 🚧
- [ ] Ultra-aggressive parameters
- [ ] Leverage implementation
- [ ] Market regime detection
- [ ] Additional strategies

### **Phase 3: Real-Time** 📋
- [ ] Live data integration
- [ ] Order execution
- [ ] Risk management
- [ ] Performance monitoring

### **Phase 4: Advanced Features** 📋
- [ ] Web dashboard
- [ ] Mobile alerts
- [ ] Portfolio optimization
- [ ] Multi-asset support

## ⚠️ **Disclaimer**

This software is for **educational and research purposes only**. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

## 📞 **Support**

For questions or support:
- **GitHub Issues**: [Create an issue](https://github.com/yeogirlyun/meta-trading-ai/issues)
- **Documentation**: See this README and code comments
- **Performance**: Check the latest results in the output

---

**MetaTradingAI** - Aggressive Meta-Adaptive Trading for 5% Returns

*Built with ❤️ for intelligent trading* 