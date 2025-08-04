# MetaTradingAI

A sophisticated meta-adaptive trading system that uses machine learning to select the optimal trading strategy for different market conditions. The system implements a "pick-and-choose" approach where multiple strategies are evaluated and the best performing one is selected for each time period.

## 🚀 Features

- **Meta-Adaptive Strategy Selection**: Uses Random Forest classifier to predict the best strategy for current market conditions
- **Multiple Trading Strategies**: 
  - Volatility Exploitation
  - Momentum Amplification  
  - Gap Exploitation
  - Mean Reversion
- **Hourly Strategy Selection**: Dynamically selects the best strategy for each hour
- **Fast Execution**: Optimized for real-time performance
- **Comprehensive Backtesting**: Full backtesting framework with performance metrics

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Market Data   │───▶│  Feature Engine  │───▶│ Strategy Selector│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Strategy Pool   │◀───│  Selected       │
                       │                  │    │  Strategy       │
                       │ • Volatility     │    └─────────────────┘
                       │ • Momentum       │              │
                       │ • Gap Exploit    │              ▼
                       │ • Mean Reversion │    ┌─────────────────┐
                       └──────────────────┘    │   Execution     │
                                               │   Engine        │
                                               └─────────────────┘
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MetaTradingAI.git
cd MetaTradingAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scipy` - Optimization algorithms
- `scikit-learn` - Machine learning models
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization

## 🚀 Quick Start

### Basic Usage

```python
from fast_meta_system import FastMetaAdaptiveSystem, FastVolatilityExploitation

# Load your data
data = load_market_data()

# Create strategies
strategies = [
    FastVolatilityExploitation(),
    FastMomentumAmplification(),
    FastGapExploitation(),
    FastMeanReversion()
]

# Initialize system
system = FastMetaAdaptiveSystem(data, strategies)

# Run the system
results = system.run_hourly_adaptive_system(test_period_days=7)
```

### Running the Complete System

```bash
python3 fast_meta_system.py
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Strategy Selection Distribution**: Which strategies were selected most often
- **Hourly Performance**: Detailed breakdown by hour
- **Trade Analysis**: Number of trades, win rate, etc.

## 🔧 Configuration

### Strategy Parameters

Each strategy can be configured with custom parameters:

```python
strategy = FastVolatilityExploitation()
strategy.parameters = {
    'volatility_threshold': 0.008,
    'momentum_period': 5,
    'reversal_threshold': 0.008,
    'max_hold_period': 30,
    'volume_threshold': 1.3
}
```

### System Configuration

```python
# Training period (days)
training_period_days = 30

# Test period (days) 
test_period_days = 7

# Reoptimization frequency
reoptimization_frequency = 'daily'  # or 'hourly'
```

## 📊 Results Analysis

The system provides detailed analysis of:

- **Strategy Performance**: Individual strategy performance metrics
- **Selection Patterns**: Which strategies work best in different conditions
- **Market Regime Analysis**: How strategies perform in different market conditions
- **Risk Metrics**: Drawdown, volatility, and other risk measures

## 🔄 Real-Time Implementation

To implement this system for real-time trading:

1. **Data Feed Integration**: Connect to live market data (Polygon.io, Alpaca, etc.)
2. **Execution Layer**: Add order execution through your broker
3. **Risk Management**: Implement position sizing and stop losses
4. **Monitoring**: Set up alerts and performance monitoring
5. **Retraining**: Schedule periodic model retraining

## 📁 Project Structure

```
MetaTradingAI/
├── fast_meta_system.py          # Fast meta-adaptive system
├── enhanced_adaptive_system.py  # Enhanced optimization system
├── meta_adaptive_system.py      # Full meta-adaptive system
├── strategy_optimizer.py        # Basic strategy optimization
├── aggressive_strategies.py     # Aggressive trading strategies
├── market_analysis.py          # Market data analysis
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore rules
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**MetaTradingAI** - Intelligent Strategy Selection for Optimal Trading Performance 