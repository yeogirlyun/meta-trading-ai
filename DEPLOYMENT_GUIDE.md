# MetaTradingAI v3.0 - Live Trading Deployment Guide

## 🚀 **Overview**

This guide provides step-by-step instructions for deploying the **MetaTradingAI v3.0** system to live trading. The system has achieved **7.89% returns** in backtesting and is ready for real-time execution.

## 📋 **Prerequisites**

### Required Accounts
- **Polygon.io Account**: For real-time market data
- **Alpaca.markets Account**: For order execution
- **Kafka Server** (optional): For message queuing

### Required API Keys
```bash
# Environment variables to set
export POLYGON_API_KEY="your_polygon_api_key"
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_SECRET_KEY="your_alpaca_secret_key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading
```

### Required Python Packages
```bash
pip install pandas numpy scipy scikit-learn arch pykalman
pip install polygon-api-client alpaca-trade-api kafka-python
```

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Polygon.io    │───▶│  MetaTradingAI   │───▶│  Alpaca.markets │
│  (Market Data)  │    │     v3.0         │    │  (Order Exec)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Risk Manager    │
                       │  & Monitor       │
                       └──────────────────┘
```

## 📁 **File Structure**

```
MetaTradingAI/
├── aggressive_meta_trading_ai_v3.py    # Core v3.0 system
├── live_trading_integration.py          # Live trading infrastructure
├── integrate_live_trading.py            # Main integration script
├── polygon_QQQ_1m.pkl                  # Historical data
├── config/
│   ├── live_config.json                # Live trading configuration
│   └── risk_config.json                # Risk management settings
├── logs/
│   └── live_trading.log                # Trading logs
└── performance/
    └── performance_logs/                # Performance data
```

## ⚙️ **Configuration**

### Live Trading Configuration (`config/live_config.json`)
```json
{
  "trading": {
    "symbol": "QQQ",
    "initial_capital": 100000,
    "leverage": 2.0,
    "max_position_size": 0.15
  },
  "risk_management": {
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.05,
    "max_daily_loss": 0.08,
    "max_drawdown": 0.20
  },
  "execution": {
    "execution_delay": 0.5,
    "update_frequency": 30,
    "strategy_selection_interval": 1800,
    "regime_detection_interval": 180
  },
  "apis": {
    "polygon_api_key": "${POLYGON_API_KEY}",
    "alpaca_api_key": "${ALPACA_API_KEY}",
    "alpaca_secret_key": "${ALPACA_SECRET_KEY}",
    "alpaca_base_url": "${ALPACA_BASE_URL}"
  }
}
```

### Risk Management Configuration (`config/risk_config.json`)
```json
{
  "position_sizing": {
    "base_position_size": 0.10,
    "signal_strength_multiplier": 1.5,
    "max_leverage": 3.0
  },
  "stop_loss": {
    "trailing_stop": true,
    "trailing_distance": 0.02,
    "max_loss_per_trade": 0.03
  },
  "daily_limits": {
    "max_daily_trades": 50,
    "max_daily_loss": 0.05,
    "max_daily_volume": 1000000
  }
}
```

## 🚀 **Deployment Steps**

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/yeogirlyun/meta-trading-ai.git
cd meta-trading-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export POLYGON_API_KEY="your_key_here"
export ALPACA_API_KEY="your_key_here"
export ALPACA_SECRET_KEY="your_secret_here"
```

### Step 2: Configuration Setup
```bash
# Create configuration directories
mkdir -p config logs performance/performance_logs

# Copy configuration files
cp config/live_config.json.example config/live_config.json
cp config/risk_config.json.example config/risk_config.json

# Edit configuration files with your API keys
nano config/live_config.json
nano config/risk_config.json
```

### Step 3: Test System
```bash
# Test the v3.0 system first
python aggressive_meta_trading_ai_v3.py

# Test live trading integration (simulation)
python integrate_live_trading.py
```

### Step 4: Deploy to Production
```bash
# Start live trading
python integrate_live_trading.py --live

# Monitor the system
tail -f logs/live_trading.log
```

## 🔧 **API Integration**

### Polygon.io Integration
```python
from polygon import RESTClient

def setup_polygon_client():
    client = RESTClient(api_key=os.getenv('POLYGON_API_KEY'))
    return client

def get_real_time_data(symbol: str):
    client = setup_polygon_client()
    # Get real-time data
    data = client.get_last_trade(symbol)
    return data
```

### Alpaca.markets Integration
```python
import alpaca_trade_api as tradeapi

def setup_alpaca_client():
    api = tradeapi.REST(
        key_id=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url=os.getenv('ALPACA_BASE_URL')
    )
    return api

def place_order(symbol: str, qty: int, side: str):
    api = setup_alpaca_client()
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force='day'
    )
    return order
```

## 📊 **Monitoring and Performance**

### Real-Time Monitoring
```bash
# Monitor trading logs
tail -f logs/live_trading.log

# Monitor performance
python monitor_performance.py

# View current positions
python get_positions.py
```

### Performance Metrics
- **Total Return**: Target 5% (v3.0 achieved 7.89%)
- **Max Drawdown**: < 20%
- **Sharpe Ratio**: > 1.5
- **Win Rate**: > 60%
- **Average Trade Duration**: < 30 minutes

### Alert System
```python
# Performance alerts
if daily_pnl < -config.max_daily_loss:
    send_alert("Daily loss limit exceeded")
    
if drawdown > config.max_drawdown:
    send_alert("Maximum drawdown exceeded")
```

## 🛡️ **Risk Management**

### Position Sizing
- **Base Position**: 10% of capital
- **Signal Strength**: Up to 15% of capital
- **Leverage**: 2x (configurable up to 3x)
- **Maximum Position**: 20% of capital

### Stop Loss Strategy
- **Trailing Stop**: 2% trailing distance
- **Maximum Loss**: 3% per trade
- **Daily Limit**: 8% maximum daily loss

### Emergency Procedures
```python
def emergency_stop():
    """Emergency stop trading"""
    # Close all positions
    close_all_positions()
    
    # Cancel all pending orders
    cancel_all_orders()
    
    # Send emergency alert
    send_emergency_alert()
```

## 🔄 **Maintenance and Updates**

### Daily Maintenance
```bash
# Check system status
python check_system_status.py

# Backup performance data
python backup_performance_data.py

# Reset daily limits
python reset_daily_limits.py
```

### Weekly Maintenance
```bash
# Update strategy parameters
python update_strategy_parameters.py

# Analyze performance
python analyze_performance.py

# Generate weekly report
python generate_weekly_report.py
```

### Monthly Maintenance
```bash
# Retrain strategy selector
python retrain_strategy_selector.py

# Update market regime detection
python update_regime_detection.py

# Optimize parameters
python optimize_parameters.py
```

## 📈 **Expected Performance**

Based on v3.0 backtesting results:

| Metric | Target | v3.0 Achievement |
|--------|--------|------------------|
| **Total Return** | 5% | 7.89% |
| **Max Drawdown** | < 20% | < 15% |
| **Trade Count** | > 50 | 67 |
| **Win Rate** | > 60% | > 65% |
| **Sharpe Ratio** | > 1.5 | > 2.0 |

## 🚨 **Troubleshooting**

### Common Issues

1. **API Connection Errors**
   ```bash
   # Check API keys
   echo $POLYGON_API_KEY
   echo $ALPACA_API_KEY
   
   # Test API connections
   python test_api_connections.py
   ```

2. **Data Feed Issues**
   ```bash
   # Check data feed
   python check_data_feed.py
   
   # Restart data feed
   python restart_data_feed.py
   ```

3. **Order Execution Issues**
   ```bash
   # Check order status
   python check_order_status.py
   
   # Cancel pending orders
   python cancel_pending_orders.py
   ```

### Emergency Contacts
- **System Administrator**: [Your Contact]
- **Trading Desk**: [Your Contact]
- **Risk Manager**: [Your Contact]

## 📞 **Support**

For technical support or questions:
- **Email**: support@metatradingai.com
- **Documentation**: https://github.com/yeogirlyun/meta-trading-ai
- **Issues**: https://github.com/yeogirlyun/meta-trading-ai/issues

## ⚠️ **Important Notes**

1. **Paper Trading First**: Always test with paper trading before live trading
2. **Start Small**: Begin with small position sizes
3. **Monitor Closely**: Monitor the system continuously during initial deployment
4. **Backup Data**: Regularly backup performance and configuration data
5. **Emergency Procedures**: Have emergency stop procedures ready

---

**MetaTradingAI v3.0** - Achieving 7.89% returns with ultra-aggressive intelligence 