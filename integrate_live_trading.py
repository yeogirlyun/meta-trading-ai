#!/usr/bin/env python3
"""
MetaTradingAI v3.0 - Live Trading Integration
Connects the proven v3.0 system to real-time trading execution
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our MetaTradingAI v3.0 system
from aggressive_meta_trading_ai_v3 import AggressiveMetaTradingAI, detect_market_regime

# Import live trading components
from live_trading_integration import (
    LiveTradingConfig, 
    LiveTradingEngine, 
    LiveTradingMonitor,
    create_live_trading_system
)

# ============================================================================
# CONFIGURATION
# ============================================================================

def setup_logging():
    """Setup logging for live trading"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('live_trading.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_live_config() -> LiveTradingConfig:
    """Create live trading configuration"""
    config = LiveTradingConfig()
    
    # Ultra-aggressive settings based on v3.0 success
    config.leverage = 2.0  # 2x leverage for higher returns
    config.max_position_size = 0.15  # 15% of capital per trade
    config.max_daily_loss = 0.08  # 8% max daily loss
    config.max_drawdown = 0.20  # 20% max drawdown
    
    # Execution settings
    config.execution_delay = 0.5  # Faster execution
    config.update_frequency = 30  # 30-second updates
    config.strategy_selection_interval = 1800  # 30 minutes
    config.regime_detection_interval = 180  # 3 minutes
    config.performance_evaluation_interval = 900  # 15 minutes
    
    return config

# ============================================================================
# LIVE TRADING INTEGRATION
# ============================================================================

class MetaTradingLiveIntegration:
    """Integrates MetaTradingAI v3.0 with live trading execution"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.config = create_live_config()
        self.trading_engine = None
        self.monitor = None
        self.meta_trading_system = None
        
    def initialize_system(self):
        """Initialize the complete live trading system"""
        self.logger.info("Initializing MetaTradingAI v3.0 Live Trading System")
        
        # Create live trading components
        self.trading_engine, self.monitor = create_live_trading_system()
        
        # Initialize MetaTradingAI v3.0 system
        self.logger.info("Loading MetaTradingAI v3.0 system...")
        self.meta_trading_system = AggressiveMetaTradingAI()
        
        # Connect the systems
        self.trading_engine.initialize_trading_system(self.meta_trading_system)
        
        self.logger.info("System initialization complete")
        
    def start_live_trading(self):
        """Start live trading session"""
        if not self.trading_engine or not self.meta_trading_system:
            self.logger.error("System not initialized")
            return
        
        self.logger.info("Starting live trading session...")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start trading engine
        self.trading_engine.start_trading()
        
    def stop_live_trading(self):
        """Stop live trading session"""
        self.logger.info("Stopping live trading session...")
        
        if self.trading_engine:
            self.trading_engine.stop_trading()
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        self.logger.info("Live trading session stopped")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.trading_engine:
            return {}
        
        return {
            'capital': self.trading_engine.risk_manager.current_capital,
            'daily_pnl': self.trading_engine.risk_manager.daily_pnl,
            'max_drawdown': self.trading_engine.risk_manager.max_drawdown,
            'total_return': (self.trading_engine.risk_manager.current_capital - 
                           self.config.initial_capital) / self.config.initial_capital,
            'positions': self.trading_engine.order_manager.get_positions(),
            'total_orders': len(self.trading_engine.order_manager.get_order_history()),
            'active_strategy': self.trading_engine.last_strategy.name if self.trading_engine.last_strategy else None,
            'market_regime': self.trading_engine.last_regime
        }

# ============================================================================
# API INTEGRATION HELPERS
# ============================================================================

def setup_polygon_connection():
    """Setup Polygon.io connection for real-time data"""
    # In production, this would use the actual Polygon.io API
    # For now, we'll use the existing data file
    print("Polygon.io connection would be configured here")
    print("Using historical data for simulation")

def setup_alpaca_connection():
    """Setup Alpaca.markets connection for order execution"""
    # In production, this would use the actual Alpaca API
    # For now, we'll simulate order execution
    print("Alpaca.markets connection would be configured here")
    print("Using simulated order execution")

def setup_kafka_connection():
    """Setup Kafka for message queuing"""
    # In production, this would use Kafka for communication
    # For now, we'll use direct function calls
    print("Kafka connection would be configured here")
    print("Using direct function calls for communication")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run live trading integration"""
    
    print("="*80)
    print("MetaTradingAI v3.0 - Live Trading Integration")
    print("="*80)
    print(f"Start Time: {datetime.now()}")
    print(f"Target Return: 5% (v3.0 achieved 7.89% in backtesting)")
    print(f"Leverage: 2x")
    print(f"Risk Management: Active")
    print("="*80)
    
    # Setup API connections
    setup_polygon_connection()
    setup_alpaca_connection()
    setup_kafka_connection()
    
    # Create and initialize the live trading system
    live_system = MetaTradingLiveIntegration()
    live_system.initialize_system()
    
    try:
        # Start live trading
        live_system.start_live_trading()
        
        # Keep the system running
        while True:
            time.sleep(60)  # Check every minute
            
            # Display performance summary
            performance = live_system.get_performance_summary()
            if performance:
                print(f"\nPerformance Summary:")
                print(f"  Capital: ${performance['capital']:,.2f}")
                print(f"  Daily P&L: ${performance['daily_pnl']:,.2f}")
                print(f"  Total Return: {performance['total_return']:.2%}")
                print(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
                print(f"  Total Orders: {performance['total_orders']}")
                print(f"  Active Strategy: {performance['active_strategy']}")
                print(f"  Market Regime: {performance['market_regime']}")
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    except Exception as e:
        print(f"Error in live trading: {e}")
    finally:
        # Stop live trading
        live_system.stop_live_trading()
        
        # Final performance summary
        final_performance = live_system.get_performance_summary()
        if final_performance:
            print("\n" + "="*80)
            print("FINAL PERFORMANCE SUMMARY")
            print("="*80)
            print(f"Final Capital: ${final_performance['capital']:,.2f}")
            print(f"Total Return: {final_performance['total_return']:.2%}")
            print(f"Max Drawdown: {final_performance['max_drawdown']:.2%}")
            print(f"Total Orders: {final_performance['total_orders']}")
            print(f"Session Duration: {datetime.now()}")
            print("="*80)

if __name__ == "__main__":
    main() 