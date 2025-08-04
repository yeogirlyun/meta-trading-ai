import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StrategyVisualizer:
    """Visualize strategy performance and optimization results"""
    
    def __init__(self, results_file: str = 'strategy_optimization_results.pkl'):
        self.results = self.load_results(results_file)
        
    def load_results(self, file_path: str) -> dict:
        """Load optimization results"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Results file {file_path} not found. Run strategy_optimizer.py first.")
            return None
    
    def plot_equity_curves(self, strategies_to_plot: list = None):
        """Plot equity curves for all strategies"""
        if not self.results:
            return
            
        plt.figure(figsize=(15, 8))
        
        optimization_results = self.results['optimization_results']
        
        if strategies_to_plot is None:
            strategies_to_plot = list(optimization_results.keys())
        
        for strategy_name in strategies_to_plot:
            if strategy_name in optimization_results:
                equity_curve = optimization_results[strategy_name]['final_metrics']['equity_curve']
                plt.plot(equity_curve.index, equity_curve.values, 
                        label=strategy_name, linewidth=2)
        
        plt.title('Strategy Equity Curves (Optimization Period)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('equity_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self):
        """Plot performance comparison bar chart"""
        if not self.results:
            return
            
        comparison_df = self.results['comparison_df']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total Return
        axes[0, 0].bar(comparison_df['Strategy'], comparison_df['Total Return'])
        axes[0, 0].set_title('Total Return')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sharpe Ratio
        axes[0, 1].bar(comparison_df['Strategy'], comparison_df['Sharpe Ratio'])
        axes[0, 1].set_title('Sharpe Ratio')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Max Drawdown
        axes[1, 0].bar(comparison_df['Strategy'], comparison_df['Max Drawdown'])
        axes[1, 0].set_title('Maximum Drawdown')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Win Rate
        axes[1, 1].bar(comparison_df['Strategy'], comparison_df['Win Rate'])
        axes[1, 1].set_title('Win Rate')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_parameter_optimization(self, strategy_name: str):
        """Plot parameter optimization results for a specific strategy"""
        if not self.results or strategy_name not in self.results['optimization_results']:
            print(f"Strategy {strategy_name} not found in results")
            return
            
        opt_result = self.results['optimization_results'][strategy_name]
        optimal_params = opt_result['optimal_parameters']
        
        # Create parameter comparison plot
        param_names = list(optimal_params.keys())
        param_values = list(optimal_params.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(param_names, param_values)
        plt.title(f'{strategy_name} - Optimal Parameters', fontsize=14)
        plt.ylabel('Parameter Value', fontsize=12)
        plt.xlabel('Parameter', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, param_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{strategy_name.lower().replace(" ", "_")}_parameters.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_trade_analysis(self, strategy_name: str):
        """Plot trade analysis for a specific strategy"""
        if not self.results or strategy_name not in self.results['optimization_results']:
            print(f"Strategy {strategy_name} not found in results")
            return
            
        trades = self.results['optimization_results'][strategy_name]['final_metrics']['trades']
        
        if not trades:
            print(f"No trades found for {strategy_name}")
            return
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60  # minutes
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # PnL distribution
        axes[0, 0].hist(trades_df['pnl'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('PnL Distribution')
        axes[0, 0].set_xlabel('PnL')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # Trade duration
        axes[0, 1].hist(trades_df['duration'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Trade Duration Distribution')
        axes[0, 1].set_xlabel('Duration (minutes)')
        axes[0, 1].set_ylabel('Frequency')
        
        # PnL over time
        axes[1, 0].scatter(trades_df['entry_time'], trades_df['pnl'], alpha=0.6)
        axes[1, 0].set_title('PnL Over Time')
        axes[1, 0].set_xlabel('Entry Time')
        axes[1, 0].set_ylabel('PnL')
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.7)
        
        # Cumulative PnL
        cumulative_pnl = trades_df['pnl'].cumsum()
        axes[1, 1].plot(trades_df['entry_time'], cumulative_pnl, linewidth=2)
        axes[1, 1].set_title('Cumulative PnL')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Cumulative PnL')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{strategy_name.lower().replace(" ", "_")}_trade_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        if not self.results:
            print("No results available")
            return
            
        print("="*80)
        print("STRATEGY OPTIMIZATION SUMMARY REPORT")
        print("="*80)
        
        # Data info
        data_info = self.results['data_info']
        print(f"\nData Information:")
        print(f"  Total records: {data_info['shape'][0]:,}")
        print(f"  Date range: {data_info['date_range'][0]} to {data_info['date_range'][1]}")
        
        # Optimization results
        print(f"\nOptimization Results:")
        optimization_results = self.results['optimization_results']
        for strategy_name, result in optimization_results.items():
            print(f"\n{strategy_name}:")
            print(f"  Optimal Parameters: {result['optimal_parameters']}")
            print(f"  Optimization Success: {result['optimization_success']}")
            metrics = result['final_metrics']
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  Total Return: {metrics['total_return']:.3f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.3f}")
            print(f"  Win Rate: {metrics['win_rate']:.3f}")
            print(f"  Number of Trades: {metrics['num_trades']}")
        
        # Out-of-sample comparison
        print(f"\nOut-of-Sample Comparison:")
        comparison_df = self.results['comparison_df']
        print(comparison_df.to_string(index=False))
        
        # Best performing strategy
        best_strategy = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
        print(f"\nBest Strategy (by Sharpe Ratio):")
        print(f"  Strategy: {best_strategy['Strategy']}")
        print(f"  Sharpe Ratio: {best_strategy['Sharpe Ratio']:.3f}")
        print(f"  Total Return: {best_strategy['Total Return']:.3f}")
        print(f"  Max Drawdown: {best_strategy['Max Drawdown']:.3f}")
        print(f"  Win Rate: {best_strategy['Win Rate']:.3f}")


def main():
    """Main visualization function"""
    visualizer = StrategyVisualizer()
    
    if visualizer.results is None:
        print("Please run strategy_optimizer.py first to generate results")
        return
    
    # Create all visualizations
    print("Creating visualizations...")
    
    # Plot equity curves
    visualizer.plot_equity_curves()
    
    # Plot performance comparison
    visualizer.plot_performance_comparison()
    
    # Plot parameter optimization for each strategy
    strategies = list(visualizer.results['optimization_results'].keys())
    for strategy in strategies:
        visualizer.plot_parameter_optimization(strategy)
        visualizer.plot_trade_analysis(strategy)
    
    # Create summary report
    visualizer.create_summary_report()
    
    print("\nAll visualizations saved as PNG files")


if __name__ == "__main__":
    main() 