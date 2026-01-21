"""
Strategy Optimizer Module

Implements walk-forward optimization for the trading strategy:
- Rolling training/validation windows
- Signal weight optimization
- Out-of-sample performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import product
from dataclasses import dataclass

from .alpha_signals import AlphaSignals
from .portfolio import PortfolioConstructor
from .backtester import Backtester


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    optimal_weights: Dict[str, float]
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_return: float
    out_of_sample_return: float
    window_results: List[Dict]


class StrategyOptimizer:
    """
    Optimizes strategy parameters using walk-forward analysis.
    
    Walk-Forward Process:
    1. Split data into rolling windows
    2. Train (optimize weights) on training period
    3. Validate on out-of-sample period
    4. Roll forward and repeat
    """
    
    def __init__(
        self,
        training_months: int = 12,
        validation_months: int = 3,
        step_months: int = 3
    ):
        """
        Initialize StrategyOptimizer.
        
        Args:
            training_months: Number of months in training window
            validation_months: Number of months in validation window
            step_months: Number of months to step forward each iteration
        """
        self.training_months = training_months
        self.validation_months = validation_months
        self.step_months = step_months
        
        # Weight grid for optimization
        self.weight_options = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
    def generate_weight_combinations(self) -> List[Dict[str, float]]:
        """
        Generate all valid weight combinations for three signals.
        
        Returns:
            List of weight dictionaries that sum to 1
        """
        combinations = []
        
        for w1, w2 in product(self.weight_options, repeat=2):
            w3 = 1.0 - w1 - w2
            
            if 0 <= w3 <= 1.0 and abs(w1 + w2 + w3 - 1.0) < 0.01:
                combinations.append({
                    'momentum': w1,
                    'mean_reversion': w2,
                    'volatility': w3
                })
                
        return combinations
    
    def calculate_strategy_sharpe(
        self,
        prices: pd.DataFrame,
        weights: Dict[str, float],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        benchmark_prices: Optional[pd.Series] = None
    ) -> Tuple[float, float]:
        """
        Calculate Sharpe ratio for given weight combination.
        
        Args:
            prices: DataFrame of close prices
            weights: Signal weight dictionary
            start_date: Start of period
            end_date: End of period
            benchmark_prices: Optional benchmark prices for regime filter
            
        Returns:
            Tuple of (sharpe_ratio, cumulative_return)
        """
        # Filter prices to date range
        mask = (prices.index >= start_date) & (prices.index <= end_date)
        period_prices = prices[mask]
        
        if len(period_prices) < 50:  # Minimum data requirement
            return float('-inf'), 0.0
        
        # Generate signals with custom weights
        signals_gen = AlphaSignals(signal_weights=weights)
        signals = signals_gen.generate_all_signals(period_prices)
        combined = signals_gen.combine_signals(signals, weights)
        
        # Construct portfolio
        pc = PortfolioConstructor()
        
        # Calculate strategy returns using vector operations from portfolio.py
        # This is more robust than the manual loop and ensures consistency with main backtest
        trading_dates = period_prices.index
        
        # If benchmark_prices is provided, align it
        period_benchmark_prices = None
        if benchmark_prices is not None:
             period_benchmark_prices = benchmark_prices[prices.index[0]:prices.index[-1]]
             # Reindex to match specific period if needed, or let generate_weight_history handle it
             # It's safer to pass the full series and let the method handle alignment? 
             # No, generate_weight_history expects the series to match the trading_dates length or be reindexed
             # Actually, generate_weight_history aligns internally. 
             # Let's just pass the full benchmark_prices and let portfolio.py handle date intersection.
        
        weight_history = pc.generate_weight_history(
            combined, trading_dates, benchmark_prices
        )
        
        # Calculate returns
        returns = period_prices.pct_change()
        portfolio_returns = (weight_history.shift(1) * returns).sum(axis=1)
        
        # Transaction costs approximation (simplified for speed)
        # In optimization we often skip granular transaction costs for speed, 
        # or we can apply a drag.
        # Let's apply the standard 10bps cost
        turnover = (weight_history - weight_history.shift(1)).abs().sum(axis=1)
        costs = turnover * 0.0010
        net_returns = portfolio_returns - costs
        
        # Trim to requested period
        # mask is aligned to prices, but returns lose the first row
        net_returns = net_returns[start_date:end_date]
        
        if len(net_returns) < 10:
             return float('-inf'), 0.0
             
        # Returns Series matching the manual loop return type
        returns = net_returns
        
        # Calculate Sharpe ratio
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        
        if annualized_vol > 0:
            sharpe = (annualized_return - 0.02) / annualized_vol
        else:
            sharpe = 0.0
        
        cumulative_return = (1 + returns).prod() - 1
        
        return sharpe, cumulative_return
    
    def optimize_weights(
        self,
        prices: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        benchmark_prices: Optional[pd.Series] = None
    ) -> Tuple[Dict[str, float], float]:
        """
        Find optimal weights for the training period.
        
        Args:
            prices: DataFrame of close prices
            start_date: Training period start
            end_date: Training period end
            benchmark_prices: Optional benchmark prices for regime filter
            
        Returns:
            Tuple of (optimal_weights, best_sharpe)
        """
        weight_combinations = self.generate_weight_combinations()
        
        best_sharpe = float('-inf')
        best_weights = {'momentum': 1/3, 'mean_reversion': 1/3, 'volatility': 1/3}
        
        for weights in weight_combinations:
            sharpe, _ = self.calculate_strategy_sharpe(
                prices, weights, start_date, end_date, benchmark_prices
            )
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights.copy()
                
        return best_weights, best_sharpe
    
    def walk_forward_optimize(
        self,
        prices: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_prices: Optional[pd.Series] = None
    ) -> OptimizationResult:
        """
        Perform walk-forward optimization.
        
        Args:
            prices: DataFrame of close prices
            benchmark_returns: Optional benchmark for comparison
            benchmark_prices: Optional benchmark prices for regime filter
            
        Returns:
            OptimizationResult with all optimization details
        """
        print("Starting walk-forward optimization...")
        
        window_results = []
        all_is_sharpes = []
        all_oos_sharpes = []
        all_is_returns = []
        all_oos_returns = []
        
        # Get date range
        start_date = prices.index[0]
        end_date = prices.index[-1]
        
        training_days = self.training_months * 21  # Approximate trading days
        validation_days = self.validation_months * 21
        step_days = self.step_months * 21
        
        current_start = start_date
        window_num = 0
        
        while True:
            # Define training and validation periods
            training_end = current_start + pd.Timedelta(days=training_days * 1.5)
            validation_start = training_end + pd.Timedelta(days=1)
            validation_end = validation_start + pd.Timedelta(days=validation_days * 1.5)
            
            # Check if we have enough data
            if validation_end > end_date:
                break
                
            window_num += 1
            print(f"\nWindow {window_num}:")
            print(f"  Training: {current_start.date()} to {training_end.date()}")
            print(f"  Validation: {validation_start.date()} to {validation_end.date()}")
            
            # Optimize on training period
            # Note: We pass benchmark_prices to also optimize WITH the regime filter active
            # This ensures the weights are optimal for the regime-adjusted strategy
            optimal_weights, is_sharpe = self.optimize_weights(
                prices, current_start, training_end, benchmark_prices
            )
            
            print(f"  Optimal weights: {optimal_weights}")
            print(f"  In-sample Sharpe: {is_sharpe:.2f}")
            
            # Validate on out-of-sample period
            oos_sharpe, oos_return = self.calculate_strategy_sharpe(
                prices, optimal_weights, validation_start, validation_end, benchmark_prices
            )
            
            _, is_return = self.calculate_strategy_sharpe(
                prices, optimal_weights, current_start, training_end, benchmark_prices
            )
            
            print(f"  Out-of-sample Sharpe: {oos_sharpe:.2f}")
            print(f"  Out-of-sample Return: {oos_return:.2%}")
            
            # Store results
            window_results.append({
                'window': window_num,
                'training_start': current_start,
                'training_end': training_end,
                'validation_start': validation_start,
                'validation_end': validation_end,
                'optimal_weights': optimal_weights,
                'is_sharpe': is_sharpe,
                'oos_sharpe': oos_sharpe,
                'is_return': is_return,
                'oos_return': oos_return
            })
            
            all_is_sharpes.append(is_sharpe)
            all_oos_sharpes.append(oos_sharpe)
            all_is_returns.append(is_return)
            all_oos_returns.append(oos_return)
            
            # Step forward
            current_start += pd.Timedelta(days=step_days * 1.5)
        
        # Calculate average optimal weights across all windows
        avg_weights = {
            'momentum': np.mean([w['optimal_weights']['momentum'] for w in window_results]),
            'mean_reversion': np.mean([w['optimal_weights']['mean_reversion'] for w in window_results]),
            'volatility': np.mean([w['optimal_weights']['volatility'] for w in window_results])
        }
        
        # Normalize to sum to 1
        total = sum(avg_weights.values())
        if total > 0:
            avg_weights = {k: v/total for k, v in avg_weights.items()}
        
        print("\n" + "="*50)
        print("OPTIMIZATION SUMMARY")
        print("="*50)
        print(f"Total windows analyzed: {window_num}")
        print(f"\nAverage optimal weights:")
        for signal, weight in avg_weights.items():
            print(f"  {signal}: {weight:.2%}")
        print(f"\nAverage in-sample Sharpe: {np.mean(all_is_sharpes):.2f}")
        print(f"Average out-of-sample Sharpe: {np.mean(all_oos_sharpes):.2f}")
        print(f"\nAverage in-sample return: {np.mean(all_is_returns):.2%}")
        print(f"Average out-of-sample return: {np.mean(all_oos_returns):.2%}")
        
        return OptimizationResult(
            optimal_weights=avg_weights,
            in_sample_sharpe=np.mean(all_is_sharpes),
            out_of_sample_sharpe=np.mean(all_oos_sharpes),
            in_sample_return=np.mean(all_is_returns),
            out_of_sample_return=np.mean(all_oos_returns),
            window_results=window_results
        )
    
    def plot_optimization_results(
        self,
        result: OptimizationResult,
        output_dir: str = 'output'
    ) -> None:
        """
        Plot optimization results.
        
        Args:
            result: OptimizationResult from walk_forward_optimize
            output_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: In-sample vs Out-of-sample Sharpe
        ax1 = axes[0, 0]
        windows = [w['window'] for w in result.window_results]
        is_sharpes = [w['is_sharpe'] for w in result.window_results]
        oos_sharpes = [w['oos_sharpe'] for w in result.window_results]
        
        ax1.plot(windows, is_sharpes, 'b-o', label='In-Sample', linewidth=2)
        ax1.plot(windows, oos_sharpes, 'r-o', label='Out-of-Sample', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('In-Sample vs Out-of-Sample Sharpe Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Weight evolution
        ax2 = axes[0, 1]
        mom_weights = [w['optimal_weights']['momentum'] for w in result.window_results]
        mr_weights = [w['optimal_weights']['mean_reversion'] for w in result.window_results]
        vol_weights = [w['optimal_weights']['volatility'] for w in result.window_results]
        
        ax2.stackplot(windows, mom_weights, mr_weights, vol_weights,
                     labels=['Momentum', 'Mean Reversion', 'Volatility'],
                     colors=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Weight')
        ax2.set_title('Optimal Weight Evolution')
        ax2.legend(loc='upper right')
        ax2.set_ylim(0, 1)
        
        # Plot 3: Returns comparison
        ax3 = axes[1, 0]
        is_returns = [w['is_return'] * 100 for w in result.window_results]
        oos_returns = [w['oos_return'] * 100 for w in result.window_results]
        
        x = np.arange(len(windows))
        width = 0.35
        
        ax3.bar(x - width/2, is_returns, width, label='In-Sample', color='#3498db')
        ax3.bar(x + width/2, oos_returns, width, label='Out-of-Sample', color='#e74c3c')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Return (%)')
        ax3.set_title('In-Sample vs Out-of-Sample Returns')
        ax3.set_xticks(x)
        ax3.set_xticklabels(windows)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final optimal weights pie chart
        ax4 = axes[1, 1]
        weights = result.optimal_weights
        labels = [f"{k.replace('_', ' ').title()}\n({v:.1%})" for k, v in weights.items()]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        ax4.pie([v for v in weights.values()], labels=labels, colors=colors,
               autopct='', startangle=90)
        ax4.set_title('Optimized Signal Weights')
        
        plt.tight_layout()
        fig.savefig(output_path / 'optimization_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Optimization results saved to {output_path / 'optimization_results.png'}")


if __name__ == "__main__":
    # Test optimizer
    from data_collector import DataCollector
    
    print("Downloading data for optimization test...")
    collector = DataCollector(years=3)  # Use 3 years for faster testing
    collector.download_price_data()
    
    prices = collector.get_combined_prices()
    
    optimizer = StrategyOptimizer(
        training_months=12,
        validation_months=3,
        step_months=3
    )
    
    result = optimizer.walk_forward_optimize(prices)
    optimizer.plot_optimization_results(result)
