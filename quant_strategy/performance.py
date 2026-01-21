"""
Performance Analysis Module

Calculates performance metrics and generates visualizations:
- Cumulative returns chart
- Drawdown analysis
- Signal contribution breakdown
- Monthly returns heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')


class PerformanceAnalyzer:
    """
    Analyzes strategy performance and generates visualizations.
    
    Provides comprehensive analysis including:
    - Performance metrics
    - Return attribution
    - Risk analysis
    - Visual reports
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize PerformanceAnalyzer.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_summary_report(
        self,
        result,  # BacktestResult
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Generate a summary report of strategy performance.
        
        Args:
            result: BacktestResult from backtester
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dictionary of performance metrics
        """
        returns = result.daily_returns
        
        report = {
            'Performance Metrics': {
                'Cumulative Return': f"{result.cumulative_return:.2%}",
                'Annualized Return': f"{result.annualized_return:.2%}",
                'Annualized Volatility': f"{result.volatility:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
            },
            'Risk Metrics': {
                'Alpha': f"{result.alpha:.2%}",
                'Beta': f"{result.beta:.2f}",
                'Skewness': f"{returns.skew():.2f}",
                'Kurtosis': f"{returns.kurtosis():.2f}",
            },
            'Trading Statistics': {
                'Total Trades': len(result.trades),
                'Trading Days': len(returns),
                'Win Rate': self._calculate_win_rate(returns),
                'Best Day': f"{returns.max():.2%}",
                'Worst Day': f"{returns.min():.2%}",
            }
        }
        
        return report
    
    def _calculate_win_rate(self, returns: pd.Series) -> str:
        """Calculate percentage of positive return days."""
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = positive_days / total_days if total_days > 0 else 0
        return f"{win_rate:.2%}"
    
    def plot_cumulative_returns(
        self,
        result,
        benchmark_returns: Optional[pd.Series] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot cumulative returns of strategy vs benchmark.
        
        Args:
            result: BacktestResult from backtester
            benchmark_returns: Optional benchmark returns
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Strategy cumulative returns
        strategy_cumulative = (1 + result.daily_returns).cumprod()
        ax.plot(strategy_cumulative.index, strategy_cumulative.values, 
                label='Strategy', linewidth=2, color='#2ecc71')
        
        # Benchmark cumulative returns
        if benchmark_returns is not None:
            # Align dates
            aligned_benchmark = benchmark_returns.reindex(result.daily_returns.index).dropna()
            benchmark_cumulative = (1 + aligned_benchmark).cumprod()
            ax.plot(benchmark_cumulative.index, benchmark_cumulative.values,
                   label='S&P 500', linewidth=2, color='#3498db', alpha=0.7)
        
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Cumulative Returns: Strategy vs Benchmark', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'cumulative_returns.png', dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_drawdown(self, result, save: bool = True) -> plt.Figure:
        """
        Plot drawdown chart.
        
        Args:
            result: BacktestResult from backtester
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        
        portfolio_values = result.portfolio_values
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        
        ax.fill_between(drawdowns.index, 0, drawdowns.values, 
                       color='#e74c3c', alpha=0.5)
        ax.plot(drawdowns.index, drawdowns.values, color='#c0392b', linewidth=1)
        
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Annotate max drawdown
        max_dd_idx = drawdowns.idxmin()
        max_dd_val = drawdowns.min()
        ax.annotate(f'Max DD: {max_dd_val:.1f}%', 
                   xy=(max_dd_idx, max_dd_val),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=10, color='#c0392b',
                   arrowprops=dict(arrowstyle='->', color='#c0392b'))
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'drawdown.png', dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_monthly_returns_heatmap(self, result, save: bool = True) -> plt.Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            result: BacktestResult from backtester
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        returns = result.daily_returns
        
        # Calculate monthly returns
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table for heatmap
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values * 100
        })
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(pivot, annot=True, fmt='.1f', center=0,
                   cmap='RdYlGn', linewidths=0.5,
                   cbar_kws={'label': 'Return (%)'}, ax=ax)
        
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'monthly_returns_heatmap.png', dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_rolling_sharpe(
        self,
        result,
        window: int = 63,  # ~3 months
        save: bool = True
    ) -> plt.Figure:
        """
        Plot rolling Sharpe ratio.
        
        Args:
            result: BacktestResult from backtester
            window: Rolling window in days
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        returns = result.daily_returns
        
        # Calculate rolling Sharpe
        rolling_mean = returns.rolling(window=window).mean() * 252
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean - 0.02) / rolling_std
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
               color='#9b59b6', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Sharpe = -1')
        
        ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                       where=rolling_sharpe.values > 0, color='#2ecc71', alpha=0.3)
        ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                       where=rolling_sharpe.values < 0, color='#e74c3c', alpha=0.3)
        
        ax.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'rolling_sharpe.png', dpi=150, bbox_inches='tight')
            
        return fig
    
    def analyze_signal_contribution(
        self,
        prices: pd.DataFrame,
        result,
        save: bool = True
    ) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Analyze the contribution of each alpha signal to performance.
        
        Args:
            prices: Price data used in backtest
            result: BacktestResult from backtester
            save: Whether to save the plot
            
        Returns:
            Tuple of (contribution DataFrame, matplotlib figure)
        """
        from .alpha_signals import AlphaSignals
        from .portfolio import PortfolioConstructor
        
        # Generate signals
        signals_gen = AlphaSignals()
        signals = signals_gen.generate_all_signals(prices)
        
        # For each signal, calculate the return if only that signal was used
        signal_returns = {}
        
        for signal_name, signal_df in signals.items():
            # Create portfolio using only this signal
            pc = PortfolioConstructor()
            
            # Get valid dates
            valid_dates = signal_df.dropna(how='all').index
            
            returns_series = []
            
            for i, date in enumerate(valid_dates[:-1]):
                try:
                    signal_scores = signal_df.loc[date].dropna()
                    if len(signal_scores) >= 6:
                        long_tickers, short_tickers = pc.select_positions(signal_scores)
                        
                        next_date = valid_dates[i + 1]
                        
                        # Calculate next day return
                        long_return = prices.loc[next_date, long_tickers].pct_change().mean() if i > 0 else 0
                        short_return = -prices.loc[next_date, short_tickers].pct_change().mean() if i > 0 else 0
                        
                        daily_return = (long_return + short_return) / 2
                        returns_series.append({'date': next_date, 'return': daily_return})
                except:
                    continue
                    
            if returns_series:
                signal_returns[signal_name] = pd.DataFrame(returns_series).set_index('date')['return']
        
        # Create contribution analysis
        contribution_df = pd.DataFrame(signal_returns)
        
        # Calculate cumulative returns for each signal
        cumulative_returns = {}
        for col in contribution_df.columns:
            clean_returns = contribution_df[col].dropna()
            if len(clean_returns) > 0:
                cumulative_returns[col] = (1 + clean_returns).prod() - 1
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cumulative returns by signal
        ax1 = axes[0]
        for signal_name in contribution_df.columns:
            clean_returns = contribution_df[signal_name].dropna()
            if len(clean_returns) > 0:
                cumulative = (1 + clean_returns).cumprod()
                ax1.plot(cumulative.index, cumulative.values, label=signal_name.title(), linewidth=2)
        
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Cumulative Returns by Alpha Signal', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bar chart of total contribution
        ax2 = axes[1]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax2.bar(cumulative_returns.keys(), 
                       [v * 100 for v in cumulative_returns.values()],
                       color=colors[:len(cumulative_returns)])
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Total Return Contribution by Signal', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        
        # Add value labels on bars
        for bar, val in zip(bars, cumulative_returns.values()):
            height = bar.get_height()
            ax2.annotate(f'{val*100:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'signal_contribution.png', dpi=150, bbox_inches='tight')
        
        return contribution_df, fig
    
    def analyze_periods(
        self,
        result,
        benchmark_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Analyze periods of outperformance and underperformance.
        
        Args:
            result: BacktestResult from backtester
            benchmark_returns: Optional benchmark returns
            
        Returns:
            DataFrame with period analysis
        """
        returns = result.daily_returns
        
        # Define periods (quarters)
        quarterly_returns = returns.resample('QE').apply(lambda x: (1 + x).prod() - 1)
        
        periods = []
        
        for period_end, ret in quarterly_returns.items():
            period_start = period_end - pd.offsets.QuarterEnd()
            
            period_data = {
                'Period Start': period_start.strftime('%Y-%m-%d'),
                'Period End': period_end.strftime('%Y-%m-%d'),
                'Strategy Return': f"{ret:.2%}",
            }
            
            if benchmark_returns is not None:
                period_mask = (benchmark_returns.index >= period_start) & \
                             (benchmark_returns.index <= period_end)
                bench_ret = benchmark_returns[period_mask]
                if len(bench_ret) > 0:
                    bench_return = (1 + bench_ret).prod() - 1
                    period_data['Benchmark Return'] = f"{bench_return:.2%}"
                    period_data['Excess Return'] = f"{ret - bench_return:.2%}"
                    period_data['Status'] = 'Outperform' if ret > bench_return else 'Underperform'
            
            periods.append(period_data)
        
        return pd.DataFrame(periods)
    
    def plot_long_short_performance(self, result, save: bool = True) -> plt.Figure:
        """
        Plot performance breakdown of long vs short positions.
        
        Args:
            result: BacktestResult from backtester
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        positions_history = result.positions_history
        
        if positions_history.empty:
            print("No position history available for long/short analysis")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Sum long and short positions separately
        long_cols = [c for c in positions_history.columns if c.startswith('L_')]
        short_cols = [c for c in positions_history.columns if c.startswith('S_')]
        
        if long_cols:
            long_values = positions_history[long_cols].sum(axis=1)
            ax.plot(long_values.index, long_values.values, 
                   label='Long Exposure', color='#2ecc71', linewidth=2)
        
        if short_cols:
            short_values = positions_history[short_cols].sum(axis=1)
            ax.plot(short_values.index, short_values.values,
                   label='Short Exposure', color='#e74c3c', linewidth=2)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Long vs Short Exposure Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Exposure ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'long_short_exposure.png', dpi=150, bbox_inches='tight')
            
        return fig
    
    def generate_all_visualizations(
        self,
        result,
        prices: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> None:
        """
        Generate all performance visualizations.
        
        Args:
            result: BacktestResult from backtester
            prices: Price data used in backtest
            benchmark_returns: Optional benchmark returns
        """
        print("Generating performance visualizations...")
        
        print("  - Cumulative returns chart...")
        self.plot_cumulative_returns(result, benchmark_returns)
        
        print("  - Drawdown chart...")
        self.plot_drawdown(result)
        
        print("  - Monthly returns heatmap...")
        self.plot_monthly_returns_heatmap(result)
        
        print("  - Rolling Sharpe ratio...")
        self.plot_rolling_sharpe(result)
        
        print("  - Signal contribution analysis...")
        self.analyze_signal_contribution(prices, result)
        
        print("  - Long/Short exposure...")
        self.plot_long_short_performance(result)
        
        print(f"\nAll visualizations saved to: {self.output_dir.absolute()}")


if __name__ == "__main__":
    # Test performance analyzer
    print("This module should be run via main.py for full analysis")
