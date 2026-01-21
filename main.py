"""
Quantitative Long-Short Trading Strategy
=========================================

This script runs the complete quantitative trading strategy including:
1. Data collection from Yahoo Finance
2. Alpha signal generation (momentum, mean reversion, volatility)
3. Portfolio construction (long top 3, short bottom 3)
4. Backtesting with monthly rebalancing
5. Risk management (stop-loss, dollar-neutral)
6. Performance analysis and visualization
7. Walk-forward optimization (optional)

Usage:
    python main.py [--optimize]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from quant_strategy.data_collector import DataCollector
from quant_strategy.alpha_signals import AlphaSignals
from quant_strategy.portfolio import PortfolioConstructor
from quant_strategy.backtester import Backtester
from quant_strategy.risk_manager import RiskManager
from quant_strategy.performance import PerformanceAnalyzer
from quant_strategy.optimizer import StrategyOptimizer


def print_header():
    """Print strategy header."""
    print("=" * 60)
    print("  QUANTITATIVE LONG-SHORT TRADING STRATEGY")
    print("=" * 60)
    print(f"  Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60 + "\n")


def main(run_optimization: bool = False):
    """
    Run the complete quantitative trading strategy.
    
    Args:
        run_optimization: Whether to run walk-forward optimization
    """
    print_header()
    
    # =========================================================================
    # STEP 1: DATA COLLECTION
    # =========================================================================
    print_section("STEP 1: DATA COLLECTION")
    
    collector = DataCollector(years=5)
    collector.download_all()
    
    prices = collector.get_combined_prices()
    benchmark_returns = collector.get_benchmark_returns()
    benchmark_data = collector.get_benchmark_data()
    benchmark_prices = benchmark_data['Close'] if benchmark_data is not None else None
    
    print(f"\nPrice data shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"\nFundamental data summary:")
    for ticker, data in collector.fundamental_data.items():
        pe = data.get('pe_ratio', 'N/A')
        pe_str = f"{pe:.1f}" if pe and pe != 'N/A' else 'N/A'
        mcap = data.get('market_cap', 0)
        mcap_str = f"${mcap/1e9:.1f}B" if mcap else 'N/A'
        print(f"  {ticker}: P/E={pe_str}, MarketCap={mcap_str}")
    
    # =========================================================================
    # STEP 2: ALPHA SIGNAL ANALYSIS
    # =========================================================================
    print_section("STEP 2: ALPHA SIGNAL ANALYSIS")
    
    signals_gen = AlphaSignals()
    signals = signals_gen.generate_all_signals(prices)
    combined = signals_gen.combine_signals(signals)
    
    print("Latest Alpha Scores (higher = more bullish):")
    latest_alpha = combined.iloc[-1].sort_values(ascending=False)
    for ticker, score in latest_alpha.items():
        print(f"  {ticker}: {score:+.3f}")
    
    print("\nSignal Contributions for Latest Date:")
    contributions = signals_gen.get_signal_contributions(prices)
    print(contributions.sort_values('combined', ascending=False).to_string())
    
    # =========================================================================
    # STEP 3: PORTFOLIO CONSTRUCTION
    # =========================================================================
    print_section("STEP 3: PORTFOLIO CONSTRUCTION")
    
    pc = PortfolioConstructor(n_long=3, n_short=3)
    long_tickers, short_tickers = pc.select_positions(combined.iloc[-1])
    
    print(f"Long Positions (Top 3):  {', '.join(long_tickers)}")
    print(f"Short Positions (Bottom 3): {', '.join(short_tickers)}")
    
    weights = pc.calculate_target_weights(
        long_tickers, 
        short_tickers, 
        prices.columns.tolist(),
        market_regime='BULL' # Default to showing bull weights for display
    )
    print("\nPortfolio Weights (Example in BULL regime):")
    for ticker, weight in weights[weights != 0].sort_values(ascending=False).items():
        position_type = "LONG" if weight > 0 else "SHORT"
        print(f"  {ticker}: {weight:+.2%} ({position_type})")
    
    # =========================================================================
    # STEP 4: BACKTESTING
    # =========================================================================
    print_section("STEP 4: BACKTESTING")
    
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost_bps=10,
        risk_free_rate=0.02
    )
    
    result = backtester.run(prices, benchmark_returns, benchmark_prices)
    
    print("Backtest Configuration:")
    print(f"  Initial Capital: $100,000")
    print(f"  Transaction Costs: 10 bps (0.10%)")
    print(f"  Rebalancing: Monthly")
    print(f"  Stop-Loss: -10% per position")
    
    print(f"\nBacktest Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Total Trading Days: {len(result.daily_returns)}")
    print(f"Total Trades Executed: {len(result.trades)}")
    
    # =========================================================================
    # STEP 5: PERFORMANCE ANALYSIS
    # =========================================================================
    print_section("STEP 5: PERFORMANCE ANALYSIS")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    analyzer = PerformanceAnalyzer(output_dir=str(output_dir))
    
    # Generate summary report
    report = analyzer.generate_summary_report(result, benchmark_returns)
    
    print("Performance Metrics:")
    for category, metrics in report.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.generate_all_visualizations(result, prices, benchmark_returns)
    
    # Analyze periods of outperformance/underperformance
    print("\nQuarterly Performance Analysis:")
    periods_df = analyzer.analyze_periods(result, benchmark_returns)
    print(periods_df.to_string(index=False))
    
    # =========================================================================
    # STEP 6: WALK-FORWARD OPTIMIZATION (OPTIONAL)
    # =========================================================================
    if run_optimization:
        print_section("STEP 6: WALK-FORWARD OPTIMIZATION")
        
        optimizer = StrategyOptimizer(
            training_months=12,
            validation_months=3,
            step_months=3
        )
        
        opt_result = optimizer.walk_forward_optimize(prices, benchmark_returns, benchmark_prices)
        optimizer.plot_optimization_results(opt_result, output_dir=str(output_dir))
        
        print("\nOptimized Strategy Comparison:")
        print(f"Average In-Sample Sharpe: {opt_result.in_sample_sharpe:.2f}")
        print(f"Average Out-of-Sample Sharpe: {opt_result.out_of_sample_sharpe:.2f}")
        print(f"\nRecommended Signal Weights:")
        for signal, weight in opt_result.optimal_weights.items():
            print(f"  {signal}: {weight:.2%}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("STRATEGY SUMMARY")
    
    print("Key Insights:")
    print(f"  • Strategy generated {result.cumulative_return:.2%} cumulative return")
    print(f"  • Sharpe ratio of {result.sharpe_ratio:.2f} indicates {'good' if result.sharpe_ratio > 1 else 'moderate' if result.sharpe_ratio > 0 else 'poor'} risk-adjusted returns")
    print(f"  • Maximum drawdown of {result.max_drawdown:.2%} shows downside risk")
    print(f"  • Alpha of {result.alpha:.2%} vs S&P 500 benchmark")
    print(f"  • Beta of {result.beta:.2f} shows {'low' if abs(result.beta) < 0.3 else 'moderate'} market correlation (dollar-neutral)")
    
    print(f"\nOutput files saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file in output_dir.glob('*.png'):
        print(f"  • {file.name}")
    
    print("\n" + "=" * 60)
    print("  STRATEGY EXECUTION COMPLETE")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Quantitative Long-Short Trading Strategy"
    )
    parser.add_argument(
        '--no-optimize',
        dest='optimize',
        action='store_false',
        help='Skip walk-forward optimization (runs faster)',
        default=True
    )
    
    args = parser.parse_args()
    
    result = main(run_optimization=args.optimize)
