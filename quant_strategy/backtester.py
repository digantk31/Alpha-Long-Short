"""
Backtesting Module

Event-driven backtesting engine for the long-short strategy.
Simulates portfolio performance over historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .alpha_signals import AlphaSignals
from .portfolio import PortfolioConstructor
from .risk_manager import RiskManager, Position


@dataclass
class Trade:
    """Represents a single trade."""
    date: pd.Timestamp
    ticker: str
    action: str  # 'BUY', 'SELL', 'SHORT', 'COVER'
    shares: float
    price: float
    value: float
    transaction_cost: float


@dataclass
class BacktestResult:
    """Container for backtest results."""
    portfolio_values: pd.Series
    daily_returns: pd.Series
    positions_history: pd.DataFrame
    weights_history: pd.DataFrame
    trades: List[Trade]
    
    # Performance metrics
    cumulative_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0


class Backtester:
    """
    Event-driven backtesting engine for the long-short strategy.
    
    Features:
    - Daily portfolio value tracking
    - Monthly rebalancing
    - Transaction cost modeling
    - Stop-loss execution
    - Performance metrics calculation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 10.0,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize the Backtester.
        
        Args:
            initial_capital: Starting portfolio value
            transaction_cost_bps: Transaction costs in basis points
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.risk_free_rate = risk_free_rate
        
        # Components
        self.alpha_signals = AlphaSignals()
        self.portfolio_constructor = PortfolioConstructor()
        self.risk_manager = RiskManager()
        
        # State tracking
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
    def run(
        self,
        prices: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        benchmark_prices: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run the backtest over the given price data.
        
        Args:
            prices: DataFrame of close prices (dates x tickers)
            benchmark_returns: Optional benchmark returns for alpha calculation
            benchmark_prices: Optional benchmark prices for regime filter
            
        Returns:
            BacktestResult with all metrics and history
        """
        # Generate alpha signals
        signals = self.alpha_signals.generate_all_signals(prices)
        combined_alpha = self.alpha_signals.combine_signals(signals)
        
        # Get weight history with monthly rebalancing
        trading_dates = prices.index
        weight_history = self.portfolio_constructor.generate_weight_history(
            combined_alpha, trading_dates, benchmark_prices
        )
        
        # Initialize tracking
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        
        portfolio_values = []
        positions_history = []
        
        # Get rebalance dates
        rebalance_dates = set(self.portfolio_constructor.get_rebalance_dates(
            trading_dates[0], trading_dates[-1], trading_dates
        ))
        
        # Determine valid start date (need enough data for signals)
        warmup_period = max(
            self.alpha_signals.momentum_window,
            self.alpha_signals.ma_window,
            self.alpha_signals.volatility_window
        )
        
        valid_dates = trading_dates[warmup_period:]
        
        for date in valid_dates:
            current_prices = prices.loc[date]
            
            # Update position prices
            self._update_position_prices(current_prices)
            
            # Check stop-losses
            stopped_out = self.risk_manager.get_stopped_out_positions(self.positions)
            for ticker in stopped_out:
                self._close_position(date, ticker, current_prices[ticker], 'STOP_LOSS')
            
            # Rebalance if needed
            if date in rebalance_dates:
                target_weights = weight_history.loc[date]
                self._rebalance(date, target_weights, current_prices)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices)
            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })
            
            # Record positions
            positions_snapshot = self._get_positions_snapshot(date)
            positions_history.append(positions_snapshot)
        
        # Create result DataFrames
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
        portfolio_series = portfolio_df['value']
        
        positions_df = pd.DataFrame(positions_history)
        if not positions_df.empty:
            positions_df = positions_df.set_index('date')
        
        # Calculate returns
        daily_returns = portfolio_series.pct_change().dropna()
        
        # Create result object
        result = BacktestResult(
            portfolio_values=portfolio_series,
            daily_returns=daily_returns,
            positions_history=positions_df,
            weights_history=weight_history.loc[valid_dates],
            trades=self.trades
        )
        
        # Calculate performance metrics
        self._calculate_metrics(result, benchmark_returns)
        
        return result
    
    def _update_position_prices(self, current_prices: pd.Series) -> None:
        """Update all position prices to current values."""
        for ticker, position in self.positions.items():
            if ticker in current_prices.index:
                position.current_price = current_prices[ticker]
    
    def _calculate_portfolio_value(self, current_prices: pd.Series) -> float:
        """Calculate total portfolio value (cash + positions)."""
        positions_value = 0.0
        
        for ticker, position in self.positions.items():
            if position.is_long:
                positions_value += position.shares * current_prices[ticker]
            else:
                # Short position value: entry_value + (entry_price - current_price) * shares
                entry_value = position.shares * position.entry_price
                pnl = (position.entry_price - current_prices[ticker]) * position.shares
                positions_value += entry_value + pnl
                
        return self.cash + positions_value
    
    def _rebalance(
        self,
        date: pd.Timestamp,
        target_weights: pd.Series,
        current_prices: pd.Series
    ) -> None:
        """Rebalance portfolio to target weights."""
        # Calculate current portfolio value
        portfolio_value = self._calculate_portfolio_value(current_prices)
        
        # Close positions that are no longer in target
        for ticker in list(self.positions.keys()):
            if target_weights.get(ticker, 0) == 0:
                self._close_position(date, ticker, current_prices[ticker], 'REBALANCE')
        
        # Recalculate portfolio value after closes
        portfolio_value = self._calculate_portfolio_value(current_prices)
        
        # Open/adjust positions for non-zero weights
        for ticker, weight in target_weights.items():
            if weight == 0:
                continue
                
            if ticker not in current_prices.index or pd.isna(current_prices[ticker]):
                continue
                
            target_value = abs(weight) * portfolio_value
            current_price = current_prices[ticker]
            target_shares = target_value / current_price
            
            current_position = self.positions.get(ticker)
            current_shares = current_position.shares if current_position else 0
            
            is_long = weight > 0
            
            # Adjust position
            if current_position is None:
                # Open new position
                self._open_position(
                    date, ticker, target_shares, current_price, is_long
                )
            elif current_position.is_long != is_long:
                # Direction change: close and reopen
                self._close_position(date, ticker, current_price, 'REBALANCE')
                self._open_position(date, ticker, target_shares, current_price, is_long)
            else:
                # Adjust size
                share_diff = target_shares - current_shares
                if abs(share_diff) > 0.01:  # Minimum trade threshold
                    self._adjust_position(
                        date, ticker, share_diff, current_price, is_long
                    )
    
    def _open_position(
        self,
        date: pd.Timestamp,
        ticker: str,
        shares: float,
        price: float,
        is_long: bool
    ) -> None:
        """Open a new position."""
        trade_value = shares * price
        transaction_cost = self.risk_manager.apply_transaction_costs(
            trade_value, self.transaction_cost_bps
        )
        
        if is_long:
            # Deduct cash for long purchase
            self.cash -= trade_value + transaction_cost
            action = 'BUY'
        else:
            # For short: receive cash from short sale (minus borrow cost in reality)
            self.cash += trade_value - transaction_cost
            action = 'SHORT'
        
        self.positions[ticker] = Position(
            ticker=ticker,
            shares=shares,
            entry_price=price,
            current_price=price,
            is_long=is_long,
            entry_date=date
        )
        
        self.trades.append(Trade(
            date=date,
            ticker=ticker,
            action=action,
            shares=shares,
            price=price,
            value=trade_value,
            transaction_cost=transaction_cost
        ))
    
    def _close_position(
        self,
        date: pd.Timestamp,
        ticker: str,
        price: float,
        reason: str = 'CLOSE'
    ) -> None:
        """Close an existing position."""
        if ticker not in self.positions:
            return
            
        position = self.positions[ticker]
        trade_value = position.shares * price
        transaction_cost = self.risk_manager.apply_transaction_costs(
            trade_value, self.transaction_cost_bps
        )
        
        if position.is_long:
            # Receive cash from selling
            self.cash += trade_value - transaction_cost
            action = 'SELL'
        else:
            # Cover short: pay to buy back shares
            self.cash -= trade_value + transaction_cost
            action = 'COVER'
        
        self.trades.append(Trade(
            date=date,
            ticker=ticker,
            action=action,
            shares=position.shares,
            price=price,
            value=trade_value,
            transaction_cost=transaction_cost
        ))
        
        del self.positions[ticker]
    
    def _adjust_position(
        self,
        date: pd.Timestamp,
        ticker: str,
        share_change: float,
        price: float,
        is_long: bool
    ) -> None:
        """Adjust position size."""
        if ticker not in self.positions:
            return
            
        position = self.positions[ticker]
        trade_value = abs(share_change) * price
        transaction_cost = self.risk_manager.apply_transaction_costs(
            trade_value, self.transaction_cost_bps
        )
        
        if share_change > 0:
            # Increasing position
            if is_long:
                self.cash -= trade_value + transaction_cost
                action = 'BUY'
            else:
                self.cash += trade_value - transaction_cost
                action = 'SHORT'
        else:
            # Decreasing position
            if is_long:
                self.cash += trade_value - transaction_cost
                action = 'SELL'
            else:
                self.cash -= trade_value + transaction_cost
                action = 'COVER'
        
        position.shares += share_change
        
        self.trades.append(Trade(
            date=date,
            ticker=ticker,
            action=action,
            shares=abs(share_change),
            price=price,
            value=trade_value,
            transaction_cost=transaction_cost
        ))
    
    def _get_positions_snapshot(self, date: pd.Timestamp) -> Dict:
        """Get a snapshot of current positions."""
        snapshot = {'date': date}
        
        for ticker, position in self.positions.items():
            prefix = 'L_' if position.is_long else 'S_'
            snapshot[f'{prefix}{ticker}'] = position.shares * position.current_price
            
        return snapshot
    
    def _calculate_metrics(
        self,
        result: BacktestResult,
        benchmark_returns: Optional[pd.Series]
    ) -> None:
        """Calculate all performance metrics."""
        returns = result.daily_returns
        portfolio_values = result.portfolio_values
        
        if len(returns) == 0:
            return
        
        # Cumulative return
        result.cumulative_return = (
            portfolio_values.iloc[-1] / self.initial_capital - 1
        )
        
        # Annualized return
        n_years = len(returns) / 252
        result.annualized_return = (1 + result.cumulative_return) ** (1 / n_years) - 1
        
        # Volatility (annualized)
        result.volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        if result.volatility > 0:
            excess_return = result.annualized_return - self.risk_free_rate
            result.sharpe_ratio = excess_return / result.volatility
        
        # Maximum drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        result.max_drawdown = drawdowns.min()
        
        # Alpha and Beta (if benchmark provided)
        if benchmark_returns is not None:
            # Align dates
            aligned_returns, aligned_benchmark = returns.align(
                benchmark_returns, join='inner'
            )
            
            if len(aligned_returns) > 1:
                # Calculate beta
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                variance = np.var(aligned_benchmark)
                
                if variance > 0:
                    result.beta = covariance / variance
                
                # Calculate alpha (annualized)
                benchmark_annual_return = (1 + aligned_benchmark.mean()) ** 252 - 1
                expected_return = self.risk_free_rate + result.beta * (
                    benchmark_annual_return - self.risk_free_rate
                )
                result.alpha = result.annualized_return - expected_return


if __name__ == "__main__":
    # Test backtester
    from data_collector import DataCollector
    
    print("Downloading data...")
    collector = DataCollector(years=2)  # Use 2 years for faster testing
    collector.download_price_data()
    collector.download_benchmark_data()
    
    prices = collector.get_combined_prices()
    benchmark_returns = collector.get_benchmark_returns()
    
    print(f"\nRunning backtest on {len(prices)} days of data...")
    
    backtester = Backtester(initial_capital=100000)
    result = backtester.run(prices, benchmark_returns)
    
    print("\n=== Backtest Results ===")
    print(f"Cumulative Return: {result.cumulative_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Alpha: {result.alpha:.2%}")
    print(f"Beta: {result.beta:.2f}")
    print(f"Total Trades: {len(result.trades)}")
