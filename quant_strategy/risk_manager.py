"""
Risk Management Module

Implements risk controls including:
- Stop-loss mechanism for individual positions
- Dollar-neutral constraint enforcement
- Position size limits
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    ticker: str
    shares: float
    entry_price: float
    current_price: float
    is_long: bool
    entry_date: pd.Timestamp
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.shares * self.current_price * (1 if self.is_long else -1)
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.is_long:
            return self.shares * (self.current_price - self.entry_price)
        else:
            return self.shares * (self.entry_price - self.current_price)
    
    @property
    def return_pct(self) -> float:
        """Percentage return on the position."""
        if self.is_long:
            return (self.current_price / self.entry_price) - 1
        else:
            return (self.entry_price / self.current_price) - 1


class RiskManager:
    """
    Manages portfolio risk through various controls.
    
    Features:
    - Individual position stop-losses
    - Dollar-neutral constraint
    - Position size limits
    """
    
    def __init__(
        self,
        stop_loss_pct: float = -0.10,
        max_position_pct: float = 0.20,
        dollar_neutral_tolerance: float = 0.05
    ):
        """
        Initialize RiskManager.
        
        Args:
            stop_loss_pct: Stop-loss threshold as negative percentage (e.g., -0.10 = -10%)
            max_position_pct: Maximum position size as percentage of portfolio
            dollar_neutral_tolerance: Acceptable deviation from dollar-neutral
        """
        self.stop_loss_pct = stop_loss_pct
        self.max_position_pct = max_position_pct
        self.dollar_neutral_tolerance = dollar_neutral_tolerance
        
    def check_stop_loss(self, position: Position) -> bool:
        """
        Check if a position has hit its stop-loss.
        
        Args:
            position: Position to check
            
        Returns:
            True if stop-loss triggered, False otherwise
        """
        return position.return_pct <= self.stop_loss_pct
    
    def get_stopped_out_positions(
        self,
        positions: Dict[str, Position]
    ) -> list:
        """
        Get list of positions that have triggered stop-loss.
        
        Args:
            positions: Dictionary of current positions
            
        Returns:
            List of tickers that should be closed
        """
        stopped_out = []
        
        for ticker, position in positions.items():
            if self.check_stop_loss(position):
                stopped_out.append(ticker)
                
        return stopped_out
    
    def check_dollar_neutral(
        self,
        positions: Dict[str, Position]
    ) -> Tuple[bool, float, float]:
        """
        Check if portfolio maintains dollar-neutral constraint.
        
        Args:
            positions: Dictionary of current positions
            
        Returns:
            Tuple of (is_balanced, long_exposure, short_exposure)
        """
        long_exposure = sum(
            pos.market_value for pos in positions.values() 
            if pos.is_long and pos.market_value > 0
        )
        
        short_exposure = abs(sum(
            pos.market_value for pos in positions.values() 
            if not pos.is_long
        ))
        
        total_exposure = long_exposure + short_exposure
        
        if total_exposure == 0:
            return True, 0.0, 0.0
            
        # Calculate imbalance
        imbalance = abs(long_exposure - short_exposure) / total_exposure
        is_balanced = imbalance <= self.dollar_neutral_tolerance
        
        return is_balanced, long_exposure, short_exposure
    
    def calculate_position_sizes(
        self,
        capital: float,
        long_tickers: list,
        short_tickers: list
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate dollar-neutral position sizes.
        
        Allocates equal weight to each position within long/short buckets,
        with total long exposure = total short exposure.
        
        Args:
            capital: Total portfolio capital
            long_tickers: List of tickers to go long
            short_tickers: List of tickers to go short
            
        Returns:
            Tuple of (long_allocations, short_allocations) as dictionaries
        """
        n_long = len(long_tickers)
        n_short = len(short_tickers)
        
        if n_long == 0 or n_short == 0:
            return {}, {}
            
        # Allocate 50% to longs, 50% to shorts for dollar-neutral
        long_capital = capital * 0.5
        short_capital = capital * 0.5
        
        # Equal weight within each bucket
        long_per_stock = long_capital / n_long
        short_per_stock = short_capital / n_short
        
        # Apply max position constraint
        max_position = capital * self.max_position_pct
        long_per_stock = min(long_per_stock, max_position)
        short_per_stock = min(short_per_stock, max_position)
        
        long_allocations = {ticker: long_per_stock for ticker in long_tickers}
        short_allocations = {ticker: short_per_stock for ticker in short_tickers}
        
        return long_allocations, short_allocations
    
    def apply_transaction_costs(
        self,
        trade_value: float,
        cost_bps: float = 10
    ) -> float:
        """
        Calculate transaction costs for a trade.
        
        Args:
            trade_value: Absolute dollar value of trade
            cost_bps: Transaction cost in basis points (default 10 = 0.1%)
            
        Returns:
            Transaction cost in dollars
        """
        return abs(trade_value) * (cost_bps / 10000)
    
    def generate_risk_report(
        self,
        positions: Dict[str, Position]
    ) -> Dict:
        """
        Generate a risk summary report for the portfolio.
        
        Args:
            positions: Dictionary of current positions
            
        Returns:
            Dictionary containing risk metrics
        """
        if not positions:
            return {
                'total_positions': 0,
                'long_positions': 0,
                'short_positions': 0,
                'long_exposure': 0,
                'short_exposure': 0,
                'net_exposure': 0,
                'gross_exposure': 0,
                'is_dollar_neutral': True,
                'positions_at_stop_loss': []
            }
            
        is_balanced, long_exp, short_exp = self.check_dollar_neutral(positions)
        stopped_out = self.get_stopped_out_positions(positions)
        
        return {
            'total_positions': len(positions),
            'long_positions': sum(1 for p in positions.values() if p.is_long),
            'short_positions': sum(1 for p in positions.values() if not p.is_long),
            'long_exposure': long_exp,
            'short_exposure': short_exp,
            'net_exposure': long_exp - short_exp,
            'gross_exposure': long_exp + short_exp,
            'is_dollar_neutral': is_balanced,
            'positions_at_stop_loss': stopped_out
        }


if __name__ == "__main__":
    # Test risk manager
    rm = RiskManager(stop_loss_pct=-0.10)
    
    # Create test positions
    positions = {
        'AAPL': Position('AAPL', 100, 150.0, 160.0, True, pd.Timestamp('2024-01-01')),
        'MSFT': Position('MSFT', 50, 300.0, 260.0, True, pd.Timestamp('2024-01-01')),  # -13%, stopped
        'TSLA': Position('TSLA', 30, 200.0, 190.0, False, pd.Timestamp('2024-01-01')),
    }
    
    print("Risk Report:")
    report = rm.generate_risk_report(positions)
    for key, value in report.items():
        print(f"  {key}: {value}")
        
    print("\nPosition Sizes for $100,000 portfolio:")
    long_alloc, short_alloc = rm.calculate_position_sizes(
        100000,
        ['AAPL', 'MSFT', 'GOOGL'],
        ['TSLA', 'META', 'NVDA']
    )
    print(f"  Long allocations: {long_alloc}")
    print(f"  Short allocations: {short_alloc}")
