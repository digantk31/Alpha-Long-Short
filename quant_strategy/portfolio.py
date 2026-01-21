"""
Portfolio Construction Module

Implements portfolio construction and rebalancing logic:
- Rank stocks based on alpha signals
- Go long top 3, short bottom 3
- Monthly rebalancing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class PortfolioConstructor:
    """
    Constructs and rebalances the long-short portfolio.
    
    Strategy:
    - Rank stocks based on combined alpha score
    - Long: Top N ranked stocks
    - Short: Bottom N ranked stocks
    - Equal weight within long/short buckets
    - Monthly rebalancing
    """
    
    def __init__(
        self,
        n_long: int = 3,
        n_short: int = 3,
        rebalance_frequency: str = 'ME'  # 'ME' for monthly, 'W' for weekly
    ):
        """
        Initialize PortfolioConstructor.
        
        Args:
            n_long: Number of stocks to go long
            n_short: Number of stocks to go short
            rebalance_frequency: Pandas frequency string for rebalancing
        """
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_frequency = rebalance_frequency
        
    def get_rebalance_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        trading_dates: pd.DatetimeIndex
    ) -> List[pd.Timestamp]:
        """
        Get the list of rebalancing dates.
        
        Args:
            start_date: Strategy start date
            end_date: Strategy end date
            trading_dates: Index of valid trading dates
            
        Returns:
            List of rebalancing dates
        """
        # Generate month-end dates
        rebalance_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq=self.rebalance_frequency
        )
        
        # Map to valid trading dates (use last trading day of each period)
        valid_rebalance_dates = []
        
        for date in rebalance_dates:
            # Find the closest trading date on or before this date
            valid_dates = trading_dates[trading_dates <= date]
            if len(valid_dates) > 0:
                valid_rebalance_dates.append(valid_dates[-1])
                
        return valid_rebalance_dates
    
    def select_positions(
        self,
        combined_alpha: pd.Series
    ) -> Tuple[List[str], List[str]]:
        """
        Select long and short positions based on alpha rankings.
        
        Args:
            combined_alpha: Series of alpha scores indexed by ticker
            
        Returns:
            Tuple of (long_tickers, short_tickers)
        """
        # Remove any NaN values
        alpha_clean = combined_alpha.dropna()
        
        if len(alpha_clean) < self.n_long + self.n_short:
            raise ValueError(
                f"Not enough stocks with valid alpha scores. "
                f"Need {self.n_long + self.n_short}, have {len(alpha_clean)}"
            )
        
        # Sort by alpha score (descending)
        sorted_alpha = alpha_clean.sort_values(ascending=False)
        
        # Top N go long, bottom N go short
        long_tickers = sorted_alpha.head(self.n_long).index.tolist()
        short_tickers = sorted_alpha.tail(self.n_short).index.tolist()
        
        return long_tickers, short_tickers
    
    def calculate_target_weights(
        self,
        long_tickers: List[str],
        short_tickers: List[str],
        all_tickers: List[str],
        market_regime: str = 'NEUTRAL'  # 'BULL', 'BEAR', or 'NEUTRAL'
    ) -> pd.Series:
        """
        Calculate target portfolio weights based on market regime.
        
        Regimes:
        - BULL: Long 100% (or n_long stocks), Short 0%
        - BEAR/NEUTRAL: Long 50%, Short 50% (Dollar Neutral)
        
        Args:
            long_tickers: Stocks to go long
            short_tickers: Stocks to go short
            all_tickers: All tickers in the universe
            market_regime: Current market environment
            
        Returns:
            Series of portfolio weights indexed by ticker
        """
        weights = pd.Series(0.0, index=all_tickers)
        
        if market_regime == 'BULL':
            # Long-only mode: 100% long, 0% short
            long_weight = 1.0 / len(long_tickers) if long_tickers else 0
            short_weight = 0.0
        else:
            # Dollar-neutral mode: 50% long, 50% short
            long_weight = 0.5 / len(long_tickers) if long_tickers else 0
            short_weight = -0.5 / len(short_tickers) if short_tickers else 0
        
        for ticker in long_tickers:
            weights[ticker] = long_weight
            
        for ticker in short_tickers:
            weights[ticker] = short_weight
            
        return weights
    
    def generate_weight_history(
        self,
        combined_alpha: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
        benchmark_prices: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate the complete history of portfolio weights with monthly rebalancing.
        
        Args:
            combined_alpha: DataFrame of combined alpha scores (dates x tickers)
            trading_dates: Index of valid trading dates
            benchmark_prices: Optional series of benchmark prices for regime filter
            
        Returns:
            DataFrame of portfolio weights over time
        """
        # Ensure combined_alpha is a DataFrame
        if isinstance(combined_alpha, pd.Series):
            combined_alpha = combined_alpha.to_frame().T
        
        start_date = combined_alpha.index[0]
        end_date = combined_alpha.index[-1]
        all_tickers = combined_alpha.columns.tolist()
        
        # Get rebalancing dates
        rebalance_dates = self.get_rebalance_dates(start_date, end_date, trading_dates)
        
        # Initialize weight history
        weight_history = pd.DataFrame(
            0.0,
            index=trading_dates,
            columns=all_tickers
        )
        
        current_weights = pd.Series(0.0, index=all_tickers)
        
        # Calculate Benchmark MA200 if provided
        ma200 = None
        if benchmark_prices is not None:
            ma200 = benchmark_prices.rolling(window=200).mean()
        
        for i, date in enumerate(trading_dates):
            if date in rebalance_dates:
                # Rebalance: calculate new weights based on alpha
                try:
                    # Determine Market Regime
                    market_regime = 'NEUTRAL'
                    if ma200 is not None and date in ma200.index:
                        current_price = benchmark_prices.loc[date]
                        current_ma = ma200.loc[date]
                        if not pd.isna(current_ma) and current_price > current_ma:
                            market_regime = 'BULL'
                        else:
                            market_regime = 'BEAR'
                    
                    alpha_scores = combined_alpha.loc[date]
                    long_tickers, short_tickers = self.select_positions(alpha_scores)
                    current_weights = self.calculate_target_weights(
                        long_tickers, short_tickers, all_tickers, market_regime
                    )
                except (KeyError, ValueError) as e:
                    # Keep previous weights if we can't calculate new ones
                    pass
                    
            weight_history.loc[date] = current_weights
            
        return weight_history
    
    def get_position_changes(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Determine which positions to open, close, or hold.
        
        Args:
            old_weights: Previous portfolio weights
            new_weights: New target weights
            
        Returns:
            Tuple of (positions_to_open, positions_to_close, positions_to_hold)
        """
        old_positions = set(old_weights[old_weights != 0].index)
        new_positions = set(new_weights[new_weights != 0].index)
        
        positions_to_open = list(new_positions - old_positions)
        positions_to_close = list(old_positions - new_positions)
        positions_to_hold = list(old_positions & new_positions)
        
        return positions_to_open, positions_to_close, positions_to_hold
    
    def calculate_turnover(
        self,
        weight_history: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate portfolio turnover at each rebalance.
        
        Args:
            weight_history: DataFrame of portfolio weights
            
        Returns:
            Series of turnover values at each rebalance
        """
        weight_changes = weight_history.diff().abs()
        turnover = weight_changes.sum(axis=1) / 2  # Divide by 2 to avoid double counting
        
        return turnover


if __name__ == "__main__":
    # Test portfolio construction
    import numpy as np
    
    # Create sample alpha scores
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'WMT']
    
    np.random.seed(42)
    alpha_data = np.random.randn(len(dates), len(tickers))
    combined_alpha = pd.DataFrame(alpha_data, index=dates, columns=tickers)
    
    # Initialize portfolio constructor
    pc = PortfolioConstructor(n_long=3, n_short=3)
    
    # Test position selection
    latest_alpha = combined_alpha.iloc[-1]
    long_tickers, short_tickers = pc.select_positions(latest_alpha)
    
    print("Latest Alpha Scores:")
    print(latest_alpha.sort_values(ascending=False))
    print(f"\nLong: {long_tickers}")
    print(f"Short: {short_tickers}")
    
    # Test weight calculation
    weights = pc.calculate_target_weights(long_tickers, short_tickers, tickers)
    print(f"\nPortfolio Weights:")
    print(weights[weights != 0])
    
    # Test weight history generation
    weight_history = pc.generate_weight_history(combined_alpha, dates)
    print(f"\nWeight History Shape: {weight_history.shape}")
    print(f"Final Weights:")
    print(weight_history.iloc[-1][weight_history.iloc[-1] != 0])
