"""
Alpha Signals Module

Implements three alpha signals for stock ranking:
1. Momentum - 20-day returns
2. Mean Reversion - Deviation from 50-day moving average
3. Volatility - 30-day historical volatility
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class AlphaSignals:
    """
    Generates alpha signals for stock ranking.
    
    Each signal produces a z-score normalized ranking that can be combined
    into a composite alpha score.
    """
    
    def __init__(
        self,
        momentum_window: int = 20,
        ma_window: int = 50,
        volatility_window: int = 30,
        signal_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize AlphaSignals generator.
        
        Args:
            momentum_window: Lookback period for momentum calculation (days)
            ma_window: Moving average window for mean reversion (days)
            volatility_window: Window for volatility calculation (days)
            signal_weights: Dictionary of signal weights (default: equal weights)
        """
        self.momentum_window = momentum_window
        self.ma_window = ma_window
        self.volatility_window = volatility_window
        
        # Default optimized weights (from Walk-Forward Analysis)
        self.signal_weights = signal_weights or {
            'momentum': 0.61,       # Strongest signal
            'mean_reversion': 0.23, # Moderate signal
            'volatility': 0.16      # Weakest signal
        }
        
    def calculate_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum signal based on N-day returns.
        
        Higher returns = higher rank (positive signal for going long)
        
        Args:
            prices: DataFrame of close prices (tickers as columns)
            
        Returns:
            DataFrame of momentum signals (z-scored)
        """
        # Calculate N-day returns
        returns = prices.pct_change(self.momentum_window)
        
        # Z-score normalize across stocks for each day
        def zscore_row(row):
            if row.std() > 0:
                return (row - row.mean()) / row.std()
            else:
                return pd.Series(0, index=row.index)
        
        momentum_zscore = returns.apply(zscore_row, axis=1)
        
        return momentum_zscore
    
    def calculate_mean_reversion(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion signal based on deviation from moving average.
        
        Stocks trading below their MA are expected to revert up (positive signal)
        
        Args:
            prices: DataFrame of close prices (tickers as columns)
            
        Returns:
            DataFrame of mean reversion signals (z-scored)
        """
        # Calculate moving average
        ma = prices.rolling(window=self.ma_window).mean()
        
        # Calculate percentage deviation from MA
        # Negative deviation = price below MA = expect mean reversion up
        deviation = (prices - ma) / ma
        
        # Invert so that negative deviation (below MA) gives positive signal
        mean_reversion = -deviation
        
        # Z-score normalize across stocks for each day
        def zscore_row(row):
            if row.std() > 0:
                return (row - row.mean()) / row.std()
            else:
                return pd.Series(0, index=row.index)
        
        mr_zscore = mean_reversion.apply(zscore_row, axis=1)
        
        return mr_zscore
    
    def calculate_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility signal based on historical volatility.
        
        Lower volatility = higher rank (quality/low-risk preference)
        
        Args:
            prices: DataFrame of close prices (tickers as columns)
            
        Returns:
            DataFrame of volatility signals (z-scored)
        """
        # Calculate daily returns
        returns = prices.pct_change()
        
        # Calculate rolling volatility (annualized)
        volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        # Invert so that lower volatility gets higher score
        inv_volatility = -volatility
        
        # Z-score normalize across stocks for each day
        def zscore_row(row):
            if row.std() > 0:
                return (row - row.mean()) / row.std()
            else:
                return pd.Series(0, index=row.index)
        
        vol_zscore = inv_volatility.apply(zscore_row, axis=1)
        
        return vol_zscore
    
    def generate_all_signals(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate all three alpha signals.
        
        Args:
            prices: DataFrame of close prices (tickers as columns)
            
        Returns:
            Dictionary mapping signal name to signal DataFrame
        """
        signals = {
            'momentum': self.calculate_momentum(prices),
            'mean_reversion': self.calculate_mean_reversion(prices),
            'volatility': self.calculate_volatility(prices)
        }
        
        return signals
    
    def combine_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Combine multiple signals into a composite alpha score.
        
        Args:
            signals: Dictionary of signal DataFrames
            weights: Optional custom weights (defaults to instance weights)
            
        Returns:
            DataFrame of combined alpha scores
        """
        weights = weights or self.signal_weights
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Initialize combined signal with zeros
        combined = None
        
        for signal_name, signal_df in signals.items():
            weight = weights.get(signal_name, 0)
            
            if combined is None:
                combined = signal_df * weight
            else:
                # Align indices before adding
                combined, signal_aligned = combined.align(signal_df, join='inner')
                combined = combined + signal_aligned * weight
                
        return combined
    
    def get_rankings(
        self,
        prices: pd.DataFrame,
        date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Get stock rankings for a specific date based on combined alpha.
        
        Args:
            prices: DataFrame of close prices
            date: Date to get rankings for (defaults to last available date)
            
        Returns:
            Series of rankings (1 = highest alpha, N = lowest)
        """
        # Generate and combine signals
        signals = self.generate_all_signals(prices)
        combined = self.combine_signals(signals)
        
        # Get scores for specified date
        if date is None:
            date = combined.index[-1]
            
        if date not in combined.index:
            raise ValueError(f"Date {date} not in signal index")
            
        scores = combined.loc[date]
        
        # Rank stocks (higher score = lower rank number = better)
        rankings = scores.rank(ascending=False)
        
        return rankings
    
    def get_signal_contributions(
        self,
        prices: pd.DataFrame,
        date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Get individual signal contributions for each stock on a given date.
        
        Args:
            prices: DataFrame of close prices
            date: Date to analyze (defaults to last available)
            
        Returns:
            DataFrame with signal values for each stock
        """
        signals = self.generate_all_signals(prices)
        
        if date is None:
            date = prices.index[-1]
            
        contributions = pd.DataFrame()
        
        for signal_name, signal_df in signals.items():
            if date in signal_df.index:
                contributions[signal_name] = signal_df.loc[date]
                
        # Add combined score
        combined = self.combine_signals(signals)
        if date in combined.index:
            contributions['combined'] = combined.loc[date]
            
        return contributions


if __name__ == "__main__":
    # Test alpha signals
    from data_collector import DataCollector
    
    collector = DataCollector()
    collector.download_price_data()
    prices = collector.get_combined_prices()
    
    signals_gen = AlphaSignals()
    signals = signals_gen.generate_all_signals(prices)
    combined = signals_gen.combine_signals(signals)
    
    print("Latest Combined Alpha Scores:")
    print(combined.iloc[-1].sort_values(ascending=False))
    
    print("\nStock Rankings (1=best):")
    rankings = signals_gen.get_rankings(prices)
    print(rankings.sort_values())
    
    print("\nSignal Contributions:")
    contributions = signals_gen.get_signal_contributions(prices)
    print(contributions.sort_values('combined', ascending=False))
