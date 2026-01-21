"""
Data Collection Module

Downloads historical stock data using yfinance for the specified universe of stocks.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os


class DataCollector:
    """
    Collects historical stock data from Yahoo Finance.
    
    Attributes:
        tickers: List of stock tickers to download
        benchmark: Benchmark ticker (S&P 500)
        start_date: Start date for data collection
        end_date: End date for data collection
    """
    
    # Default universe of 10 stocks
    DEFAULT_TICKERS = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'AMZN',   # Amazon
        'GOOGL',  # Google
        'META',   # Facebook/Meta
        'TSLA',   # Tesla
        'NVDA',   # NVIDIA
        'JPM',    # JPMorgan Chase
        'JNJ',    # Johnson & Johnson
        'WMT'     # Walmart
    ]
    
    BENCHMARK = '^GSPC'  # S&P 500
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        years: int = 5,
        end_date: Optional[datetime] = None
    ):
        """
        Initialize the DataCollector.
        
        Args:
            tickers: List of stock tickers (defaults to DEFAULT_TICKERS)
            years: Number of years of historical data to collect
            end_date: End date for data collection (defaults to today)
        """
        self.tickers = tickers or self.DEFAULT_TICKERS
        self.end_date = end_date or datetime.now()
        self.start_date = self.end_date - timedelta(days=years * 365)
        
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.fundamental_data: Dict[str, dict] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        
    def download_price_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for all tickers.
        
        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV data
        """
        print(f"Downloading price data from {self.start_date.date()} to {self.end_date.date()}...")
        
        for ticker in self.tickers:
            try:
                print(f"  Downloading {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True  # Adjust for splits and dividends
                )
                
                if df.empty:
                    print(f"  Warning: No data available for {ticker}")
                    continue
                    
                # Keep only OHLCV columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                self.price_data[ticker] = df
                
            except Exception as e:
                print(f"  Error downloading {ticker}: {e}")
                
        return self.price_data
    
    def download_benchmark_data(self) -> pd.DataFrame:
        """
        Download S&P 500 data as benchmark.
        
        Returns:
            DataFrame with benchmark OHLCV data
        """
        print(f"Downloading benchmark ({self.BENCHMARK})...")
        
        try:
            benchmark = yf.Ticker(self.BENCHMARK)
            df = benchmark.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True
            )
            self.benchmark_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"Error downloading benchmark: {e}")
            
        return self.benchmark_data
    
    def download_fundamental_data(self) -> Dict[str, dict]:
        """
        Download fundamental data for all tickers.
        
        Returns:
            Dictionary mapping ticker to fundamental data dict
        """
        print("Downloading fundamental data...")
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                self.fundamental_data[ticker] = {
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'dividend_yield': info.get('dividendYield'),
                    'beta': info.get('beta'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry')
                }
                
            except Exception as e:
                print(f"  Error getting fundamentals for {ticker}: {e}")
                self.fundamental_data[ticker] = {}
                
        return self.fundamental_data
    
    def get_combined_prices(self) -> pd.DataFrame:
        """
        Combine all ticker close prices into a single DataFrame.
        
        Returns:
            DataFrame with close prices for all tickers (columns) by date (index)
        """
        if not self.price_data:
            self.download_price_data()
            
        close_prices = pd.DataFrame()
        
        for ticker, df in self.price_data.items():
            close_prices[ticker] = df['Close']
            
        # Forward fill missing values, then drop any remaining NaNs
        close_prices = close_prices.ffill().dropna()
        
        return close_prices
    
    def get_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns for all tickers.
        
        Returns:
            DataFrame with daily returns for all tickers
        """
        close_prices = self.get_combined_prices()
        returns = close_prices.pct_change().dropna()
        return returns
    
    def get_benchmark_returns(self) -> pd.Series:
        """
        Calculate daily returns for the benchmark.
        
        Returns:
            Series with daily benchmark returns
        """
        if self.benchmark_data is None:
            self.download_benchmark_data()
            
        returns = self.benchmark_data['Close'].pct_change().dropna()
        returns.name = 'Benchmark'
        return returns
        
    def get_benchmark_data(self) -> pd.DataFrame:
        """
        Get the raw benchmark data.
        
        Returns:
            DataFrame with benchmark OHLCV data
        """
        if self.benchmark_data is None:
            self.download_benchmark_data()
            
        return self.benchmark_data
    
    def download_all(self) -> None:
        """Download all data (prices, benchmark, fundamentals)."""
        self.download_price_data()
        self.download_benchmark_data()
        self.download_fundamental_data()
        print(f"\nData collection complete!")
        print(f"  Stocks: {len(self.price_data)}")
        print(f"  Date range: {self.start_date.date()} to {self.end_date.date()}")
        

if __name__ == "__main__":
    # Test data collection
    collector = DataCollector()
    collector.download_all()
    
    print("\nSample close prices:")
    print(collector.get_combined_prices().tail())
    
    print("\nFundamental data:")
    for ticker, data in collector.fundamental_data.items():
        print(f"  {ticker}: P/E={data.get('pe_ratio')}, MarketCap={data.get('market_cap')}")
