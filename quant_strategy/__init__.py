"""
Quantitative Long-Short Trading Strategy Package

This package implements a quantitative trading strategy that generates alpha
by going long on stocks expected to outperform and short on stocks expected
to underperform.
"""

from .data_collector import DataCollector
from .alpha_signals import AlphaSignals
from .portfolio import PortfolioConstructor
from .backtester import Backtester
from .risk_manager import RiskManager
from .performance import PerformanceAnalyzer

__all__ = [
    'DataCollector',
    'AlphaSignals', 
    'PortfolioConstructor',
    'Backtester',
    'RiskManager',
    'PerformanceAnalyzer'
]
