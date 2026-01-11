"""
Real-time Network Traffic Analysis Module
Модуль для анализа сетевого трафика в реальном времени
"""

from .capture import PacketCapture
from .flow_aggregator import FlowAggregator
from .feature_extractor import FeatureExtractor
from .analyzer import TrafficAnalyzer
from .pipeline import RealtimePipeline

__all__ = [
    'PacketCapture',
    'FlowAggregator',
    'FeatureExtractor',
    'TrafficAnalyzer',
    'RealtimePipeline'
]

__version__ = '1.0.0'
