"""
Real-time Network Traffic Analysis Module
Module for real-time network traffic analysis
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
