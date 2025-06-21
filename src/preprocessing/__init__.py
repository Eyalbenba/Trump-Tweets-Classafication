"""
Data preprocessing module for Trump tweets classification.
"""

from .data_loader import load_data
from .text_cleaner import clean_text
from .feature_extractor import extract_features

__all__ = ['load_data', 'clean_text', 'extract_features']