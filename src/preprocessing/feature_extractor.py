"""
Feature extraction utilities for tweet classification.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re

class TweetFeatureExtractor:
    """Feature extractor for tweet classification."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
    
    def extract_text_features(self, texts: List[str], method: str = 'tfidf', 
                            max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)) -> np.ndarray:
        """
        Extract text-based features using TF-IDF or Count vectorization.
        
        Args:
            texts (List[str]): List of tweet texts
            method (str): 'tfidf' or 'count'
            max_features (int): Maximum number of features
            ngram_range (Tuple[int, int]): N-gram range
            
        Returns:
            np.ndarray: Feature matrix
        """
        if method == 'tfidf':
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words='english'
                )
                features = self.tfidf_vectorizer.fit_transform(texts)
            else:
                features = self.tfidf_vectorizer.transform(texts)
        
        elif method == 'count':
            if self.count_vectorizer is None:
                self.count_vectorizer = CountVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words='english'
                )
                features = self.count_vectorizer.fit_transform(texts)
            else:
                features = self.count_vectorizer.transform(texts)
        
        return features.toarray()
    
    def extract_stylistic_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract stylistic features from tweets.
        
        Args:
            texts (List[str]): List of tweet texts
            
        Returns:
            np.ndarray: Stylistic feature matrix
        """
        features = []
        
        for text in texts:
            tweet_features = {}
            
            # Length features
            tweet_features['char_count'] = len(text)
            tweet_features['word_count'] = len(text.split())
            tweet_features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
            # Capitalization features
            tweet_features['caps_count'] = sum(1 for c in text if c.isupper())
            tweet_features['caps_ratio'] = tweet_features['caps_count'] / len(text) if len(text) > 0 else 0
            tweet_features['all_caps_words'] = len([word for word in text.split() if word.isupper() and len(word) > 1])
            
            # Punctuation features
            tweet_features['exclamation_count'] = text.count('!')
            tweet_features['question_count'] = text.count('?')
            tweet_features['period_count'] = text.count('.')
            tweet_features['comma_count'] = text.count(',')
            tweet_features['ellipsis_count'] = text.count('...')
            
            # Social media features
            tweet_features['hashtag_count'] = len(re.findall(r'#\w+', text))
            tweet_features['mention_count'] = len(re.findall(r'@\w+', text))
            tweet_features['url_count'] = len(re.findall(r'http[s]?://\S+', text))
            
            # Emotional features
            tweet_features['emoticon_count'] = len(re.findall(r'[:\-;=8][)(\[\]{}|\\\/DpP]', text))
            
            features.append(list(tweet_features.values()))
        
        return np.array(features)
    
    def extract_temporal_features(self, timestamps: List[str]) -> np.ndarray:
        """
        Extract temporal features from tweet timestamps.
        
        Args:
            timestamps (List[str]): List of timestamp strings
            
        Returns:
            np.ndarray: Temporal feature matrix
        """
        features = []
        
        for timestamp_str in timestamps:
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                temp_features = {}
                temp_features['hour'] = timestamp.hour
                temp_features['day_of_week'] = timestamp.weekday()
                temp_features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
                temp_features['is_morning'] = 1 if 6 <= timestamp.hour <= 11 else 0
                temp_features['is_afternoon'] = 1 if 12 <= timestamp.hour <= 17 else 0
                temp_features['is_evening'] = 1 if 18 <= timestamp.hour <= 23 else 0
                temp_features['is_night'] = 1 if timestamp.hour <= 5 else 0
                
                features.append(list(temp_features.values()))
                
            except ValueError:
                # Handle parsing errors
                features.append([0] * 7)
        
        return np.array(features)
    
    def combine_features(self, text_features: np.ndarray, stylistic_features: np.ndarray, 
                        temporal_features: np.ndarray = None) -> np.ndarray:
        """
        Combine different types of features.
        
        Args:
            text_features (np.ndarray): Text-based features
            stylistic_features (np.ndarray): Stylistic features
            temporal_features (np.ndarray): Temporal features (optional)
            
        Returns:
            np.ndarray: Combined feature matrix
        """
        features_list = [text_features, stylistic_features]
        
        if temporal_features is not None:
            features_list.append(temporal_features)
        
        return np.hstack(features_list)

def extract_features(df: pd.DataFrame, feature_types: List[str] = ['text', 'stylistic', 'temporal'], 
                    text_column: str = 'tweet_text') -> Dict[str, Any]:
    """
    Main function to extract features from tweet DataFrame.
    
    Args:
        df (pd.DataFrame): Tweet DataFrame
        feature_types (List[str]): Types of features to extract
        text_column (str): Name of the text column
        
    Returns:
        Dict[str, Any]: Dictionary containing features and extractor
    """
    extractor = TweetFeatureExtractor()
    features_dict = {}
    
    if 'text' in feature_types:
        text_features = extractor.extract_text_features(df[text_column].tolist())
        features_dict['text'] = text_features
    
    if 'stylistic' in feature_types:
        stylistic_features = extractor.extract_stylistic_features(df[text_column].tolist())
        features_dict['stylistic'] = stylistic_features
    
    if 'temporal' in feature_types and 'timestamp' in df.columns:
        temporal_features = extractor.extract_temporal_features(df['timestamp'].tolist())
        features_dict['temporal'] = temporal_features
    
    # Combine all features
    all_features = []
    for feature_type in feature_types:
        if feature_type in features_dict:
            all_features.append(features_dict[feature_type])
    
    if all_features:
        features_dict['combined'] = np.hstack(all_features)
    
    features_dict['extractor'] = extractor
    features_dict['labels'] = df['label'].values if 'label' in df.columns else None
    
    return features_dict


def get_feature_names(extractor: TweetFeatureExtractor, feature_types: List[str]) -> List[str]:
    """
    Get feature names for interpretability.
    
    Args:
        extractor (TweetFeatureExtractor): Fitted feature extractor
        feature_types (List[str]): Types of features extracted
        
    Returns:
        List[str]: Feature names
    """
    feature_names = []
    
    if 'text' in feature_types and extractor.tfidf_vectorizer is not None:
        text_names = [f"tfidf_{name}" for name in extractor.tfidf_vectorizer.get_feature_names_out()]
        feature_names.extend(text_names)
    
    if 'stylistic' in feature_types:
        stylistic_names = [
            'char_count', 'word_count', 'avg_word_length', 'caps_count', 'caps_ratio', 
            'all_caps_words', 'exclamation_count', 'question_count', 'period_count', 
            'comma_count', 'ellipsis_count', 'hashtag_count', 'mention_count', 
            'url_count', 'emoticon_count'
        ]
        feature_names.extend(stylistic_names)
    
    if 'temporal' in feature_types:
        temporal_names = [
            'hour', 'day_of_week', 'is_weekend', 'is_morning', 
            'is_afternoon', 'is_evening', 'is_night'
        ]
        feature_names.extend(temporal_names)
    
    return feature_names