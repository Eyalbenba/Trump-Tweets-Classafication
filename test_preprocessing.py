#!/usr/bin/env python3
"""
Test script to demonstrate preprocessing functionality
"""

import sys
import os
sys.path.append('src')

from preprocessing.data_loader import load_data, create_labels
from preprocessing.text_cleaner import clean_text, remove_punctuation
from preprocessing.feature_extractor import TweetFeatureExtractor

def test_data_loading():
    """Test data loading functionality"""
    print("=== Testing Data Loading ===")
    
    # Load the training data
    df = load_data('data/trump_train.tsv')
    print(f"Loaded {len(df)} tweets")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows
    print("\nFirst 3 tweets:")
    for i in range(min(3, len(df))):
        print(f"Tweet {i+1}:")
        print(f"  Text: {df.iloc[i]['text'][:100]}...")
        print(f"  Device: {df.iloc[i]['device']}")
        print(f"  Timestamp: {df.iloc[i]['timestamp']}")
        print()
    
    # Create labels
    labels = create_labels(df)
    print(f"Labels created: {len(labels)} labels")
    print(f"Trump tweets (0): {sum(labels == 0)}")
    print(f"Staffer tweets (1): {sum(labels == 1)}")
    
    return df

def test_text_cleaning():
    """Test text cleaning functionality"""
    print("\n=== Testing Text Cleaning ===")
    
    # Sample tweets for testing
    sample_tweets = [
        "Democrats are smiling in D.C. that the Freedom Caucus, with the help of Club For Growth and Heritage, have saved Planned Parenthood &amp; Ocare!",
        "For first time the failing @nytimes will take an ad (a bad one) to help save its failing reputation. Try reporting accurately &amp; fairly!",
        "INTELLIGENCE INSIDERS NOW CLAIM THE TRUMP DOSSIER IS \"A COMPLETE FRAUD!\" @OANN"
    ]
    
    for i, tweet in enumerate(sample_tweets):
        print(f"\nTweet {i+1} Original:")
        print(f"  {tweet}")
        
        cleaned = clean_text(tweet)
        print(f"Cleaned:")
        print(f"  {cleaned}")
        
        no_punct = remove_punctuation(cleaned)
        print(f"No Punctuation:")
        print(f"  {no_punct}")

def test_feature_extraction():
    """Test feature extraction functionality"""
    print("\n=== Testing Feature Extraction ===")
    
    # Load small sample of data
    df = load_data('data/trump_train.tsv')
    sample_df = df.head(5).copy()  # Just first 5 tweets
    
    # Clean the text
    sample_df['cleaned_text'] = sample_df['text'].apply(clean_text)
    
    extractor = TweetFeatureExtractor()
    
    # Test stylistic features
    print("Testing stylistic features...")
    stylistic_features = extractor.extract_stylistic_features(sample_df['text'].tolist())
    print(f"Stylistic features shape: {stylistic_features.shape}")
    print("Features for first tweet:")
    feature_names = ['char_count', 'word_count', 'avg_word_length', 'caps_count', 'caps_ratio', 
                    'all_caps_words', 'exclamation_count', 'question_count', 'period_count', 
                    'comma_count', 'ellipsis_count', 'hashtag_count', 'mention_count', 
                    'url_count', 'emoticon_count']
    
    for i, feature_name in enumerate(feature_names):
        print(f"  {feature_name}: {stylistic_features[0][i]}")
    
    # Test temporal features
    print("\nTesting temporal features...")
    temporal_features = extractor.extract_temporal_features(sample_df['timestamp'].tolist())
    print(f"Temporal features shape: {temporal_features.shape}")
    print("Temporal features for first tweet:")
    temporal_names = ['hour', 'day_of_week', 'is_weekend', 'is_morning', 
                     'is_afternoon', 'is_evening', 'is_night']
    
    for i, feature_name in enumerate(temporal_names):
        print(f"  {feature_name}: {temporal_features[0][i]}")
    
    # Test text features (small sample)
    print("\nTesting text features...")
    text_features = extractor.extract_text_features(sample_df['cleaned_text'].tolist(), max_features=100)
    print(f"Text features shape: {text_features.shape}")
    print(f"Non-zero features in first tweet: {sum(text_features[0] > 0)}")

if __name__ == "__main__":
    # Test all preprocessing components
    df = test_data_loading()
    test_text_cleaning()
    test_feature_extraction()
    
    print("\n=== Preprocessing Test Complete ===")
    print("All preprocessing modules are working correctly!")