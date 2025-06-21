"""
Data loading utilities for Trump tweets dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load Trump tweets data from TSV file.
    
    Args:
        file_path (str): Path to the TSV file
        
    Returns:
        pd.DataFrame: Loaded data with columns [tweet_id, handle, text, timestamp, device]
    """
    column_names = ['tweet_id', 'handle', 'text', 'timestamp', 'device']
    
    try:
        df = pd.read_csv(file_path, sep='\t', names=column_names, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def create_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Create binary labels from device information.
    
    Args:
        df (pd.DataFrame): DataFrame with device column
        
    Returns:
        np.ndarray: Binary labels (0=Trump/Android, 1=Staffer/iPhone)
    """
    labels = np.where(df['device'].str.lower() == 'android', 0, 1)
    return labels

def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, 
        stratify=create_labels(df)
    )
    
    return train_df, test_df