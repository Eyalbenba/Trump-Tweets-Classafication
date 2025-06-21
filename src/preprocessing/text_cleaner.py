"""
Text cleaning utilities for preprocessing tweets.
"""

import re
import string
from typing import List, Optional

def clean_text(text: str, remove_urls: bool = True, remove_mentions: bool = True, 
               remove_hashtags: bool = False, lowercase: bool = True) -> str:
    """
    Clean tweet text with various preprocessing options.
    
    Args:
        text (str): Input text to clean
        remove_urls (bool): Remove URLs
        remove_mentions (bool): Remove @mentions
        remove_hashtags (bool): Remove #hashtags
        lowercase (bool): Convert to lowercase
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove mentions
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (but keep the text)
    if remove_hashtags:
        text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove HTML entities
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    return text

def remove_punctuation(text: str, keep_emoticons: bool = True) -> str:
    """
    Remove punctuation from text with option to keep emoticons.
    
    Args:
        text (str): Input text
        keep_emoticons (bool): Whether to preserve emoticons
        
    Returns:
        str: Text with punctuation removed
    """
    if keep_emoticons:
        # Keep common emoticons
        emoticon_pattern = r'[:\-;=8][)(\[\]{}|\\\/DpP]|[)(\[\]{}|\\\/DpP][:\-;=8]'
        emoticons = re.findall(emoticon_pattern, text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Add emoticons back
        text = text + ' ' + ' '.join(emoticons)
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text.strip()

def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization of text.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of tokens
    """
    return text.split()