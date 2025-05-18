# Import libraries
import pandas as pd
import numpy as np
import re
import os
import pickle

# NLTK for text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Vectorization & encoding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
def preprocess_data(input_file):
    # Load data
    try:
        df = pd.read_csv(input_file)
        print("File loaded successfully!")
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        return None

    # Clean sentiment
    df['clean_sentiment'] = df['sentiment'].apply(clean_sentiment)
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Handle missing values
    df = df.dropna(subset=['text'])
    
    # Text preprocessing
    df['text_clean'] = df['text'].copy()
    df['text_clean'] = df['text_clean'].apply(lambda x: str(x).lower())
    df['text_clean'] = df['text_clean'].apply(remove_special_characters)
    df['text_clean'] = df['text_clean'].apply(remove_stopwords)
    df['tokenized'] = df['text_clean'].apply(tokenize_text)
    
    # Encode sentiment
    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['clean_sentiment'])
    
    # Save label encoder
    with open('preprocessed_artifacts/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    text_features = vectorizer.fit_transform(df['text_clean'])
    
    # Save vectorizer
    with open('preprocessed_artifacts/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return df, vectorizer, label_encoder

# Helper functions
def clean_sentiment(text):
    text = str(text).lower()
    if 'negative' in text:
        return 'negative'
    elif 'positive' in text:
        return 'positive'
    elif 'neutral' in text:
        return 'neutral'
    else:
        return 'unknown'

def remove_special_characters(text):
    if isinstance(text, str):
        # Remove emojis
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            u"\U0001F800-\U0001F8FF"  # Supplemental Symbols and Pictographs
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return text

def remove_stopwords(text):
    if isinstance(text, str):
        st_words = stopwords.words('english')
        return ' '.join([w for w in text.split() if w not in st_words])
    return text

def tokenize_text(text):
    if not isinstance(text, str):
        return []
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

if __name__ == "__main__":
    # Create directories for outputs
    os.makedirs('preprocessed_artifacts', exist_ok=True)
    os.makedirs('preprocessed_data', exist_ok=True)
    
    # Process data
    input_file = '../cryptonews_raw/cryptonews.csv'
    df, vectorizer, label_encoder = preprocess_data(input_file)
    
    if df is not None:
        # Save processed data
        columns_to_save = [
            'date', 'source', 'subject', 'title',
            'text_clean', 'tokenized', 'clean_sentiment', 'sentiment_encoded'
        ]
        df_filtered = df[columns_to_save]
        df_filtered.to_csv('preprocessed_data/preprocessed_cryptonews.csv', index=False)
        print("Preprocessing completed successfully!")