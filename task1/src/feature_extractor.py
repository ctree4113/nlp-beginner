import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter
import re

class FeatureExtractor:
    def __init__(self, feature_type: str = 'bow', ngram_range: Tuple[int, int] = (1, 1), max_features: int = 5000):
        """
        Initialize feature extractor
        
        Args:
            feature_type: Feature type, 'bow' for Bag-of-Words, 'binary' for binary features, 'tfidf' for TF-IDF
            ngram_range: N-gram range, e.g., (1, 2) for unigrams and bigrams
            max_features: Maximum number of features
        """
        self.feature_type = feature_type
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocabulary = {}  # Vocabulary mapping words to indices
        self.feature_names = []  # List of feature names
        self.idf = None  # For TF-IDF
        
    def fit(self, texts: List[str]) -> 'FeatureExtractor':
        """
        Build vocabulary from training texts
        
        Args:
            texts: List of texts
            
        Returns:
            self
        """
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Extract all n-grams
        all_ngrams = []
        for text in processed_texts:
            text_ngrams = self._extract_ngrams(text)
            all_ngrams.extend(text_ngrams)
        
        # Count n-gram frequencies
        ngram_counts = Counter(all_ngrams)
        
        # Select most common n-grams as features
        most_common = ngram_counts.most_common(self.max_features)
        
        # Build vocabulary
        self.vocabulary = {ngram: idx for idx, (ngram, _) in enumerate(most_common)}
        self.feature_names = [ngram for ngram, _ in most_common]
        
        # Compute IDF (if needed)
        if self.feature_type == 'tfidf':
            self._compute_idf(processed_texts)
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix with shape (n_samples, n_features)
        """
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Initialize feature matrix
        n_samples = len(processed_texts)
        n_features = len(self.vocabulary)
        X = np.zeros((n_samples, n_features))
        
        # Fill feature matrix
        for i, text in enumerate(processed_texts):
            ngrams = self._extract_ngrams(text)
            ngram_counts = Counter(ngrams)
            
            for ngram, count in ngram_counts.items():
                if ngram in self.vocabulary:
                    feature_idx = self.vocabulary[ngram]
                    
                    if self.feature_type == 'bow':
                        X[i, feature_idx] = count
                    elif self.feature_type == 'binary':
                        X[i, feature_idx] = 1
                    elif self.feature_type == 'tfidf':
                        tf = count / len(ngrams)  # Term frequency
                        X[i, feature_idx] = tf * self.idf[feature_idx]
        
        return X
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text
        
        Args:
            text: Original text
            
        Returns:
            Processed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_ngrams(self, text: str) -> List[str]:
        """
        Extract N-grams from text
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of N-grams
        """
        words = text.split()
        ngrams = []
        
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
        
        return ngrams
    
    def _compute_idf(self, processed_texts: List[str]) -> None:
        """
        Compute Inverse Document Frequency (IDF)
        
        Args:
            processed_texts: List of preprocessed texts
        """
        n_samples = len(processed_texts)
        self.idf = np.zeros(len(self.vocabulary))
        
        # Count document frequency for each feature
        doc_freq = np.zeros(len(self.vocabulary))
        
        for text in processed_texts:
            ngrams = set(self._extract_ngrams(text))  # Use set to remove duplicates
            for ngram in ngrams:
                if ngram in self.vocabulary:
                    doc_freq[self.vocabulary[ngram]] += 1
        
        # Compute IDF
        self.idf = np.log(n_samples / (doc_freq + 1)) + 1  # Smoothing to avoid division by zero 