import numpy as np
import os
import re
from typing import Tuple, List, Dict, Set

class DataProcessor:
    def __init__(self, data_dir: str):
        """
        Initialize data processor
        
        Args:
            data_dir: Directory containing the dataset
        """
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'train.tsv')
        self.test_path = os.path.join(data_dir, 'test.tsv')
        
    def load_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Load training and test datasets
        
        Returns:
            train_texts: List of training texts
            train_labels: List of training labels
            test_texts: List of test texts
            test_labels: List of test labels
        """
        train_texts, train_labels = self._read_tsv(self.train_path)
        test_texts, test_labels = self._read_tsv(self.test_path)
        
        return train_texts, train_labels, test_texts, test_labels
    
    def _read_tsv(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        Read TSV file
        
        Args:
            file_path: Path to the TSV file
            
        Returns:
            texts: List of texts
            labels: List of labels
        """
        texts = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    text, label = parts
                    texts.append(text)
                    labels.append(int(label))
        
        return texts, labels
    
    def preprocess_text(self, text: str) -> str:
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
    
    def split_train_valid(self, texts: List[str], labels: List[int], valid_ratio: float = 0.2) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Split training data into training and validation sets
        
        Args:
            texts: List of texts
            labels: List of labels
            valid_ratio: Validation set ratio
            
        Returns:
            train_texts: Training texts
            train_labels: Training labels
            valid_texts: Validation texts
            valid_labels: Validation labels
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Get dataset size
        n_samples = len(texts)
        
        # Generate random indices
        indices = np.random.permutation(n_samples)
        valid_size = int(n_samples * valid_ratio)
        
        # Split indices
        valid_indices = indices[:valid_size]
        train_indices = indices[valid_size:]
        
        # Split data
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        valid_texts = [texts[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        
        return train_texts, train_labels, valid_texts, valid_labels 