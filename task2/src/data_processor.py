import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import pandas as pd
from collections import Counter
import string

class TextDataset(Dataset):
    """Text dataset class for PyTorch DataLoader"""
    
    def __init__(self, texts, labels, tokenizer, max_length=None):
        """
        Initialize text dataset
        
        Args:
            texts: List of texts
            labels: List of labels
            tokenizer: Tokenizer object
            max_length: Maximum sequence length, None means no truncation
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Process text using tokenizer
        tokens = self.tokenizer.tokenize(text)
        
        # Truncate if necessary
        if self.max_length is not None:
            tokens = tokens[:self.max_length]
        
        # Convert tokens to indices
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Convert label to integer if it's a string
        if isinstance(label, str):
            try:
                label = int(label)
            except ValueError:
                # If conversion fails, handle it appropriately
                # For example, you might need to map string labels to integers
                # This is a simple example assuming labels are already numeric strings
                print(f"Warning: Could not convert label '{label}' to integer")
                label = 0
        
        return {
            'text': text,
            'tokens': tokens,
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader
        
        Args:
            batch: List of samples
            
        Returns:
            Batched tensors
        """
        # Get max sequence length in this batch
        max_len = max([len(sample['token_ids']) for sample in batch])
        
        # Prepare batched tensors
        texts = [sample['text'] for sample in batch]
        tokens = [sample['tokens'] for sample in batch]
        token_ids = []
        labels = []
        
        # Pad sequences
        for sample in batch:
            # Pad token_ids
            padded_ids = sample['token_ids'].tolist()
            padded_ids = padded_ids + [0] * (max_len - len(padded_ids))  # 0 is <PAD> token
            token_ids.append(torch.tensor(padded_ids, dtype=torch.long))
            
            # Get label
            labels.append(sample['label'])
        
        # Stack tensors
        token_ids = torch.stack(token_ids)
        labels = torch.stack(labels)
        
        return {
            'text': texts,
            'tokens': tokens,
            'token_ids': token_ids,
            'label': labels
        }


class Tokenizer:
    """Tokenizer class for text tokenization and conversion to indices"""
    
    def __init__(self, vocab=None, min_freq=1, max_vocab_size=None):
        """
        Initialize tokenizer
        
        Args:
            vocab: Predefined vocabulary, None means it needs to be built from data
            min_freq: Minimum word frequency
            max_vocab_size: Maximum vocabulary size
        """
        self.vocab = vocab
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        
        # Vocabulary and index mappings
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # If a predefined vocabulary is provided, use it directly
        if vocab is not None:
            self.build_vocab_from_list(vocab)
    
    def build_vocab(self, texts):
        """
        Build vocabulary from text list
        
        Args:
            texts: List of texts
        """
        # Count word frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        # Filter low-frequency words
        filtered_words = [word for word, freq in self.word_freq.items() 
                         if freq >= self.min_freq]
        
        # Limit vocabulary size
        if self.max_vocab_size is not None:
            filtered_words = sorted(filtered_words, 
                                   key=lambda x: self.word_freq[x], 
                                   reverse=True)[:self.max_vocab_size-2]  # -2 for PAD and UNK
        
        # Build vocabulary
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        
        for idx, word in enumerate(filtered_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def build_vocab_from_list(self, vocab_list):
        """
        Build vocabulary from word list
        
        Args:
            vocab_list: List of words
        """
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        
        for idx, word in enumerate(vocab_list, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def tokenize(self, text):
        """
        Tokenize text
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        """
        Convert tokens to indices
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of indices
        """
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """
        Convert indices to tokens
        
        Args:
            ids: List of indices
            
        Returns:
            List of tokens
        """
        return [self.idx2word.get(idx, self.unk_token) for idx in ids]
    
    def __len__(self):
        return len(self.word2idx)


class DataProcessor:
    """Data processor class"""
    
    def __init__(self, data_dir):
        """
        Initialize data processor
        
        Args:
            data_dir: Data directory
        """
        self.data_dir = data_dir
    
    def load_data(self):
        """
        Load training and test datasets
        
        Returns:
            train_texts: Training texts
            train_labels: Training labels
            test_texts: Test texts
            test_labels: Test labels
        """
        # Load training set
        train_file = os.path.join(self.data_dir, 'train.tsv')
        train_df = pd.read_csv(train_file, sep='\t', header=None)
        # First column is text, second column is label
        train_texts = train_df[0].tolist()
        train_labels = train_df[1].tolist()
        
        # Load test set
        test_file = os.path.join(self.data_dir, 'test.tsv')
        test_df = pd.read_csv(test_file, sep='\t', header=None)
        # First column is text, second column is label
        test_texts = test_df[0].tolist()
        test_labels = test_df[1].tolist()
        
        return train_texts, train_labels, test_texts, test_labels
    
    def preprocess_text(self, text):
        """
        Preprocess text
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_train_valid(self, texts, labels, valid_ratio=0.2, random_state=42):
        """
        Split training and validation sets
        
        Args:
            texts: List of texts
            labels: List of labels
            valid_ratio: Validation set ratio
            random_state: Random seed
            
        Returns:
            train_texts: Training texts
            train_labels: Training labels
            valid_texts: Validation texts
            valid_labels: Validation labels
        """
        # Set random seed
        np.random.seed(random_state)
        
        # Get dataset size
        n_samples = len(texts)
        
        # Generate indices and shuffle
        indices = np.random.permutation(n_samples)
        
        # Calculate validation set size
        n_valid = int(n_samples * valid_ratio)
        
        # Split training and validation sets
        valid_indices = indices[:n_valid]
        train_indices = indices[n_valid:]
        
        # Get training and validation sets
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        valid_texts = [texts[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        
        return train_texts, train_labels, valid_texts, valid_labels
    
    def create_dataloaders(self, train_texts, train_labels, valid_texts, valid_labels, 
                          test_texts, test_labels, tokenizer, batch_size=32, max_length=None):
        """
        Create DataLoaders
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            valid_texts: Validation texts
            valid_labels: Validation labels
            test_texts: Test texts
            test_labels: Test labels
            tokenizer: Tokenizer
            batch_size: Batch size
            max_length: Maximum sequence length
            
        Returns:
            train_loader: Training DataLoader
            valid_loader: Validation DataLoader
            test_loader: Test DataLoader
        """
        # Create datasets
        train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
        valid_dataset = TextDataset(valid_texts, valid_labels, tokenizer, max_length)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)
        
        # Create DataLoaders with custom collate function
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 collate_fn=TextDataset.collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                 collate_fn=TextDataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=TextDataset.collate_fn)
        
        return train_loader, valid_loader, test_loader
    
    def create_embedding_matrix(self, tokenizer, embedding_dim=100, embedding_path=None):
        """
        Create embedding matrix
        
        Args:
            tokenizer: Tokenizer
            embedding_dim: Embedding dimension
            embedding_path: Pre-trained embedding path, None means random initialization
            
        Returns:
            embedding_matrix: Embedding matrix
        """
        vocab_size = len(tokenizer)
        embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
        
        # Set <PAD> embedding to zeros
        embedding_matrix[0] = np.zeros(embedding_dim)
        
        # If pre-trained embeddings are provided, load them
        if embedding_path is not None:
            print(f"Loading pre-trained embeddings from {embedding_path}")
            word_embeddings = {}
            
            with open(embedding_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.strip().split()
                    word = values[0]
                    vector = np.array(values[1:], dtype='float32')
                    word_embeddings[word] = vector
            
            # Update embedding matrix
            found_words = 0
            for word, idx in tokenizer.word2idx.items():
                if word in word_embeddings:
                    embedding_matrix[idx] = word_embeddings[word]
                    found_words += 1
            
            print(f"Found {found_words}/{vocab_size} words in pre-trained embeddings")
        
        return torch.FloatTensor(embedding_matrix) 