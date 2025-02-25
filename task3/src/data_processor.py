import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nltk
nltk.download('punkt', quiet=True)  # Ensure NLTK data is downloaded


class Tokenizer:
    """
    Tokenizer for text processing
    
    Handles tokenization, vocabulary building, and conversion between tokens and indices
    """
    
    def __init__(self, min_freq=1, max_vocab_size=None):
        """
        Initialize tokenizer
        
        Args:
            min_freq: Minimum frequency for a token to be included in vocabulary
            max_vocab_size: Maximum vocabulary size
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_freq = {}
        self.vocab_size = 2  # Start with pad and unk tokens
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            tokens: List of tokens
        """
        return nltk.word_tokenize(text.lower())
    
    def build_vocab(self, texts):
        """
        Build vocabulary from texts
        
        Args:
            texts: List of texts
        """
        # Count word frequencies
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                self.word_freq[token] = self.word_freq.get(token, 0) + 1
        
        # Filter by minimum frequency
        words = [word for word, freq in self.word_freq.items() if freq >= self.min_freq]
        
        # Sort by frequency (descending)
        words = sorted(words, key=lambda x: self.word_freq[x], reverse=True)
        
        # Limit vocabulary size if specified
        if self.max_vocab_size is not None:
            words = words[:self.max_vocab_size - 2]  # -2 for <pad> and <unk>
        
        # Build word to index mapping
        for word in words:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
    
    def convert_tokens_to_ids(self, tokens):
        """
        Convert tokens to indices
        
        Args:
            tokens: List of tokens
            
        Returns:
            indices: List of indices
        """
        return [self.word2idx.get(token, 1) for token in tokens]  # 1 is <unk>
    
    def __len__(self):
        """Return vocabulary size"""
        return self.vocab_size


class SNLIDataset(Dataset):
    """
    Dataset for Stanford Natural Language Inference (SNLI) corpus
    
    Handles loading and preprocessing of premise-hypothesis pairs and their labels
    """
    
    def __init__(self, data, tokenizer, max_seq_len=50):
        """
        Initialize dataset
        
        Args:
            data: List of dictionaries containing premise, hypothesis, and label
            tokenizer: Tokenizer for text processing
            max_seq_len: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Label mapping
        self.label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    def __len__(self):
        """Return dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Args:
            idx: Index
            
        Returns:
            item: Dictionary containing processed premise, hypothesis, and label
        """
        item = self.data[idx]
        
        # Get premise and hypothesis
        premise = item['sentence1']
        hypothesis = item['sentence2']
        
        # Tokenize
        premise_tokens = self.tokenizer.tokenize(premise)
        hypothesis_tokens = self.tokenizer.tokenize(hypothesis)
        
        # Truncate if necessary
        premise_tokens = premise_tokens[:self.max_seq_len]
        hypothesis_tokens = hypothesis_tokens[:self.max_seq_len]
        
        # Convert tokens to indices
        premise_ids = self.tokenizer.convert_tokens_to_ids(premise_tokens)
        hypothesis_ids = self.tokenizer.convert_tokens_to_ids(hypothesis_tokens)
        
        # Create masks (1 for real tokens, 0 for padding)
        premise_mask = [1] * len(premise_ids)
        hypothesis_mask = [1] * len(hypothesis_ids)
        
        # Store original lengths
        premise_len = len(premise_ids)
        hypothesis_len = len(hypothesis_ids)
        
        # Pad sequences
        premise_ids = premise_ids + [0] * (self.max_seq_len - premise_len)
        hypothesis_ids = hypothesis_ids + [0] * (self.max_seq_len - hypothesis_len)
        premise_mask = premise_mask + [0] * (self.max_seq_len - premise_len)
        hypothesis_mask = hypothesis_mask + [0] * (self.max_seq_len - hypothesis_len)
        
        # Get label
        label = self.label_map.get(item['gold_label'], 1)  # Default to neutral for unknown labels
        
        return {
            'premise_ids': torch.tensor(premise_ids, dtype=torch.long),
            'hypothesis_ids': torch.tensor(hypothesis_ids, dtype=torch.long),
            'premise_mask': torch.tensor(premise_mask, dtype=torch.bool),
            'hypothesis_mask': torch.tensor(hypothesis_mask, dtype=torch.bool),
            'label': torch.tensor(label, dtype=torch.long)
        }


class DataProcessor:
    """
    Data processor for SNLI dataset
    
    Handles loading, preprocessing, and creating data loaders
    """
    
    def __init__(self, data_dir, max_seq_len=50, batch_size=32):
        """
        Initialize data processor
        
        Args:
            data_dir: Directory containing the dataset
            max_seq_len: Maximum sequence length
            batch_size: Batch size
        """
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.debug = False
    
    def load_data(self):
        """
        Load SNLI data from files
        
        Returns:
            train_data: Training data
            valid_data: Validation data
            test_data: Test data
        """
        train_path = os.path.join(self.data_dir, 'snli_1.0_train.jsonl')
        valid_path = os.path.join(self.data_dir, 'snli_1.0_dev.jsonl')
        test_path = os.path.join(self.data_dir, 'snli_1.0_test.jsonl')
        
        train_data = self._read_jsonl(train_path)
        valid_data = self._read_jsonl(valid_path)
        test_data = self._read_jsonl(test_path)
        
        # Filter out examples with '-' label
        train_data = [item for item in train_data if item['gold_label'] != '-']
        valid_data = [item for item in valid_data if item['gold_label'] != '-']
        test_data = [item for item in test_data if item['gold_label'] != '-']
        
        # If in debug mode, use smaller datasets
        if self.debug:
            train_data = train_data[:1000]
            valid_data = valid_data[:200]
            test_data = test_data[:200]
        
        return train_data, valid_data, test_data
    
    def _read_jsonl(self, file_path):
        """
        Read JSONL file
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            data: List of dictionaries
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def create_dataloaders(self, debug=False):
        """
        Create data loaders for training, validation, and testing
        
        Args:
            debug: Whether to use debug mode (smaller datasets)
            
        Returns:
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            vocab_size: Vocabulary size
        """
        # Set debug mode
        self.debug = debug
        
        # Load data
        train_data, valid_data, test_data = self.load_data()
        
        # Create tokenizer
        tokenizer = Tokenizer(min_freq=3)
        
        # Build vocabulary from training data
        all_texts = []
        for item in train_data:
            all_texts.append(item['sentence1'])
            all_texts.append(item['sentence2'])
        tokenizer.build_vocab(all_texts)
        
        # Create datasets
        train_dataset = SNLIDataset(train_data, tokenizer, self.max_seq_len)
        valid_dataset = SNLIDataset(valid_data, tokenizer, self.max_seq_len)
        test_dataset = SNLIDataset(test_data, tokenizer, self.max_seq_len)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, valid_loader, test_loader, tokenizer.vocab_size 