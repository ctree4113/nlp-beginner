import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
import numpy as np


class CoNLLDataset(Dataset):
    """
    CoNLL dataset for NER
    """
    def __init__(self, data, word2idx, tag2idx, max_seq_len, char2idx=None, max_word_len=None):
        """
        Initialize CoNLL dataset
        
        Args:
            data: List of tuples (sentence, tags), where both are lists
            word2idx: Word to index mapping
            tag2idx: Tag to index mapping
            max_seq_len: Maximum sequence length
            char2idx: Character to index mapping
            max_word_len: Maximum word length
        """
        self.data = data
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_seq_len = max_seq_len
        self.char2idx = char2idx
        self.max_word_len = max_word_len
    
    def __len__(self):
        """
        Return dataset size
        
        Returns:
            length: Dataset size
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a data item
        
        Args:
            idx: Index
            
        Returns:
            sample: Sample containing word_ids, tag_ids and seq_length
        """
        sentence, tags = self.data[idx]
        
        # Truncate sequence if too long
        if len(sentence) > self.max_seq_len:
            sentence = sentence[:self.max_seq_len]
            tags = tags[:self.max_seq_len]
        
        # Get actual sequence length
        seq_length = len(sentence)
        
        # Convert words to indices
        word_ids = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence]
        
        # Convert tags to indices
        tag_ids = [self.tag2idx[tag] for tag in tags]
        
        # Create sample
        sample = {
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
            "seq_length": seq_length
        }
        
        # Add character-level features if char2idx is provided
        if self.char2idx is not None and self.max_word_len is not None:
            char_ids = []
            for word in sentence:
                word_char_ids = []
                for char in word[:self.max_word_len]:
                    word_char_ids.append(self.char2idx.get(char, self.char2idx["<UNK>"]))
                
                # Pad word to max_word_len
                while len(word_char_ids) < self.max_word_len:
                    word_char_ids.append(0)  # Pad with 0
                
                char_ids.append(word_char_ids)
            
            # Pad sequence to max_seq_len
            while len(char_ids) < self.max_seq_len:
                char_ids.append([0] * self.max_word_len)
            
            sample["char_ids"] = torch.tensor(char_ids, dtype=torch.long)
        
        return sample


class DataProcessor:
    """
    Data processor for NER task
    """
    def __init__(self, data_dir, language="eng", max_seq_len=128, max_word_len=20):
        """
        Initialize data processor
        
        Args:
            data_dir: Data directory
            language: Language, either 'eng' or 'deu'
            max_seq_len: Maximum sequence length
            max_word_len: Maximum word length
        """
        self.data_dir = data_dir
        self.language = language
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        
        # These will be built in load_data
        self.word2idx = None
        self.tag2idx = None
        self.char2idx = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
    
    def load_data(self):
        """
        Load dataset and create datasets
        """
        # Read training set
        train_file = os.path.join(self.data_dir, f"{self.language}.train")
        train_sentences, train_tags = self._read_conll_file(train_file)
        
        # Read validation set
        valid_file = os.path.join(self.data_dir, f"{self.language}.testa")
        valid_sentences, valid_tags = self._read_conll_file(valid_file)
        
        # Read test set
        test_file = os.path.join(self.data_dir, f"{self.language}.testb")
        test_sentences, test_tags = self._read_conll_file(test_file)
        
        # Build vocabulary
        self.word2idx = self._build_vocab(train_sentences)
        
        # Build character vocabulary
        self.char2idx = self._build_char_vocab(train_sentences)
        
        # Build tag map
        self.tag2idx = self._build_tag_map(train_tags)
        
        # Create datasets
        self.train_dataset = CoNLLDataset(
            list(zip(train_sentences, train_tags)),
            self.word2idx,
            self.tag2idx,
            self.max_seq_len,
            self.char2idx,
            self.max_word_len
        )
        
        self.valid_dataset = CoNLLDataset(
            list(zip(valid_sentences, valid_tags)),
            self.word2idx,
            self.tag2idx,
            self.max_seq_len,
            self.char2idx,
            self.max_word_len
        )
        
        self.test_dataset = CoNLLDataset(
            list(zip(test_sentences, test_tags)),
            self.word2idx,
            self.tag2idx,
            self.max_seq_len,
            self.char2idx,
            self.max_word_len
        )
    
    def _read_conll_file(self, file_path):
        """
        Read CoNLL format file
        
        Args:
            file_path: File path
            
        Returns:
            sentences: List of sentences, each containing a list of words
            tags: List of tag lists, each corresponding to a sentence
        """
        sentences = []
        tags = []
        
        sentence = []
        sentence_tags = []
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                sentences = []
                tags = []
                sentence = []
                sentence_tags = []
                
                with open(file_path, 'r', encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        
                        # Empty line indicates end of sentence
                        if not line:
                            if sentence:
                                sentences.append(sentence)
                                tags.append(sentence_tags)
                                sentence = []
                                sentence_tags = []
                            continue
                        
                        # Parse line
                        fields = line.split()
                        
                        # CoNLL format: word POS chunk NER_tag
                        # We only care about word and NER tag
                        if len(fields) >= 4:
                            word = fields[0]
                            tag = fields[3]
                            
                            sentence.append(word)
                            sentence_tags.append(tag)
                
                # Add last sentence if not empty
                if sentence:
                    sentences.append(sentence)
                    tags.append(sentence_tags)
                
                # If we got here without error, break the loop
                break
            except UnicodeDecodeError:
                continue
        
        if not sentences:
            raise ValueError(f"Could not read file {file_path} with any of the attempted encodings")
        
        return sentences, tags
    
    def _build_vocab(self, sentences, min_freq=1):
        """
        Build vocabulary
        
        Args:
            sentences: List of sentences, each containing a list of words
            min_freq: Minimum word frequency, words with lower frequency will be filtered
            
        Returns:
            word2idx: Word to index mapping
        """
        # Count word frequencies
        word_counter = Counter()
        for sentence in sentences:
            word_counter.update(sentence)
        
        # Filter words by frequency
        valid_words = [word for word, count in word_counter.items() if count >= min_freq]
        
        # Build mapping
        word2idx = {
            "<PAD>": 0,  # Padding token
            "<UNK>": 1   # Unknown word token
        }
        
        # Add other words
        for i, word in enumerate(valid_words, start=2):
            word2idx[word] = i
        
        return word2idx
    
    def _build_tag_map(self, tags_list):
        """
        Build tag mapping
        
        Args:
            tags_list: List of tag lists, each corresponding to a sentence
            
        Returns:
            tag2idx: Tag to index mapping
        """
        # Build tag set
        tag_set = set()
        for tags in tags_list:
            tag_set.update(tags)
        
        # Build mapping
        tag2idx = {
            "O": 0  # Non-entity tag
        }
        
        # Add other tags
        for i, tag in enumerate(sorted(tag_set), start=1):
            if tag != "O":  # O was already added
                tag2idx[tag] = i
        
        return tag2idx
    
    def _build_char_vocab(self, sentences, min_freq=1):
        """
        Build character vocabulary
        
        Args:
            sentences: List of sentences
            min_freq: Minimum frequency for a character to be included
            
        Returns:
            char2idx: Character to index mapping
        """
        # Count character frequencies
        char_counter = Counter()
        for sentence in sentences:
            for word in sentence:
                for char in word:
                    char_counter[char] += 1
        
        # Create character to index mapping
        char2idx = {"<PAD>": 0, "<UNK>": 1}
        for char, freq in char_counter.items():
            if freq >= min_freq:
                char2idx[char] = len(char2idx)
        
        print(f"Character vocabulary size: {len(char2idx)}")
        
        return char2idx
    
    def create_dataloaders(self, batch_size=32):
        """
        Create data loaders
        
        Args:
            batch_size: Batch size
            
        Returns:
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
        """
        def collate_fn(batch):
            """
            Collate function for data loader
            
            Args:
                batch: Batch of samples
                
            Returns:
                batch_dict: Batch dictionary
            """
            # Get batch size
            batch_size = len(batch)
            
            # Get max sequence length in batch
            max_len = max([sample["seq_length"] for sample in batch])
            
            # Prepare tensors
            word_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
            tag_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
            mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
            seq_lengths = torch.zeros(batch_size, dtype=torch.long)
            
            # Fill tensors
            for i, sample in enumerate(batch):
                seq_len = sample["seq_length"]
                word_ids[i, :seq_len] = sample["word_ids"]
                tag_ids[i, :seq_len] = sample["tag_ids"]
                mask[i, :seq_len] = 1
                seq_lengths[i] = seq_len
            
            # Create batch dictionary
            batch_dict = {
                "word_ids": word_ids,
                "tag_ids": tag_ids,
                "mask": mask,
                "seq_lengths": seq_lengths
            }
            
            # Add character-level features if available
            if "char_ids" in batch[0]:
                char_ids = torch.zeros(batch_size, max_len, self.max_word_len, dtype=torch.long)
                for i, sample in enumerate(batch):
                    seq_len = sample["seq_length"]
                    char_ids[i, :seq_len] = sample["char_ids"][:seq_len]
                
                batch_dict["char_ids"] = char_ids
            
            return batch_dict
        
        # Create DataLoaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader, test_loader 