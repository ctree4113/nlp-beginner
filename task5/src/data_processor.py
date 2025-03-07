import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
from torch.nn.utils.rnn import pad_sequence


class TextDataset(Dataset):
    """
    Enhanced character-level text dataset with variable-length sequence support
    """
    def __init__(self, data, seq_length, stride=1, min_seq_length=None, data_augmentation=False):
        """
        Initialize dataset
        
        Args:
            data: Text data indices
            seq_length: Maximum sequence length
            stride: Stride for sliding window
            min_seq_length: Minimum sequence length (for variable length)
            data_augmentation: Whether to use data augmentation
        """
        self.data = data
        self.seq_length = seq_length
        self.stride = stride
        self.data_augmentation = data_augmentation
        self.min_seq_length = min_seq_length if min_seq_length else seq_length
        
        # Create indices for accessing data with stride
        self.indices = list(range(0, len(data) - seq_length, stride))
        
    def __len__(self):
        """
        Return dataset size
        """
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a data item
        
        Args:
            idx: Index
            
        Returns:
            dict: Dictionary containing input and target sequences
        """
        # Get position index
        pos = self.indices[idx]
        
        # Determine sequence length (fixed or variable)
        if self.min_seq_length < self.seq_length and self.data_augmentation:
            # Variable length for augmentation
            curr_seq_length = random.randint(self.min_seq_length, self.seq_length)
        else:
            # Fixed length
            curr_seq_length = self.seq_length
        
        # Get input sequence (x) and target sequence (y)
        x = self.data[pos:pos + curr_seq_length]
        y = self.data[pos + 1:pos + curr_seq_length + 1]
        
        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return {
            'input': x_tensor,
            'target': y_tensor,
            'seq_len': curr_seq_length
        }


def custom_collate_fn(batch):
    """
    Custom collate function for variable-length sequences
    
    Args:
        batch: Batch of data
        
    Returns:
        batch_dict: Dictionary containing batched data
    """
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    seq_lens = [item['seq_len'] for item in batch]
    
    # Pad sequences
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return {
        'input': inputs_padded,
        'target': targets_padded,
        'seq_len': torch.tensor(seq_lens, dtype=torch.long)
    }


class DataProcessor:
    """
    Enhanced data processor for character-level language model
    """
    def __init__(self, args):
        """
        Initialize data processor
        
        Args:
            args: Command line arguments
        """
        self.data_path = args.data_path
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.min_seq_length = getattr(args, 'min_seq_length', args.seq_length)  # Default to seq_length
        self.stride = getattr(args, 'stride', 1)  # Default stride is 1
        self.data_augmentation = getattr(args, 'data_augmentation', False)  # Default no augmentation
        self.args = args  # Store the entire args object
        
        # Initialize data attributes
        self.text = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = 0
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """
        Load and preprocess text data with improved poetry handling
        """
        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Read text file
        print(f"Reading data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # For faster training, limit text size if in debug mode
        if hasattr(self.args, 'debug') and self.args.debug:
            max_chars = 5000  # Limit to 5000 characters in debug mode
            self.text = self.text[:max_chars]
            print(f"Debug mode: Text limited to {len(self.text)} characters")
        
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(self.data_path))[0]
        vocab_path = os.path.join(os.path.dirname(self.data_path), f"{filename}_vocab.json")
        
        # Option to force rebuild vocabulary
        force_rebuild = hasattr(self.args, 'rebuild_vocab') and self.args.rebuild_vocab
        if force_rebuild and os.path.exists(vocab_path):
            print(f"Forcing vocabulary rebuild. Removing existing vocabulary file: {vocab_path}")
            os.remove(vocab_path)
        
        # Preprocess text
        self.text = self._preprocess_text(self.text)
        
        # Build or load vocabulary
        if os.path.exists(vocab_path) and not force_rebuild:
            print(f"Loading vocabulary from {vocab_path}")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                self.char_to_idx = vocab_data['char_to_idx']
                self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
                self.vocab_size = len(self.char_to_idx)
                
                # Check and ensure special tokens are in vocabulary
                special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<LINE>']
                missing_tokens = [token for token in special_tokens if token not in self.char_to_idx]
                
                if missing_tokens:
                    print(f"Adding missing special tokens to vocabulary: {missing_tokens}")
                    # Get current maximum index
                    max_idx = max(int(idx) for idx in self.idx_to_char.keys())
                    
                    # Add missing special tokens
                    for token in missing_tokens:
                        max_idx += 1
                        self.char_to_idx[token] = max_idx
                        self.idx_to_char[max_idx] = token
                    
                    self.vocab_size = len(self.char_to_idx)
                    
                    # Update and save corrected vocabulary
                    vocab_data = {
                        'char_to_idx': self.char_to_idx,
                        'idx_to_char': self.idx_to_char
                    }
                    with open(vocab_path, 'w', encoding='utf-8') as f:
                        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
                    print(f"Updated vocabulary saved to {vocab_path}")
        else:
            print(f"Building vocabulary from text")
            self._build_vocab()
            
            # Save vocabulary for future use
            vocab_data = {
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char
            }
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            print(f"Vocabulary saved to {vocab_path}")
        
        # Process special tokens as complete tokens, not individual characters
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<LINE>']
        
        # Convert text to indices, handling special tokens as complete tokens
        data = []
        i = 0
        while i < len(self.text):
            # Check if current position starts a special token
            is_special_token = False
            for token in special_tokens:
                if self.text[i:i+len(token)] == token:
                    # Add the special token index
                    data.append(self.char_to_idx[token])
                    i += len(token)
                    is_special_token = True
                    break
            
            # If not a special token, process as a regular character
            if not is_special_token:
                char = self.text[i]
                if char in self.char_to_idx:
                    data.append(self.char_to_idx[char])
                elif '<UNK>' in self.char_to_idx:
                    data.append(self.char_to_idx['<UNK>'])
                else:
                    # Use first index (usually <PAD>) if <UNK> doesn't exist
                    data.append(0)
                i += 1
        
        # Check if data is empty
        if len(data) == 0:
            raise ValueError("Processed data is empty, cannot continue training")
        
        # Split data by poems, not randomly
        # Find all poem start and end positions
        poem_boundaries = []
        start_idx = None
        
        for i in range(len(data)):
            # Detect poem start marker <BOS>
            if data[i] == self.char_to_idx.get('<BOS>') and start_idx is None:
                start_idx = i
            # Detect poem end marker <EOS>
            elif data[i] == self.char_to_idx.get('<EOS>') and start_idx is not None:
                poem_boundaries.append((start_idx, i + 1))  # Include end marker
                start_idx = None
        
        # If no complete poems found, use simple sequence splitting
        if not poem_boundaries:
            print("Warning: No complete poem boundaries found, using simple sequence splitting")
            # Split data into fixed-length sequences
            seq_length = self.seq_length
            poem_boundaries = [(i, min(i + seq_length, len(data))) for i in range(0, len(data) - seq_length // 2, seq_length // 2)]
        
        # Shuffle poem order but keep each poem intact
        random.seed(self.args.seed if hasattr(self.args, 'seed') else 42)
        random.shuffle(poem_boundaries)
        
        # Split dataset by ratio
        n_poems = len(poem_boundaries)
        n_train = max(1, int(n_poems * 0.8))  # Ensure at least 1 sample
        n_val = max(1, int(n_poems * 0.1))    # Ensure at least 1 sample
        
        train_boundaries = poem_boundaries[:n_train]
        val_boundaries = poem_boundaries[n_train:n_train + n_val]
        test_boundaries = poem_boundaries[n_train + n_val:] if n_train + n_val < n_poems else [poem_boundaries[-1]]
        
        # Build datasets
        self.train_data = []
        for start, end in train_boundaries:
            self.train_data.extend(data[start:end])
        
        self.val_data = []
        for start, end in val_boundaries:
            self.val_data.extend(data[start:end])
        
        self.test_data = []
        for start, end in test_boundaries:
            self.test_data.extend(data[start:end])
        
        # Ensure all datasets are non-empty
        if not self.train_data:
            self.train_data = data[:int(len(data) * 0.8)]
        if not self.val_data:
            self.val_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
        if not self.test_data:
            self.test_data = data[int(len(data) * 0.9):]
        
        print(f"Data loading complete: {len(self.text)} characters, {self.vocab_size} unique characters")
        print(f"Training set: {len(self.train_data)} characters, {len(train_boundaries)} poems")
        print(f"Validation set: {len(self.val_data)} characters, {len(val_boundaries)} poems")
        print(f"Test set: {len(self.test_data)} characters, {len(test_boundaries)} poems")
    
    def _preprocess_text(self, text):
        """
        Preprocess text data with improved handling for poetry
        
        Args:
            text: Raw text data
            
        Returns:
            Preprocessed text data
        """
        # Split text by poems (empty lines as separators)
        poems = text.split('\n\n')
        poems = [p.strip() for p in poems if p.strip()]
        print(f"Found {len(poems)} poems in the dataset")
        
        # Filter non-Chinese characters
        def is_valid_char(c):
            # Keep only Chinese characters, Chinese punctuation, and special tokens
            if '\u4e00' <= c <= '\u9fff':  # Chinese character range
                return True
            if c in '，。；：？！、""''（）《》【】—…「」':  # Chinese punctuation
                return True
            if c in ['\n', ' ']:  # Keep newlines and spaces
                return True
            # Don't keep individual letters - they'll be part of special tokens
            return False
        
        # Add special tokens
        processed_poems = []
        skipped_poems = 0
        for poem in poems:
            # Filter non-Chinese characters
            filtered_poem = ''.join(c for c in poem if is_valid_char(c))
            
            # Add poem start and end markers
            lines = filtered_poem.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            
            # 跳过过短的诗句
            if len(lines) < 2:
                skipped_poems += 1
                continue
                
            # 过滤并规范化诗句
            # 唐诗通常为五言或七言，每首四句或八句
            standardized_lines = []
            line_lengths = []
            
            for line in lines:
                # 如果行太长，可能需要分割
                # 移除行中的标点符号计算实际字数
                line_chars = ''.join(c for c in line if '\u4e00' <= c <= '\u9fff')
                line_lengths.append(len(line_chars))
                standardized_lines.append(line)
            
            # 如果诗句长度不一致，跳过不规则的诗
            if len(set(line_lengths)) > 1 and max(line_lengths) - min(line_lengths) > 2:
                skipped_poems += 1
                continue
                
            # 规范化为四行或八行诗
            # 如果行数在3-5之间，填充或截断为四行
            # 如果行数在6-9之间，填充或截断为八行
            if 3 <= len(standardized_lines) <= 5:
                target_lines = 4
            elif 6 <= len(standardized_lines) <= 9:
                target_lines = 8
            else:
                # 行数太多或太少，使用原始行数
                target_lines = len(standardized_lines)
            
            # 填充或截断行数
            if len(standardized_lines) > target_lines:
                standardized_lines = standardized_lines[:target_lines]
            
            # 构建诗歌文本
            processed_poem = '<BOS>' + '<LINE>'.join(standardized_lines) + '<EOS>'
            processed_poems.append(processed_poem)
        
        print(f"Processed {len(processed_poems)} poems, skipped {skipped_poems} irregular poems")
        
        # 重新组合文本 - 使用换行符重新连接保持原有格式
        processed_text = '\n\n'.join(processed_poems)
        
        return processed_text
    
    def _build_vocab(self):
        """
        Build vocabulary from text data with poetry-specific tokens
        """
        # Define special tokens first
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<LINE>']
        
        # Get unique characters from text, excluding those that are part of special tokens
        chars = []
        for c in sorted(list(set(self.text))):
            # Only include Chinese characters, Chinese punctuation, and whitespace
            if '\u4e00' <= c <= '\u9fff' or c in '，。；：？！、""''（）《》【】—…「」' or c in ['\n', ' ']:
                chars.append(c)
        
        # Create mappings
        all_tokens = special_tokens + chars
        self.char_to_idx = {c: i for i, c in enumerate(all_tokens)}
        self.idx_to_char = {i: c for i, c in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)
        
        print(f"Built vocabulary with {self.vocab_size} tokens")
        
        # Print statistics about the vocabulary
        chinese_chars = sum(1 for c in chars if '\u4e00' <= c <= '\u9fff')
        punctuation = sum(1 for c in chars if c in '，。；：？！、""''（）《》【】—…「」')
        special = len(special_tokens)
        other = len(chars) - chinese_chars - punctuation
        
        print(f"Vocabulary statistics: {chinese_chars} Chinese characters, {punctuation} punctuation marks, {special} special tokens, {other} other characters")
        
        # Check for any remaining non-Chinese characters
        if other > 0:
            non_chinese = [c for c in chars if not ('\u4e00' <= c <= '\u9fff') and 
                          c not in '，。；：？！、""''（）《》【】—…「」' and 
                          c not in ['\n', ' ']]
            if non_chinese:
                print(f"Examples of non-Chinese characters: {non_chinese[:10]}")
                
                # Remove these characters if strict_chinese is enabled
                if hasattr(self.args, 'strict_chinese') and self.args.strict_chinese:
                    print("Strict Chinese mode enabled. Removing non-Chinese characters from vocabulary.")
                    # Rebuild vocabulary without non-Chinese characters
                    filtered_chars = [c for c in chars if c not in non_chinese]
                    all_tokens = special_tokens + filtered_chars
                    self.char_to_idx = {c: i for i, c in enumerate(all_tokens)}
                    self.idx_to_char = {i: c for i, c in enumerate(all_tokens)}
                    self.vocab_size = len(all_tokens)
                    print(f"Final vocabulary size after filtering: {self.vocab_size}")
    
    def create_dataloaders(self):
        """
        Create data loaders for training, validation, and test sets
        
        Returns:
            tuple: (train_loader, val_loader, test_loader, vocab_size, idx_to_char)
        """
        # Create datasets with different configurations
        train_dataset = TextDataset(
            self.train_data, 
            self.seq_length, 
            stride=self.stride,
            min_seq_length=self.min_seq_length,
            data_augmentation=self.data_augmentation
        )
        
        val_dataset = TextDataset(
            self.val_data, 
            self.seq_length, 
            stride=1,  # No stride for validation
            min_seq_length=self.seq_length,  # Fixed length for validation
            data_augmentation=False  # No augmentation for validation
        )
        
        test_dataset = TextDataset(
            self.test_data, 
            self.seq_length, 
            stride=1,  # No stride for test
            min_seq_length=self.seq_length,  # Fixed length for test
            data_augmentation=False  # No augmentation for test
        )
        
        # Use custom collate function if variable length is enabled
        collate_fn = custom_collate_fn if self.data_augmentation else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to more for faster loading if needed
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader, test_loader, self.vocab_size, self.idx_to_char 