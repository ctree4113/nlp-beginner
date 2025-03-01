import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import re


class RNNModel(nn.Module):
    """
    Enhanced RNN Language Model (LSTM/GRU) with attention mechanism and poetry-specific improvements
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5, model_type='LSTM', 
                 tie_weights=True, use_layer_norm=True, use_residual=True, bidirectional=False,
                 use_attention=True):
        """
        Initialize RNN language model
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of RNN layers
            dropout: Dropout rate
            model_type: Model type ('LSTM' or 'GRU')
            tie_weights: Whether to tie embedding and output weights
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            bidirectional: Whether to use bidirectional RNN
            use_attention: Whether to use attention mechanism
        """
        super(RNNModel, self).__init__()
        
        # Save parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.bidirectional = bidirectional
        self.tie_weights = tie_weights
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = use_attention
        self.dropout = dropout  # Store dropout rate for dynamic adjustment
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer (LSTM or GRU)
        rnn_class = nn.LSTM if model_type == 'LSTM' else nn.GRU
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim // self.num_directions,  # Divide hidden size if bidirectional
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Linear(hidden_dim, hidden_dim)
            self.attention_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Tie weights if enabled
        if tie_weights and embedding_dim == hidden_dim:
            self.fc.weight = self.embedding.weight
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """
        Initialize model weights using Xavier uniform initialization
        """
        # Xavier uniform initialization for embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        # Initialize attention weights if used
        if self.use_attention:
            nn.init.xavier_uniform_(self.attention.weight)
            nn.init.zeros_(self.attention.bias)
            nn.init.xavier_uniform_(self.attention_combine.weight)
            nn.init.zeros_(self.attention_combine.bias)
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state
        
        Args:
            batch_size: Batch size
            device: Device
            
        Returns:
            hidden: Initial hidden state
        """
        hidden_shape = (self.num_layers * self.num_directions, batch_size, self.hidden_dim // self.num_directions)
        
        if self.model_type == 'LSTM':
            # LSTM needs (h_0, c_0)
            return (torch.zeros(hidden_shape, device=device),
                    torch.zeros(hidden_shape, device=device))
        else:
            # GRU only needs h_0
            return torch.zeros(hidden_shape, device=device)
    
    def attention_mechanism(self, rnn_output, hidden):
        """
        Apply attention mechanism
        
        Args:
            rnn_output: RNN output, shape [batch_size, seq_length, hidden_dim]
            hidden: Hidden state, shape [num_layers * num_directions, batch_size, hidden_dim // num_directions]
            
        Returns:
            context: Context vector, shape [batch_size, seq_length, hidden_dim]
        """
        # Get the last hidden state
        if self.model_type == 'LSTM':
            # For LSTM, hidden is a tuple (h_n, c_n)
            last_hidden = hidden[0]
        else:
            # For GRU, hidden is just h_n
            last_hidden = hidden
        
        # Get the last layer's hidden state
        last_hidden = last_hidden[-self.num_directions:].transpose(0, 1)  # [batch_size, num_directions, hidden_dim // num_directions]
        last_hidden = last_hidden.reshape(last_hidden.size(0), 1, -1)  # [batch_size, 1, hidden_dim]
        
        # Calculate attention weights
        attn_weights = torch.bmm(self.attention(rnn_output), last_hidden.transpose(1, 2))  # [batch_size, seq_length, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Apply attention weights to get context vector
        context = torch.bmm(attn_weights.transpose(1, 2), rnn_output)  # [batch_size, 1, hidden_dim]
        context = context.squeeze(1)  # [batch_size, hidden_dim]
        
        # Expand context to match rnn_output shape
        context = context.unsqueeze(1).expand(-1, rnn_output.size(1), -1)  # [batch_size, seq_length, hidden_dim]
        
        # Combine context and rnn_output
        combined = torch.cat((rnn_output, context), dim=2)  # [batch_size, seq_length, hidden_dim * 2]
        combined = self.attention_combine(combined)  # [batch_size, seq_length, hidden_dim]
        
        return combined
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape [batch_size, seq_length]
            hidden: Hidden state
            
        Returns:
            output: Output tensor, shape [batch_size, seq_length, vocab_size]
            hidden: New hidden state
        """
        batch_size = x.size(0)
        
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # Initialize hidden state if not provided
        if hidden is None:
            device = x.device
            hidden = self.init_hidden(batch_size, device)
        
        # RNN layer
        rnn_output, hidden = self.rnn(embedded, hidden)  # [batch_size, seq_length, hidden_dim]
        
        # Apply attention if enabled
        if self.use_attention:
            rnn_output = self.attention_mechanism(rnn_output, hidden)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            rnn_output = self.layer_norm(rnn_output)
            
        # Apply residual connection if enabled
        if self.use_residual and self.embedding_dim == self.hidden_dim:
            rnn_output = rnn_output + embedded
            
        # Dropout layer
        rnn_output = self.dropout(rnn_output)
        
        # Fully connected layer
        logits = self.fc(rnn_output)  # [batch_size, seq_length, vocab_size]
        
        return logits, hidden
    
    def generate(self, initial_seq, max_length, device, temperature=1.0, idx_to_char=None, top_k=0, top_p=0.0, char_to_idx=None):
        """
        Generate text with improved sampling strategies for poetry
        
        Args:
            initial_seq: Initial sequence, shape [1, seq_length]
            max_length: Maximum generation length
            device: Device
            temperature: Sampling temperature
            idx_to_char: Index to character mapping
            top_k: Number of top k predictions to sample from (0 to disable)
            top_p: Nucleus sampling threshold (0.0 to disable)
            char_to_idx: Character to index mapping
            
        Returns:
            generated_seq: Generated text
        """
        self.eval()
        with torch.no_grad():
            # Initialize sequence and hidden state
            hidden = self.init_hidden(1, device)
            x = initial_seq.to(device)
            
            # Store generated sequence
            generated_indices = []
            
            # Get special token indices
            line_token_idx = char_to_idx['<LINE>'] if char_to_idx and '<LINE>' in char_to_idx else None
            eos_token_idx = char_to_idx['<EOS>'] if char_to_idx and '<EOS>' in char_to_idx else None
            pad_token_idx = char_to_idx['<PAD>'] if char_to_idx and '<PAD>' in char_to_idx else None
            unk_token_idx = char_to_idx['<UNK>'] if char_to_idx and '<UNK>' in char_to_idx else None
            bos_token_idx = char_to_idx['<BOS>'] if char_to_idx and '<BOS>' in char_to_idx else None
            
            # 创建特殊标记集合，用于过滤
            special_tokens = {pad_token_idx, unk_token_idx, bos_token_idx}
            if None in special_tokens:
                special_tokens.remove(None)
            
            # 创建标点符号集合，用于控制标点符号的生成频率
            punctuation_chars = {'，', '。', '；', '：', '？', '！', '、', '"', '"', ''', ''', '（', '）', '《', '》', '【', '】', '—', '…', '「', '」'}
            punctuation_indices = {char_to_idx[char] for char in punctuation_chars if char in char_to_idx}
            
            # Track line lengths for enforcing poetry structure
            current_line_length = 0
            target_line_length = 5  # 默认每行5个字，符合五言诗格式
            line_count = 0
            last_char_was_punct = False  # 跟踪上一个字符是否为标点
            
            # 记录已生成的字符，用于避免重复
            recent_chars = []
            
            # Generate text
            for i in range(max_length):
                # Forward pass
                output, hidden = self.forward(x, hidden)
                
                # Get output from the last time step
                logits = output[:, -1, :]
                
                # Apply temperature annealing - gradually reduce temperature as generation progresses
                # This helps maintain coherence in longer sequences
                adaptive_temp = temperature * (1.0 - 0.3 * min(1.0, i / (max_length * 0.7)))
                logits = logits / adaptive_temp
                
                # 应用诗歌特定约束
                if line_token_idx is not None:
                    # 如果达到目标行长度，增加换行符的概率
                    if current_line_length >= target_line_length:
                        logits[0, line_token_idx] += 8.0  # 大幅提高换行概率
                    
                    # 如果行刚开始，降低换行符的概率
                    if current_line_length < 2:
                        logits[0, line_token_idx] -= 15.0  # 大幅降低换行概率
                
                # 降低特殊标记的生成概率
                for token_idx in special_tokens:
                    if token_idx is not None:
                        logits[0, token_idx] -= 50.0  # 极大降低特殊标记的概率
                
                # 控制标点符号的生成频率
                if last_char_was_punct:
                    # 如果上一个字符是标点，降低再次生成标点的概率
                    for punct_idx in punctuation_indices:
                        logits[0, punct_idx] -= 15.0
                elif current_line_length == target_line_length - 1:
                    # 在行尾增加逗号或句号的概率
                    if char_to_idx.get('，') is not None:
                        logits[0, char_to_idx['，']] += 5.0
                    if char_to_idx.get('。') is not None:
                        logits[0, char_to_idx['。']] += 5.0
                
                # 避免重复字符
                if len(recent_chars) > 0:
                    for recent_idx in recent_chars[-3:]:  # 检查最近3个字符
                        logits[0, recent_idx] -= 8.0  # 降低重复的概率
                
                # Apply nucleus (top-p) sampling
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = float('-inf')
                
                # Apply top-k sampling
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))  # Safety check
                    values, _ = torch.topk(logits, top_k)
                    min_value = values[:, -1].unsqueeze(1).expand_as(logits)
                    logits[logits < min_value] = float('-inf')
                
                # Convert output to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample next character index
                next_char_idx = torch.multinomial(probs, 1).item()
                
                # 检查是否需要重新采样（避免特殊标记）
                retry_count = 0
                while next_char_idx in special_tokens and retry_count < 5:
                    next_char_idx = torch.multinomial(probs, 1).item()
                    retry_count += 1
                
                # 如果多次重试后仍然是特殊标记，选择概率最高的非特殊字符
                if next_char_idx in special_tokens:
                    # 获取所有非特殊字符的索引
                    valid_indices = [idx for idx in range(len(probs[0])) 
                                    if idx not in special_tokens]
                    if valid_indices:
                        # 从有效索引中选择概率最高的
                        valid_probs = probs[0, valid_indices]
                        next_char_idx = valid_indices[torch.argmax(valid_probs).item()]
                    else:
                        # 如果没有有效索引，选择一个常见汉字
                        common_chars = ['山', '水', '风', '云', '花', '月', '日', '天', '人', '心']
                        for char in common_chars:
                            if char in char_to_idx:
                                next_char_idx = char_to_idx[char]
                                break
                
                # 更新行跟踪
                if next_char_idx == line_token_idx:
                    # 检测到换行
                    line_count += 1
                    
                    # 重置当前行长度
                    current_line_length = 0
                    last_char_was_punct = False
                else:
                    # 常规字符，增加行长度
                    current_line_length += 1
                    
                    # 更新标点符号状态
                    last_char_was_punct = next_char_idx in punctuation_indices
                
                # 检查序列结束
                if next_char_idx == eos_token_idx:
                    break
                
                # 添加新字符到生成序列
                generated_indices.append(next_char_idx)
                
                # 更新最近生成的字符
                recent_chars.append(next_char_idx)
                if len(recent_chars) > 10:  # 只保留最近10个字符
                    recent_chars.pop(0)
                
                # 准备下一个时间步的输入
                x = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
                
                # 每4行后添加一个空行，形成诗歌段落
                if line_count > 0 and line_count % 4 == 0 and current_line_length == 0:
                    generated_indices.append(line_token_idx)
            
            # 转换为文本（如果提供了idx_to_char）
            if idx_to_char is not None:
                generated_text = ''
                for idx in generated_indices:
                    char = idx_to_char[idx]
                    if char == '<LINE>':
                        generated_text += '\n'
                    elif char not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                        generated_text += char
                
                # 后处理：清理连续的换行符和行尾的标点
                generated_text = re.sub(r'\n{3,}', '\n\n', generated_text)  # 将3个以上连续换行符替换为2个
                generated_text = re.sub(r'([，。；：？！、])\n', r'\1\n', generated_text)  # 确保标点符号后的换行符保留
                
                # 格式化为诗歌形式（每4行为一个段落）
                lines = generated_text.split('\n')
                formatted_lines = []
                for i, line in enumerate(lines):
                    if line.strip():  # 只处理非空行
                        formatted_lines.append(line)
                        if (i + 1) % 4 == 0 and i < len(lines) - 1:
                            formatted_lines.append('')  # 每4行后添加一个空行
                
                generated_text = '\n'.join(formatted_lines)
                
                return generated_text
            
            return generated_indices

    def increase_dropout(self, increase_by):
        """
        Increase the dropout rate by a specified amount and update all dropout layers
        
        Args:
            increase_by: Amount to increase the dropout rate
        """
        # Update stored dropout rate
        old_dropout_p = self.dropout.p
        new_dropout_p = min(old_dropout_p + increase_by, 0.9)  # Cap at 0.9
        
        # Update all dropout layers in the model
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = min(module.p + increase_by, 0.9)  # Cap at 0.9
        
        return new_dropout_p


class TransformerModel(nn.Module):
    """
    Transformer-based language model
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, nhead, dropout=0.5):
        """
        Initialize Transformer language model
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            nhead: Number of attention heads
            dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()
        
        # Save parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = nhead
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """
        Initialize model weights
        """
        init_range = 0.1
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -init_range, init_range)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape [batch_size, seq_length]
            hidden: Not used (for compatibility with RNN models)
            
        Returns:
            output: Output tensor, shape [batch_size, seq_length, vocab_size]
            None: For compatibility with RNN models
        """
        # Create source mask for transformer (to prevent attention to padding tokens)
        src_mask = None
        src_key_padding_mask = None
        
        # Embedding layer with positional encoding
        embedded = self.embedding(x) * math.sqrt(self.embedding_dim)
        embedded = self.pos_encoder(embedded)
        
        # Transformer encoder
        transformer_output = self.transformer_encoder(embedded, src_mask, src_key_padding_mask)
        
        # Dropout layer
        output = self.dropout(transformer_output)
        
        # Fully connected layer
        output = self.fc(output)
        
        return output, None
    
    def generate(self, initial_seq, max_length, device, temperature=1.0, idx_to_char=None, top_k=0, top_p=0.0, char_to_idx=None):
        """
        Generate text with improved sampling strategies for poetry
        
        Args:
            initial_seq: Initial sequence, shape [1, seq_length]
            max_length: Maximum generation length
            device: Device
            temperature: Sampling temperature
            idx_to_char: Index to character mapping
            top_k: Number of top k predictions to sample from (0 to disable)
            top_p: Nucleus sampling threshold (0.0 to disable)
            char_to_idx: Character to index mapping
            
        Returns:
            generated_seq: Generated text
        """
        self.eval()
        with torch.no_grad():
            # Store generated sequence and its indices
            generated_indices = initial_seq.tolist()[0]
            current_input = initial_seq.to(device)
            
            # Get special token indices
            line_token_idx = char_to_idx['<LINE>'] if char_to_idx and '<LINE>' in char_to_idx else None
            eos_token_idx = char_to_idx['<EOS>'] if char_to_idx and '<EOS>' in char_to_idx else None
            pad_token_idx = char_to_idx['<PAD>'] if char_to_idx and '<PAD>' in char_to_idx else None
            unk_token_idx = char_to_idx['<UNK>'] if char_to_idx and '<UNK>' in char_to_idx else None
            bos_token_idx = char_to_idx['<BOS>'] if char_to_idx and '<BOS>' in char_to_idx else None
            
            # 创建特殊标记集合，用于过滤
            special_tokens = {pad_token_idx, unk_token_idx, bos_token_idx, eos_token_idx}
            if None in special_tokens:
                special_tokens.remove(None)
            
            # 创建标点符号集合，用于控制标点符号的生成频率
            punctuation_chars = {'，', '。', '；', '：', '？', '！', '、', '"', '"', ''', ''', '（', '）', '《', '》', '【', '】', '—', '…', '「', '」'}
            punctuation_indices = {char_to_idx[char] for char in punctuation_chars if char in char_to_idx}
            
            # Track line lengths for enforcing poetry structure
            current_line_length = 0
            target_line_length = 5  # 默认每行5个字，符合五言诗格式
            line_count = 0
            last_char_was_punct = False  # 跟踪上一个字符是否为标点
            
            # 记录已生成的字符，用于避免重复
            recent_chars = []
            
            # Generate text
            for i in range(max_length):
                # Forward pass
                output, _ = self.forward(current_input)
                
                # Get output from the last time step
                logits = output[:, -1, :]
                
                # Apply temperature annealing - gradually reduce temperature as generation progresses
                adaptive_temp = temperature * (1.0 - 0.3 * min(1.0, i / (max_length * 0.7)))
                logits = logits / adaptive_temp
                
                # 应用诗歌特定约束
                if line_token_idx is not None:
                    # 如果达到目标行长度，增加换行符的概率
                    if current_line_length >= target_line_length:
                        logits[0, line_token_idx] += 8.0  # 大幅提高换行概率
                    
                    # 如果行刚开始，降低换行符的概率
                    if current_line_length < 2:
                        logits[0, line_token_idx] -= 15.0  # 大幅降低换行概率
                
                # 降低特殊标记的生成概率
                for token_idx in special_tokens:
                    if token_idx is not None:
                        logits[0, token_idx] -= 50.0  # 极大降低特殊标记的概率
                
                # 控制标点符号的生成频率
                if last_char_was_punct:
                    # 如果上一个字符是标点，降低再次生成标点的概率
                    for punct_idx in punctuation_indices:
                        logits[0, punct_idx] -= 15.0
                elif current_line_length == target_line_length - 1:
                    # 在行尾增加逗号或句号的概率
                    if char_to_idx.get('，') is not None:
                        logits[0, char_to_idx['，']] += 5.0
                    if char_to_idx.get('。') is not None:
                        logits[0, char_to_idx['。']] += 5.0
                
                # 避免重复字符
                if len(recent_chars) > 0:
                    for recent_idx in recent_chars[-3:]:  # 检查最近3个字符
                        logits[0, recent_idx] -= 8.0  # 降低重复的概率
                
                # Apply nucleus (top-p) sampling
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = float('-inf')
                
                # Apply top-k sampling
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))  # Safety check
                    values, _ = torch.topk(logits, top_k)
                    min_value = values[:, -1].unsqueeze(1).expand_as(logits)
                    logits[logits < min_value] = float('-inf')
                
                # Convert output to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample next character index
                next_char_idx = torch.multinomial(probs, 1).item()
                
                # 检查是否需要重新采样（避免特殊标记）
                retry_count = 0
                while next_char_idx in special_tokens and retry_count < 5:
                    next_char_idx = torch.multinomial(probs, 1).item()
                    retry_count += 1
                
                # 如果多次重试后仍然是特殊标记，选择概率最高的非特殊字符
                if next_char_idx in special_tokens:
                    # 获取所有非特殊字符的索引
                    valid_indices = [idx for idx in range(len(probs[0])) 
                                    if idx not in special_tokens]
                    if valid_indices:
                        # 从有效索引中选择概率最高的
                        valid_probs = probs[0, valid_indices]
                        next_char_idx = valid_indices[torch.argmax(valid_probs).item()]
                    else:
                        # 如果没有有效索引，选择一个常见汉字
                        common_chars = ['山', '水', '风', '云', '花', '月', '日', '天', '人', '心']
                        for char in common_chars:
                            if char in char_to_idx:
                                next_char_idx = char_to_idx[char]
                                break
                
                # 更新行跟踪
                if next_char_idx == line_token_idx:
                    # 检测到换行
                    line_count += 1
                    
                    # 重置当前行长度
                    current_line_length = 0
                    last_char_was_punct = False
                else:
                    # 常规字符，增加行长度
                    current_line_length += 1
                    
                    # 更新标点符号状态
                    last_char_was_punct = next_char_idx in punctuation_indices
                
                # 检查序列结束
                if next_char_idx == eos_token_idx:
                    break
                
                # 添加新字符到生成序列
                generated_indices.append(next_char_idx)
                
                # 更新最近生成的字符
                recent_chars.append(next_char_idx)
                if len(recent_chars) > 10:  # 只保留最近10个字符
                    recent_chars.pop(0)
                
                # Prepare input for next time step (append new token to sequence)
                new_token = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
                current_input = torch.cat([current_input, new_token], dim=1)
                
                # Limit context length if growing too large
                if current_input.size(1) > 128:  # Use a reasonable context size
                    current_input = current_input[:, -128:]
            
            # 转换为文本（如果提供了idx_to_char）
            if idx_to_char is not None:
                generated_text = ''
                for idx in generated_indices[len(initial_seq.tolist()[0]):]:  # 移除初始序列
                    char = idx_to_char[idx]
                    if char == '<LINE>':
                        generated_text += '\n'
                    elif char not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                        generated_text += char
                
                # 后处理：清理连续的换行符和行尾的标点
                generated_text = re.sub(r'\n{3,}', '\n\n', generated_text)  # 将3个以上连续换行符替换为2个
                generated_text = re.sub(r'([，。；：？！、])\n', r'\1\n', generated_text)  # 确保标点符号后的换行符保留
                
                # 格式化为诗歌形式（每4行为一个段落）
                lines = generated_text.split('\n')
                formatted_lines = []
                for i, line in enumerate(lines):
                    if line.strip():  # 只处理非空行
                        formatted_lines.append(line)
                        if (i + 1) % 4 == 0 and i < len(lines) - 1:
                            formatted_lines.append('')  # 每4行后添加一个空行
                
                generated_text = '\n'.join(formatted_lines)
                
                return generated_text
            
            return generated_indices[len(initial_seq.tolist()[0]):]


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a model parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            x: Output tensor with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x) 