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
            model_type: Model type (LSTM or GRU)
            tie_weights: Tie embedding and output weights
            use_layer_norm: Use layer normalization
            use_residual: Use residual connections
            bidirectional: Use bidirectional RNN
            use_attention: Use attention mechanism
        """
        super(RNNModel, self).__init__()
        
        # Save parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.model_type = model_type
        self.tie_weights = tie_weights
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Calculate number of directions
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 位置编码 - 默认关闭，需要的话可以在后续更新中启用
        self.position_encoding = False
        self.max_seq_len = 100  # 最大序列长度
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # RNN layer (LSTM or GRU)
        rnn_class = nn.LSTM if model_type == 'LSTM' else nn.GRU
        self.rnn = rnn_class(
            embedding_dim, 
            hidden_dim // self.num_directions,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Attention layer
        if use_attention:
            self.attention = nn.Linear(hidden_dim, hidden_dim)
            self.attention_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 添加门控重置机制
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Tie weights
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
        应用增强的注意力机制
        
        Args:
            rnn_output: RNN输出，形状 [batch_size, seq_length, hidden_dim]
            hidden: 隐藏状态，形状 [num_layers * num_directions, batch_size, hidden_dim // num_directions]
            
        Returns:
            context: 上下文向量，形状 [batch_size, seq_length, hidden_dim]
        """
        # 获取最后的隐藏状态
        if self.model_type == 'LSTM':
            # 对于LSTM，hidden是元组 (h_n, c_n)
            last_hidden = hidden[0]
        else:
            # 对于GRU，hidden只是h_n
            last_hidden = hidden
        
        # 获取最后一层的隐藏状态
        last_hidden = last_hidden[-self.num_directions:].transpose(0, 1)  # [batch_size, num_directions, hidden_dim // num_directions]
        last_hidden = last_hidden.reshape(last_hidden.size(0), 1, -1)  # [batch_size, 1, hidden_dim]
        
        # 计算注意力权重
        attn_weights = torch.bmm(self.attention(rnn_output), last_hidden.transpose(1, 2))  # [batch_size, seq_length, 1]
        
        # 应用缩放 - 如果hidden_dim很大，可能需要缩放
        scale_factor = math.sqrt(self.hidden_dim) if self.hidden_dim > 512 else 1.0
        if scale_factor != 1.0:
            attn_weights = attn_weights / scale_factor
            
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 应用注意力权重获取上下文向量
        context = torch.bmm(attn_weights.transpose(1, 2), rnn_output)  # [batch_size, 1, hidden_dim]
        
        # 扩展上下文向量以匹配rnn_output形状
        context = context.expand(-1, rnn_output.size(1), -1)  # [batch_size, seq_length, hidden_dim]
        
        # 合并上下文和rnn_output
        combined = torch.cat((rnn_output, context), dim=2)  # [batch_size, seq_length, hidden_dim * 2]
        combined = self.attention_combine(combined)  # [batch_size, seq_length, hidden_dim]
        
        # 添加非线性激活
        combined = F.relu(combined)
        
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
        batch_size, seq_len = x.size()
        
        # 嵌入层
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # 添加位置编码 - 如果启用
        if self.position_encoding and hasattr(self, 'position_embeddings') and seq_len <= self.max_seq_len:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.position_embeddings(position_ids)
            embedded = embedded + position_embeddings
        
        # 应用dropout
        embedded = self.dropout(embedded)
        
        # 初始化隐藏状态（如果未提供）
        if hidden is None:
            device = x.device
            hidden = self.init_hidden(batch_size, device)
        
        # RNN层
        rnn_output, hidden = self.rnn(embedded, hidden)  # [batch_size, seq_length, hidden_dim]
        
        # 应用注意力机制（如果启用）
        if self.use_attention:
            attended_output = self.attention_mechanism(rnn_output, hidden)
            
            # 使用门控机制组合原始输出和注意力输出
            if hasattr(self, 'gate'):
                gate = torch.sigmoid(self.gate(rnn_output))
                rnn_output = gate * attended_output + (1 - gate) * rnn_output
            else:
                rnn_output = attended_output
        
        # 应用层归一化（如果启用）
        if self.use_layer_norm:
            rnn_output = self.layer_norm(rnn_output)
        
        # 应用残差连接（如果启用）
        if self.use_residual and embedded.size(-1) == rnn_output.size(-1):
            # 确保维度匹配
            rnn_output = rnn_output + embedded
        
        # 应用dropout
        output = self.dropout(rnn_output)
        
        # 应用全连接层
        output = self.fc(output)
        
        return output, hidden
    
    def generate(self, initial_seq, max_length, device, temperature=1.0, idx_to_char=None, top_k=0, top_p=0.0, char_to_idx=None, poem_type='auto'):
        """
        Generate sequence from initial sequence
        
        Args:
            initial_seq: Initial sequence as tensor
            max_length: Maximum length of generated sequence
            device: Device to use
            temperature: Temperature for sampling
            idx_to_char: Mapping from indices to characters
            top_k: Use top-k sampling
            top_p: Use top-p sampling
            char_to_idx: Mapping from characters to indices
            poem_type: Poem type to generate ('5', '7', or 'auto')
            
        Returns:
            Generated text
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            current_input = initial_seq.to(device)
            hidden = None
            generated_indices = initial_seq.tolist()[0].copy()
            
            # Reduce overfitting caused by gradient accumulation
            if hasattr(self, 'dropout'):
                original_dropout = self.dropout.p
                # Use slightly higher dropout during generation to reduce overfitting
                increased_dropout = min(original_dropout + 0.1, 0.8)
                for module in self.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = increased_dropout
            
            # Loop until we reach max_length or generate the EOS token
            for _ in range(max_length):
                # Get model output
                output, hidden = self(current_input, hidden)
                
                # Get the last output of the sequence
                last_output = output[:, -1, :]
                
                # Apply temperature scaling
                if temperature != 1.0:
                    last_output = last_output / temperature
                
                # Apply sampling strategy
                if top_k > 0:
                    # Top-k sampling
                    indices_to_remove = last_output < torch.topk(last_output, top_k)[0][..., -1, None]
                    last_output[indices_to_remove] = -float('inf')
                
                if top_p > 0.0:
                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(last_output, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = torch.zeros_like(last_output, dtype=torch.bool).scatter_(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    last_output[indices_to_remove] = -float('inf')
                
                # Sample from the distribution
                probs = F.softmax(last_output, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to generated indices
                generated_indices.append(next_token.item())
                
                # Check if we've generated an EOS token
                if char_to_idx is not None and next_token.item() == char_to_idx.get('<EOS>', 0):
                    break
                
                # Update input for next timestep
                # Ensure dimension matching: current_input is [batch_size, seq_length], next_token needs to be [batch_size, 1]
                next_token_reshaped = next_token.view(1, 1)  # Explicitly specify shape as [1, 1]
                current_input = torch.cat((current_input, next_token_reshaped), dim=1)
                
                # Truncate if input gets too long
                if current_input.size(1) > 128:
                    current_input = current_input[:, -128:]
            
            # Restore original dropout
            if hasattr(self, 'dropout'):
                for module in self.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = original_dropout
            
            # Convert to text
            if idx_to_char is not None:
                generated_text = ''
                for idx in generated_indices[len(initial_seq.tolist()[0]):]:
                    char = idx_to_char.get(idx, '')
                    if char == '<LINE>':
                        generated_text += '\n'
                    elif char not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>']:
                        generated_text += char
                
                # Clean text: Remove extra newlines and punctuation at the end of lines
                import re
                generated_text = re.sub(r'\n{3,}', '\n\n', generated_text)
                generated_text = re.sub(r'([，。；：？！、])\n', r'\1\n', generated_text)
                
                # Format as Tang Poetry, keeping original structure but standardizing character count
                lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
                
                # Determine if the poem is five or seven characters
                line_lengths = []
                for line in lines:
                    # Only count Chinese characters, not punctuation
                    chinese_chars = [c for c in line if '\u4e00' <= c <= '\u9fff']
                    if chinese_chars:
                        line_lengths.append(len(chinese_chars))
                
                # According to poem_type parameter, determine target length
                if poem_type == '5':
                    target_length = 5  # Force five-character poem
                elif poem_type == '7':
                    target_length = 7  # Force seven-character poem
                else:  # 'auto' mode
                    # Automatic detection, default to five-character
                    target_length = 5
                    if line_lengths:
                        # Determine target line length (five or seven characters)
                        if sum(1 for l in line_lengths if l > 5) > len(line_lengths) / 2:
                            target_length = 7  # If most lines exceed 5 characters, set to seven
                
                # Prepare characters and punctuation
                # Extract all Chinese characters, to supplement insufficient lines
                all_chars = [c for c in generated_text if '\u4e00' <= c <= '\u9fff']
                if not all_chars:
                    all_chars = ['山', '水', '云', '月', '风', '花', '雪', '日', '天', '地']
                
                # Build standard format Tang Poetry
                formatted_lines = []
                used_char_index = 0
                
                # First ensure at least 2 lines
                min_lines = max(2, min(len(lines), 10))  # Minimum 2 lines, maximum 10 lines
                
                # Process existing lines
                for i in range(min_lines):
                    if i < len(lines) and lines[i].strip():
                        # Use existing line, extract pure Chinese characters
                        chinese_chars = [c for c in lines[i] if '\u4e00' <= c <= '\u9fff']
                        
                        # Adjust character count to target length
                        if len(chinese_chars) > target_length:
                            # If Chinese characters exceed target length, truncate
                            chinese_chars = chinese_chars[:target_length]
                        elif len(chinese_chars) < target_length:
                            # If Chinese characters are insufficient, supplement from all Chinese characters
                            while len(chinese_chars) < target_length:
                                if used_char_index < len(all_chars):
                                    chinese_chars.append(all_chars[used_char_index])
                                    used_char_index = (used_char_index + 1) % len(all_chars)
                                else:
                                    # Loop through vocabulary
                                    used_char_index = 0
                                    if all_chars:
                                        chinese_chars.append(all_chars[0])
                    else:
                        # If there are not enough lines, create new line
                        chinese_chars = []
                        for _ in range(target_length):
                            if used_char_index < len(all_chars):
                                chinese_chars.append(all_chars[used_char_index])
                                used_char_index = (used_char_index + 1) % len(all_chars)
                            else:
                                # Loop through vocabulary
                                used_char_index = 0
                                if all_chars:
                                    chinese_chars.append(all_chars[0])
                    
                    # Add appropriate punctuation
                    if i == min_lines - 1:  # Last line
                        line = ''.join(chinese_chars) + '。'
                    else:
                        if (i + 1) % 2 == 0:  # Even line
                            line = ''.join(chinese_chars) + '。'
                        else:  # Odd line
                            line = ''.join(chinese_chars) + '，'
                    
                    formatted_lines.append(line)
                
                # Maintain original format style, grouping every two lines (according to sample format)
                final_output = []
                for i, line in enumerate(formatted_lines):
                    final_output.append(line)
                    # If the last line of a group (even line) and not the last line, add newline
                    if (i + 1) % 2 == 0 and i < len(formatted_lines) - 1:
                        final_output.append('')
                
                return '\n'.join(final_output)
            
            return generated_indices[len(initial_seq.tolist()[0]):]

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