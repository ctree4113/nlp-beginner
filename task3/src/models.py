import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embedding(nn.Module):
    """Word embedding layer"""
    
    def __init__(self, vocab_size, embedding_dim, padding_idx=0, embedding_matrix=None, freeze_embedding=False):
        """
        Initialize embedding layer
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            padding_idx: Padding index
            embedding_matrix: Pre-trained embedding matrix
            freeze_embedding: Whether to freeze embedding
        """
        super(Embedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # If pre-trained embedding matrix is provided, use it to initialize the embedding layer
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # If freeze embedding, disable gradient update
        if freeze_embedding:
            self.embedding.weight.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, seq_len)
            
        Returns:
            embedded: Embedding tensor, shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(x)


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        """
        Initialize bidirectional LSTM encoder
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(BiLSTMEncoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, input_dim)
            mask: Mask tensor, shape (batch_size, seq_len)
            
        Returns:
            outputs: LSTM output, shape (batch_size, seq_len, hidden_dim * 2)
        """
        # If mask is provided, compute sequence lengths
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            # Pack sequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            # Pass through LSTM
            packed_outputs, _ = self.lstm(packed_x)
            # Unpack sequence
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, _ = self.lstm(x)
        
        return outputs


class TreeLSTMCell(nn.Module):
    """Tree-LSTM Cell as described in the paper"""
    
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize Tree-LSTM cell
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
        """
        super(TreeLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i_l = nn.Linear(hidden_dim, hidden_dim)
        self.U_i_r = nn.Linear(hidden_dim, hidden_dim)
        
        # Forget gate (left child)
        self.W_fl = nn.Linear(input_dim, hidden_dim)
        self.U_fl_l = nn.Linear(hidden_dim, hidden_dim)
        self.U_fl_r = nn.Linear(hidden_dim, hidden_dim)
        
        # Forget gate (right child)
        self.W_fr = nn.Linear(input_dim, hidden_dim)
        self.U_fr_l = nn.Linear(hidden_dim, hidden_dim)
        self.U_fr_r = nn.Linear(hidden_dim, hidden_dim)
        
        # Output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o_l = nn.Linear(hidden_dim, hidden_dim)
        self.U_o_r = nn.Linear(hidden_dim, hidden_dim)
        
        # Cell update
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u_l = nn.Linear(hidden_dim, hidden_dim)
        self.U_u_r = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, h_l=None, c_l=None, h_r=None, c_r=None):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            h_l: Hidden state of left child, shape (batch_size, hidden_dim)
            c_l: Cell state of left child, shape (batch_size, hidden_dim)
            h_r: Hidden state of right child, shape (batch_size, hidden_dim)
            c_r: Cell state of right child, shape (batch_size, hidden_dim)
            
        Returns:
            h: Hidden state, shape (batch_size, hidden_dim)
            c: Cell state, shape (batch_size, hidden_dim)
        """
        # If no child states are provided, initialize with zeros
        batch_size = x.size(0)
        if h_l is None:
            h_l = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        if c_l is None:
            c_l = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        if h_r is None:
            h_r = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        if c_r is None:
            c_r = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Input gate
        i = torch.sigmoid(
            self.W_i(x) + self.U_i_l(h_l) + self.U_i_r(h_r)
        )
        
        # Forget gate (left child)
        f_l = torch.sigmoid(
            self.W_fl(x) + self.U_fl_l(h_l) + self.U_fl_r(h_r)
        )
        
        # Forget gate (right child)
        f_r = torch.sigmoid(
            self.W_fr(x) + self.U_fr_l(h_l) + self.U_fr_r(h_r)
        )
        
        # Output gate
        o = torch.sigmoid(
            self.W_o(x) + self.U_o_l(h_l) + self.U_o_r(h_r)
        )
        
        # Cell update
        u = torch.tanh(
            self.W_u(x) + self.U_u_l(h_l) + self.U_u_r(h_r)
        )
        
        # Cell state
        c = i * u + f_l * c_l + f_r * c_r
        
        # Hidden state
        h = o * torch.tanh(c)
        
        return h, c


class LinearTreeLSTM(nn.Module):
    """Linear Tree-LSTM that processes sequences linearly (for compatibility with ESIM)"""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        """
        Initialize Linear Tree-LSTM
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(LinearTreeLSTM, self).__init__()
        
        self.cell = TreeLSTMCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, input_dim)
            mask: Mask tensor, shape (batch_size, seq_len)
            
        Returns:
            outputs: Tree-LSTM output, shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.size()
        hidden_dim = self.cell.hidden_dim
        
        # Initialize outputs
        outputs = torch.zeros(batch_size, seq_len, hidden_dim, device=x.device)
        
        # Initialize states
        h = torch.zeros(batch_size, hidden_dim, device=x.device)
        c = torch.zeros(batch_size, hidden_dim, device=x.device)
        
        # Process sequence from left to right
        for t in range(seq_len):
            # Apply dropout to input
            x_t = self.dropout(x[:, t, :])
            
            # Update states
            h, c = self.cell(x_t, h, c, None, None)
            
            # Store output
            outputs[:, t, :] = h
            
            # Apply mask if provided
            if mask is not None:
                mask_t = mask[:, t].unsqueeze(1)
                h = h * mask_t
                c = c * mask_t
        
        return outputs


class AttentionLayer(nn.Module):
    """
    Attention layer for computing attention between premise and hypothesis.
    """
    def __init__(self):
        super(AttentionLayer, self).__init__()
    
    def forward(self, premise, hypothesis, premise_mask=None, hypothesis_mask=None):
        """
        Compute attention between premise and hypothesis.
        
        Args:
            premise: Premise tensor, shape (batch_size, premise_len, hidden_dim)
            hypothesis: Hypothesis tensor, shape (batch_size, hypothesis_len, hidden_dim)
            premise_mask: Premise mask, shape (batch_size, premise_len)
            hypothesis_mask: Hypothesis mask, shape (batch_size, hypothesis_len)
            
        Returns:
            tuple containing:
                - tuple of (attended_premise, attended_hypothesis): Attention-weighted representations
                - p2h_attention: Premise-to-hypothesis attention weights
        """
        # Calculate actual dimensions
        batch_size, premise_len, hidden_dim = premise.size()
        _, hypothesis_len, _ = hypothesis.size()
        
        # Compute attention scores, shape (batch_size, premise_len, hypothesis_len)
        attention_scores = torch.bmm(premise, hypothesis.transpose(1, 2))
        
        # Apply masks if provided
        if premise_mask is not None and hypothesis_mask is not None:
            # Convert masks to boolean type
            premise_mask = premise_mask.bool()
            hypothesis_mask = hypothesis_mask.bool()
            
            # Important: Truncate masks to match actual sequence lengths
            # Prevent dimension mismatch between masks and attention scores
            premise_mask = premise_mask[:, :premise_len]
            hypothesis_mask = hypothesis_mask[:, :hypothesis_len]
            
            # Create masks suitable for attention score shapes
            premise_mask_expanded = premise_mask.unsqueeze(2)  # (batch_size, premise_len, 1)
            hypothesis_mask_expanded = hypothesis_mask.unsqueeze(1)  # (batch_size, 1, hypothesis_len)
            
            # Ensure dimension matching, using broadcasting mechanism to generate attention mask
            # This will generate a mask of shape (batch_size, premise_len, hypothesis_len)
            attention_mask = premise_mask_expanded & hypothesis_mask_expanded
            
            # Apply mask, setting non-valid positions to a very small negative value
            attention_scores = attention_scores.masked_fill(~attention_mask, -1e9)
        
        # Compute softmax along different dimensions to get attention weights
        # Premise-to-hypothesis attention (calculate weights for each word in hypothesis)
        p2h_attention = F.softmax(attention_scores, dim=2)  # (batch_size, premise_len, hypothesis_len)
        
        # Hypothesis-to-premise attention (calculate weights for each word in premise)
        h2p_attention = F.softmax(attention_scores.transpose(1, 2), dim=2)  # (batch_size, hypothesis_len, premise_len)
        
        # Calculate weighted representations
        # Weighted hypothesis representation for each premise word
        attended_hypothesis = torch.bmm(p2h_attention, hypothesis)  # (batch_size, premise_len, hidden_dim)
        
        # Weighted premise representation for each hypothesis word
        attended_premise = torch.bmm(h2p_attention, premise)  # (batch_size, hypothesis_len, hidden_dim)
        
        return (attended_premise, attended_hypothesis), p2h_attention


class ESIM(nn.Module):
    """Enhanced Sequential Inference Model (ESIM)"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=3, dropout=0.5, 
                 padding_idx=0, embedding_matrix=None, freeze_embedding=False, use_tree_lstm=False):
        """
        Initialize ESIM model
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_classes: Number of classes
            dropout: Dropout rate
            padding_idx: Padding index
            embedding_matrix: Pre-trained embedding matrix
            freeze_embedding: Whether to freeze embedding
            use_tree_lstm: Whether to use Tree-LSTM instead of BiLSTM
        """
        super(ESIM, self).__init__()
        
        # Word embedding layer
        self.embedding = Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx,
            embedding_matrix=embedding_matrix,
            freeze_embedding=freeze_embedding
        )
        
        # Input encoder
        if use_tree_lstm:
            self.input_encoder = LinearTreeLSTM(embedding_dim, hidden_dim, dropout=dropout)
            # For Tree-LSTM, we need to adjust the composition input dimension
            # since the output of input_encoder is hidden_dim, not hidden_dim*2 as in BiLSTM
            self.composition = LinearTreeLSTM(hidden_dim * 4, hidden_dim, dropout=dropout)
            # For Tree-LSTM, the output dimension is hidden_dim (not hidden_dim*2 as in BiLSTM)
            self.output_dim = hidden_dim
        else:
            self.input_encoder = BiLSTMEncoder(embedding_dim, hidden_dim, dropout=dropout)
            self.composition = BiLSTMEncoder(hidden_dim * 8, hidden_dim, dropout=dropout)
            # For BiLSTM, the output dimension is hidden_dim*2
            self.output_dim = hidden_dim * 2
        
        # Attention layer
        self.attention = AttentionLayer()
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Store model configuration
        self.use_tree_lstm = use_tree_lstm
    
    def forward(self, premise_ids, hypothesis_ids, premise_mask=None, hypothesis_mask=None, return_attention=False):
        """
        Forward pass
        
        Args:
            premise_ids: Premise token ids, shape (batch_size, seq_len)
            hypothesis_ids: Hypothesis token ids, shape (batch_size, seq_len)
            premise_mask: Premise mask, shape (batch_size, seq_len)
            hypothesis_mask: Hypothesis mask, shape (batch_size, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Logits for each class, shape (batch_size, num_classes)
            attention_weights: Attention weights if return_attention is True
        """
        # Embedding
        premise_embedded = self.embedding(premise_ids)
        hypothesis_embedded = self.embedding(hypothesis_ids)
        
        # Encoding
        if self.use_tree_lstm:
            premise_encoded = self.input_encoder(premise_embedded, premise_mask)
            hypothesis_encoded = self.input_encoder(hypothesis_embedded, hypothesis_mask)
        else:
            premise_encoded = self.input_encoder(premise_embedded, premise_mask)
            hypothesis_encoded = self.input_encoder(hypothesis_embedded, hypothesis_mask)
        
        # Get actual lengths from encoded representations
        batch_size, premise_len, hidden_dim = premise_encoded.size()
        _, hypothesis_len, _ = hypothesis_encoded.size()
        
        # Attention
        attention_output, attention_weights = self.attention(
            premise_encoded, 
            hypothesis_encoded, 
            premise_mask, 
            hypothesis_mask
        )
        premise_attended, hypothesis_attended = attention_output
        
        # Ensure dimensions match (attended representations might have different lengths)
        # The issue is usually that the attended tensors might be shorter due to mask truncation
        if premise_encoded.size(1) != premise_attended.size(1):
            # Truncate the longer one to match
            min_premise_len = min(premise_encoded.size(1), premise_attended.size(1))
            premise_encoded = premise_encoded[:, :min_premise_len, :]
            premise_attended = premise_attended[:, :min_premise_len, :]
            if premise_mask is not None:
                premise_mask = premise_mask[:, :min_premise_len]
        
        if hypothesis_encoded.size(1) != hypothesis_attended.size(1):
            # Truncate the longer one to match
            min_hypothesis_len = min(hypothesis_encoded.size(1), hypothesis_attended.size(1))
            hypothesis_encoded = hypothesis_encoded[:, :min_hypothesis_len, :]
            hypothesis_attended = hypothesis_attended[:, :min_hypothesis_len, :]
            if hypothesis_mask is not None:
                hypothesis_mask = hypothesis_mask[:, :min_hypothesis_len]
        
        # Create enhanced representation
        premise_enhanced = torch.cat([
            premise_encoded, 
            premise_attended, 
            premise_encoded - premise_attended, 
            premise_encoded * premise_attended
        ], dim=-1)
        
        hypothesis_enhanced = torch.cat([
            hypothesis_encoded, 
            hypothesis_attended, 
            hypothesis_encoded - hypothesis_attended, 
            hypothesis_encoded * hypothesis_attended
        ], dim=-1)
        
        # Apply dropout
        premise_enhanced = self.dropout(premise_enhanced)
        hypothesis_enhanced = self.dropout(hypothesis_enhanced)
        
        # Composition
        if self.use_tree_lstm:
            premise_composed = self.composition(premise_enhanced, premise_mask)
            hypothesis_composed = self.composition(hypothesis_enhanced, hypothesis_mask)
        else:
            premise_composed = self.composition(premise_enhanced, premise_mask)
            hypothesis_composed = self.composition(hypothesis_enhanced, hypothesis_mask)
        
        # Pooling
        # Max pooling
        premise_max_pooled = self._max_pooling(premise_composed, premise_mask)
        hypothesis_max_pooled = self._max_pooling(hypothesis_composed, hypothesis_mask)
        
        # Mean pooling
        premise_mean_pooled = self._mean_pooling(premise_composed, premise_mask)
        hypothesis_mean_pooled = self._mean_pooling(hypothesis_composed, hypothesis_mask)
        
        # Concatenate
        pooled = torch.cat([
            premise_max_pooled, premise_mean_pooled,
            hypothesis_max_pooled, hypothesis_mean_pooled
        ], dim=-1)
        
        # Classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits

    def _max_pooling(self, x, mask=None):
        """
        Max pooling operation
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, hidden_dim)
            mask: Mask tensor, shape (batch_size, seq_len)
            
        Returns:
            max_pooled: Max pooled tensor, shape (batch_size, hidden_dim)
        """
        if mask is not None:
            # Ensure mask has the correct length (truncate or pad)
            seq_len = x.size(1)
            if mask.size(1) != seq_len:
                # Truncate or pad mask to match x's sequence length
                if mask.size(1) > seq_len:
                    mask = mask[:, :seq_len]
                else:
                    # This shouldn't happen, but just in case
                    pad_size = seq_len - mask.size(1)
                    mask = F.pad(mask, (0, pad_size), value=0)
                    
            # Convert mask to float and expand to match x's dimensions
            mask = mask.float().unsqueeze(-1).expand_as(x)
            # Set masked positions to large negative value
            x = x * mask + (1 - mask) * -1e9
        
        # Apply max pooling
        max_pooled, _ = torch.max(x, dim=1)
        return max_pooled
    
    def _mean_pooling(self, x, mask=None):
        """
        Mean pooling operation
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, hidden_dim)
            mask: Mask tensor, shape (batch_size, seq_len)
            
        Returns:
            mean_pooled: Mean pooled tensor, shape (batch_size, hidden_dim)
        """
        if mask is not None:
            # Ensure mask has the correct length (truncate or pad)
            seq_len = x.size(1)
            if mask.size(1) != seq_len:
                # Truncate or pad mask to match x's sequence length
                if mask.size(1) > seq_len:
                    mask = mask[:, :seq_len]
                else:
                    # This shouldn't happen, but just in case
                    pad_size = seq_len - mask.size(1)
                    mask = F.pad(mask, (0, pad_size), value=0)
            
            # Convert mask to float and expand to match x's dimensions
            mask = mask.float().unsqueeze(-1).expand_as(x)
            # Apply mask (zeros will not contribute to mean)
            x = x * mask
            # Compute sum and divide by the sum of mask (along seq_len dimension)
            sum_mask = torch.sum(mask, dim=1)
            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_pooled = torch.sum(x, dim=1) / sum_mask
        else:
            # Simple mean if no mask provided
            mean_pooled = torch.mean(x, dim=1)
        
        return mean_pooled 