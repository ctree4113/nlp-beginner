import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """
    Character-level CNN for word representations
    """
    def __init__(self, num_chars, char_embedding_dim, char_channel_size, kernel_size=3):
        """
        Initialize character-level CNN
        
        Args:
            num_chars: Number of characters
            char_embedding_dim: Character embedding dimension
            char_channel_size: CNN output channel size
            kernel_size: CNN kernel size
        """
        super(CharCNN, self).__init__()
        
        self.char_embedding = nn.Embedding(num_chars, char_embedding_dim, padding_idx=0)
        self.char_conv = nn.Conv1d(
            in_channels=char_embedding_dim,
            out_channels=char_channel_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Character indices, shape (batch_size, seq_len, max_word_len)
            
        Returns:
            char_features: Character-level features, shape (batch_size, seq_len, char_channel_size)
        """
        batch_size, seq_len, max_word_len = x.size()
        
        # Reshape for character embedding
        x = x.view(batch_size * seq_len, max_word_len)  # (batch_size * seq_len, max_word_len)
        
        # Character embedding
        char_embeds = self.char_embedding(x)  # (batch_size * seq_len, max_word_len, char_embedding_dim)
        
        # Transpose for CNN
        char_embeds = char_embeds.transpose(1, 2)  # (batch_size * seq_len, char_embedding_dim, max_word_len)
        
        # Apply CNN
        char_features = self.char_conv(char_embeds)  # (batch_size * seq_len, char_channel_size, max_word_len)
        
        # Max pooling
        char_features = F.max_pool1d(char_features, char_features.size(2))  # (batch_size * seq_len, char_channel_size, 1)
        
        # Reshape back
        char_features = char_features.squeeze(2)  # (batch_size * seq_len, char_channel_size)
        char_features = char_features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, char_channel_size)
        
        # Apply dropout
        char_features = self.dropout(char_features)
        
        return char_features


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.5, 
                 use_char_cnn=True, num_chars=None, char_embedding_dim=30, char_channel_size=50):
        """
        Initialize Bidirectional LSTM encoder
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_char_cnn: Whether to use character-level CNN
            num_chars: Number of characters (required if use_char_cnn=True)
            char_embedding_dim: Character embedding dimension
            char_channel_size: Character CNN output channel size
        """
        super(BiLSTMEncoder, self).__init__()
        
        self.use_char_cnn = use_char_cnn
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Character-level CNN
        if use_char_cnn:
            assert num_chars is not None, "num_chars must be provided when use_char_cnn=True"
            self.char_cnn = CharCNN(num_chars, char_embedding_dim, char_channel_size)
            lstm_input_dim = embedding_dim + char_channel_size
        else:
            lstm_input_dim = embedding_dim
        
        # BiLSTM
        self.lstm = nn.LSTM(
            lstm_input_dim, 
            hidden_dim // 2,  # Half hidden dim for each direction
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, seq_lengths, char_x=None):
        """
        Forward pass
        
        Args:
            x: Input sequence, shape (batch_size, seq_len)
            seq_lengths: Actual sequence lengths
            char_x: Character indices, shape (batch_size, seq_len, max_word_len)
            
        Returns:
            lstm_out: LSTM output, shape (batch_size, seq_len, hidden_dim)
        """
        # Word embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Combine with character-level features if enabled
        if self.use_char_cnn and char_x is not None:
            char_features = self.char_cnn(char_x)  # (batch_size, seq_len, char_channel_size)
            embedded = torch.cat([embedded, char_features], dim=-1)  # (batch_size, seq_len, embedding_dim + char_channel_size)
        
        # Pack sequence to handle variable lengths
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process through LSTM
        packed_output, _ = self.lstm(packed_embedded)
        
        # Unpack sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        
        return lstm_out


class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling
    """
    def __init__(self, num_tags, optimize_constraints=True):
        """
        Initialize CRF layer
        
        Args:
            num_tags: Number of tags
            optimize_constraints: Whether to optimize constraints
        """
        super(CRF, self).__init__()
        
        # Transition matrix: transitions[i, j] = score of transitioning from j to i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Initialize transitions with constraints for BIO tagging scheme
        self._initialize_constraints(optimize_constraints)
    
    def _initialize_constraints(self, optimize_constraints):
        """
        Initialize transition matrix with constraints for BIO tagging scheme
        
        Constraints:
        1. No transitions from O to I-X
        2. No transitions from B-X/I-X to I-Y where X != Y
        3. No transitions from START to I-X (handled implicitly in decode method)
        4. Special handling for <UNK> tag
        """
        # Get number of tags
        num_tags = self.transitions.size(0)
        
        # Initialize with small random values
        nn.init.xavier_normal_(self.transitions)
        
        # Assume standard order of tags in BIO scheme:
        # 0: O (Outside)
        # 1: <UNK> (Unknown tag)
        # 2+: B-X, I-X, B-Y, I-Y, etc.
        
        # Find all B- and I- tags
        o_index = 0  # O tag index
        unk_index = 1  # <UNK> tag index
        b_indices = []
        i_indices = []
        tag_types = {}
        
        # Identify tag types based on BIO scheme
        # Starting from index 2 (after O and <UNK>)
        for i in range(2, num_tags):
            if (i - 2) % 2 == 0:  # Even offsets are B-X tags
                tag_type = (i - 2) // 2
                b_indices.append(i)
                tag_types[i] = tag_type
            else:  # Odd offsets are I-X tags
                tag_type = (i - 3) // 2  # Maps to corresponding B-X
                i_indices.append(i)
                tag_types[i] = tag_type
        
        # Apply constraints
        with torch.no_grad():
            # Constraint 1: No transitions from O to I-X
            for i in i_indices:
                self.transitions[i, o_index] = -10000.0
            
            # Special handling for <UNK> tag
            # Allow transitions to and from <UNK> with small penalty
            for i in range(num_tags):
                if i != unk_index:
                    # Slightly penalize transitions to <UNK>
                    self.transitions[unk_index, i] = -1.0
                    # Slightly penalize transitions from <UNK>
                    self.transitions[i, unk_index] = -1.0
            
            # Constraint 2: No transitions from B-X/I-X to I-Y where X != Y
            for i in b_indices + i_indices:  # For each B-X or I-X tag
                tag_type_i = tag_types[i]
                for j in i_indices:  # For each I-Y tag
                    tag_type_j = tag_types[j]
                    if tag_type_i != tag_type_j:  # X != Y
                        self.transitions[j, i] = -10000.0
            
            # Constraint 3: Add small bonus for B-X to I-X transitions
            for b_idx in b_indices:
                tag_type = tag_types[b_idx]
                for i_idx in i_indices:
                    if tag_types[i_idx] == tag_type:
                        self.transitions[i_idx, b_idx] += 2.0
            
            if optimize_constraints:
                # Optimize constraints
                # This is a placeholder and should be replaced with actual optimization logic
                pass
    
    def forward(self, emissions, tags, mask):
        """
        Calculate negative log likelihood loss
        
        Args:
            emissions: Model emission scores, shape (batch_size, seq_len, num_tags)
            tags: True tags, shape (batch_size, seq_len)
            mask: Mask tensor, shape (batch_size, seq_len)
            
        Returns:
            loss: Negative log likelihood loss, scalar
        """
        # Calculate scores
        gold_score = self._score_sentence(emissions, tags, mask)  # (batch_size,)
        forward_score = self._partition_function(emissions, mask)  # (batch_size,)
        
        # Calculate negative log likelihood, take batch mean
        loss = (forward_score - gold_score).mean()
        
        return loss
    
    def _score_sentence(self, emissions, tags, mask):
        """
        Calculate score of the true path
        
        Args:
            emissions: Model emission scores, shape (batch_size, seq_len, num_tags)
            tags: True tags, shape (batch_size, seq_len)
            mask: Mask tensor, shape (batch_size, seq_len)
            
        Returns:
            score: Score of the true path, shape (batch_size,)
        """
        batch_size, seq_len, _ = emissions.size()
        score = torch.zeros(batch_size, device=emissions.device)
        
        # Convert tags to long for indexing
        tags = tags.long()
        
        # First step: extract emission scores for first tags
        first_tags = tags[:, 0]  # (batch_size,)
        score = score + emissions[torch.arange(batch_size), 0, first_tags] * mask[:, 0]
        
        # Rest of the steps: calculate transition scores + emission scores
        for i in range(1, seq_len):
            # Current and previous tags
            curr_tags = tags[:, i]  # (batch_size,)
            prev_tags = tags[:, i-1]  # (batch_size,)
            
            # Apply mask to handle variable sequence lengths
            mask_i = mask[:, i]  # (batch_size,)
            
            # Add transition scores for each sample in batch
            # Use advanced indexing to extract transitions
            transition_scores = self.transitions[curr_tags, prev_tags]  # (batch_size,)
            
            # Add emission and transition scores where mask is true
            score = score + (transition_scores + emissions[torch.arange(batch_size), i, curr_tags]) * mask_i
        
        return score
    
    def _partition_function(self, emissions, mask):
        """
        Calculate the partition function (log sum exp of all possible paths)
        
        Args:
            emissions: Model emission scores, shape (batch_size, seq_len, num_tags)
            mask: Mask tensor, shape (batch_size, seq_len)
            
        Returns:
            log_sum_exp: Log sum exp of all possible paths, shape (batch_size,)
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # Initialize score with first emission
        score = emissions[:, 0]  # (batch_size, num_tags)
        
        # Apply mask for first timestep
        score = score * mask[:, 0].unsqueeze(-1)
        
        # Iterate through the rest of the sequence
        for i in range(1, seq_len):
            # Broadcast previous score and transitions matrix
            # prev_score: (batch_size, num_tags, 1)
            # broadcast_transitions: (1, num_tags, num_tags)
            prev_score = score.unsqueeze(-1)  # (batch_size, num_tags, 1)
            
            # Calculate next score using log-sum-exp
            # next_score[b, j] = log sum_i exp(prev_score[b, i] + transitions[j, i] + emissions[b, i, j])
            next_score = prev_score + self.transitions.unsqueeze(0)  # (batch_size, num_tags, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)  # (batch_size, num_tags)
            
            # Add current emissions and apply mask
            emit_score = emissions[:, i]  # (batch_size, num_tags)
            next_score = next_score + emit_score * mask[:, i].unsqueeze(-1)
            
            # Update score
            score = next_score
        
        # Final score: log sum exp over all final tags
        return torch.logsumexp(score, dim=1)  # (batch_size,)
    
    def decode(self, emissions, mask):
        """
        Decode the best tag sequence using Viterbi algorithm
        
        Args:
            emissions: Model emission scores, shape (batch_size, seq_len, num_tags)
            mask: Mask tensor, shape (batch_size, seq_len)
            
        Returns:
            best_tags: Decoded best tag sequences
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # Initialize score and backpointers
        score = emissions[:, 0]  # (batch_size, num_tags)
        history = []
        
        # Viterbi algorithm
        for i in range(1, seq_len):
            # Broadcast score and transitions
            broadcast_score = score.unsqueeze(2)  # (batch_size, num_tags, 1)
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # (batch_size, 1, num_tags)
            
            # Calculate next score and backpointers
            # next_score[b, j, i] = score[b, i] + transitions[j, i] + emissions[b, i, j]
            next_score = broadcast_score + self.transitions.unsqueeze(0)  # (batch_size, num_tags, num_tags)
            
            # Find best previous tags
            best_tags = next_score.argmax(dim=1)  # (batch_size, num_tags)
            history.append(best_tags)
            
            # Get best score for each tag
            next_score = next_score.max(dim=1)[0]  # (batch_size, num_tags)
            
            # Add current emissions and apply mask
            score = next_score + broadcast_emissions.squeeze(1) * mask[:, i].unsqueeze(-1)
        
        # Find best final tag
        best_last_tags = score.argmax(dim=1)  # (batch_size,)
        
        # Backtrace to get best tag sequences
        best_paths = []
        for b in range(batch_size):
            # Start with best final tag
            best_path = [best_last_tags[b].item()]
            
            # Backtrace through history
            for h in reversed(history):
                best_path.append(h[b, best_path[-1]].item())
            
            # Reverse path to get correct order
            best_path.reverse()
            
            best_paths.append(best_path)
        
        return best_paths


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF model for sequence labeling
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, num_layers=1, dropout=0.5,
                 use_char_cnn=True, num_chars=None, char_embedding_dim=30, char_channel_size=50,
                 use_crf=True, optimize_crf=True):
        """
        Initialize BiLSTM-CRF model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_tags: Number of tags
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_char_cnn: Whether to use character-level CNN
            num_chars: Number of characters (required if use_char_cnn=True)
            char_embedding_dim: Character embedding dimension
            char_channel_size: Character CNN output channel size
            use_crf: Whether to use CRF layer
            optimize_crf: Whether to optimize CRF transition matrix
        """
        super(BiLSTMCRF, self).__init__()
        
        self.bilstm = BiLSTMEncoder(
            vocab_size, embedding_dim, hidden_dim, num_layers, dropout,
            use_char_cnn, num_chars, char_embedding_dim, char_channel_size
        )
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.use_crf = use_crf
        
        if use_crf:
            self.crf = CRF(num_tags, optimize_constraints=optimize_crf)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, tags, mask=None, seq_lengths=None, char_x=None):
        """
        Forward pass
        
        Args:
            x: Input sequence, shape (batch_size, seq_len)
            tags: True tags, shape (batch_size, seq_len)
            mask: Mask tensor, shape (batch_size, seq_len)
            seq_lengths: Actual sequence lengths
            char_x: Character indices, shape (batch_size, seq_len, max_word_len)
            
        Returns:
            loss: CRF loss or cross-entropy loss
        """
        # BiLSTM encoding
        lstm_out = self.bilstm(x, seq_lengths, char_x)  # (batch_size, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        
        # Projection to tag space
        emissions = self.fc(lstm_out)  # (batch_size, seq_len, num_tags)
        
        if self.use_crf:
            # CRF loss
            loss = self.crf(emissions, tags, mask)
        else:
            # For BiLSTM (without CRF), use cross-entropy loss
            batch_size, seq_len, num_tags = emissions.size()
            emissions_flat = emissions.view(-1, num_tags)
            tags_flat = tags.view(-1)
            
            # Create mask to ignore padding tokens
            if mask is None:
                mask = torch.ones_like(tags, dtype=torch.bool)
            
            mask_flat = mask.view(-1)
            
            # Calculate cross-entropy loss only on non-padded tokens
            loss = F.cross_entropy(
                emissions_flat[mask_flat], 
                tags_flat[mask_flat],
                reduction='mean'
            )
        
        return loss
    
    def predict(self, x, mask=None, seq_lengths=None, char_x=None):
        """
        Predict tag sequence
        
        Args:
            x: Input sequence, shape (batch_size, seq_len)
            mask: Mask tensor, shape (batch_size, seq_len)
            seq_lengths: Actual sequence lengths
            char_x: Character indices, shape (batch_size, seq_len, max_word_len)
            
        Returns:
            predictions: Predicted tag sequences
        """
        # BiLSTM encoding
        lstm_out = self.bilstm(x, seq_lengths, char_x)  # (batch_size, seq_len, hidden_dim)
        
        # Projection to tag space
        emissions = self.fc(lstm_out)  # (batch_size, seq_len, num_tags)
        
        if self.use_crf:
            # Decode using CRF
            predictions = self.crf.decode(emissions, mask)
        else:
            # For BiLSTM (without CRF), use argmax decoding
            predictions = []
            
            # Get batch size and sequence length
            batch_size = emissions.size(0)
            
            # For each sequence in the batch
            for i in range(batch_size):
                if seq_lengths is not None:
                    length = seq_lengths[i]
                else:
                    length = emissions.size(1)
                
                # Apply argmax to each position
                pred = emissions[i, :length].argmax(dim=-1).cpu().numpy().tolist()
                predictions.append(pred)
        
        # Replace unknown tag (index 1) with O tag (index 0)
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                if predictions[i][j] == 1:  # <UNK> tag
                    predictions[i][j] = 0   # O tag
                    
        return predictions 