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
    def __init__(self, num_tags, start_idx=None, stop_idx=None):
        """
        Initialize CRF layer
        
        Args:
            num_tags: Number of tags
            start_idx: Index of START tag, if None, will be set to num_tags
            stop_idx: Index of STOP tag, if None, will be set to num_tags + 1
        """
        super(CRF, self).__init__()
        
        # Set START and STOP indices
        self.start_idx = num_tags if start_idx is None else start_idx
        self.stop_idx = num_tags + 1 if stop_idx is None else stop_idx
        
        # Adjust num_tags to include START and STOP if they're not already included
        self.num_tags = num_tags
        if start_idx is None or stop_idx is None:
            num_tags_with_special = num_tags + 2  # Add START and STOP
        else:
            num_tags_with_special = max(num_tags, self.stop_idx + 1)
        
        # Transition matrix: transitions[i, j] = score of transitioning from j to i
        self.transitions = nn.Parameter(torch.zeros(num_tags_with_special, num_tags_with_special))
        
        # Initialize transitions with constraints for BIO tagging scheme
        self._initialize_constraints()
    
    def _initialize_constraints(self):
        """
        Initialize transition matrix with constraints for BIO tagging scheme
        
        Constraints:
        1. No transitions from O to I-X
        2. No transitions from B-X/I-X to I-Y where X != Y
        3. No transitions from START to I-X
        4. No transitions to START or from STOP
        """
        # Reset transitions with small random values
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        
        num_tags = self.num_tags  # Actual number of tags (excluding START and STOP)
        start_idx = self.start_idx
        stop_idx = self.stop_idx
        
        # Penalty value for invalid transitions
        CONSTRAINT_VALUE = -100.0
        
        # Assume tag mapping follows this format:
        # 0: O
        # 1: <UNK>
        # 2: B-PER
        # 3: I-PER
        # 4: B-LOC
        # 5: I-LOC
        # etc...
        
        # Apply constraints
        with torch.no_grad():
            # Basic constraints: no transitions to START, no transitions from STOP
            self.transitions[start_idx, :] = CONSTRAINT_VALUE
            self.transitions[:, stop_idx] = CONSTRAINT_VALUE
            
            # If there are enough tags, apply BIO constraints
            if num_tags > 2:  # At least O and one entity type
                # Assume tag order: O(0), UNK(1), B-X1(2), I-X1(3), B-X2(4), I-X2(5), ...
                
                # 1. No transitions from O to any I-X
                for i in range(3, num_tags, 2):  # All I-X tags
                    self.transitions[i, 0] = CONSTRAINT_VALUE  # From O to I-X
                
                # 2. No transitions from B-X or I-X to different type I-Y
                for i in range(2, num_tags, 2):  # All B-X tags
                    b_idx = i
                    i_idx = i + 1 if i + 1 < num_tags else None
                    
                    if i_idx is not None:
                        # For each B-X, find corresponding I-X and all other I-Y
                        for j in range(3, num_tags, 2):  # All I-Y tags
                            if j != i_idx:  # If not corresponding I-X
                                self.transitions[j, b_idx] = CONSTRAINT_VALUE  # Forbid B-X to I-Y (X!=Y)
                                self.transitions[j, i_idx] = CONSTRAINT_VALUE  # Forbid I-X to I-Y (X!=Y)
                
                # 3. No transitions from START to any I-X
                for i in range(3, num_tags, 2):  # All I-X tags
                    self.transitions[i, start_idx] = CONSTRAINT_VALUE
                
                # 4. Encourage valid transitions
                for i in range(2, num_tags, 2):  # All B-X tags
                    b_idx = i
                    i_idx = i + 1 if i + 1 < num_tags else None
                    
                    if i_idx is not None:
                        # Encourage B-X to I-X transitions
                        self.transitions[i_idx, b_idx] += 2.0
                        
                        # Encourage I-X to I-X transitions (entity continuation)
                        self.transitions[i_idx, i_idx] += 1.0
                
                # 5. Encourage transitions from START to O or B-X
                self.transitions[0, start_idx] += 1.0  # START to O
                for i in range(2, num_tags, 2):  # All B-X tags
                    self.transitions[i, start_idx] += 1.0  # START to B-X
                
                # 6. Encourage transitions from any tag to STOP
                for i in range(num_tags):
                    self.transitions[stop_idx, i] += 0.5
    
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
        # Ensure mask is float type for calculations
        if mask.dtype == torch.bool:
            mask = mask.float()
            
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
            mask: Mask tensor, shape (batch_size, seq_len) as float
            
        Returns:
            score: Score of the true path, shape (batch_size,)
        """
        batch_size, seq_len, num_tags = emissions.size()
        score = torch.zeros(batch_size, device=emissions.device)
        
        # Define START and STOP indices
        start_idx = self.start_idx  # <START> tag index
        stop_idx = self.stop_idx    # <STOP> tag index
        
        # Convert tags to long for indexing
        tags = tags.long()
        
        # First step: extract emission scores for first tags and add transition from START
        # This explicitly models the transition from the START tag to the first tag in the sequence
        first_tags = tags[:, 0]  # (batch_size,)
        score = score + emissions[torch.arange(batch_size), 0, first_tags] * mask[:, 0]
        score = score + self.transitions[first_tags, start_idx] * mask[:, 0]
        
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
        
        # Add transition to STOP for the last valid tag in each sequence
        # This explicitly models the transition from the last tag to the STOP tag
        # Get the length of each sequence
        seq_lengths = mask.sum(dim=1).long()  # (batch_size,)
        
        # Get the last valid tag for each sequence
        last_tags = tags[torch.arange(batch_size), seq_lengths - 1]  # (batch_size,)
        
        # Add transition score to STOP
        score = score + self.transitions[stop_idx, last_tags]
        
        return score
    
    def _partition_function(self, emissions, mask):
        """
        Calculate the partition function (log sum exp of all possible paths)
        
        Args:
            emissions: Model emission scores, shape (batch_size, seq_len, num_tags)
            mask: Mask tensor, shape (batch_size, seq_len) as float
            
        Returns:
            log_sum_exp: Log sum exp of all possible paths, shape (batch_size,)
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # Define START and STOP indices
        start_idx = self.start_idx  # <START> tag index
        stop_idx = self.stop_idx    # <STOP> tag index
        
        # Initialize score with first emission and transition from START
        # This explicitly models the transition from the START tag to all possible first tags
        # score[b, i] = emissions[b, 0, i] + transitions[i, START]
        score = emissions[:, 0] + self.transitions[:num_tags, start_idx].unsqueeze(0)  # (batch_size, num_tags)
        
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
            next_score = prev_score + self.transitions[:num_tags, :num_tags].unsqueeze(0)  # (batch_size, num_tags, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)  # (batch_size, num_tags)
            
            # Add current emissions and apply mask
            emit_score = emissions[:, i]  # (batch_size, num_tags)
            next_score = next_score + emit_score * mask[:, i].unsqueeze(-1)
            
            # Update score
            score = next_score
        
        # Add transition to STOP for all tags
        # This explicitly models the transition from all possible last tags to the STOP tag
        # score[b] = log sum_i exp(score[b, i] + transitions[STOP, i])
        score = score + self.transitions[stop_idx, :num_tags].unsqueeze(0)  # (batch_size, num_tags)
        
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
        
        # Ensure mask is float type for calculations
        if mask.dtype == torch.bool:
            mask = mask.float()
        
        # Initialize score and history for Viterbi
        # score[b, t] = max score of paths ending at tag t at current step
        score = emissions[:, 0]  # (batch_size, num_tags)
        
        # Add transition from START tag to each tag
        # For each valid tag i, add transition score from START to i
        start_transitions = self.transitions[:num_tags, self.start_idx].unsqueeze(0)
        score += start_transitions  # (batch_size, num_tags)
        
        # Apply mask for first timestep
        score = score * mask[:, 0].unsqueeze(-1)
        
        # Store backpointers for each step and tag
        history = []
        
        # Iterate through sequence, calculating most likely tag at each step
        for i in range(1, seq_len):
            # Previous score
            broadcast_score = score.unsqueeze(2)  # (batch_size, num_tags, 1)
            
            # Current emissions
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # (batch_size, 1, num_tags)
            
            # Calculate all possible scores
            # We need to use only the relevant part of the transitions matrix
            # next_score[b, i, j] = score[b, i] + transitions[j, i]
            next_score = broadcast_score + self.transitions[:num_tags, :num_tags].unsqueeze(0)  # (batch_size, num_tags, num_tags)
            
            # Get the best previous tag for each current tag
            best_score, best_prev_tags = next_score.max(dim=1)  # (batch_size, num_tags)
            
            # Store backpointers
            history.append(best_prev_tags)
            
            # Update score with current emissions
            score = best_score + broadcast_emissions.squeeze(1)
            
            # Apply mask - convert bool mask to float for arithmetic operations
            mask_i = mask[:, i].unsqueeze(-1).float()  # (batch_size, 1) as float
            score = score * mask_i + (1 - mask_i) * broadcast_score.squeeze(2)
        
        # Add transition to STOP tag
        score += self.transitions[self.stop_idx, :num_tags].unsqueeze(0)  # (batch_size, num_tags)
        
        # Get best final tag
        best_final_score, best_final_tag = score.max(dim=1)  # (batch_size,)
        
        # Backtrace for the best path
        best_paths = []
        for b in range(batch_size):
            # Length of the sequence (without padding)
            seq_length = int(mask[b].sum().item())
            if seq_length == 0:
                # Empty sequence
                best_paths.append([])
                continue
                
            # Start with the best final tag
            best_tag = best_final_tag[b].item()
            best_path = [best_tag]
            
            # Backtrace through history
            for h_idx in range(len(history) - 1, -1, -1):
                # Skip if beyond actual sequence length
                step_idx = h_idx + 1
                if step_idx >= seq_length:
                    continue
                    
                best_tag = history[h_idx][b, best_tag].item()
                best_path.insert(0, best_tag)
            
            # Remove special tags
            best_path = [tag for tag in best_path if tag != self.start_idx and tag != self.stop_idx]
            best_paths.append(best_path)
        
        return best_paths


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF model for sequence labeling
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, num_layers=1, dropout=0.5,
                 use_char_cnn=True, num_chars=None, char_embedding_dim=30, char_channel_size=50,
                 use_crf=True, start_idx=None, stop_idx=None):
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
            start_idx: Index of START tag, if None, will be set to num_tags
            stop_idx: Index of STOP tag, if None, will be set to num_tags + 1
        """
        super(BiLSTMCRF, self).__init__()
        
        self.bilstm = BiLSTMEncoder(
            vocab_size, embedding_dim, hidden_dim, num_layers, dropout,
            use_char_cnn, num_chars, char_embedding_dim, char_channel_size
        )
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.use_crf = use_crf
        
        if use_crf:
            self.crf = CRF(num_tags, start_idx=start_idx, stop_idx=stop_idx)
        
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
        # Create mask if not provided
        if mask is None:
            if seq_lengths is not None:
                # Create mask from sequence lengths
                batch_size, seq_len = x.size()
                mask = torch.zeros(batch_size, seq_len, device=x.device)
                for i, length in enumerate(seq_lengths):
                    mask[i, :length] = 1
            else:
                # Assume all tokens are valid (no padding)
                mask = torch.ones_like(x, dtype=torch.float, device=x.device)
        elif mask.dtype == torch.bool:
            # Convert boolean mask to float
            mask = mask.float()
        
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
            mask_flat = mask.view(-1)
            
            # Calculate cross-entropy loss only on non-padded tokens
            loss = F.cross_entropy(
                emissions_flat[mask_flat.bool()], 
                tags_flat[mask_flat.bool()],
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
        # Create mask if not provided
        if mask is None:
            if seq_lengths is not None:
                # Create mask from sequence lengths
                batch_size, seq_len = x.size()
                mask = torch.zeros(batch_size, seq_len, device=x.device)
                for i, length in enumerate(seq_lengths):
                    mask[i, :length] = 1
            else:
                # Assume all tokens are valid (no padding)
                mask = torch.ones_like(x, dtype=torch.float, device=x.device)
        elif mask.dtype == torch.bool:
            # Convert boolean mask to float
            mask = mask.float()
        
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