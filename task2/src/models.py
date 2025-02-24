import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class TextCNN(nn.Module):
    """Text CNN model"""
    
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, 
                 dropout=0.5, padding_idx=0, embedding_matrix=None, freeze_embedding=False):
        """
        Initialize TextCNN model
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes
            num_classes: Number of classes
            dropout: Dropout rate
            padding_idx: Padding index
            embedding_matrix: Pre-trained embedding matrix, None means random initialization
            freeze_embedding: Whether to freeze embedding
        """
        super(TextCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # If pre-trained embedding is provided, initialize with it
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            
        # Whether to freeze embedding
        if freeze_embedding:
            self.embedding.weight.requires_grad = False
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, seq_len)
            
        Returns:
            Output tensor, shape (batch_size, num_classes)
        """
        # Embedding, shape (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # Change dimension, shape (batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Convolution and pooling
        # conv_outputs shape [(batch_size, num_filters, seq_len - filter_size + 1), ...]
        conv_outputs = [F.relu(conv(embedded)) for conv in self.convs]
        
        # Max pooling, shape [(batch_size, num_filters, 1), ...]
        pooled_outputs = [F.max_pool1d(conv_output, conv_output.size(2)) for conv_output in conv_outputs]
        
        # Concatenate, shape (batch_size, num_filters * len(filter_sizes))
        cat = torch.cat([pooled_output.squeeze(2) for pooled_output in pooled_outputs], dim=1)
        
        # Dropout
        dropped = self.dropout(cat)
        
        # Fully connected layer, shape (batch_size, num_classes)
        logits = self.fc(dropped)
        
        return logits


class TextRNN(nn.Module):
    """Text RNN model"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1, 
                 bidirectional=True, dropout=0.5, padding_idx=0, embedding_matrix=None, 
                 freeze_embedding=False, rnn_type='lstm'):
        """
        Initialize TextRNN model
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_classes: Number of classes
            num_layers: Number of RNN layers
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout rate
            padding_idx: Padding index
            embedding_matrix: Pre-trained embedding matrix, None means random initialization
            freeze_embedding: Whether to freeze embedding
            rnn_type: RNN type, 'lstm' or 'gru'
        """
        super(TextRNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # If pre-trained embedding is provided, initialize with it
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            
        # Whether to freeze embedding
        if freeze_embedding:
            self.embedding.weight.requires_grad = False
        
        # RNN layer
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                              bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, 
                             bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, seq_len)
            
        Returns:
            Output tensor, shape (batch_size, num_classes)
        """
        # Embedding, shape (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # RNN, output shape (batch_size, seq_len, hidden_dim * num_directions)
        if self.rnn_type == 'lstm':
            output, (hidden, _) = self.rnn(embedded)
        else:  # GRU
            output, hidden = self.rnn(embedded)
        
        # Get the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward last hidden states
            # hidden shape (num_layers * num_directions, batch_size, hidden_dim)
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Get the last layer's hidden state
            last_hidden = hidden[-1, :, :]
        
        # Dropout
        dropped = self.dropout(last_hidden)
        
        # Fully connected layer, shape (batch_size, num_classes)
        logits = self.fc(dropped)
        
        return logits


class TextRCNN(nn.Module):
    """Text RCNN model, combining features of RNN and CNN"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, 
                 dropout=0.5, padding_idx=0, embedding_matrix=None, freeze_embedding=False):
        """
        Initialize TextRCNN model
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_classes: Number of classes
            dropout: Dropout rate
            padding_idx: Padding index
            embedding_matrix: Pre-trained embedding matrix, None means random initialization
            freeze_embedding: Whether to freeze embedding
        """
        super(TextRCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # If pre-trained embedding is provided, initialize with it
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(embedding_matrix)
            
        # Whether to freeze embedding
        if freeze_embedding:
            self.embedding.weight.requires_grad = False
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Linear transformation
        self.linear = nn.Linear(embedding_dim + 2 * hidden_dim, hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, seq_len)
            
        Returns:
            Output tensor, shape (batch_size, num_classes)
        """
        # Embedding, shape (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM, output shape (batch_size, seq_len, hidden_dim * 2)
        output, _ = self.lstm(embedded)
        
        # Concatenate embedding and LSTM output, shape (batch_size, seq_len, embedding_dim + hidden_dim * 2)
        x = torch.cat([embedded, output], dim=2)
        
        # Linear transformation, shape (batch_size, seq_len, hidden_dim)
        x = self.linear(x)
        
        # Activation function
        x = torch.tanh(x)
        
        # Max pooling, shape (batch_size, hidden_dim)
        x = F.max_pool1d(x.permute(0, 2, 1), x.size(1)).squeeze(2)
        
        # Dropout
        x = self.dropout(x)
        
        # Fully connected layer, shape (batch_size, num_classes)
        logits = self.fc(x)
        
        return logits 