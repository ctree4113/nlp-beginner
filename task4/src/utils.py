import os
import random
import argparse
import torch
import numpy as np


def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    """
    Get command line arguments
    
    Returns:
        args: Arguments
    """
    parser = argparse.ArgumentParser(description='Named Entity Recognition with LSTM+CRF')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../dataset', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='../output', help='Output directory')
    parser.add_argument('--language', type=str, default='eng', choices=['eng', 'deu'], help='Language selection (eng or deu)')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--max_word_len', type=int, default=20, help='Maximum word length for character-level features')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=100, help='Word embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--use_char_cnn', action='store_true', help='Use character-level CNN')
    parser.add_argument('--char_embedding_dim', type=int, default=30, help='Character embedding dimension')
    parser.add_argument('--char_channel_size', type=int, default=50, help='Character CNN output channel size')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=3, help='Patience for learning rate scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for learning rate scheduler')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--early_stopping', type=int, default=5, help='Early stopping patience (0 to disable)')
    
    # Other parameters
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    # Comparison mode parameters
    parser.add_argument('--model_type', type=str, default='bilstm_crf_char', 
                      choices=['bilstm', 'bilstm_crf', 'bilstm_crf_char'],
                      help='Model type for comparison')
    parser.add_argument('--comparison_mode', action='store_true', help='Run in comparison mode')
    parser.add_argument('--comparison_visualization', action='store_true', help='Generate comparison visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    return args


def print_args(args):
    """
    Print arguments

    Args:
        args: Arguments
    """
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")


def count_parameters(model):
    """
    Count the number of parameters in the model
    
    Args:
        model: Model
        
    Returns:
        params_count: Number of parameters
    """
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params_count:,}")
    return params_count


def epoch_time(start_time, end_time):
    """
    Calculate epoch time
    
    Args:
        start_time: Start time
        end_time: End time
        
    Returns:
        elapsed_mins: Minutes
        elapsed_secs: Seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs 