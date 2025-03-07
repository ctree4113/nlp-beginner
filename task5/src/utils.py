import os
import random
import argparse
import time
import torch
import numpy as np
import math


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
    parser = argparse.ArgumentParser(description='Character-level Language Model with LSTM/GRU')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='../dataset/poetryFromTang.txt', help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='../output', help='Output directory')
    parser.add_argument('--seq_length', type=int, default=48, help='Sequence length')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'GRU'], help='Model type')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Generation parameters
    parser.add_argument('--generate_length', type=int, default=100, help='Length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    
    # Experiment parameters
    parser.add_argument('--experiment', action='store_true', help='Run comparison experiment between LSTM and GRU')
    
    # Other parameters
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    
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


def calculate_perplexity(loss_value):
    """
    Calculate perplexity from loss
    
    Args:
        loss_value: Loss value
        
    Returns:
        float: Perplexity
    """
    # Convert input to Python float
    if isinstance(loss_value, torch.Tensor):
        loss_value = loss_value.item()
    
    # Check for invalid values
    if math.isnan(loss_value) or math.isinf(loss_value) or loss_value < 0:
        print(f"Warning: Invalid loss value ({loss_value}), returning maximum perplexity value")
        return 10000.0  # Return a large but finite value
    
    # Prevent numerical overflow, set a reasonable upper limit
    if loss_value > 20:  # ln(10000) ≈ 9.21, set a lower threshold to preserve some distinctiveness
        print(f"Warning: Loss value too large ({loss_value:.4f}), limiting perplexity calculation")
        # Use a progressive scaling method to preserve some gradient information
        return math.exp(20) * (loss_value - 20 + 1)
    
    # Normal perplexity calculation
    return math.exp(loss_value)


def epoch_time(start_time, end_time):
    """
    Calculate time taken for one epoch
    
    Args:
        start_time: Start time
        end_time: End time
        
    Returns:
        tuple: Minutes and seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs 