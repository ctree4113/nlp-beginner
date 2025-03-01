import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from data_processor import DataProcessor
from models import RNNModel, TransformerModel
from trainer import Trainer
from utils import set_seed, print_args, count_parameters, epoch_time


def train_model(args, model_type):
    """
    Train a single model
    
    Args:
        args: Command line arguments
        model_type: Model type (LSTM, GRU or Transformer)
        
    Returns:
        tuple: (test_loss, test_ppl, best_model_path)
    """
    # Set model type
    args.model_type = model_type
    
    # Print header
    print("\n" + "=" * 60)
    print(f"Training {model_type} Language Model")
    print("=" * 60)
    
    # Process data
    print("Loading and processing data...")
    data_processor = DataProcessor(args)
    train_loader, val_loader, test_loader, vocab_size, idx_to_char = data_processor.create_dataloaders()
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(train_loader) * args.batch_size}")
    print(f"Validation samples: {len(val_loader) * args.batch_size}")
    print(f"Test samples: {len(test_loader) * args.batch_size}")
    print("-" * 60)
    
    # Create model
    print(f"Creating {model_type} model...")
    if model_type == 'Transformer':
        model = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            nhead=args.nhead,
            dropout=args.dropout
        )
    else:  # LSTM or GRU
        model = RNNModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            model_type=model_type,
            tie_weights=args.tie_weights,
            use_layer_norm=args.use_layer_norm,
            use_residual=args.use_residual,
            bidirectional=args.bidirectional,
            use_attention=args.use_attention
        )
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    print("-" * 60)
    
    # Create trainer
    trainer = Trainer(model, vocab_size, idx_to_char, args)
    
    # Train model
    best_model_path = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.load_checkpoint(best_model_path)
    test_loss, test_ppl, test_accuracy = trainer.test(test_loader)
    
    # Generate text with different sampling strategies
    print("\nGenerating text with the best model:")
    
    # Generate with different temperatures
    for temp in [0.5, 0.8, 1.0, 1.2]:
        print(f"\n=== Temperature: {temp} ===")
        generated_text = trainer.generate_text(
            max_length=args.generate_length,
            temperature=temp,
            top_p=args.top_p
        )
        print(f"\nGenerated text (temp={temp}):\n{generated_text}")
        
        # Save generated text to file
        output_file = os.path.join(args.output_dir, f"{model_type}_generated_text_temp{temp}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"Generated text saved to {output_file}")
    
    # Generate with different sampling strategies
    sampling_strategies = [
        {"name": "greedy", "temp": 0.01, "top_k": 0, "top_p": 0.0},
        {"name": "temperature", "temp": 0.8, "top_k": 0, "top_p": 0.0},
        {"name": "top_k", "temp": 1.0, "top_k": 5, "top_p": 0.0},
        {"name": "nucleus", "temp": 1.0, "top_k": 0, "top_p": 0.9},
        {"name": "combined", "temp": 0.8, "top_k": 10, "top_p": 0.8}
    ]
    
    for strategy in sampling_strategies:
        print(f"\n=== Sampling Strategy: {strategy['name']} ===")
        generated_text = trainer.generate_text(
            max_length=args.generate_length,
            temperature=strategy["temp"],
            top_k=strategy["top_k"],
            top_p=strategy["top_p"]
        )
        print(f"\nGenerated text ({strategy['name']}):\n{generated_text}")
        
        # Save generated text to file
        output_file = os.path.join(args.output_dir, f"{model_type}_generated_text_{strategy['name']}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"Generated text saved to {output_file}")
    
    # Generate with seed text if provided
    if hasattr(args, 'seed_text') and args.seed_text:
        print(f"\n=== Using seed text: {args.seed_text} ===")
        generated_text = trainer.generate_text(
            seed_text=args.seed_text,
            max_length=args.generate_length,
            temperature=1.0,
            top_p=0.9
        )
        print(f"\nGenerated text with seed:\n{generated_text}")
        
        # Save generated text to file
        output_file = os.path.join(args.output_dir, f"{model_type}_generated_text_seeded.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
    
    return test_loss, test_ppl, best_model_path


def run_experiment_model_comparison(args):
    """
    Run experiment to compare LSTM, GRU and Transformer models
    
    Args:
        args: Command line arguments
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Model Type Comparison")
    print("=" * 60)
    
    # Store results
    results = {}
    
    # Define model types to compare
    model_types = ['LSTM', 'GRU']
    if args.include_transformer:
        model_types.append('Transformer')
    
    # Train and evaluate each model type
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        test_loss, test_ppl, _ = train_model(args, model_type)
        results[model_type] = {"loss": test_loss, "perplexity": test_ppl}
    
    # Print comparison results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS: Model Comparison")
    print("=" * 60)
    print(f"{'Model Type':<12} | {'Test Loss':<12} | {'Test Perplexity':<15}")
    print("-" * 60)
    for model_type, metrics in results.items():
        print(f"{model_type:<12} | {metrics['loss']:<12.4f} | {metrics['perplexity']:<15.2f}")
    
    # Determine best model
    best_model = min(results.items(), key=lambda x: x[1]["perplexity"])
    print("-" * 60)
    print(f"Best model: {best_model[0]} with perplexity {best_model[1]['perplexity']:.2f}")
    print("=" * 60)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "model_comparison_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("EXPERIMENT RESULTS: Model Comparison\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Model Type':<12} | {'Test Loss':<12} | {'Test Perplexity':<15}\n")
        f.write("-" * 60 + "\n")
        for model_type, metrics in results.items():
            f.write(f"{model_type:<12} | {metrics['loss']:<12.4f} | {metrics['perplexity']:<15.2f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Best model: {best_model[0]} with perplexity {best_model[1]['perplexity']:.2f}\n")
    
    print(f"Results saved to {results_file}")
    
    # Plot comparison chart
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    perplexities = [results[model]["perplexity"] for model in model_names]
    
    plt.bar(model_names, perplexities)
    plt.title('Model Comparison: Perplexity')
    plt.xlabel('Model Type')
    plt.ylabel('Perplexity (lower is better)')
    
    # Add value labels to bars
    for i, perplexity in enumerate(perplexities):
        plt.text(i, perplexity + 1, f'{perplexity:.2f}', ha='center')
    
    # Save chart
    chart_path = os.path.join(args.output_dir, "model_comparison_chart.png")
    plt.savefig(chart_path)
    print(f"Comparison chart saved to {chart_path}")


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        args: Command line arguments
    """
    parser = argparse.ArgumentParser(description='Character-level Language Model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='../dataset/poetryFromTang.txt',
                        help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='../output',
                        help='Output directory')
    parser.add_argument('--seq_length', type=int, default=64,
                        help='Sequence length')
    parser.add_argument('--min_seq_length', type=int, default=16,
                        help='Minimum sequence length for data augmentation')
    parser.add_argument('--stride', type=int, default=3,
                        help='Stride for sequence generation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_augmentation', action='store_true',
                        help='Enable data augmentation with variable sequence lengths')
    parser.add_argument('--rebuild_vocab', action='store_true',
                        help='Force rebuild vocabulary even if it exists')
    parser.add_argument('--strict_chinese', action='store_true',
                        help='Strictly filter non-Chinese characters from vocabulary')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU', 'Transformer'],
                        help='Model type')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--tie_weights', action='store_true',
                        help='Tie embedding and output weights')
    parser.add_argument('--use_layer_norm', action='store_true',
                        help='Use layer normalization')
    parser.add_argument('--use_residual', action='store_true',
                        help='Use residual connections')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional RNN')
    parser.add_argument('--use_attention', action='store_true',
                        help='Use attention mechanism')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads (Transformer only)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Print log info every N batches')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0.0 to disable)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use exponential moving average of model parameters')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    
    # Generation parameters
    parser.add_argument('--generate_length', type=int, default=200,
                        help='Maximum length for text generation')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k sampling parameter (0 to disable)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling parameter (0.0 to disable)')
    parser.add_argument('--seed_text', type=str, default=None,
                        help='Seed text for generation')
    
    # Experiment parameters
    parser.add_argument('--experiment', type=str, default=None,
                        choices=['model_comparison', 'hyperparameter_search', None],
                        help='Experiment to run')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with limited data')
    
    return parser.parse_args()


def main():
    """
    Main function
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set random seed
    set_seed(args.seed)
    
    # Print arguments
    print_args(args)
    
    # Run experiment or train single model
    if args.experiment == 'model_comparison':
        run_experiment_model_comparison(args)
    elif args.experiment == 'hyperparameter_search':
        run_experiment_hyperparameter_search(args)
    else:
        train_model(args, args.model_type)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 