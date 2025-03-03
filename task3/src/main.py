import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import re

from data_processor import DataProcessor, Tokenizer
from models import ESIM
from trainer import Trainer, Evaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Text Matching with Attention Mechanism")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../dataset",
                        help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="../output",
                        help="Directory to save the model and results")
    parser.add_argument("--max_seq_len", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="esim", choices=["esim"],
                        help="Model type")
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=300, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--use_tree_lstm", action="store_true", help="Use Tree-LSTM instead of BiLSTM")
    
    # Embedding arguments
    parser.add_argument("--pretrained_embedding", type=str, default=None, 
                        choices=[None, "glove"], help="Type of pretrained embedding")
    parser.add_argument("--embedding_path", type=str, default="../glove/glove.6B.100d.txt",
                        help="Path to pretrained embedding file")
    parser.add_argument("--freeze_embedding", action="store_true", 
                        help="Freeze embedding weights during training")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0004, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=10.0, help="Gradient clipping")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="Learning rate scheduler patience")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Optimizer")
    
    # Experiment arguments
    parser.add_argument("--experiment", type=str, default=None, 
                      choices=[None, "encoder_type", "hidden_dim", "dropout", "max_seq_len", 
                              "batch_size", "learning_rate", "optimizer", "epochs", "pretrained_embedding"],
                        help="Experiment type")
    parser.add_argument("--param_values", nargs="+", default=None,
                      help="Parameter values for experiment")
    
    # Example and analysis arguments
    parser.add_argument("--example", action="store_true", help="Run model on example sentences")
    parser.add_argument("--premise", type=str, default=None, help="Example premise")
    parser.add_argument("--hypothesis", type=str, default=None, help="Example hypothesis")
    parser.add_argument("--analyze_attention", action="store_true", 
                      help="Analyze attention weights on test examples")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA if available")
    parser.add_argument("--debug", action="store_true", help="Use debug mode with smaller dataset")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_model(args, data_loaders=None):
    """Run single model"""
    print("=" * 50)
    encoder_type = "Tree-LSTM" if args.use_tree_lstm else "BiLSTM"
    embedding_type = args.pretrained_embedding if args.pretrained_embedding else "Random"
    print(f"Running ESIM with {encoder_type} encoder and {embedding_type} embeddings")
    print("=" * 50)
    
    # Check if this is part of a comparison experiment
    is_comparison = hasattr(args, 'param_name') and hasattr(args, 'param_value')
    param_suffix = f"_{args.param_name}_{args.param_value}" if is_comparison else ""
    
    # Create output directory
    if not is_comparison:
        # For single model experiments, create a subdirectory
        output_dir = os.path.join(args.output_dir, f"esim_{encoder_type.lower()}_{embedding_type.lower()}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        # For comparison experiments, use the parent directory directly
        output_dir = args.output_dir
    
    # Load and preprocess data if not provided
    if data_loaders is None:
        print("Loading and preprocessing data...")
        data_processor = DataProcessor(
            data_dir=args.data_dir,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size
        )
        train_loader, valid_loader, test_loader, vocab_size = data_processor.create_dataloaders(debug=args.debug)
    else:
        train_loader, valid_loader, test_loader, vocab_size = data_loaders
    
    # Print dataset information
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(valid_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Load pretrained embeddings if specified
    embedding_matrix = None
    if args.pretrained_embedding == "glove":
        word2idx = train_loader.dataset.tokenizer.word2idx
        embedding_matrix = load_pretrained_embeddings(
            embedding_path=args.embedding_path,
            word2idx=word2idx,
            embedding_dim=args.embedding_dim
        )
    
    # Create model
    print("Creating model...")
    if args.model_type == "esim":
        model = ESIM(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=3,  # entailment, neutral, contradiction
            dropout=args.dropout,
            use_tree_lstm=args.use_tree_lstm,
            padding_idx=0,
            embedding_matrix=embedding_matrix,
            freeze_embedding=args.freeze_embedding
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Print model information
    print(f"Model type: {args.model_type}")
    print(f"Encoder type: {encoder_type}")
    print(f"Embedding type: {embedding_type}")
    if args.pretrained_embedding:
        print(f"Embedding path: {args.embedding_path}")
        print(f"Freeze embedding: {args.freeze_embedding}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Use modified output paths for trainer to handle parameter-specific file naming
    modified_args = copy.deepcopy(args)
    modified_args.output_dir = output_dir
    modified_args.param_suffix = param_suffix
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        args=modified_args
    )
    
    # Train model
    print("Training model...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    # Extract evaluation metrics
    report_file = os.path.join(output_dir, f'classification_report{param_suffix}.txt')
    with open(report_file, 'r') as f:
        report_text = f.read()
    
    # Parse classification report to extract metrics
    lines = report_text.strip().split('\n')
    # Get weighted avg line (last line)
    weighted_avg_line = [line for line in lines if 'weighted avg' in line]
    if weighted_avg_line:
        parts = weighted_avg_line[0].split()
        precision = float(parts[2])
        recall = float(parts[3])
        f1 = float(parts[4])
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    
    # Get accuracy from test set evaluation
    test_acc_line = [line for line in lines if 'accuracy' in line.lower()]
    if test_acc_line:
        accuracy = float(test_acc_line[0].split()[-1])
    else:
        # Fallback to trainer's best validation accuracy
        accuracy = trainer.best_valid_acc
    
    # Collect metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time
    }
    
    # Save results
    results = {
        'args': vars(args),
        'metrics': metrics,
        'train_time': train_time,
        'encoder_type': encoder_type
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_dir}")
    
    return metrics


def run_comparison_experiment(args, param_name, param_values):
    """Run comparison experiment with different parameter values"""
    print(f"Running comparison experiment: {param_name}")
    print(f"Parameter values: {param_values}")
    
    # Create output directory (single directory for all parameter values)
    output_dir = os.path.join(args.output_dir, f"compare_{param_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_processor = DataProcessor(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size
    )
    data_loaders = data_processor.create_dataloaders(debug=args.debug)
    
    # Store results
    results = []
    
    # Run each parameter value
    for val in param_values:
        print("=" * 50)
        print(f"Running with {param_name} = {val}")
        print("=" * 50)
        
        # Create a copy of arguments with updated parameter
        current_args = copy.deepcopy(args)
        
        # Set output directory to main output_dir (will be appended with parameter value in run_single_model)
        current_args.output_dir = output_dir
        
        # Add parameter value information to use in file naming
        current_args.param_name = param_name
        current_args.param_value = val
        
        # Special handling for parameter names
        if param_name == "pretrained_embedding":
            # For pretrained embeddings experiment
            if val == "none":
                current_args.pretrained_embedding = None
            else:
                current_args.pretrained_embedding = val
        else:
            # For other parameters
            setattr(current_args, param_name, val)
        
        # Run model with current parameter value
        run_single_model(current_args, data_loaders)
        
        # Load results from file with parameter-specific name
        results_file = os.path.join(output_dir, f'results_{param_name}_{val}.json')
        with open(results_file, 'r') as f:
            result = json.load(f)
        
        # Add parameter value to results
        result[param_name] = val
        results.append(result)
    
    # Create table of results
    print("=" * 50)
    print(f"Results for {param_name} comparison experiment:")
    print("=" * 50)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Reorder columns to show parameter first
    cols = df.columns.tolist()
    cols.remove(param_name)
    cols = [param_name] + cols
    df = df[cols]
    
    # Print results
    print(df.to_string(index=False))
    
    # Save results to CSV
    csv_file = os.path.join(output_dir, 'results.csv')
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")
    
    # Plot learning curves
    plot_learning_curves(output_dir, param_name, param_values)
    
    # Plot metrics comparison
    plot_metrics_comparison(df, output_dir, param_name)
    
    # Plot confusion matrices
    plot_confusion_matrices(output_dir, param_name, param_values)
    
    # Plot attention matrices
    plot_attention_matrices(output_dir, param_name, param_values)
    
    print(f"Experiment completed. Results saved to {output_dir}")


def load_pretrained_embeddings(embedding_path, word2idx, embedding_dim):
    """
    Load pretrained word embeddings from file
    
    Args:
        embedding_path: Path to pretrained embedding file
        word2idx: Word to index mapping
        embedding_dim: Dimension of embeddings
        
    Returns:
        embedding_matrix: Numpy array of shape (vocab_size, embedding_dim)
    """
    print(f"Loading pretrained embeddings from {embedding_path}...")
    
    # Initialize embedding matrix
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    # Count words found in pretrained embeddings
    found_count = 0
    dimension_mismatch_count = 0
    
    # Check embedding file name for dimension info
    file_dim = None
    if 'glove' in embedding_path.lower():
        # Try to extract dimension from filename (e.g., glove.6B.100d.txt → 100)
        match = re.search(r'(\d+)d', embedding_path)
        if match:
            file_dim = int(match.group(1))
            if file_dim != embedding_dim:
                print(f"WARNING: Embedding dimension mismatch! File appears to be {file_dim}d but {embedding_dim}d was specified.")
                print(f"Please ensure these dimensions match to avoid errors.")
    
    # Load embeddings
    try:
        with open(embedding_path, 'r', encoding='utf-8') as f:
            # Read first line to determine actual dimension
            first_line = f.readline().strip().split()
            actual_dim = len(first_line) - 1  # Subtract 1 for the word itself
            
            if actual_dim != embedding_dim:
                print(f"ERROR: Embedding dimension mismatch! File contains {actual_dim}d vectors but {embedding_dim}d was specified.")
                print(f"Please correct --embedding_dim parameter to match the file or use a different embedding file.")
                raise ValueError(f"Embedding dimension mismatch: expected {embedding_dim}, got {actual_dim}")
            
            # Process first line
            word = first_line[0]
            if word in word2idx:
                vector = np.array([float(val) for val in first_line[1:]])
                embedding_matrix[word2idx[word]] = vector
                found_count += 1
            
            # Process rest of the file
            for line in f:
                values = line.strip().split()
                word = values[0]
                
                # Skip words not in vocabulary
                if word not in word2idx:
                    continue
                    
                # Convert embedding values to float
                vector = np.array([float(val) for val in values[1:]])
                
                # Update embedding matrix
                embedding_matrix[word2idx[word]] = vector
                found_count += 1
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        raise
    
    # Print statistics
    coverage = found_count / vocab_size * 100
    print(f"Found embeddings for {found_count}/{vocab_size} words ({coverage:.2f}% coverage)")
    
    if coverage < 1.0:
        print("WARNING: Very low embedding coverage. Make sure the embedding file is correct.")
    
    return embedding_matrix


def run_experiment(args):
    """Run experiment based on arguments"""
    print(f"Running experiment: {args.experiment}")
    
    if args.experiment == "encoder_type":
        # Compare BiLSTM vs Tree-LSTM
        param_values = [False, True]  # [BiLSTM, Tree-LSTM]
        run_comparison_experiment(args, "use_tree_lstm", param_values)
    elif args.experiment == "hidden_dim":
        # Compare different hidden dimensions
        param_values = args.param_values if args.param_values else [100, 200, 300, 400, 500]
        param_values = [int(val) for val in param_values]  # Convert string to int
        run_comparison_experiment(args, "hidden_dim", param_values)
    elif args.experiment == "dropout":
        # Compare different dropout rates
        param_values = args.param_values if args.param_values else [0.2, 0.3, 0.4, 0.5, 0.6]
        param_values = [float(val) for val in param_values]  # Convert string to float
        run_comparison_experiment(args, "dropout", param_values)
    elif args.experiment == "max_seq_len":
        # Compare different maximum sequence lengths
        param_values = args.param_values if args.param_values else [50, 75, 100, 125, 150]
        param_values = [int(val) for val in param_values]  # Convert string to int
        run_comparison_experiment(args, "max_seq_len", param_values)
    elif args.experiment == "batch_size":
        # Compare different batch sizes
        param_values = args.param_values if args.param_values else [32, 64, 128, 256, 512]
        param_values = [int(val) for val in param_values]  # Convert string to int
        run_comparison_experiment(args, "batch_size", param_values)
    elif args.experiment == "learning_rate":
        # Compare different learning rates
        param_values = args.param_values if args.param_values else [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        param_values = [float(val) for val in param_values]  # Convert string to float
        run_comparison_experiment(args, "learning_rate", param_values)
    elif args.experiment == "optimizer":
        # Compare different optimizers
        param_values = args.param_values if args.param_values else ["adam", "sgd"]
        run_comparison_experiment(args, "optimizer", param_values)
    elif args.experiment == "epochs":
        # Compare different numbers of training epochs
        param_values = args.param_values if args.param_values else [5, 10, 15, 20, 25]
        param_values = [int(val) for val in param_values]  # Convert string to int
        run_comparison_experiment(args, "epochs", param_values)
    elif args.experiment == "pretrained_embedding":
        # Compare with and without pretrained embeddings
        param_values = args.param_values if args.param_values else ["none", "glove"]
        run_comparison_experiment(args, "pretrained_embedding", param_values)
    else:
        # Run single model
        run_single_model(args)


def main():
    """Main entry point"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle special modes
    if args.example:
        if args.premise is None or args.hypothesis is None:
            print("Please provide both premise and hypothesis for example mode")
            return
        run_example(args)
        return
        
    if args.analyze_attention:
        run_attention_analysis(args)
        return
    
    # Run experiment
    run_experiment(args)
    

def run_example(args):
    """Run model on example sentences"""
    # Load data processor to get tokenizer
    data_processor = DataProcessor(args.data_dir, args.max_seq_len, args.batch_size)
    data_processor.load_data()
    tokenizer = data_processor.tokenizer
    
    # Build a model with default parameters
    model = ESIM(
        vocab_size=len(tokenizer),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_tree_lstm=args.use_tree_lstm
    )
    
    # Check if model exists
    model_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded model from {model_path}")
    else:
        print(f"No saved model found at {model_path}. Using untrained model.")
    
    # Process inputs
    premise_tokens = tokenizer.tokenize(args.premise)
    hypothesis_tokens = tokenizer.tokenize(args.hypothesis)
    
    premise_ids = tokenizer.convert_tokens_to_ids(premise_tokens)
    hypothesis_ids = tokenizer.convert_tokens_to_ids(hypothesis_tokens)
    
    # Pad sequences
    if len(premise_ids) < args.max_seq_len:
        premise_ids += [0] * (args.max_seq_len - len(premise_ids))
    else:
        premise_ids = premise_ids[:args.max_seq_len]
        
    if len(hypothesis_ids) < args.max_seq_len:
        hypothesis_ids += [0] * (args.max_seq_len - len(hypothesis_ids))
    else:
        hypothesis_ids = hypothesis_ids[:args.max_seq_len]
    
    # Create masks
    premise_mask = [1 if id != 0 else 0 for id in premise_ids]
    hypothesis_mask = [1 if id != 0 else 0 for id in hypothesis_ids]
    
    # Convert to tensors
    premise_tensor = torch.tensor([premise_ids], dtype=torch.long)
    hypothesis_tensor = torch.tensor([hypothesis_ids], dtype=torch.long)
    premise_mask_tensor = torch.tensor([premise_mask], dtype=torch.float)
    hypothesis_mask_tensor = torch.tensor([hypothesis_mask], dtype=torch.float)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        logits, attention_weights = model(
            premise_tensor, 
            hypothesis_tensor, 
            premise_mask_tensor, 
            hypothesis_mask_tensor,
            return_attention=True
        )
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    
    # Map prediction to label
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    predicted_label = label_map[prediction]
    
    # Print results
    print("\nResults:")
    print(f"Premise: {args.premise}")
    print(f"Hypothesis: {args.hypothesis}")
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {probs[0][prediction]:.4f}")
    
    # Print attention visualization
    print("\nAttention Visualization:")
    visualize_attention(
        premise_tokens, 
        hypothesis_tokens, 
        attention_weights[0].numpy()
    )


def run_attention_analysis(args):
    """Analyze attention weights on test examples"""
    # Load data processor
    data_processor = DataProcessor(args.data_dir, args.max_seq_len, args.batch_size)
    data_processor.load_data()
    train_loader, valid_loader, test_loader = data_processor.create_dataloaders()
    tokenizer = data_processor.tokenizer
    
    # Build model
    model = ESIM(
        vocab_size=len(tokenizer),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_tree_lstm=args.use_tree_lstm
    )
    
    # Load trained model if exists
    model_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded model from {model_path}")
    else:
        print(f"No saved model found at {model_path}. Using untrained model.")
    
    # Create evaluator
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(model, test_loader, device)
    
    # Analyze attention on random examples
    print("Analyzing attention weights on random test examples...")
    for i in range(5):  # Analyze 5 random examples
        evaluator.visualize_attention(None, None, tokenizer)
        print("\n" + "-"*80 + "\n")


def visualize_attention(premise_tokens, hypothesis_tokens, attention_matrix):
    """Visualize attention between premise and hypothesis"""
    max_display_length = 30  # Limit display length for cleaner output
    
    # Truncate tokens if needed
    p_tokens = premise_tokens[:max_display_length]
    h_tokens = hypothesis_tokens[:max_display_length]
    
    # Truncate attention matrix
    attention = attention_matrix[:len(p_tokens), :len(h_tokens)]
    
    # Find the max attention value for each premise token
    max_attentions = np.max(attention, axis=1)
    
    # Print premise tokens with their max attention value
    print("Premise tokens and their importance:")
    for i, token in enumerate(p_tokens):
        attention_str = "#" * int(max_attentions[i] * 10)
        print(f"{token:>12}: {attention_str} ({max_attentions[i]:.2f})")
    
    print("\nHypothesis tokens and their importance:")
    # Find the max attention value for each hypothesis token
    max_attentions = np.max(attention, axis=0)
    for i, token in enumerate(h_tokens):
        attention_str = "#" * int(max_attentions[i] * 10)
        print(f"{token:>12}: {attention_str} ({max_attentions[i]:.2f})")
    
    print("\nTop 5 premise-hypothesis token pairs with highest attention:")
    # Get top 5 attention values
    flat_attention = attention.flatten()
    top_indices = np.argsort(flat_attention)[-5:][::-1]
    for idx in top_indices:
        p_idx, h_idx = np.unravel_index(idx, attention.shape)
        print(f"{p_tokens[p_idx]} -> {h_tokens[h_idx]}: {attention[p_idx, h_idx]:.4f}")


def plot_learning_curves(output_dir, param_name, param_values):
    """
    Plot learning curves for different parameter values
    
    Args:
        output_dir: Directory to save plots
        param_name: Name of parameter being compared
        param_values: List of parameter values
    """
    plt.figure(figsize=(12, 8))
    
    for val in param_values:
        # Construct parameter-specific filename
        history_path = os.path.join(output_dir, f'training_history_{param_name}_{val}.json')
        
        if not os.path.exists(history_path):
            print(f"Warning: No training history found for {param_name} = {val}")
            continue
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Plot training and validation loss
        epochs = range(1, len(history['train_loss']) + 1)
        plt.subplot(2, 1, 1)
        plt.plot(epochs, history['train_loss'], 'o-', label=f'{val} (train)')
        plt.plot(epochs, history['val_loss'], 's-', label=f'{val} (val)')
    
    plt.title(f'Loss Curves for Different {param_name.replace("_", " ").title()} Values')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    for val in param_values:
        # Construct parameter-specific filename
        history_path = os.path.join(output_dir, f'training_history_{param_name}_{val}.json')
        
        if not os.path.exists(history_path):
            continue
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Plot training and validation accuracy
        epochs = range(1, len(history['train_accuracy']) + 1)
        plt.subplot(2, 1, 2)
        plt.plot(epochs, history['train_accuracy'], 'o-', label=f'{val} (train)')
        plt.plot(epochs, history['val_accuracy'], 's-', label=f'{val} (val)')
    
    plt.title(f'Accuracy Curves for Different {param_name.replace("_", " ").title()} Values')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()


def plot_metrics_comparison(df, output_dir, param_name):
    """
    Plot comparison of metrics for different parameter values
    
    Args:
        df: DataFrame containing results
        output_dir: Directory to save plots
        param_name: Name of parameter being compared
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'train_time']
    
    # Filter out metrics that are not in the dataframe
    metrics = [m for m in metrics if m in df.columns]
    
    # Create subplots for each metric
    n_metrics = len(metrics)
    fig_rows = (n_metrics + 1) // 2  # Ceiling division
    
    plt.figure(figsize=(15, 5 * fig_rows))
    
    for i, metric in enumerate(metrics):
        plt.subplot(fig_rows, 2, i + 1)
        
        if metric == 'train_time':
            # For training time, use bar plot
            sns.barplot(x=param_name, y=metric, data=df)
            plt.title(f'Training Time for Different {param_name.replace("_", " ").title()} Values')
            plt.ylabel('Training Time (s)')
        else:
            # For other metrics, use bar plot with percentage y-axis
            sns.barplot(x=param_name, y=metric, data=df)
            plt.title(f'{metric.title()} for Different {param_name.replace("_", " ").title()} Values')
            plt.ylabel(f'{metric.title()}')
            plt.ylim(0, 1.0)
            
            # Add percentage labels
            for p in plt.gca().patches:
                plt.gca().annotate(f"{p.get_height():.2%}", 
                                  (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='bottom')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()
    
    # Create heatmap for performance metrics (excluding training time)
    performance_metrics = [m for m in metrics if m != 'train_time']
    if performance_metrics:
        plt.figure(figsize=(10, 8))
        
        # Pivot the dataframe for heatmap
        heatmap_df = df.pivot(index=param_name, columns=None, values=performance_metrics)
        
        # Plot heatmap
        sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', fmt='.2%', linewidths=.5)
        plt.title(f'Performance Metrics for Different {param_name.replace("_", " ").title()} Values')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
        plt.close()


def plot_confusion_matrices(output_dir, param_name, param_values):
    """
    Plot confusion matrices for different parameter values
    
    Args:
        output_dir: Directory to save plots
        param_name: Name of parameter being compared
        param_values: List of parameter values
    """
    # Determine grid size
    n_values = len(param_values)
    grid_size = int(np.ceil(np.sqrt(n_values)))
    
    plt.figure(figsize=(5 * grid_size, 5 * grid_size))
    
    for i, val in enumerate(param_values):
        # Construct parameter-specific filename
        cm_path = os.path.join(output_dir, f'confusion_matrix_{param_name}_{val}.npy')
        
        if not os.path.exists(cm_path):
            print(f"Warning: No confusion matrix found for {param_name} = {val}")
            continue
        
        # Load confusion matrix
        cm = np.load(cm_path)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.subplot(grid_size, grid_size, i + 1)
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                   xticklabels=['E', 'N', 'C'], yticklabels=['E', 'N', 'C'])
        plt.title(f'{param_name} = {val}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()


def plot_attention_matrices(output_dir, param_name, param_values):
    """
    Plot attention matrices for different parameter values using example sentences
    
    Args:
        output_dir: Directory to save plots
        param_name: Name of parameter being compared
        param_values: List of parameter values
    """
    # Example sentences for visualization (simple sentences that are easy to understand)
    premise = "A man is walking his dog in the park."
    hypothesis = "A person is in the park with a pet."
    
    # Tokenize sentences
    premise_tokens = premise.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    
    # Create plot
    plt.figure(figsize=(20, 5 * len(param_values)))
    
    for i, val in enumerate(param_values):
        # Construct parameter-specific filename
        model_path = os.path.join(output_dir, f'best_model_{param_name}_{val}.pt')
        
        if not os.path.exists(model_path):
            print(f"Warning: No model found for {param_name} = {val}")
            continue
        
        # Try to load the attention matrices
        attention_file = os.path.join(output_dir, f'attention_matrix_example_{param_name}_{val}.npy')
        
        if os.path.exists(attention_file):
            # Load pre-computed attention matrix
            attention_matrix = np.load(attention_file)
        else:
            # If attention matrix doesn't exist, we need to compute it
            # This would require loading the model and tokenizer, which is complex
            print(f"Warning: No attention matrix found for {param_name} = {val}. Generating one...")
            
            try:
                # Load config to get necessary parameters
                config_file = os.path.join(output_dir, f'config_{param_name}_{val}.json')
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                else:
                    # Use default config
                    config = {
                        'embedding_dim': 100,
                        'hidden_dim': 300,
                        'dropout': 0.5,
                        'use_tree_lstm': False
                    }
                
                # Load model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Create a simple tokenizer just for these sentences
                vocab = set(premise_tokens + hypothesis_tokens + ['<pad>', '<unk>'])
                word2idx = {word: idx for idx, word in enumerate(vocab)}
                
                # Load model based on config
                model = ESIM(
                    vocab_size=len(vocab), 
                    embedding_dim=config.get('embedding_dim', 100),
                    hidden_dim=config.get('hidden_dim', 300),
                    dropout=config.get('dropout', 0.5),
                    use_tree_lstm=config.get('use_tree_lstm', False)
                )
                
                # Load state dict
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                # Convert tokens to ids
                premise_ids = torch.tensor([[word2idx.get(w, 1) for w in premise_tokens]])
                hypothesis_ids = torch.tensor([[word2idx.get(w, 1) for w in hypothesis_tokens]])
                
                # Create masks
                premise_mask = torch.ones_like(premise_ids)
                hypothesis_mask = torch.ones_like(hypothesis_ids)
                
                # Get attention matrix
                with torch.no_grad():
                    _, attention_matrix = model(
                        premise_ids, hypothesis_ids, 
                        premise_mask, hypothesis_mask, 
                        return_attention=True
                    )
                
                attention_matrix = attention_matrix.squeeze(0).cpu().numpy()
                
                # Save for future use
                np.save(attention_file, attention_matrix)
            except Exception as e:
                print(f"Error generating attention matrix: {e}")
                continue
        
        # Plot attention matrix
        plt.subplot(len(param_values), 1, i+1)
        sns.heatmap(
            attention_matrix, 
            annot=True, 
            cmap='Blues', 
            xticklabels=hypothesis_tokens,
            yticklabels=premise_tokens
        )
        plt.title(f'{param_name} = {val}')
        plt.xlabel('Hypothesis')
        plt.ylabel('Premise')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_matrices.png'))
    plt.close()


if __name__ == "__main__":
    main()