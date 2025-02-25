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
    parser.add_argument("--experiment", type=str, default=None, choices=[None, "encoder_type"],
                        help="Experiment type")
    
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
    print(f"Running ESIM with {encoder_type}")
    print("=" * 50)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"esim_{encoder_type.lower()}")
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
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
    
    # Create model
    print("Creating model...")
    if args.model_type == "esim":
        model = ESIM(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=3,  # entailment, neutral, contradiction
            dropout=args.dropout,
            use_tree_lstm=args.use_tree_lstm
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Print model information
    print(f"Model type: {args.model_type}")
    print(f"Encoder type: {encoder_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        args=args
    )
    
    # Train model
    print("Training model...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    # Extract evaluation metrics
    with open(os.path.join(output_dir, 'classification_report.txt'), 'r') as f:
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
    """Run comparison experiment"""
    print("=" * 50)
    print(f"Comparing different {param_name} values")
    print("=" * 50)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"compare_{param_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed results directory
    detailed_dir = os.path.join(output_dir, "detailed_results")
    os.makedirs(detailed_dir, exist_ok=True)
    
    # Load data once to reuse
    print("Loading and preprocessing data...")
    data_processor = DataProcessor(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size
    )
    data_loaders = data_processor.create_dataloaders(debug=args.debug)
    
    # Run models with different parameter values
    model_metrics = {}
    param_display_values = []
    accuracy_values = []
    train_times = []
    precision_values = []
    recall_values = []
    f1_values = []
    
    for param_value in param_values:
        # Set parameter
        new_args = argparse.Namespace(**vars(args))
        setattr(new_args, param_name, param_value)
        
        # Run model
        print(f"\n{param_name} = {param_value}")
        
        # Create output directory for this parameter value
        param_output_dir = os.path.join(args.output_dir, f"{param_name}_{param_value}")
        new_args.output_dir = param_output_dir
        
        # Run model
        metrics = run_single_model(new_args, data_loaders)
        
        # Record results
        display_value = "Tree-LSTM" if param_value else "BiLSTM"
        model_metrics[display_value] = metrics
        param_display_values.append(display_value)
        accuracy_values.append(metrics['accuracy'])
        train_times.append(metrics.get('train_time', 0))
        precision_values.append(metrics['precision'])
        recall_values.append(metrics['recall'])
        f1_values.append(metrics['f1'])
        
        # Save detailed results
        detailed_results = {
            'parameter_name': param_name,
            'parameter_value': param_value,
            'display_value': display_value,
            'args': vars(new_args),
            'metrics': metrics
        }
        
        with open(os.path.join(detailed_dir, f"{param_name}_{display_value}.json"), 'w') as f:
            json.dump(detailed_results, f, indent=4)
    
    # Create and save detailed comparison table
    comparison_data = {
        'Encoder Type': param_display_values,
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values,
        'Training Time (s)': train_times
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)
    
    # Create visualizations for all metrics
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy, precision, recall, and F1 score
    plt.subplot(2, 2, 1)
    plt.bar(param_display_values, accuracy_values, color='blue')
    plt.title('Accuracy comparison between BiLSTM and Tree-LSTM')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.bar(param_display_values, precision_values, color='green')
    plt.title('Precision comparison between BiLSTM and Tree-LSTM')
    plt.ylabel('Precision')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    plt.bar(param_display_values, recall_values, color='orange')
    plt.title('Recall comparison between BiLSTM and Tree-LSTM')
    plt.ylabel('Recall')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    plt.bar(param_display_values, f1_values, color='red')
    plt.title('F1 score comparison between BiLSTM and Tree-LSTM')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_comparison.png'))
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(param_display_values, train_times, color='purple')
    plt.title('Training time comparison between BiLSTM and Tree-LSTM')
    plt.ylabel('Training Time (s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'))
    
    # Create metrics heatmap
    metrics_df = pd.DataFrame({
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values
    }, index=param_display_values)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title('Performance metrics heatmap for BiLSTM vs Tree-LSTM')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
    
    # Save results
    results = {
        'args': vars(args),
        'param_name': param_name,
        'param_values': [str(val) for val in param_values],
        'display_values': param_display_values,
        'metrics': model_metrics,
        'summary': {
            'accuracy': {param: metrics['accuracy'] for param, metrics in model_metrics.items()},
            'precision': {param: metrics['precision'] for param, metrics in model_metrics.items()},
            'recall': {param: metrics['recall'] for param, metrics in model_metrics.items()},
            'f1': {param: metrics['f1'] for param, metrics in model_metrics.items()},
            'train_time': {param: metrics.get('train_time', 0) for param, metrics in model_metrics.items()}
        }
    }
    
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Find and report best encoder for each metric
    best_accuracy_param = param_display_values[np.argmax(accuracy_values)]
    best_precision_param = param_display_values[np.argmax(precision_values)]
    best_recall_param = param_display_values[np.argmax(recall_values)]
    best_f1_param = param_display_values[np.argmax(f1_values)]
    fastest_param = param_display_values[np.argmin(train_times)]
    
    best_params = {
        'best_accuracy': {'encoder': best_accuracy_param, 'value': max(accuracy_values)},
        'best_precision': {'encoder': best_precision_param, 'value': max(precision_values)},
        'best_recall': {'encoder': best_recall_param, 'value': max(recall_values)},
        'best_f1': {'encoder': best_f1_param, 'value': max(f1_values)},
        'fastest_training': {'encoder': fastest_param, 'value': min(train_times)}
    }
    
    with open(os.path.join(output_dir, 'best_encoder.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Encoder Type Comparison Summary")
    print("=" * 50)
    print(f"Best Accuracy: {max(accuracy_values):.4f} with encoder = {best_accuracy_param}")
    print(f"Best Precision: {max(precision_values):.4f} with encoder = {best_precision_param}")
    print(f"Best Recall: {max(recall_values):.4f} with encoder = {best_recall_param}")
    print(f"Best F1 Score: {max(f1_values):.4f} with encoder = {best_f1_param}")
    print(f"Fastest Training Time: {min(train_times):.2f}s with encoder = {fastest_param}")
    print("=" * 50)
    
    print(f"\nComparison results saved to {output_dir}")
    print(f"Detailed results for each encoder type saved to {detailed_dir}")


def run_experiment(args):
    """Run experiment"""
    if args.experiment is None:
        # Run single model
        run_single_model(args)
    elif args.experiment == "encoder_type":
        # Compare BiLSTM and Tree-LSTM encoders
        param_values = [False, True]  # [BiLSTM, Tree-LSTM]
        run_comparison_experiment(args, "use_tree_lstm", param_values)


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    run_experiment(args)
    
    print(f"All experiments completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()