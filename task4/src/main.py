import os
import time
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from models import BiLSTMCRF
from trainer import Trainer
from utils import get_args, set_seed, print_args, count_parameters, epoch_time


def create_model(args, data_processor):
    """
    Create a model based on the specified model type
    
    Args:
        args: Command line arguments
        data_processor: Data processor instance
        
    Returns:
        model: Initialized model
    """
    vocab_size = len(data_processor.word2idx)
    num_tags = len(data_processor.tag2idx)
    
    if args.model_type == 'bilstm':
        # Basic BiLSTM without CRF and character features
        model = BiLSTMCRF(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_tags=num_tags,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_char_cnn=False,
            use_crf=False
        )
    elif args.model_type == 'bilstm_crf':
        # BiLSTM with CRF but without character features
        model = BiLSTMCRF(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_tags=num_tags,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_char_cnn=False,
            use_crf=True,
            optimize_crf=False
        )
    elif args.model_type == 'bilstm_crf_char':
        # BiLSTM with CRF and character features
        model = BiLSTMCRF(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_tags=num_tags,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_char_cnn=True,
            num_chars=len(data_processor.char2idx),
            char_embedding_dim=args.char_embedding_dim,
            char_channel_size=args.char_channel_size,
            use_crf=True,
            optimize_crf=False
        )
    else:  # bilstm_crf_char_opt
        # BiLSTM with optimized CRF and character features
        model = BiLSTMCRF(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_tags=num_tags,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_char_cnn=True,
            num_chars=len(data_processor.char2idx),
            char_embedding_dim=args.char_embedding_dim,
            char_channel_size=args.char_channel_size,
            use_crf=True,
            optimize_crf=True
        )
    
    return model


def generate_comparison_visualizations(output_dir):
    """
    Generate visualizations comparing different model architectures
    
    Args:
        output_dir: Output directory where comparison results are stored
    """
    comparison_dir = os.path.join(output_dir, 'comparison')
    results_dir = os.path.join(comparison_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Model directories and names for legend
    model_dirs = [
        'base_bilstm',
        'bilstm_crf',
        'bilstm_crf_char',
        'bilstm_crf_char_opt'
    ]
    
    model_names = [
        'BiLSTM (Base)',
        'BiLSTM-CRF',
        'BiLSTM-CRF + Char CNN',
        'BiLSTM-CRF + Char CNN + Opt CRF'
    ]
    
    # Collect metrics from all models
    models_metrics = []
    for model_dir in model_dirs:
        model_path = os.path.join(comparison_dir, model_dir)
        metrics_path = os.path.join(model_path, 'metrics', 'metrics.csv')
        
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            models_metrics.append(metrics_df)
    
    if not models_metrics:
        print("No metrics found for comparison visualization")
        return
    
    # Combine metrics into a single dataframe
    combined_metrics = pd.concat(models_metrics, ignore_index=True)
    
    # Add model names
    combined_metrics['model_name'] = model_names[:len(models_metrics)]
    
    # Save combined metrics
    combined_metrics.to_csv(os.path.join(results_dir, 'combined_metrics.csv'), index=False)
    
    # Extract test metrics
    test_precision = []
    test_recall = []
    test_f1 = []
    
    for metrics in models_metrics:
        test_precision.append(float(metrics['test_precision'].iloc[0].strip('%')))
        test_recall.append(float(metrics['test_recall'].iloc[0].strip('%')))
        test_f1.append(float(metrics['test_f1'].iloc[0].strip('%')))
    
    # Create bar chart comparing test metrics
    plt.figure(figsize=(12, 8))
    x = range(len(model_names[:len(models_metrics)]))
    width = 0.25
    
    plt.bar([i - width for i in x], test_precision, width, label='Precision', color='crimson')
    plt.bar(x, test_recall, width, label='Recall', color='forestgreen')
    plt.bar([i + width for i in x], test_f1, width, label='F1 Score', color='royalblue')
    
    plt.xlabel('Model Architecture')
    plt.ylabel('Score (%)')
    plt.title('Performance Comparison of Different Model Architectures')
    plt.xticks(x, model_names[:len(models_metrics)])
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save comparison chart
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
    
    # Create a table-like figure
    fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Model', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']
    ]
    
    for i, name in enumerate(model_names[:len(models_metrics)]):
        table_data.append([
            name,
            f"{test_precision[i]:.2f}",
            f"{test_recall[i]:.2f}",
            f"{test_f1[i]:.2f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.4, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('Model Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_table.png'))
    
    print(f"Comparison visualizations saved to {results_dir}")


def main():
    # Get arguments
    args = get_args()
    
    # Handle comparison visualization mode separately
    if args.comparison_visualization:
        print("\nGenerating comparison visualizations...")
        generate_comparison_visualizations(args.output_dir)
        return
    
    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    # Print header
    print("\n" + "="*60)
    print(f"BiLSTM-CRF Model for Named Entity Recognition")
    print("="*60 + "\n")
    
    # Print arguments
    print_args(args)
    
    # Load data
    print("\nLoading data...")
    start_time = time.time()
    data_processor = DataProcessor(
        args.data_dir, 
        args.language, 
        args.max_seq_len,
        args.max_word_len
    )
    data_processor.load_data()
    
    # Create data loaders
    train_loader, dev_loader, test_loader = data_processor.create_dataloaders(args.batch_size)
    end_time = time.time()
    mins, secs = epoch_time(start_time, end_time)
    print(f"Data loaded in: {mins}m {secs}s")
    
    # Print dataset information
    print(f"\nDataset Information:")
    print(f"  Vocabulary size: {len(data_processor.word2idx)}")
    print(f"  Character vocabulary size: {len(data_processor.char2idx)}")
    print(f"  Tag set size: {len(data_processor.tag2idx)}")
    print(f"  Training samples: {len(data_processor.train_dataset)}")
    print(f"  Validation samples: {len(data_processor.valid_dataset)}")
    print(f"  Test samples: {len(data_processor.test_dataset)}")
    
    # Initialize model based on the model type
    print(f"\nInitializing {args.model_type} model...")
    model = create_model(args, data_processor)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(model, data_processor.tag2idx, args)
    
    # Train model
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    start_time = time.time()
    best_model_path = trainer.train(train_loader, dev_loader, test_loader)
    end_time = time.time()
    
    mins, secs = epoch_time(start_time, end_time)
    print(f"\nTotal training time: {mins}m {secs}s")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main() 