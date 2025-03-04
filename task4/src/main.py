import os
import time
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
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
    
    # Get START and STOP indices from tag2idx if they exist
    start_idx = data_processor.tag2idx.get("<START>")
    stop_idx = data_processor.tag2idx.get("<STOP>")
    
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
            use_crf=False,
            start_idx=start_idx,
            stop_idx=stop_idx
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
            start_idx=start_idx,
            stop_idx=stop_idx
        )
    else:  # bilstm_crf_char (default)
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
            start_idx=start_idx,
            stop_idx=stop_idx
        )
    
    return model


def generate_comparison_matrix(output_dir):
    """
    generate the comparison matrix of the models
    
    Args:
        output_dir: the path of the output directory
    """
    print("Generating model comparison matrix...")
    
    comparison_dir = os.path.join(output_dir, 'comparison')
    results_dir = os.path.join(comparison_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # model directories and names
    model_dirs = [
        'base_bilstm',
        'bilstm_crf',
        'bilstm_crf_char'
    ]
    
    model_names = [
        'BiLSTM (Base)',
        'BiLSTM-CRF',
        'BiLSTM-CRF + Char CNN'
    ]
    
    # metric names and display names
    metrics = ['test_precision', 'test_recall', 'test_f1']
    metric_names = ['Precision', 'Recall', 'F1 Score']
    
    # collect all model metrics
    all_metrics = []
    
    for model_dir in model_dirs:
        model_path = os.path.join(comparison_dir, model_dir)
        metrics_path = os.path.join(model_path, 'eng', 'metrics', 'metrics.csv')
        
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            model_metrics = {}
            
            for metric in metrics:
                # extract value from percentage string
                value = float(metrics_df[metric].iloc[0].strip('%'))
                model_metrics[metric] = value
                
            all_metrics.append(model_metrics)
        else:
            print(f"Warning: {metrics_path} not found")
            all_metrics.append({metric: 0 for metric in metrics})
    
    if not all_metrics:
        print("Error: No metrics data found")
        return
    
    # create data matrix
    data_matrix = np.zeros((len(model_names), len(metrics)))
    
    for i, model_metrics in enumerate(all_metrics):
        for j, metric in enumerate(metrics):
            data_matrix[i, j] = model_metrics[metric]
    
    # save data to CSV
    df = pd.DataFrame(data_matrix, index=model_names, columns=metric_names)
    df.to_csv(os.path.join(results_dir, 'test_metrics_comparison.csv'))
    
    # generate heatmap
    plt.figure(figsize=(10, 6))
    
    # create custom color mapping - from light blue to dark blue
    cmap = LinearSegmentedColormap.from_list('blue_gradient', ['#DCECF9', '#1E5AA8'])
    
    # draw heatmap
    ax = sns.heatmap(data_matrix, annot=True, fmt='.2f', 
                     xticklabels=metric_names, yticklabels=model_names,
                     cmap=cmap, linewidths=0.5, cbar_kws={'label': 'Score (%)'})
    
    # set title and labels
    plt.title('Model Performance Comparison Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    
    # save image
    plt.savefig(os.path.join(results_dir, 'test_metrics_matrix.png'), dpi=300, bbox_inches='tight')
    
    # generate bar chart comparison
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    # draw bar chart
    for i, metric in enumerate(metrics):
        values = [model_metrics[metric] for model_metrics in all_metrics]
        plt.bar(x + (i - 1) * width, values, width, 
                label=metric_names[i], 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i])
    
    # set title and labels
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Model Architecture', fontsize=14)
    plt.ylabel('Score (%)', fontsize=14)
    plt.xticks(x, model_names)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # save image
    plt.savefig(os.path.join(results_dir, 'test_metrics_bars.png'), dpi=300, bbox_inches='tight')
    
    # generate table image
    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111, frame_on=False)
    ax.axis('off')
    
    # create table data
    table_data = [metric_names]
    for i, name in enumerate(model_names):
        table_data.append([f"{data_matrix[i, j]:.2f}" for j in range(len(metrics))])
    
    # draw table
    table = plt.table(cellText=table_data[1:],
                      rowLabels=model_names,
                      colLabels=table_data[0],
                      loc='center',
                      cellLoc='center',
                      rowLoc='center')
    
    # set table style
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # save table image
    plt.savefig(os.path.join(results_dir, 'test_metrics_table.png'), dpi=300, bbox_inches='tight')
    
    print(f"Model comparison matrix saved to {results_dir}")


def generate_comparison_visualizations(output_dir):
    """
    Generate visualizations comparing different model architectures
    
    Args:
        output_dir: the path of the output directory
    """
    
    # Generate comparison matrix visualization
    generate_comparison_matrix(output_dir)


def main():
    """
    Main function
    """
    # Parse arguments
    args = get_args()
    
    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")
    
    # Check if comparison visualization mode
    if args.comparison_visualization:
        generate_comparison_visualizations(args.output_dir)
        return
    
    # Print arguments
    print_args(args)
    
    # Load data
    print("Loading data...")
    start_time = time.time()
    data_processor = DataProcessor(
        args.data_dir,
        args.language,
        args.max_seq_len,
        args.max_word_len
    )
    data_processor.load_data()
    end_time = time.time()
    print(f"Data loaded in: {int((end_time - start_time) // 60)}m {int((end_time - start_time) % 60)}s")
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"  Vocabulary size: {len(data_processor.word2idx)}")
    print(f"  Character vocabulary size: {len(data_processor.char2idx)}")
    print(f"  Tag set size: {len(data_processor.tag2idx)}")
    print(f"  Training samples: {len(data_processor.train_dataset)}")
    print(f"  Validation samples: {len(data_processor.valid_dataset)}")
    print(f"  Test samples: {len(data_processor.test_dataset)}")
    
    # Create data loaders
    train_loader, dev_loader, test_loader = data_processor.create_dataloaders(args.batch_size)
    
    # Create model
    print(f"\nInitializing {args.model_type} model...")
    model = create_model(args, data_processor)
    print(f"Model parameters: {count_parameters(model)}")
    
    # Create trainer
    trainer = Trainer(model, data_processor.tag2idx, args)
    
    # Analyze entity type distribution in datasets
    print("\nAnalyzing entity type distribution in datasets...")
    trainer.analyze_dataset_differences(train_loader, dev_loader)
    
    # Train model
    print("\n============================================================")
    print("Starting Training")
    print("============================================================")
    best_model_path = trainer.train(train_loader, dev_loader, test_loader)
    
    # Generate comparison visualizations if in comparison mode
    if args.comparison_mode:
        print("Generating comparison visualizations...")
        generate_comparison_visualizations(args.output_dir)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 