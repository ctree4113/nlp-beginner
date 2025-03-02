import os
import argparse
import torch
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
import tempfile

from data_processor import DataProcessor, Tokenizer
from models import TextCNN, TextRNN, TextRCNN
from trainer import Trainer, Evaluator


def parse_args():
    """
    Parse command line arguments for the text classification application.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Text Classification Based on Deep Learning')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../dataset', 
                        help='Directory containing the dataset files')
    parser.add_argument('--output_dir', type=str, default='../output', 
                        help='Directory to save model outputs and results')
    parser.add_argument('--valid_ratio', type=float, default=0.2, 
                        help='Proportion of training data to use for validation')
    
    # Embedding parameters
    parser.add_argument('--embedding_type', type=str, default='random', choices=['random', 'glove'], 
                        help='Type of word embeddings: random (random initialization) or glove (pre-trained embeddings)')
    parser.add_argument('--embedding_dim', type=int, default=100, 
                        choices=[50, 100, 200, 300], help='Dimension of word embeddings')
    parser.add_argument('--glove_dir', type=str, default='../glove', 
                        help='Directory containing GloVe pre-trained embeddings')
    parser.add_argument('--freeze_embedding', action='store_true', 
                        help='Whether to freeze embedding weights during training')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'rnn', 'rcnn'], 
                        help='Neural network architecture: CNN, RNN, or RCNN')
    parser.add_argument('--max_length', type=int, default=100, 
                        help='Maximum sequence length for input texts')
    parser.add_argument('--vocab_size', type=int, default=20000, 
                        help='Maximum vocabulary size')
    parser.add_argument('--min_freq', type=int, default=1, 
                        help='Minimum word frequency for inclusion in vocabulary')
    
    # CNN parameters
    parser.add_argument('--num_filters', type=int, default=100, 
                        help='Number of convolutional filters for CNN')
    parser.add_argument('--filter_sizes', type=str, default='1,2,3', 
                        help='Sizes of convolutional filters, separated by commas')
    
    # RNN parameters
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='Hidden dimension size for RNN')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of recurrent layers')
    parser.add_argument('--bidirectional', action='store_true', 
                        help='Whether to use bidirectional RNN architecture')
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], 
                        help='Type of recurrent cell: LSTM or GRU')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training and evaluation')
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Initial learning rate for optimizer')
    parser.add_argument('--dropout', type=float, default=0.5, 
                        help='Dropout probability for regularization')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='L2 regularization coefficient')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], 
                        help='Optimization algorithm: Adam or SGD')
    parser.add_argument('--scheduler', type=str, default=None, 
                        choices=[None, 'step', 'cosine', 'plateau'], 
                        help='Learning rate scheduler type: StepLR, CosineAnnealingLR, or ReduceLROnPlateau')
    
    # Experiment parameters
    parser.add_argument('--experiment', type=str, default=None, 
                        choices=[None, 'model_type', 'embedding_type', 'embedding_dim', 'max_length', 
                                'num_filters', 'filter_sizes', 'hidden_dim', 'num_layers', 
                                'bidirectional', 'rnn_type', 'batch_size', 'learning_rate', 
                                'dropout', 'optimizer', 'scheduler'], 
                        help='Type of experiment to run for hyperparameter comparison')
    parser.add_argument('--param_name', type=str, default=None, 
                        help='Parameter name to compare (alternative to --experiment)')
    parser.add_argument('--param_values', type=str, default=None, 
                        help='Comma-separated parameter values to compare')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true', default=True, 
                        help='Whether to use GPU acceleration if available')
    parser.add_argument('--clean_temp', action='store_true', default=True, help='Clean temporary directories after experiment')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_experiment(args):
    """
    Run experiment (single model or comparison of multiple models).
    Results and visualizations will be saved in model-specific or parameter-specific subdirectories
    under the main output directory.
    
    Args:
        args: Command line arguments
    """
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle experiment parameter
    if hasattr(args, 'experiment') and args.experiment:
        # Set param_name based on experiment type
        if not hasattr(args, 'param_name') or args.param_name is None:
            args.param_name = args.experiment
        
        # Set default param_values based on experiment type
        if not hasattr(args, 'param_values') or not args.param_values:
            # Add param_values attribute if it doesn't exist
            if args.experiment == 'model_type':
                args.param_values = 'cnn,rnn,rcnn'
            elif args.experiment == 'embedding_type':
                args.param_values = 'random,glove'
            elif args.experiment == 'embedding_dim':
                args.param_values = '50,100,200,300'
            elif args.experiment == 'filter_sizes':
                args.param_values = '1-2-3,2-3-4,3-4-5,2-4-6,3-5-7'
            elif args.experiment == 'num_filters':
                args.param_values = '50,100,150,200'
            elif args.experiment == 'hidden_dim':
                args.param_values = '64,128,256'
            elif args.experiment == 'num_layers':
                args.param_values = '1,2,3'
            elif args.experiment == 'bidirectional':
                args.param_values = 'False,True'
            elif args.experiment == 'rnn_type':
                args.param_values = 'lstm,gru'
            elif args.experiment == 'dropout':
                args.param_values = '0.3,0.4,0.5,0.6'
            elif args.experiment == 'optimizer':
                args.param_values = 'adam,sgd'
            elif args.experiment == 'scheduler':
                args.param_values = 'None,step,cosine,plateau'
    
    # Run single model or comparison experiment
    if not hasattr(args, 'param_name') or args.param_name is None:
        # Run a single model
        print(f"\nRunning single model with {args.model_type} architecture")
        model_dir = os.path.join(args.output_dir, args.model_type)
        run_single_model(args)
        print(f"\nSingle model experiment completed. Results are available in: {model_dir}")
    else:
        # Parse parameter values for comparison
        print(f"\nRunning comparison experiment for parameter: {args.param_name}")
        
        # Ensure param_values exists
        if not hasattr(args, 'param_values') or not args.param_values:
            print(f"Error: param_values is required for comparison experiment.")
            return
            
        param_values = args.param_values.split(',')
        
        # Convert parameter values to appropriate type
        if args.param_name in ['dropout', 'learning_rate', 'weight_decay']:
            param_values = [float(v) for v in param_values]
        elif args.param_name in ['embedding_dim', 'hidden_dim', 'num_filters', 'num_epochs', 
                                'batch_size', 'num_layers', 'patience', 'valid_ratio', 
                                'min_freq', 'vocab_size', 'max_length']:
            param_values = [int(v) for v in param_values]
        elif args.param_name in ['freeze_embedding', 'bidirectional']:
            # Convert 'True'/'False' strings to boolean values
            param_values = [v.lower() == 'true' for v in param_values]
        elif args.param_name == 'filter_sizes':
            # Handle special case for filter_sizes where values might contain dashes
            param_values = [v.replace('-', ',') for v in param_values]
        
        # Run comparison experiment
        param_dir = os.path.join(args.output_dir, f'compare_{args.param_name}')
        run_comparison_experiment(args, args.param_name, param_values)
        print(f"\nComparison experiment completed. Results are available in: {param_dir}")
    
    print("\nAll experiments completed successfully.")


def run_single_model(args):
    """
    Run a single model with the specified parameters
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Evaluation metrics
    """
    # Create model-specific output directory
    model_output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Create a temporary copy of args with the model-specific output directory
    new_args = argparse.Namespace(**vars(args))
    new_args.output_dir = model_output_dir
    
    # Get metrics, history and confusion matrix data
    metrics, history, cm, class_names = run_single_model_collect_data(new_args)
    
    # Create and save learning curves to model-specific output directory
    if history:
        # Create learning curves figure
        plt.figure(figsize=(15, 7))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', linestyle='-')
        if 'valid_loss' in history and len(history['valid_loss']) > 0:
            plt.plot(history['valid_loss'], label='Valid Loss', linestyle='--')
        if 'test_loss' in history and len(history['test_loss']) > 0:
            plt.plot(history['test_loss'], label='Test Loss', linestyle=':')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc', linestyle='-')
        if 'valid_acc' in history and len(history['valid_acc']) > 0:
            plt.plot(history['valid_acc'], label='Valid Acc', linestyle='--')
        if 'test_acc' in history and len(history['test_acc']) > 0:
            plt.plot(history['test_acc'], label='Test Acc', linestyle=':')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save to model-specific output directory
        curves_path = os.path.join(model_output_dir, 'learning_curves.png')
        plt.savefig(curves_path, dpi=300)
        plt.close()
        print(f"Learning curves saved to {curves_path}")
    
    # Create and save confusion matrix to model-specific output directory
    if cm is not None and class_names:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        cm_path = os.path.join(model_output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")
    
    # Save metrics to JSON file
    metrics_path = os.path.join(model_output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    return metrics


def run_single_model_collect_data(args):
    """
    Run a single model with the specified parameters and collect training data
    without saving intermediate files.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (metrics, history, confusion_matrix, class_names)
    """
    print("=" * 50)
    print(f"Running model: {args.model_type}")
    print("=" * 50)
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_processor = DataProcessor(args.data_dir)
    train_texts, train_labels, test_texts, test_labels = data_processor.load_data()
    
    # Preprocess texts
    print("Preprocessing text data...")
    train_texts = [data_processor.preprocess_text(text) for text in train_texts]
    test_texts = [data_processor.preprocess_text(text) for text in test_texts]
    
    # Split training and validation sets
    train_texts, train_labels, valid_texts, valid_labels = data_processor.split_train_valid(
        train_texts, train_labels, args.valid_ratio)
    
    print(f"Dataset sizes - Training: {len(train_texts)}, Validation: {len(valid_texts)}, Test: {len(test_texts)}")
    
    # Create tokenizer
    tokenizer = Tokenizer(min_freq=args.min_freq, max_vocab_size=args.vocab_size)
    tokenizer.build_vocab(train_texts)
    
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, valid_loader, test_loader = data_processor.create_dataloaders(
        train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels,
        tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    
    # Create embedding matrix
    embedding_matrix = None
    if args.embedding_type == 'glove':
        print(f"Loading GloVe embeddings ({args.embedding_dim}d)...")
        glove_path = os.path.join(args.glove_dir, f'glove.6B.{args.embedding_dim}d.txt')
        embedding_matrix = data_processor.create_embedding_matrix(
            tokenizer, embedding_dim=args.embedding_dim, embedding_path=glove_path)
    
    # Create model
    print(f"Initializing {args.model_type.upper()} model...")
    if args.model_type == 'cnn':
        filter_sizes = [int(fs) for fs in args.filter_sizes.split(',')]
        model = TextCNN(
            vocab_size=len(tokenizer),
            embedding_dim=args.embedding_dim,
            num_filters=args.num_filters,
            filter_sizes=filter_sizes,
            num_classes=5,  # Assume 5 classes
            dropout=args.dropout,
            embedding_matrix=embedding_matrix,
            freeze_embedding=args.freeze_embedding
        )
    elif args.model_type == 'rnn':
        model = TextRNN(
            vocab_size=len(tokenizer),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=5,  # Assume 5 classes
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            dropout=args.dropout,
            embedding_matrix=embedding_matrix,
            freeze_embedding=args.freeze_embedding,
            rnn_type=args.rnn_type
        )
    elif args.model_type == 'rcnn':
        model = TextRCNN(
            vocab_size=len(tokenizer),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=5,  # Assume 5 classes
            dropout=args.dropout,
            embedding_matrix=embedding_matrix,
            freeze_embedding=args.freeze_embedding
        )
    
    # Create optimizer
    print(f"Using {args.optimizer.upper()} optimizer with learning rate {args.learning_rate}")
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        print("Using StepLR scheduler")
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        print("Using CosineAnnealingLR scheduler")
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        print("Using ReduceLROnPlateau scheduler")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=args.output_dir,
        model_name=args.model_type
    )
    
    # Train model
    print("\nStarting training process...")
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        patience=args.patience,
        verbose=True,
        save_best=True
    )
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, f'{args.model_type}_best.pt')
    if os.path.exists(best_model_path):
        trainer.load_model(best_model_path)
        print(f"Loaded best model from {best_model_path}")
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    class_names = [str(i) for i in range(5)]  # Assume 5 classes
    metrics = trainer.evaluate_metrics(test_loader)
    
    # Get training history
    history = trainer._plot_history()
    
    # Get confusion matrix data directly
    cm, _ = trainer.plot_confusion_matrix(test_loader, class_names)
    
    # Add training time to metrics
    metrics['train_time'] = train_time
    
    return metrics, history, cm, class_names


def run_comparison_experiment(args, param_name, param_values):
    """
    Run comparison experiment with different parameter values.
    Generate visualizations and store results in a parameter-specific subdirectory.
    
    Args:
        args: Command line arguments
        param_name: Parameter name to compare
        param_values: List of parameter values to compare
        
    Returns:
        DataFrame containing the experiment results
    """
    print("=" * 50)
    print(f"Comparing different {param_name} values")
    print("=" * 50)
    
    # Create parameter-specific output directory
    param_output_dir = os.path.join(args.output_dir, f'compare_{param_name}')
    os.makedirs(param_output_dir, exist_ok=True)
    
    # Run models with different parameter values
    param_display_values = []
    accuracy_values = []
    train_times = []
    precision_values = []
    recall_values = []
    f1_values = []
    
    # Store training histories and confusion matrices for visualization
    all_histories = {}
    all_confusion_matrices = {}
    all_class_names = None
    
    for param_value in param_values:
        print(f"\nEvaluating model with {param_name} = {param_value}")
        
        # Create a temporary copy of args with the current parameter value
        new_args = argparse.Namespace(**vars(args))
        setattr(new_args, param_name, param_value)
        
        # Use a temporary directory for each model run to avoid conflicts
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set output directory to temp directory
            new_args.output_dir = temp_dir
            
            # Run model and collect data without saving intermediate files
            metrics, history, cm, class_names = run_single_model_collect_data(new_args)
            
            # Record results
            param_display_values.append(str(param_value))
            accuracy_values.append(metrics['accuracy'])
            train_times.append(metrics.get('train_time', 0))
            precision_values.append(metrics['precision'])
            recall_values.append(metrics['recall'])
            f1_values.append(metrics['f1'])
            
            # Store training history and confusion matrix for visualization
            all_histories[str(param_value)] = history
            all_confusion_matrices[str(param_value)] = cm
            all_class_names = class_names
    
    # Create and save results table
    results_df = pd.DataFrame({
        'Parameter Value': param_display_values,
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values,
        'Training Time (s)': train_times
    })
    
    # Save as CSV to parameter-specific output directory
    csv_path = os.path.join(param_output_dir, 'results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # 1. Create metrics comparison chart (bar chart with all metrics)
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.15
    index = np.arange(len(param_display_values))
    
    # Plot bars for each metric
    plt.bar(index - bar_width*1.5, accuracy_values, bar_width, label='Accuracy', color='#1f77b4')
    plt.bar(index - bar_width*0.5, precision_values, bar_width, label='Precision', color='#ff7f0e')
    plt.bar(index + bar_width*0.5, recall_values, bar_width, label='Recall', color='#2ca02c')
    plt.bar(index + bar_width*1.5, f1_values, bar_width, label='F1 Score', color='#d62728')
    
    plt.xlabel(f'{param_name.replace("_", " ").title()}')
    plt.ylabel('Score')
    plt.title(f'Performance Metrics Comparison for Different {param_name.replace("_", " ").title()} Values')
    plt.xticks(index, param_display_values)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save metrics comparison chart to parameter-specific output directory
    comparison_path = os.path.join(param_output_dir, 'metrics_comparison.png')
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"Metrics comparison chart saved to {comparison_path}")
    
    # 2. Create confusion matrix visualization if available
    if all_confusion_matrices and all_class_names:
        print(f"Creating combined confusion matrices visualization from {len(all_confusion_matrices)} matrices")
        n_classes = next(iter(all_confusion_matrices.values())).shape[0]
        if n_classes <= 10:  # Only plot if number of classes is reasonable
            n_params = len(all_confusion_matrices)
            fig_cols = min(3, n_params)
            fig_rows = (n_params + fig_cols - 1) // fig_cols
            
            plt.figure(figsize=(5*fig_cols, 4*fig_rows))
            
            for i, (param_val, cm) in enumerate(all_confusion_matrices.items()):
                plt.subplot(fig_rows, fig_cols, i+1)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                           xticklabels=all_class_names, yticklabels=all_class_names)
                plt.title(f'{param_name} = {param_val}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
            
            plt.tight_layout()
            
            # Save confusion matrices to parameter-specific output directory
            cm_path = os.path.join(param_output_dir, 'confusion_matrices.png')
            plt.savefig(cm_path, dpi=300)
            plt.close()
            print(f"Confusion matrices saved to {cm_path}")
        else:
            print(f"Warning: Number of classes ({n_classes}) is too large for visualization")
    else:
        print("Warning: No confusion matrices found for visualization")
    
    # 3. Create learning curves visualization if available
    if all_histories:
        print(f"Creating combined learning curves visualization from {len(all_histories)} histories")
        
        # Create a figure with two subplots: one for loss, one for accuracy
        plt.figure(figsize=(15, 7))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        for param_val, history in all_histories.items():
            if 'train_loss' in history and len(history['train_loss']) > 0:
                plt.plot(history['train_loss'], label=f'{param_val} - Train', linestyle='-')
            if 'valid_loss' in history and len(history['valid_loss']) > 0:
                plt.plot(history['valid_loss'], label=f'{param_val} - Valid', linestyle='--')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves for Different {param_name.replace("_", " ").title()} Values')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        for param_val, history in all_histories.items():
            if 'train_acc' in history and len(history['train_acc']) > 0:
                plt.plot(history['train_acc'], label=f'{param_val} - Train', linestyle='-')
            if 'valid_acc' in history and len(history['valid_acc']) > 0:
                plt.plot(history['valid_acc'], label=f'{param_val} - Valid', linestyle='--')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curves for Different {param_name.replace("_", " ").title()} Values')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save learning curves to parameter-specific output directory
        curves_path = os.path.join(param_output_dir, 'learning_curves.png')
        plt.savefig(curves_path, dpi=300)
        plt.close()
        print(f"Learning curves saved to {curves_path}")
    else:
        print("Warning: No learning curves found for visualization")
    
    # Find and report best parameter values
    best_accuracy_param = param_display_values[np.argmax(accuracy_values)]
    best_f1_param = param_display_values[np.argmax(f1_values)]
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"{param_name.replace('_', ' ').title()} Comparison Summary")
    print("=" * 50)
    print(f"Best Accuracy: {max(accuracy_values):.4f} with {param_name} = {best_accuracy_param}")
    print(f"Best F1 Score: {max(f1_values):.4f} with {param_name} = {best_f1_param}")
    
    return results_df


def main():
    """
    Main function
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Run experiment
    print("\nStarting experiment...")
    run_experiment(args)
    
    print("\nExperiment completed successfully!")


if __name__ == '__main__':
    main() 