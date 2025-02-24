import os
import argparse
import torch
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_processor import DataProcessor, Tokenizer
from models import TextCNN, TextRNN, TextRCNN
from trainer import Trainer, Evaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Text Classification Based on Deep Learning')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../dataset', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='../output', help='Output directory')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='Validation set ratio')
    
    # Embedding parameters
    parser.add_argument('--embedding_type', type=str, default='random', choices=['random', 'glove'], 
                        help='Embedding type: random(random initialization), glove(pre-trained embeddings)')
    parser.add_argument('--embedding_dim', type=int, default=100, 
                        choices=[50, 100, 200, 300], help='Embedding dimension')
    parser.add_argument('--glove_dir', type=str, default='../glove', help='GloVe embeddings directory')
    parser.add_argument('--freeze_embedding', action='store_true', help='Whether to freeze embeddings')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'rnn', 'rcnn'], 
                        help='Model type: cnn(TextCNN), rnn(TextRNN), rcnn(TextRCNN)')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, default=20000, help='Vocabulary size')
    parser.add_argument('--min_freq', type=int, default=1, help='Minimum word frequency')
    
    # CNN parameters
    parser.add_argument('--num_filters', type=int, default=100, help='Number of filters')
    parser.add_argument('--filter_sizes', type=str, default='1,2,3', help='Filter sizes, separated by commas')
    
    # RNN parameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'], 
                        help='RNN type: lstm(LSTM), gru(GRU)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], 
                        help='Optimizer: adam(Adam), sgd(SGD)')
    parser.add_argument('--scheduler', type=str, default=None, 
                        choices=[None, 'step', 'cosine', 'plateau'], 
                        help='Learning rate scheduler: step(StepLR), cosine(CosineAnnealingLR), plateau(ReduceLROnPlateau)')
    
    # Experiment parameters
    parser.add_argument('--experiment', type=str, default=None, 
                        choices=[None, 'model_type', 'embedding_type', 'embedding_dim', 'max_length', 
                                'num_filters', 'filter_sizes', 'hidden_dim', 'num_layers', 
                                'bidirectional', 'rnn_type', 'batch_size', 'learning_rate', 
                                'dropout', 'optimizer'], 
                        help='Experiment type')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='Whether to use CUDA')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_experiment(args):
    """Run experiment"""
    if args.experiment is None:
        # Run single model
        run_single_model(args)
    elif args.experiment == 'model_type':
        # Compare different model types
        model_types = ['cnn', 'rnn', 'rcnn']
        run_comparison_experiment(args, 'model_type', model_types)
    elif args.experiment == 'embedding_type':
        # Compare different embedding types
        embedding_types = ['random', 'glove']
        run_comparison_experiment(args, 'embedding_type', embedding_types)
    elif args.experiment == 'embedding_dim':
        # Compare different embedding dimensions
        embedding_dims = [50, 100, 200, 300]
        run_comparison_experiment(args, 'embedding_dim', embedding_dims)
    elif args.experiment == 'max_length':
        # Compare different maximum sequence lengths
        max_lengths = [50, 100, 150, 200]
        run_comparison_experiment(args, 'max_length', max_lengths)
    elif args.experiment == 'num_filters':
        # Compare different numbers of filters
        num_filters = [50, 100, 150, 200]
        run_comparison_experiment(args, 'num_filters', num_filters)
    elif args.experiment == 'filter_sizes':
        # Compare different filter sizes
        filter_sizes = ['2,3,4', '3,4,5', '4,5,6', '3,5,7']
        run_comparison_experiment(args, 'filter_sizes', filter_sizes)
    elif args.experiment == 'hidden_dim':
        # Compare different hidden dimensions
        hidden_dims = [64, 128, 256, 512]
        run_comparison_experiment(args, 'hidden_dim', hidden_dims)
    elif args.experiment == 'num_layers':
        # Compare different numbers of RNN layers
        num_layers = [1, 2, 3, 4]
        run_comparison_experiment(args, 'num_layers', num_layers)
    elif args.experiment == 'bidirectional':
        # Compare whether to use bidirectional RNN
        bidirectional = [False, True]
        run_comparison_experiment(args, 'bidirectional', bidirectional)
    elif args.experiment == 'rnn_type':
        # Compare different RNN types
        rnn_types = ['lstm', 'gru']
        run_comparison_experiment(args, 'rnn_type', rnn_types)
    elif args.experiment == 'batch_size':
        # Compare different batch sizes
        batch_sizes = [16, 32, 64, 128]
        run_comparison_experiment(args, 'batch_size', batch_sizes)
    elif args.experiment == 'learning_rate':
        # Compare different learning rates
        learning_rates = [0.0001, 0.001, 0.01, 0.1]
        run_comparison_experiment(args, 'learning_rate', learning_rates)
    elif args.experiment == 'dropout':
        # Compare different dropout rates
        dropouts = [0.0, 0.3, 0.5, 0.7]
        run_comparison_experiment(args, 'dropout', dropouts)
    elif args.experiment == 'optimizer':
        # Compare different optimizers
        optimizers = ['adam', 'sgd']
        run_comparison_experiment(args, 'optimizer', optimizers)


def run_single_model(args):
    """Run single model"""
    print("=" * 50)
    print(f"Running single model: {args.model_type}")
    print("=" * 50)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{args.embedding_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_processor = DataProcessor(args.data_dir)
    train_texts, train_labels, test_texts, test_labels = data_processor.load_data()
    
    # Preprocess texts
    train_texts = [data_processor.preprocess_text(text) for text in train_texts]
    test_texts = [data_processor.preprocess_text(text) for text in test_texts]
    
    # Split training and validation sets
    train_texts, train_labels, valid_texts, valid_labels = data_processor.split_train_valid(
        train_texts, train_labels, args.valid_ratio)
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(valid_texts)}")
    print(f"Test set size: {len(test_texts)}")
    
    # Create tokenizer
    tokenizer = Tokenizer(min_freq=args.min_freq, max_vocab_size=args.vocab_size)
    tokenizer.build_vocab(train_texts)
    
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Create data loaders
    train_loader, valid_loader, test_loader = data_processor.create_dataloaders(
        train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels,
        tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    
    # Create embedding matrix
    embedding_matrix = None
    if args.embedding_type == 'glove':
        glove_path = os.path.join(args.glove_dir, f'glove.6B.{args.embedding_dim}d.txt')
        embedding_matrix = data_processor.create_embedding_matrix(
            tokenizer, embedding_dim=args.embedding_dim, embedding_path=glove_path)
    
    # Create model
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
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        model_name=args.model_type
    )
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        patience=args.patience,
        verbose=True,
        save_best=True
    )
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Evaluate model
    evaluator = Evaluator(output_dir)
    
    # Load best model
    best_model_path = os.path.join(output_dir, f'{args.model_type}_best.pt')
    if os.path.exists(best_model_path):
        trainer.load_model(best_model_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    class_names = [str(i) for i in range(5)]  # Assume 5 classes
    metrics = evaluator.evaluate(trainer, test_loader, class_names)
    
    # Save results
    results = {
        'args': vars(args),
        'metrics': metrics,
        'train_time': train_time,
        'vocab_size': len(tokenizer)
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
        param_output_dir = os.path.join(output_dir, f"{param_name}_{param_value}")
        os.makedirs(param_output_dir, exist_ok=True)
        
        # Set output directory
        new_args.output_dir = param_output_dir
        
        # Run model
        metrics = run_single_model(new_args)
        
        # Record results
        model_metrics[str(param_value)] = metrics
        param_display_values.append(str(param_value))
        accuracy_values.append(metrics['accuracy'])
        train_times.append(metrics.get('train_time', 0))
        precision_values.append(metrics['precision'])
        recall_values.append(metrics['recall'])
        f1_values.append(metrics['f1'])
        
        # Save detailed results
        detailed_results = {
            'parameter_name': param_name,
            'parameter_value': param_value,
            'args': vars(new_args),
            'metrics': metrics
        }
        
        with open(os.path.join(detailed_dir, f"{param_name}_{param_value}.json"), 'w') as f:
            json.dump(detailed_results, f, indent=4)
    
    # Create evaluator
    evaluator = Evaluator(output_dir)
    
    # Compare performance across different parameters
    evaluator.compare_models(model_metrics, metric_name='accuracy', 
                           title=f'Accuracy comparison across different {param_name} values')
    evaluator.compare_models(model_metrics, metric_name='f1', 
                           title=f'F1 score comparison across different {param_name} values')
    
    # Create and save detailed comparison table
    comparison_data = {
        'Parameter Value': param_display_values,
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values,
        'Training Time (s)': train_times
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)
    
    # Create more detailed visualizations for all metrics
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy, precision, recall, and F1 score
    plt.subplot(2, 2, 1)
    plt.bar(param_display_values, accuracy_values, color='blue')
    plt.title(f'Accuracy across different {param_name} values')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.bar(param_display_values, precision_values, color='green')
    plt.title(f'Precision across different {param_name} values')
    plt.ylabel('Precision')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    plt.bar(param_display_values, recall_values, color='orange')
    plt.title(f'Recall across different {param_name} values')
    plt.ylabel('Recall')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    plt.bar(param_display_values, f1_values, color='red')
    plt.title(f'F1 score across different {param_name} values')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_comparison.png'))
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(param_display_values, train_times, color='purple')
    plt.title(f'Training time across different {param_name} values')
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
    plt.title(f'Performance metrics heatmap across different {param_name} values')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
    
    # Save results
    results = {
        'args': vars(args),
        'param_name': param_name,
        'param_values': [str(val) for val in param_values],
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
    
    # Find and report best parameter values for each metric
    best_accuracy_param = param_display_values[np.argmax(accuracy_values)]
    best_precision_param = param_display_values[np.argmax(precision_values)]
    best_recall_param = param_display_values[np.argmax(recall_values)]
    best_f1_param = param_display_values[np.argmax(f1_values)]
    fastest_param = param_display_values[np.argmin(train_times)]
    
    best_params = {
        'best_accuracy': {'param': best_accuracy_param, 'value': max(accuracy_values)},
        'best_precision': {'param': best_precision_param, 'value': max(precision_values)},
        'best_recall': {'param': best_recall_param, 'value': max(recall_values)},
        'best_f1': {'param': best_f1_param, 'value': max(f1_values)},
        'fastest_training': {'param': fastest_param, 'value': min(train_times)}
    }
    
    with open(os.path.join(output_dir, 'best_parameters.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"{param_name} Comparison Summary")
    print("=" * 50)
    print(f"Best Accuracy: {max(accuracy_values):.4f} with {param_name} = {best_accuracy_param}")
    print(f"Best Precision: {max(precision_values):.4f} with {param_name} = {best_precision_param}")
    print(f"Best Recall: {max(recall_values):.4f} with {param_name} = {best_recall_param}")
    print(f"Best F1 Score: {max(f1_values):.4f} with {param_name} = {best_f1_param}")
    print(f"Fastest Training Time: {min(train_times):.2f}s with {param_name} = {fastest_param}")
    print("=" * 50)
    
    print(f"\nComparison results saved to {output_dir}")
    print(f"Detailed results for each parameter value saved to {detailed_dir}")


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    run_experiment(args) 