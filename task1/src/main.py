import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_processor import DataProcessor
from feature_extractor import FeatureExtractor
from model import LogisticRegression, SoftmaxRegression
from evaluator import Evaluator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Text Classification Based on Machine Learning')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../dataset', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='../output', help='Output directory')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='Validation set ratio')
    
    # Feature parameters
    parser.add_argument('--feature_type', type=str, default='tfidf', choices=['bow', 'binary', 'tfidf'], 
                        help='Feature type: bow(Bag of Words), binary(Binary Features), tfidf(TF-IDF)')
    parser.add_argument('--ngram_min', type=int, default=1, help='Minimum N-gram value')
    parser.add_argument('--ngram_max', type=int, default=2, help='Maximum N-gram value')
    parser.add_argument('--max_features', type=int, default=5000, help='Maximum number of features')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='softmax', choices=['logistic', 'softmax'], 
                        help='Model type: logistic(Logistic Regression), softmax(Softmax Regression)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--num_iterations', type=int, default=2000, help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--regularization', type=str, default='l2', choices=[None, 'l1', 'l2'], 
                        help='Regularization type: None, l1, l2')
    parser.add_argument('--lambda_param', type=float, default=0.001, help='Regularization parameter')
    parser.add_argument('--batch_strategy', type=str, default='mini-batch', 
                        choices=['full-batch', 'stochastic', 'mini-batch'], 
                        help='Batch strategy: full-batch, stochastic, mini-batch')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        choices=['sgd', 'momentum'], 
                        help='Optimizer: sgd(Stochastic Gradient Descent), momentum(Momentum GD)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum coefficient for momentum optimizer')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'squared_error', 'hinge'], 
                        help='Loss function: cross_entropy, squared_error, hinge')
    parser.add_argument('--shuffle', type=bool, default=True, 
                        help='Whether to shuffle data during training')
    
    # Experiment parameters
    parser.add_argument('--experiment', type=str, default=None, 
                        choices=[None, 'model_type', 'feature_type', 'ngram', 'learning_rate', 
                                 'regularization', 'batch_size', 'batch_strategy', 'optimizer', 'loss_function', 'shuffle'], 
                        help='Experiment type')
    
    return parser.parse_args()

def run_experiment(args):
    """Run experiment"""
    if args.experiment is None:
        # Run single model
        run_single_model(args)
    elif args.experiment == 'model_type':
        # Compare different model types
        model_types = ['logistic', 'softmax']
        run_comparison_experiment(args, 'model_type', model_types)
    elif args.experiment == 'feature_type':
        # Compare different feature types
        feature_types = ['bow', 'binary', 'tfidf']
        run_comparison_experiment(args, 'feature_type', feature_types)
    elif args.experiment == 'ngram':
        # Compare different N-gram ranges
        ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]
        run_comparison_experiment(args, 'ngram', ngram_ranges)
    elif args.experiment == 'learning_rate':
        # Compare different learning rates
        learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]
        run_comparison_experiment(args, 'learning_rate', learning_rates)
    elif args.experiment == 'regularization':
        # Compare different regularization methods
        regularizations = [None, 'l1', 'l2']
        run_comparison_experiment(args, 'regularization', regularizations)
    elif args.experiment == 'batch_size':
        # Compare different batch sizes
        batch_sizes = [32, 64, 128, 256, 512, None]  # None means full batch
        run_comparison_experiment(args, 'batch_size', batch_sizes)
    elif args.experiment == 'batch_strategy':
        # Compare different batch strategies
        batch_strategies = ['full-batch', 'stochastic', 'mini-batch']
        run_comparison_experiment(args, 'batch_strategy', batch_strategies)
    elif args.experiment == 'optimizer':
        # Compare different optimization methods
        optimizers = ['sgd', 'momentum']
        run_comparison_experiment(args, 'optimizer', optimizers)
    elif args.experiment == 'loss_function':
        # Compare different loss functions
        loss_functions = ['cross_entropy', 'squared_error', 'hinge']
        run_comparison_experiment(args, 'loss_function', loss_functions)
    elif args.experiment == 'shuffle':
        # Compare shuffle vs no-shuffle
        shuffle_options = [True, False]
        run_comparison_experiment(args, 'shuffle', shuffle_options)

def create_model(args):
    """Create model based on arguments"""
    if args.model_type == 'logistic':
        model = LogisticRegression(
            learning_rate=args.learning_rate,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
            regularization=args.regularization,
            lambda_param=args.lambda_param,
            batch_strategy=args.batch_strategy,
            optimizer=args.optimizer,
            momentum=args.momentum,
            loss_function=args.loss_function,
            shuffle=args.shuffle
        )
    else:  # softmax
        model = SoftmaxRegression(
            learning_rate=args.learning_rate,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
            regularization=args.regularization,
            lambda_param=args.lambda_param,
            batch_strategy=args.batch_strategy,
            optimizer=args.optimizer,
            momentum=args.momentum,
            loss_function=args.loss_function,
            shuffle=args.shuffle
        )
    
    return model

def run_single_model(args):
    """Run a single model"""
    # Create output directory - ensure directory name is safe
    safe_model_type = str(args.model_type).replace(':', '_')
    safe_feature_type = str(args.feature_type).replace(':', '_')
    output_dir = os.path.join(args.output_dir, f"{safe_model_type}_{safe_feature_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_processor = DataProcessor(args.data_dir)
    train_texts, train_labels, test_texts, test_labels = data_processor.load_data()
    
    # Preprocess text
    train_texts = [data_processor.preprocess_text(text) for text in train_texts]
    test_texts = [data_processor.preprocess_text(text) for text in test_texts]
    
    # Split training and validation sets
    train_texts, train_labels, valid_texts, valid_labels = data_processor.split_train_valid(
        train_texts, train_labels, args.valid_ratio)
    
    print("Training set size:", len(train_texts))
    print("Validation set size:", len(valid_texts))
    print("Test set size:", len(test_texts))
    print()
    
    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor(
        feature_type=args.feature_type,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features
    )
    X_train = feature_extractor.fit_transform(train_texts)
    X_valid = feature_extractor.transform(valid_texts)
    X_test = feature_extractor.transform(test_texts)
    print("Number of features:", X_train.shape[1])
    
    # Create and train model
    print("Training model...")
    model = create_model(args)
    model.fit(X_train, np.array(train_labels), X_valid, np.array(valid_labels), X_test, np.array(test_labels))
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = Evaluator(model, feature_extractor)
    evaluator.evaluate(test_texts, test_labels)
    
    # Save results
    save_results(args, model, evaluator, output_dir)
    
    return model

def run_comparison_experiment(args, param_name, param_values):
    """
    Run comparison experiment with optimized parameters
    
    Args:
        args: Command line arguments
        param_name: Parameter name to compare
        param_values: List of parameter values to compare
    """
    print("=" * 50)
    print(f"Comparing different {param_name} values")
    print("=" * 50)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"compare_{param_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data once for all experiments
    data_processor = DataProcessor(args.data_dir)
    train_texts, train_labels, test_texts, test_labels = data_processor.load_data()
    
    # Preprocess text
    train_texts = [data_processor.preprocess_text(text) for text in train_texts]
    test_texts = [data_processor.preprocess_text(text) for text in test_texts]
    
    # Split training and validation sets
    train_texts, train_labels, valid_texts, valid_labels = data_processor.split_train_valid(
        train_texts, train_labels, args.valid_ratio)
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(valid_texts)}")
    print(f"Test set size: {len(test_texts)}")
    
    # Initialize metrics storage
    model_metrics = {}
    param_display_values = []
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    train_times = []
    
    # Store models for visualization
    models = {}
    confusion_matrices = {}
    
    for param_value in param_values:
        # Set parameter
        new_args = argparse.Namespace(**vars(args))
        if param_name == 'ngram':
            new_args.ngram_min = param_value[0]
            new_args.ngram_max = param_value[1]
            param_display = f"({param_value[0]},{param_value[1]})"
        else:
            setattr(new_args, param_name, param_value)
            param_display = str(param_value)
        
        print(f"\n{'-'*30}")
        print(f"Running with {param_name} = {param_display}")
        print(f"{'-'*30}")
        
        # Extract features
        print("Extracting features...")
        feature_extractor = FeatureExtractor(
            feature_type=new_args.feature_type,
            ngram_range=(new_args.ngram_min, new_args.ngram_max),
            max_features=new_args.max_features
        )
        
        # Transform data
        X_train = feature_extractor.fit_transform(train_texts)
        X_valid = feature_extractor.transform(valid_texts)
        X_test = feature_extractor.transform(test_texts)
        
        print(f"Number of features: {X_train.shape[1]}")
        
        # Create model
        if new_args.model_type == 'logistic':
            model = LogisticRegression(
                learning_rate=new_args.learning_rate,
                num_iterations=new_args.num_iterations,
                batch_size=new_args.batch_size,
                regularization=new_args.regularization,
                lambda_param=new_args.lambda_param,
                batch_strategy=new_args.batch_strategy,
                optimizer=new_args.optimizer,
                momentum=new_args.momentum,
                loss_function=new_args.loss_function,
                shuffle=new_args.shuffle,
                verbose=True
            )
        else:
            model = SoftmaxRegression(
                learning_rate=new_args.learning_rate,
                num_iterations=new_args.num_iterations,
                batch_size=new_args.batch_size,
                regularization=new_args.regularization,
                lambda_param=new_args.lambda_param,
                batch_strategy=new_args.batch_strategy,
                optimizer=new_args.optimizer,
                momentum=new_args.momentum,
                loss_function=new_args.loss_function,
                shuffle=new_args.shuffle,
                verbose=True
            )
        
        # Train model
        print(f"Training model...")
        start_time = time.time()
        model.fit(X_train, np.array(train_labels), X_valid, np.array(valid_labels), X_test, np.array(test_labels))
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")
        
        # Store model for visualization
        models[param_display] = model
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_pred = model.predict(X_test)
        evaluator = Evaluator(model, feature_extractor)
        test_metrics = evaluator.evaluate(X=X_test, y=np.array(test_labels))
        
        # Store confusion matrix
        cm = confusion_matrix(np.array(test_labels), test_pred)
        confusion_matrices[param_display] = cm
        
        # Print metrics
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        
        # Record results
        model_metrics[param_display] = test_metrics
        param_display_values.append(param_display)
        accuracy_values.append(test_metrics['accuracy'])
        precision_values.append(test_metrics['precision'])
        recall_values.append(test_metrics['recall'])
        f1_values.append(test_metrics['f1'])
        train_times.append(train_time)
    
    # Create evaluator for visualization
    evaluator = Evaluator(output_dir)
    
    # 1. Plot combined learning curves (loss and accuracy)
    plt.figure(figsize=(12, 10))
    
    # Plot loss curves
    plt.subplot(2, 1, 1)
    for name, model in models.items():
        # Sample loss history to avoid overcrowding
        sample_rate = max(1, len(model.loss_history) // 100)
        iterations = list(range(0, len(model.loss_history), sample_rate))
        losses = [model.loss_history[i] for i in iterations]
        plt.plot(iterations, losses, label=f"{param_name}={name}")
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves for Different {param_name} Values')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curves
    plt.subplot(2, 1, 2)
    for name, model in models.items():
        if hasattr(model, 'valid_accuracy_history') and len(model.valid_accuracy_history) > 0:
            record_interval = model.record_interval
            iterations = [i * record_interval for i in range(len(model.valid_accuracy_history))]
            plt.plot(iterations, model.valid_accuracy_history, label=f"{param_name}={name}")
    
    plt.xlabel('Iterations')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Validation Accuracy Curves for Different {param_name} Values')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    
    # 2. Plot metrics comparison
    plt.figure(figsize=(12, 8))
    
    metrics_df = pd.DataFrame({
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values
    }, index=param_display_values)
    
    # Plot metrics as grouped bar chart
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f'Performance Metrics for Different {param_name} Values')
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    
    # 3. Plot confusion matrices
    class_names = [str(i) for i in range(5)]  # Classes 0-4
    
    # Determine grid size based on number of parameters
    n_params = len(param_values)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 4*n_rows))
    
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        plt.subplot(n_rows, n_cols, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{param_name}={name}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    
    # Save results to CSV
    results_df = pd.DataFrame({
        f'{param_name}': param_display_values,
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values,
        'Training Time (s)': train_times
    })
    
    results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    
    # Find best parameter value
    best_idx = np.argmax(accuracy_values)
    best_param = param_display_values[best_idx]
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Comparison Summary for {param_name}")
    print("=" * 50)
    print(f"Best Accuracy: {accuracy_values[best_idx]:.4f} with {param_name} = {best_param}")
    print(f"Best F1 Score: {f1_values[best_idx]:.4f} with {param_name} = {best_param}")
    print("=" * 50)
    
    print(f"\nResults saved to {output_dir}")

def save_results(args, model, evaluator, output_dir):
    """Save experiment results"""
    # Get metrics
    metrics = evaluator.get_metrics()
    
    # Save results to JSON
    results = {
        'args': vars(args),
        'metrics': metrics,
        'feature_count': model.weights.shape[1] if hasattr(model, 'weights') and model.weights is not None else 0
    }
    
    # Save to file
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    run_experiment(args) 