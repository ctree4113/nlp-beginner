import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import json
import pandas as pd
import seaborn as sns

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
    parser.add_argument('--feature_type', type=str, default='bow', choices=['bow', 'binary', 'tfidf'], 
                        help='Feature type: bow(Bag of Words), binary(Binary Features), tfidf(TF-IDF)')
    parser.add_argument('--ngram_min', type=int, default=1, help='Minimum N-gram value')
    parser.add_argument('--ngram_max', type=int, default=1, help='Maximum N-gram value')
    parser.add_argument('--max_features', type=int, default=5000, help='Maximum number of features')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='softmax', choices=['logistic', 'softmax'], 
                        help='Model type: logistic(Logistic Regression), softmax(Softmax Regression)')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--num_iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--regularization', type=str, default='l2', choices=[None, 'l1', 'l2'], 
                        help='Regularization type: None, l1, l2')
    parser.add_argument('--lambda_param', type=float, default=0.001, help='Regularization parameter')
    parser.add_argument('--optimizer', type=str, default='gd', 
                        choices=['gd', 'sgd', 'mini-batch', 'momentum'], 
                        help='Optimizer: gd(Gradient Descent), sgd(Stochastic Gradient Descent), mini-batch(Mini-batch GD), momentum(Momentum GD)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum coefficient for momentum optimizer')
    
    # Experiment parameters
    parser.add_argument('--experiment', type=str, default=None, 
                        choices=[None, 'model_type', 'feature_type', 'ngram', 'learning_rate', 'regularization', 'batch_size', 'optimizer'], 
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
        batch_sizes = [1, 8, 32, 128, 512, None]  # None means full batch
        run_comparison_experiment(args, 'batch_size', batch_sizes)
    elif args.experiment == 'optimizer':
        # Compare different optimization methods
        optimizers = ['gd', 'sgd', 'mini-batch', 'momentum']
        run_comparison_experiment(args, 'optimizer', optimizers)

def run_single_model(args):
    """Run single model"""
    print("=" * 50)
    print(f"Running single model: {args.model_type}")
    print("=" * 50)
    
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
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(valid_texts)}")
    print(f"Test set size: {len(test_texts)}")
    
    # Extract features
    print("\nExtracting features...")
    feature_extractor = FeatureExtractor(
        feature_type=args.feature_type,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features
    )
    
    # Transform data
    X_train = feature_extractor.fit_transform(train_texts)
    X_valid = feature_extractor.transform(valid_texts)
    X_test = feature_extractor.transform(test_texts)
    
    print(f"Number of features: {X_train.shape[1]}")
    
    # Create model
    if args.model_type == 'logistic':
        model = LogisticRegression(
            learning_rate=args.learning_rate,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
            regularization=args.regularization,
            lambda_param=args.lambda_param,
            optimizer=args.optimizer,
            momentum=args.momentum,
            verbose=True
        )
    else:
        model = SoftmaxRegression(
            learning_rate=args.learning_rate,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
            regularization=args.regularization,
            lambda_param=args.lambda_param,
            optimizer=args.optimizer,
            momentum=args.momentum,
            verbose=True
        )
    
    # Train model with validation and test data for tracking accuracy
    print(f"\nTraining model with {args.optimizer} optimizer...")
    start_time = time.time()
    model.fit(X_train, np.array(train_labels), X_valid, np.array(valid_labels), X_test, np.array(test_labels))
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")
    
    # Evaluate model
    evaluator = Evaluator(output_dir)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    valid_pred = model.predict(X_valid)
    valid_metrics = evaluator.evaluate(np.array(valid_labels), valid_pred)
    evaluator.print_metrics(valid_metrics)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_pred = model.predict(X_test)
    test_metrics = evaluator.evaluate(np.array(test_labels), test_pred)
    evaluator.print_metrics(test_metrics)
    
    # Plot loss curve
    evaluator.plot_loss_curve(model.loss_history, title=f"{args.model_type} Model Loss Curve ({args.optimizer})")
    
    # Plot accuracy curves if available
    if hasattr(model, 'train_accuracy_history') and len(model.train_accuracy_history) > 0:
        # Plot training accuracy curve
        plt.figure(figsize=(10, 6))
        record_interval = model.record_interval
        iterations = [i * record_interval for i in range(len(model.train_accuracy_history))]
        plt.plot(iterations, model.train_accuracy_history, label='Training Accuracy')
        
        # Plot validation accuracy curve if available
        if hasattr(model, 'valid_accuracy_history') and len(model.valid_accuracy_history) > 0:
            plt.plot(iterations, model.valid_accuracy_history, label='Validation Accuracy')
        
        # Plot test accuracy curve if available
        if hasattr(model, 'test_accuracy_history') and len(model.test_accuracy_history) > 0:
            plt.plot(iterations, model.test_accuracy_history, label='Test Accuracy')
        
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title(f"{args.model_type} Model Accuracy Curves ({args.optimizer})")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.0)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'))
    
    # Plot confusion matrix
    class_names = [str(i) for i in range(5)]  # Classes 0-4
    evaluator.plot_confusion_matrix(np.array(test_labels), test_pred, class_names=class_names)
    
    # Save results
    results = {
        'args': vars(args),
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'train_time': train_time,
        'feature_count': X_train.shape[1]
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_dir}")
    
    return {**test_metrics, 'train_time': train_time}

def run_comparison_experiment(args, param_name, param_values):
    """Run comparison experiment"""
    print("=" * 50)
    print(f"Comparing different {param_name} values")
    print("=" * 50)
    
    # Create output directory - ensure directory name is safe
    safe_param_name = str(param_name).replace(':', '_')
    output_dir = os.path.join(args.output_dir, f"compare_{safe_param_name}")
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
    
    # Store models for training process comparison
    models = {}
    
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
        
        # Run model
        print(f"\n{param_name} = {param_display}")
        
        # Ensure directory name is safe, replace characters that may cause path issues
        safe_param_display = str(param_display).replace(':', '_').replace('(', '').replace(')', '').replace(',', '_')
        
        # Create output directory for this parameter value
        param_output_dir = os.path.join(output_dir, f"{param_name}_{safe_param_display}")
        os.makedirs(param_output_dir, exist_ok=True)
        
        # Extract features
        print("\nExtracting features...")
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
                optimizer=new_args.optimizer,
                momentum=new_args.momentum,
                verbose=True
            )
        else:
            model = SoftmaxRegression(
                learning_rate=new_args.learning_rate,
                num_iterations=new_args.num_iterations,
                batch_size=new_args.batch_size,
                regularization=new_args.regularization,
                lambda_param=new_args.lambda_param,
                optimizer=new_args.optimizer,
                momentum=new_args.momentum,
                verbose=True
            )
        
        # Train model with validation and test data for tracking accuracy
        print(f"\nTraining model with {new_args.optimizer} optimizer...")
        start_time = time.time()
        model.fit(X_train, np.array(train_labels), X_valid, np.array(valid_labels), X_test, np.array(test_labels))
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")
        
        # Store model for later comparison
        models[param_display] = model
        
        # Evaluate model
        evaluator = Evaluator(param_output_dir)
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        valid_pred = model.predict(X_valid)
        valid_metrics = evaluator.evaluate(np.array(valid_labels), valid_pred)
        evaluator.print_metrics(valid_metrics)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_pred = model.predict(X_test)
        test_metrics = evaluator.evaluate(np.array(test_labels), test_pred)
        evaluator.print_metrics(test_metrics)
        
        # Plot loss curve
        evaluator.plot_loss_curve(model.loss_history, title=f"{new_args.model_type} Model Loss Curve ({new_args.optimizer})")
        
        # Plot accuracy curves if available
        if hasattr(model, 'train_accuracy_history') and len(model.train_accuracy_history) > 0:
            # Plot training accuracy curve
            plt.figure(figsize=(10, 6))
            record_interval = model.record_interval
            iterations = [i * record_interval for i in range(len(model.train_accuracy_history))]
            plt.plot(iterations, model.train_accuracy_history, label='Training Accuracy')
            
            # Plot validation accuracy curve if available
            if hasattr(model, 'valid_accuracy_history') and len(model.valid_accuracy_history) > 0:
                plt.plot(iterations, model.valid_accuracy_history, label='Validation Accuracy')
            
            # Plot test accuracy curve if available
            if hasattr(model, 'test_accuracy_history') and len(model.test_accuracy_history) > 0:
                plt.plot(iterations, model.test_accuracy_history, label='Test Accuracy')
            
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.title(f"{new_args.model_type} Model Accuracy Curves ({new_args.optimizer})")
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1.0)
            
            # Save plot
            plt.savefig(os.path.join(param_output_dir, 'accuracy_curves.png'))
        
        # Plot confusion matrix
        class_names = [str(i) for i in range(5)]  # Classes 0-4
        evaluator.plot_confusion_matrix(np.array(test_labels), test_pred, class_names=class_names)
        
        # Record results
        metrics = test_metrics.copy()
        metrics['train_time'] = train_time
        model_metrics[param_display] = metrics
        param_display_values.append(param_display)
        accuracy_values.append(metrics['accuracy'])
        train_times.append(metrics['train_time'])
        precision_values.append(metrics['precision'])
        recall_values.append(metrics['recall'])
        f1_values.append(metrics['f1'])
        
        # Save detailed results for this parameter value
        detailed_results = {
            'parameter_name': param_name,
            'parameter_value': param_display,
            'args': vars(new_args),
            'metrics': metrics
        }
        
        # Ensure file name is safe, replace characters that may cause path issues
        detailed_result_path = os.path.join(detailed_dir, f"{param_name}_{safe_param_display}.json")
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(detailed_result_path), exist_ok=True)
        
        with open(detailed_result_path, 'w') as f:
            json.dump(detailed_results, f, indent=4)
    
    # Create evaluator for comparison
    evaluator = Evaluator(output_dir)
    
    # Compare training process (loss curves and accuracy curves)
    evaluator.plot_training_process_comparison(models, title_prefix=f"{param_name} Comparison: ")
    
    # Compare performance of different parameters
    evaluator.compare_models(model_metrics, metric_name='accuracy', 
                           title=f'Comparison of accuracy across different {param_name} values')
    evaluator.compare_models(model_metrics, metric_name='f1', 
                           title=f'Comparison of F1 score across different {param_name} values')
    
    # Plot combined metrics in a single figure
    combined_metrics = {
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values,
        'train_time': train_times
    }
    evaluator.plot_combined_metrics(param_display_values, combined_metrics, param_name,
                                  title=f'All Metrics Comparison for different {param_name} values')
    
    # Plot radar chart for metrics comparison
    evaluator.plot_metrics_radar(model_metrics, 
                               title=f'Performance Metrics Comparison for different {param_name} values')
    
    # Plot parameter influence curve for numeric parameters only
    if param_name == 'learning_rate' or param_name == 'lambda_param' or (param_name == 'batch_size' and None not in param_values):
        # For numeric parameters, plot curve
        numeric_param_values = [float(val) if val is not None else 0 for val in param_values]
        evaluator.compare_hyperparameters(numeric_param_values, accuracy_values, 
                                        param_name, metric_name='accuracy')
    
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
    
    # Create a more detailed visualization of all metrics
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
    plt.title(f'F1 Score across different {param_name} values')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_comparison.png'))
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(param_display_values, train_times, color='purple')
    plt.title(f'Training Time across different {param_name} values')
    plt.ylabel('Training Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'))
    
    # Create a heatmap for all metrics
    metrics_df = pd.DataFrame({
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values
    }, index=param_display_values)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5)
    plt.title(f'Performance Metrics Heatmap for different {param_name} values')
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
            'train_time': {param: metrics['train_time'] for param, metrics in model_metrics.items()}
        }
    }
    
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Find and report the best parameter value for each metric
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
    print(f"Comparison Summary for {param_name}")
    print("=" * 50)
    print(f"Best Accuracy: {max(accuracy_values):.4f} with {param_name} = {best_accuracy_param}")
    print(f"Best Precision: {max(precision_values):.4f} with {param_name} = {best_precision_param}")
    print(f"Best Recall: {max(recall_values):.4f} with {param_name} = {best_recall_param}")
    print(f"Best F1 Score: {max(f1_values):.4f} with {param_name} = {best_f1_param}")
    print(f"Fastest Training: {min(train_times):.2f}s with {param_name} = {fastest_param}")
    print("=" * 50)
    
    print(f"\nComparison results saved to {output_dir}")
    print(f"Detailed results for each parameter value saved to {detailed_dir}")

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory - ensure directory name is safe
    # Only replace characters that cause issues, but preserve path separators
    safe_output_dir = str(args.output_dir).replace('\\', '/').replace(':', '_')
    if safe_output_dir != args.output_dir:
        print(f"Warning: Output directory contains special characters. Using '{safe_output_dir}' instead of '{args.output_dir}'")
        args.output_dir = safe_output_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    run_experiment(args) 