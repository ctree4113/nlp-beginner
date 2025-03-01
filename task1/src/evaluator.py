import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os

class Evaluator:
    """Model evaluator"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize evaluator
        
        Args:
            output_dir: Output directory for saving evaluation results
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # For multi-class problems, use macro average
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Return evaluation metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Print evaluation metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("Model Evaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str] = None, title: str = "Confusion Matrix") -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
            title: Plot title
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        
        # Save plot
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        
        plt.show()
    
    def plot_loss_curve(self, loss_history: List[float], title: str = "Loss Curve") -> None:
        """
        Plot loss curve
        
        Args:
            loss_history: Loss history
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid(True)
        
        # Save plot
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'loss_curve.png'))
        
        plt.show()
    
    def compare_models(self, model_metrics: Dict[str, Dict[str, float]], 
                      metric_name: str = 'accuracy', title: str = None) -> None:
        """
        Compare different models' performance
        
        Args:
            model_metrics: Dictionary of model metrics, format is {model_name: metrics}
            metric_name: Metric name to compare
            title: Plot title
        """
        model_names = list(model_metrics.keys())
        metric_values = [metrics[metric_name] for metrics in model_metrics.values()]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel(metric_name)
        plt.title(title or f'Comparison of {metric_name} across models')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'model_comparison_{metric_name}.png'))
        
        plt.show()
    
    def compare_hyperparameters(self, param_values: List[Any], metrics: List[float], 
                               param_name: str, metric_name: str = 'accuracy', 
                               title: str = None) -> None:
        """
        Compare performance across different hyperparameter values
        
        Args:
            param_values: List of hyperparameter values
            metrics: List of corresponding evaluation metrics
            param_name: Hyperparameter name
            metric_name: Metric name
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, metrics, 'o-')
        plt.xlabel(param_name)
        plt.ylabel(metric_name)
        plt.title(title or f'Effect of {param_name} on {metric_name}')
        plt.grid(True)
        
        # Mark best point
        best_idx = np.argmax(metrics)
        best_param = param_values[best_idx]
        best_metric = metrics[best_idx]
        plt.scatter([best_param], [best_metric], color='red', s=100, zorder=5)
        plt.annotate(f'Best: {best_param}\n{metric_name}: {best_metric:.4f}',
                    (best_param, best_metric), xytext=(10, -20),
                    textcoords='offset points', color='red')
        
        # Save plot
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'{param_name}_comparison.png'))
        
        plt.show()
    
    def compare_loss_curves(self, model_loss_histories: Dict[str, List[float]], 
                           title: str = "Loss Curves Comparison") -> None:
        """
        Compare loss curves from different models
        
        Args:
            model_loss_histories: Dictionary of loss histories, format is {model_name: loss_history}
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, loss_history in model_loss_histories.items():
            # Plot every 10 points to avoid overcrowding
            iterations = list(range(0, len(loss_history), 10))
            if len(iterations) == 0:
                iterations = list(range(len(loss_history)))
                losses = loss_history
            else:
                losses = [loss_history[i] for i in iterations]
            
            plt.plot(iterations, losses, label=model_name)
        
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # Save plot
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'loss_curves_comparison.png'))
        
        plt.show()
    
    def compare_accuracy_curves(self, model_accuracy_histories: Dict[str, List[float]], 
                               accuracy_type: str = "train", 
                               title: str = None) -> None:
        """
        Compare accuracy curves from different models
        
        Args:
            model_accuracy_histories: Dictionary of accuracy histories, format is {model_name: accuracy_history}
            accuracy_type: Type of accuracy ('train', 'valid', or 'test')
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, acc_history in model_accuracy_histories.items():
            # Get iterations based on record interval (assumed to be 10)
            record_interval = 10
            iterations = [i * record_interval for i in range(len(acc_history))]
            
            plt.plot(iterations, acc_history, label=model_name)
        
        plt.xlabel('Iterations')
        plt.ylabel(f'{accuracy_type.capitalize()} Accuracy')
        plt.title(title or f'{accuracy_type.capitalize()} Accuracy Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.0)
        
        # Save plot
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'{accuracy_type}_accuracy_curves_comparison.png'))
        
        plt.show()
    
    def plot_training_process_comparison(self, models: Dict[str, Any], 
                                        title_prefix: str = "") -> None:
        """
        Plot comprehensive comparison of training process for different models
        
        Args:
            models: Dictionary of models, format is {model_name: model}
            title_prefix: Prefix for plot titles
        """
        # Compare loss curves
        loss_histories = {name: model.loss_history for name, model in models.items()}
        self.compare_loss_curves(loss_histories, 
                                title=f"{title_prefix}Loss Curves Comparison")
        
        # Compare training accuracy curves
        train_acc_histories = {name: model.train_accuracy_history for name, model in models.items() 
                              if hasattr(model, 'train_accuracy_history') and len(model.train_accuracy_history) > 0}
        if train_acc_histories:
            self.compare_accuracy_curves(train_acc_histories, "train", 
                                        title=f"{title_prefix}Training Accuracy Curves Comparison")
        
        # Compare validation accuracy curves
        valid_acc_histories = {name: model.valid_accuracy_history for name, model in models.items() 
                              if hasattr(model, 'valid_accuracy_history') and len(model.valid_accuracy_history) > 0}
        if valid_acc_histories:
            self.compare_accuracy_curves(valid_acc_histories, "valid", 
                                        title=f"{title_prefix}Validation Accuracy Curves Comparison")
        
        # Compare test accuracy curves
        test_acc_histories = {name: model.test_accuracy_history for name, model in models.items() 
                             if hasattr(model, 'test_accuracy_history') and len(model.test_accuracy_history) > 0}
        if test_acc_histories:
            self.compare_accuracy_curves(test_acc_histories, "test", 
                                        title=f"{title_prefix}Test Accuracy Curves Comparison")
    
    def plot_metrics_radar(self, model_metrics: Dict[str, Dict[str, float]], 
                          title: str = "Model Performance Comparison") -> None:
        """
        Plot radar chart comparing different metrics across models
        
        Args:
            model_metrics: Dictionary of model metrics, format is {model_name: metrics}
            title: Plot title
        """
        # Get model names and metrics
        model_names = list(model_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics
        N = len(metrics)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Plot for each model
        for i, model_name in enumerate(model_names):
            # Get values for current model
            values = [model_metrics[model_name][metric] for metric in metrics]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        plt.title(title)
        plt.legend(loc='upper right')
        
        # Save plot
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'metrics_radar_chart.png'))
        
        plt.show()
    
    def plot_combined_metrics(self, param_values: List[str], metrics: Dict[str, List[float]], 
                             param_name: str, title: str = "Performance Metrics Comparison") -> None:
        """
        Plot all metrics (accuracy, precision, recall, f1, training time) in a single figure
        
        Args:
            param_values: List of parameter values
            metrics: Dictionary of metrics lists, format is {metric_name: [values]}
            param_name: Parameter name
            title: Plot title
        """
        plt.figure(figsize=(15, 10))
        
        # Create a line plot for all metrics
        metric_names = [m for m in metrics.keys() if m != 'train_time']
        
        # Plot metrics
        for i, metric in enumerate(metric_names):
            plt.plot(param_values, metrics[metric], 'o-', label=metric)
        
        # Add labels and title
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.0)
        
        # Create a twin axis for training time
        if 'train_time' in metrics:
            ax2 = plt.twinx()
            ax2.plot(param_values, metrics['train_time'], 'r--', label='Training Time')
            ax2.set_ylabel('Training Time (s)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Add training time to legend
            lines, labels = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        # Save plot
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'combined_metrics_comparison.png'))
        
        plt.show() 