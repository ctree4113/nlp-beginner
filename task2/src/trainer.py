import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    """Model trainer class"""
    
    def __init__(self, model, device, criterion=None, optimizer=None, scheduler=None, 
                 output_dir='../output', model_name='model'):
        """
        Initialize trainer
        
        Args:
            model: Model
            device: Device
            criterion: Loss function, default is CrossEntropyLoss
            optimizer: Optimizer, default is Adam
            scheduler: Learning rate scheduler, default is None
            output_dir: Output directory
            model_name: Model name
        """
        self.model = model
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else optim.Adam(model.parameters())
        self.scheduler = scheduler
        self.output_dir = output_dir
        self.model_name = model_name
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Record training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_acc': [],
            'test_loss': [],
            'test_acc': []
        }
    
    def train(self, train_loader, valid_loader=None, test_loader=None, num_epochs=10, 
              patience=5, verbose=True, save_best=True):
        """
        Train model
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            num_epochs: Number of training epochs
            patience: Early stopping patience
            verbose: Whether to print training progress
            save_best: Whether to save the best model
            
        Returns:
            Training history
        """
        # Move model to device
        self.model.to(self.device)
        
        # Record best validation loss and epoch
        best_valid_loss = float('inf')
        best_epoch = 0
        
        # Record start time
        start_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            # Train one epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Record training loss and accuracy
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Evaluate on validation set
            if valid_loader is not None:
                valid_loss, valid_acc = self._evaluate(valid_loader)
                self.history['valid_loss'].append(valid_loss)
                self.history['valid_acc'].append(valid_acc)
            else:
                valid_loss, valid_acc = None, None
            
            # Evaluate on test set
            if test_loader is not None:
                test_loss, test_acc = self._evaluate(test_loader)
                self.history['test_loss'].append(test_loss)
                self.history['test_acc'].append(test_acc)
            else:
                test_loss, test_acc = None, None
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_loss)
                else:
                    self.scheduler.step()
            
            # Print training progress
            if verbose:
                elapsed_time = time.time() - start_time
                print(f'Epoch {epoch+1}/{num_epochs} - Time: {elapsed_time:.2f}s')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                if valid_loader is not None:
                    print(f'  Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
                if test_loader is not None:
                    print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            
            # Save best model
            if valid_loader is not None and save_best and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                self._save_model(os.path.join(self.output_dir, f'{self.model_name}_best.pt'))
            
            # Early stopping
            if valid_loader is not None and epoch - best_epoch >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Save last model
        if save_best:
            self._save_model(os.path.join(self.output_dir, f'{self.model_name}_last.pt'))
        
        # Save training history
        self._save_history()
        
        # Plot training curves
        self._plot_history()
        
        return self.history
    
    def _train_epoch(self, train_loader):
        """
        Train one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            epoch_loss: Average loss for this epoch
            epoch_acc: Accuracy for this epoch
        """
        # Set to training mode
        self.model.train()
        
        # Record loss and accuracy
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Iterate over batches
        for batch in train_loader:
            # Get inputs and labels
            token_ids = batch['token_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(token_ids)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Record loss and accuracy
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # Calculate average loss and accuracy
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        
        return epoch_loss, epoch_acc
    
    def _evaluate(self, data_loader):
        """
        Evaluate model
        
        Args:
            data_loader: Data loader
            
        Returns:
            avg_loss: Average loss
            accuracy: Accuracy
        """
        # Set to evaluation mode
        self.model.eval()
        
        # Record loss and accuracy
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # No gradient calculation
        with torch.no_grad():
            # Iterate over batches
            for batch in data_loader:
                # Get inputs and labels
                token_ids = batch['token_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(token_ids)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Record loss and accuracy
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def predict(self, data_loader):
        """
        Predict
        
        Args:
            data_loader: Data loader
            
        Returns:
            predictions: Prediction results
            true_labels: True labels
            probabilities: Prediction probabilities
        """
        # Set to evaluation mode
        self.model.eval()
        
        # Record predictions and true labels
        predictions = []
        true_labels = []
        probabilities = []
        
        # No gradient calculation
        with torch.no_grad():
            # Iterate over batches
            for batch in data_loader:
                # Get inputs and labels
                token_ids = batch['token_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(token_ids)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Record predictions and true labels
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(true_labels), np.array(probabilities)
    
    def evaluate_metrics(self, data_loader):
        """
        Evaluate model and calculate various metrics
        
        Args:
            data_loader: Data loader
            
        Returns:
            metrics: Evaluation metrics dictionary
        """
        # Predict
        predictions, true_labels, _ = self.predict(data_loader)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='macro'),
            'recall': recall_score(true_labels, predictions, average='macro'),
            'f1': f1_score(true_labels, predictions, average='macro'),
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
        }
        
        return metrics
    
    def _save_model(self, path):
        """
        Save model
        
        Args:
            path: Save path
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None
        }, path)
    
    def load_model(self, path):
        """
        Load model
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _save_history(self):
        """Save training history"""
        with open(os.path.join(self.output_dir, f'{self.model_name}_history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def _plot_history(self):
        """Plot training curves"""
        # Create figure
        plt.figure(figsize=(12, 4))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'valid_loss' in self.history and len(self.history['valid_loss']) > 0:
            plt.plot(self.history['valid_loss'], label='Valid Loss')
        if 'test_loss' in self.history and len(self.history['test_loss']) > 0:
            plt.plot(self.history['test_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        if 'valid_acc' in self.history and len(self.history['valid_acc']) > 0:
            plt.plot(self.history['valid_acc'], label='Valid Acc')
        if 'test_acc' in self.history and len(self.history['test_acc']) > 0:
            plt.plot(self.history['test_acc'], label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_curves.png'))
        plt.close()
    
    def plot_confusion_matrix(self, data_loader, class_names=None):
        """
        Plot confusion matrix
        
        Args:
            data_loader: Data loader
            class_names: Class names
        """
        # Predict
        predictions, true_labels, _ = self.predict(data_loader)
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{self.model_name}_confusion_matrix.png'))
        plt.close()


class Evaluator:
    """Model evaluator class"""
    
    def __init__(self, output_dir='../output'):
        """
        Initialize evaluator
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(self, trainer, data_loader, class_names=None):
        """
        Evaluate model
        
        Args:
            trainer: Trainer
            data_loader: Data loader
            class_names: Class names
            
        Returns:
            metrics: Evaluation metrics dictionary
        """
        # Evaluate metrics
        metrics = trainer.evaluate_metrics(data_loader)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Plot confusion matrix
        trainer.plot_confusion_matrix(data_loader, class_names)
        
        # Save metrics
        with open(os.path.join(self.output_dir, f'{trainer.model_name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print evaluation metrics
        
        Args:
            metrics: Evaluation metrics dictionary
        """
        print('Evaluation Metrics:')
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    def compare_models(self, model_metrics, metric_name='accuracy', title=None):
        """
        Compare different models' performance
        
        Args:
            model_metrics: Model metrics dictionary, format {model_name: metrics}
            metric_name: Metric name to compare
            title: Chart title
        """
        # Extract metrics
        model_names = list(model_metrics.keys())
        metric_values = [metrics[metric_name] for metrics in model_metrics.values()]
        
        # Plot bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, metric_values, color='skyblue')
        plt.xlabel('Model')
        plt.ylabel(metric_name.capitalize())
        plt.title(title or f'Comparison of {metric_name} across different models')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'model_comparison_{metric_name}.png'))
        plt.close()
    
    def compare_hyperparameters(self, param_values, metric_values, param_name, metric_name='accuracy'):
        """
        Compare performance across different hyperparameter values
        
        Args:
            param_values: Hyperparameter values list
            metric_values: Metric values list
            param_name: Hyperparameter name
            metric_name: Metric name
        """
        # Plot line chart
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, metric_values, 'o-', color='blue')
        plt.xlabel(param_name)
        plt.ylabel(metric_name.capitalize())
        plt.title(f'Effect of {param_name} on {metric_name}')
        plt.grid(True)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(param_values, metric_values)):
            plt.text(x, y + 0.01, f'{y:.4f}', ha='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{param_name}_comparison.png'))
        plt.close() 