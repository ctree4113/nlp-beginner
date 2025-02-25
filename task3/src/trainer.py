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
from sklearn.metrics import classification_report
from tqdm import tqdm


class Trainer:
    """Model trainer class"""
    
    def __init__(self, model, train_loader, valid_loader, test_loader, args):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            args: Training arguments
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.args = args
        
        # Set device
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set optimizer
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=args.learning_rate, 
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(), 
                lr=args.learning_rate, 
                momentum=0.9, 
                weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        
        # Set learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=args.lr_patience
        )
        
        # Set loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize best validation accuracy
        self.best_valid_acc = 0.0
        
        # Initialize early stopping counter
        self.early_stop_counter = 0
        
        # Initialize metrics history
        self.train_loss_history = []
        self.train_acc_history = []
        self.valid_loss_history = []
        self.valid_acc_history = []
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
    
    def train(self):
        """Train the model"""
        print(f"Training on {self.device}")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # Evaluate on validation set
            valid_loss, valid_acc = self._evaluate(self.valid_loader, "Validation")
            
            # Update learning rate scheduler
            self.scheduler.step(valid_acc)
            
            # Save metrics history
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.valid_loss_history.append(valid_loss)
            self.valid_acc_history.append(valid_acc)
            
            # Print metrics
            print(f"Epoch {epoch}/{self.args.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")
            print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save model if validation accuracy improved
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                self._save_model()
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # Early stopping
            if self.early_stop_counter >= self.args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self._load_model()
        
        # Evaluate on test set
        test_loss, test_acc = self._evaluate(self.test_loader, "Test")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        # Generate detailed evaluation report
        self._generate_report()
        
        # Plot metrics history
        self._plot_metrics()
    
    def _train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            epoch_loss: Average loss for the epoch
            epoch_acc: Accuracy for the epoch
        """
        self.model.train()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            premise_ids = batch['premise_ids'].to(self.device)
            hypothesis_ids = batch['hypothesis_ids'].to(self.device)
            premise_mask = batch['premise_mask'].to(self.device)
            hypothesis_mask = batch['hypothesis_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(premise_ids, hypothesis_ids, premise_mask, hypothesis_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / total:.2f}%"
            })
        
        # Calculate epoch metrics
        epoch_loss /= len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _evaluate(self, data_loader, split_name):
        """
        Evaluate the model
        
        Args:
            data_loader: Data loader for evaluation
            split_name: Name of the data split (e.g., "Validation", "Test")
            
        Returns:
            epoch_loss: Average loss for the evaluation
            epoch_acc: Accuracy for the evaluation
        """
        self.model.eval()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(data_loader, desc=f"[{split_name}]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Get batch data
                premise_ids = batch['premise_ids'].to(self.device)
                hypothesis_ids = batch['hypothesis_ids'].to(self.device)
                premise_mask = batch['premise_mask'].to(self.device)
                hypothesis_mask = batch['hypothesis_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(premise_ids, hypothesis_ids, premise_mask, hypothesis_mask)
                loss = self.criterion(logits, labels)
                
                # Update metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.0 * correct / total:.2f}%"
                })
        
        # Calculate epoch metrics
        epoch_loss /= len(data_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _save_model(self):
        """Save model checkpoint"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Save only necessary states, avoid saving the entire args object
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_valid_acc': self.best_valid_acc,
            # Don't save the complete args object, only save necessary configurations
            'config': {
                'hidden_dim': self.args.hidden_dim,
                'dropout': self.args.dropout,
                'model_type': self.args.model_type
            }
        }
        torch.save(checkpoint, os.path.join(self.args.output_dir, 'best_model.pt'))
        print(f"Model saved to {os.path.join(self.args.output_dir, 'best_model.pt')}")
    
    def _load_model(self):
        """Load model checkpoint"""
        model_path = os.path.join(self.args.output_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}, skipping model loading")
            return
            
        try:
            # Try using weights-only loading method (PyTorch 2.6 default)
            checkpoint = torch.load(model_path)
        except Exception as e:
            # If failed, try using non-weights-only loading method
            print(f"Warning: Failed to load with weights_only=True. Trying with weights_only=False: {str(e)}")
            checkpoint = torch.load(model_path, weights_only=False)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_valid_acc = checkpoint['best_valid_acc']
        print(f"Successfully loaded model with validation accuracy: {self.best_valid_acc:.4f}")
    
    def _generate_report(self):
        """Generate detailed evaluation report"""
        self.model.eval()
        
        # Collect predictions and labels
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Generating report"):
                # Get batch data
                premise_ids = batch['premise_ids'].to(self.device)
                hypothesis_ids = batch['hypothesis_ids'].to(self.device)
                premise_mask = batch['premise_mask'].to(self.device)
                hypothesis_mask = batch['hypothesis_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(premise_ids, hypothesis_ids, premise_mask, hypothesis_mask)
                _, predicted = torch.max(logits, 1)
                
                # Collect predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=['entailment', 'neutral', 'contradiction'])
        print("\nClassification Report:")
        print(report)
        
        # Save classification report
        with open(os.path.join(self.args.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['entailment', 'neutral', 'contradiction'],
                    yticklabels=['entailment', 'neutral', 'contradiction'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'confusion_matrix.png'))
    
    def _plot_metrics(self):
        """Plot training and validation metrics"""
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss_history, label='Train')
        plt.plot(self.valid_loss_history, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc_history, label='Train')
        plt.plot(self.valid_acc_history, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epoch')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'metrics.png'))


class Evaluator:
    """Model evaluator class"""
    
    def __init__(self, model, data_loader, device):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            device: Device to use for evaluation
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate(self):
        """
        Evaluate the model
        
        Returns:
            accuracy: Evaluation accuracy
            predictions: Model predictions
            labels: True labels
        """
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc="Evaluating"):
                # Get batch data
                premise_ids = batch['premise_ids'].to(self.device)
                hypothesis_ids = batch['hypothesis_ids'].to(self.device)
                premise_mask = batch['premise_mask'].to(self.device)
                hypothesis_mask = batch['hypothesis_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(premise_ids, hypothesis_ids, premise_mask, hypothesis_mask)
                _, predicted = torch.max(logits, 1)
                
                # Update metrics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = correct / total
        
        return accuracy, np.array(all_preds), np.array(all_labels)
    
    def visualize_attention(self, premise_text, hypothesis_text, tokenizer):
        """
        Visualize attention weights
        
        Args:
            premise_text: Premise text
            hypothesis_text: Hypothesis text
            tokenizer: Tokenizer for tokenizing text
            
        Note: This is a placeholder for attention visualization.
        The actual implementation would depend on the model architecture
        and how attention weights are accessed.
        """
        # This is a placeholder for attention visualization
        # In a real implementation, you would:
        # 1. Tokenize the texts
        # 2. Convert to model inputs
        # 3. Forward pass with attention weight extraction
        # 4. Visualize the attention weights
        
        print("Attention visualization not implemented yet.")
        print(f"Premise: {premise_text}")
        print(f"Hypothesis: {hypothesis_text}") 