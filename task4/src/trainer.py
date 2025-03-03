import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
import csv


class Trainer:
    """
    Trainer for BiLSTM-CRF model
    """
    def __init__(self, model, tag_map, args):
        """
        Initialize trainer
        
        Args:
            model: BiLSTM-CRF model
            tag_map: Mapping from tags to ids
            args: Training arguments
        """
        self.model = model
        self.tag_map = tag_map
        self.id_to_tag = {v: k for k, v in tag_map.items()}
        self.args = args
        
        # Setup device
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer with gradient clipping
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Setup LR scheduler if specified
        self.scheduler = None
        if args.use_lr_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=args.scheduler_factor,
                patience=args.scheduler_patience
            )
            # Save initial learning rate
            self.last_lr = args.lr
        
        # Setup output directory
        self.output_dir = os.path.join(args.output_dir, args.language)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup best model tracking
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Setup metrics tracking
        self.train_losses = []
        self.train_f1s = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        
        # Setup training time tracking
        self.training_start_time = None
        self.training_end_time = None
    
    def train(self, train_loader, dev_loader, test_loader=None):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            dev_loader: DataLoader for development data
            test_loader: DataLoader for test data
            
        Returns:
            best_model_path: Path to the best model checkpoint
        """
        print(f"Training on {self.device}")
        best_model_path = os.path.join(self.output_dir, "best_model.pt")
        
        # Record training start time
        self.training_start_time = time.time()
        
        for epoch in range(1, self.args.epochs + 1):
            # Train for one epoch
            train_loss, train_f1 = self._train_epoch(train_loader, epoch)
            
            # Record training metrics
            self.train_losses.append(train_loss)
            self.train_f1s.append(train_f1)
            
            # Evaluate on dev set
            dev_metrics = self.evaluate(dev_loader)
            precision, recall, f1 = dev_metrics['precision'], dev_metrics['recall'], dev_metrics['f1']
            
            # Record validation metrics (storing the original 0-1 scaled values, not percentages)
            self.val_precisions.append(precision / 100)  # Convert from percentage to ratio
            self.val_recalls.append(recall / 100)        # Convert from percentage to ratio
            self.val_f1s.append(f1 / 100)                # Convert from percentage to ratio
            
            # Print epoch summary
            print(f"Epoch {epoch}/{self.args.epochs} - "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train F1: {train_f1:.2f}% | "
                  f"Validation: P: {precision:.2f}%, R: {recall:.2f}%, F1: {f1:.2f}%")
            
            # Update best model
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': self.best_f1,
                    'metrics': dev_metrics
                }, best_model_path)
                
                print(f"New best model saved with F1 score: {f1:.2f}%")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
                
                # Early stopping
                if self.args.early_stopping > 0 and self.patience_counter >= self.args.early_stopping:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Update learning rate
            if self.scheduler:
                old_lr = self.last_lr
                self.scheduler.step(f1)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Check if learning rate changed
                if current_lr != old_lr:
                    print(f"Learning rate reduced from {old_lr:.6f} to {current_lr:.6f}")
                    self.last_lr = current_lr
                
            # Save learning curves
            self._plot_learning_curves()
        
        # Record training end time
        self.training_end_time = time.time()
        training_time = self.training_end_time - self.training_start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Best model at epoch {self.best_epoch} with F1 score: {self.best_f1:.2f}%")
        
        # Evaluate on test set if provided
        if test_loader and os.path.exists(best_model_path):
            print("Evaluating best model on test set...")
            self._load_checkpoint(best_model_path)
            test_metrics = self.evaluate(test_loader)
            precision, recall, f1 = test_metrics['precision'], test_metrics['recall'], test_metrics['f1']
            print(f"Test metrics - Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}%")
            
            # Save test metrics
            self._save_test_metrics(test_metrics, training_time)
        
        return best_model_path
    
    def _train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for the epoch
            f1: Training F1 score
        """
        self.model.train()
        total_loss = 0.0
        
        all_predictions = []
        all_true_tags = []
        
        # Create progress bar
        progress_bar = tqdm(total=len(train_loader), 
                            desc=f"Epoch {epoch}/{self.args.epochs} [Train]",
                            bar_format='{l_bar}{bar:30}{r_bar}')
        
        for i, batch in enumerate(train_loader):
            # Get batch data
            word_ids = batch['word_ids'].to(self.device)
            tag_ids = batch['tag_ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            seq_lengths = batch['seq_lengths']
            
            # Get character-level features if available
            char_ids = batch.get('char_ids', None)
            if char_ids is not None:
                char_ids = char_ids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model(word_ids, tag_ids, mask, seq_lengths, char_ids)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            # Update parameters
            self.optimizer.step()
            
            # Collect predictions and true tags for entity-level metrics
            with torch.no_grad():
                predictions = self.model.predict(word_ids, mask, seq_lengths, char_ids)
                true_tags = tag_ids.cpu().numpy()
                
                for i, length in enumerate(seq_lengths):
                    pred_tags = predictions[i][:length]
                    true_tags_i = true_tags[i][:length]
                    
                    all_predictions.append([self.id_to_tag[tag] for tag in pred_tags])
                    all_true_tags.append([self.id_to_tag[tag] for tag in true_tags_i])
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/(i+1))
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        
        # Calculate entity-level metrics
        precision, recall, f1 = self._calculate_entity_metrics(all_true_tags, all_predictions)
        
        # Update progress bar with entity metrics
        print(f"Train metrics - Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%")
        
        return avg_loss, f1 * 100
    
    def evaluate(self, data_loader):
        """
        Evaluate model on data
        
        Args:
            data_loader: DataLoader for evaluation data
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_true_tags = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
                # Get batch data
                word_ids = batch['word_ids'].to(self.device)
                tag_ids = batch['tag_ids'].to(self.device)
                mask = batch['mask'].to(self.device)
                seq_lengths = batch['seq_lengths']
                
                # Get character-level features if available
                char_ids = batch.get('char_ids', None)
                if char_ids is not None:
                    char_ids = char_ids.to(self.device)
                
                # Get predictions
                predictions = self.model.predict(word_ids, mask, seq_lengths, char_ids)
                
                # Convert tag_ids to list of lists
                true_tags = tag_ids.cpu().numpy()
                
                # Collect predictions and true tags
                for i, length in enumerate(seq_lengths):
                    pred_tags = predictions[i][:length]
                    true_tags_i = true_tags[i][:length]
                    
                    all_predictions.append([self.id_to_tag[tag] for tag in pred_tags])
                    all_true_tags.append([self.id_to_tag[tag] for tag in true_tags_i])
        
        # Calculate entity-level metrics
        precision, recall, f1 = self._calculate_entity_metrics(all_true_tags, all_predictions)
        
        return {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    def _calculate_entity_metrics(self, true_tags_list, pred_tags_list):
        """
        Calculate entity-level precision, recall, and F1 score
        
        Args:
            true_tags_list: List of lists of true tags
            pred_tags_list: List of lists of predicted tags
            
        Returns:
            precision: Entity-level precision
            recall: Entity-level recall
            f1: Entity-level F1 score
        """
        true_entities = []
        pred_entities = []
        
        # Extract entities from tags and keep track of which sentence they belong to
        for sent_idx, (true_tags, pred_tags) in enumerate(zip(true_tags_list, pred_tags_list)):
            true_ents = [(entity[0], entity[1], entity[2], sent_idx) for entity in self._extract_entities(true_tags)]
            pred_ents = [(entity[0], entity[1], entity[2], sent_idx) for entity in self._extract_entities(pred_tags)]
            true_entities.extend(true_ents)
            pred_entities.extend(pred_ents)
        
        # Create sets to avoid duplicate counting
        unique_true_entities = set([(e[0], e[1], e[2], e[3]) for e in true_entities])
        unique_pred_entities = set([(e[0], e[1], e[2], e[3]) for e in pred_entities])
        
        # Calculate correct predictions (intersection of sets)
        correct_pred_set = unique_true_entities.intersection(unique_pred_entities)
        
        # Calculate precision
        if not unique_pred_entities:
            precision = 0.0
        else:
            precision = len(correct_pred_set) / len(unique_pred_entities)
        
        # Calculate recall
        if not unique_true_entities:
            recall = 0.0
        else:
            recall = len(correct_pred_set) / len(unique_true_entities)
        
        # Calculate F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return precision, recall, f1
    
    def _extract_entities(self, tags):
        """
        Extract entities from a sequence of BIO tags
        
        Args:
            tags: List of BIO tags
            
        Returns:
            entities: List of (entity_type, start, end) tuples
        """
        entities = []
        entity_type = None
        start = None
        
        for i, tag in enumerate(tags):
            if tag == 'O':
                if entity_type is not None:
                    entities.append((entity_type, start, i - 1))
                    entity_type = None
                    start = None
            elif tag.startswith('B-'):
                if entity_type is not None:
                    entities.append((entity_type, start, i - 1))
                entity_type = tag[2:]  # Remove 'B-'
                start = i
            elif tag.startswith('I-'):
                current_type = tag[2:]  # Remove 'I-'
                if entity_type is None:
                    # This is an invalid I- tag (not preceded by B- or I-)
                    # Treat it as a B- tag
                    entity_type = current_type
                    start = i
                elif current_type != entity_type:
                    # This is an invalid I- tag (different type from previous)
                    # End the current entity and start a new one
                    entities.append((entity_type, start, i - 1))
                    entity_type = current_type
                    start = i
        
        # Handle the case where the last tag is part of an entity
        if entity_type is not None:
            entities.append((entity_type, start, len(tags) - 1))
        
        return entities
    
    def _load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint 
    
    def _plot_learning_curves(self):
        """
        Generate and save learning curve plots
        """
        # Create metrics directory
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        epochs = list(range(1, len(self.train_losses) + 1))
        
        # Create a figure with two y-axes
        fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()
        
        # Plot training loss on the first y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)
        line1 = ax1.plot(epochs, self.train_losses, color=color, marker='o', linestyle='-', label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create a second y-axis for F1 score
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('F1 Score (%)', color=color)
        line2 = ax2.plot(epochs, self.train_f1s, color=color, marker='s', linestyle='-', label='Training F1 Score')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        
        # Add title
        plt.title('Training Loss and F1 Score')
        
        # Add combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'training_curves.png'))
        plt.close()
        
        # Plot validation metrics (ensure values are between 0-100%)
        plt.figure(figsize=(10, 6))
        
        # Convert ratios back to percentages for plotting
        val_precisions_pct = [p * 100 for p in self.val_precisions]
        val_recalls_pct = [r * 100 for r in self.val_recalls]
        val_f1s_pct = [f * 100 for f in self.val_f1s]
        
        plt.plot(epochs, val_precisions_pct, 'r-', marker='o', label='Precision')
        plt.plot(epochs, val_recalls_pct, 'g-', marker='s', label='Recall')
        plt.plot(epochs, val_f1s_pct, 'b-', marker='^', label='F1 Score')
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Score (%)')
        
        # Set y-axis limits to ensure display between 0-100%
        plt.ylim(0, 100)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, 'validation_curves.png'))
        plt.close()
    
    def _save_test_metrics(self, test_metrics, training_time):
        """
        Save metrics in CSV format, combining test metrics, validation info, and summary
        
        Args:
            test_metrics: Test metrics dictionary
            training_time: Total training time in seconds
        """
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Format training time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        training_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Prepare CSV data
        csv_file = os.path.join(metrics_dir, 'metrics.csv')
        
        # Define CSV header and data
        fieldnames = [
            # Experiment info
            'language', 'timestamp', 'total_epochs', 'best_model_epoch', 
            # Training metrics
            'training_time_seconds', 'training_time_formatted',
            'final_train_loss', 'final_train_f1',
            # Validation metrics
            'best_val_precision', 'best_val_recall', 'best_val_f1', 
            # Test metrics
            'test_precision', 'test_recall', 'test_f1',
            # Model info
            'model_params', 'hidden_dim', 'num_layers', 'dropout'
        ]
        
        data = {
            # Experiment info
            'language': self.args.language,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_epochs': len(self.train_losses),
            'best_model_epoch': self.best_epoch,
            
            # Training metrics
            'training_time_seconds': f"{training_time:.2f}",
            'training_time_formatted': training_time_str,
            'final_train_loss': f"{self.train_losses[-1]:.4f}",
            'final_train_f1': f"{self.train_f1s[-1]:.2f}%",
            
            # Validation metrics (best values)
            'best_val_precision': f"{self.val_precisions[self.best_epoch-1] * 100:.2f}%",
            'best_val_recall': f"{self.val_recalls[self.best_epoch-1] * 100:.2f}%",
            'best_val_f1': f"{self.val_f1s[self.best_epoch-1] * 100:.2f}%",
            
            # Test metrics
            'test_precision': f"{test_metrics['precision']:.2f}%",
            'test_recall': f"{test_metrics['recall']:.2f}%",
            'test_f1': f"{test_metrics['f1']:.2f}%",
            
            # Model info
            'model_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'hidden_dim': self.args.hidden_dim,
            'num_layers': self.args.num_layers,
            'dropout': self.args.dropout
        }
        
        # Write to CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)
        
        print(f"All metrics saved to {csv_file}")
        
        # Save validation metrics history to separate CSV (for possible analysis)
        history_file = os.path.join(metrics_dir, 'validation_history.csv')
        
        # Write validation metrics history
        with open(history_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'loss', 'train_f1', 'precision', 'recall', 'f1'])
            
            for i in range(len(self.train_losses)):
                writer.writerow([
                    i+1, 
                    f"{self.train_losses[i]:.4f}",
                    f"{self.train_f1s[i]:.2f}",
                    f"{self.val_precisions[i] * 100:.2f}",
                    f"{self.val_recalls[i] * 100:.2f}",
                    f"{self.val_f1s[i] * 100:.2f}"
                ])
        
        print(f"Validation history saved to {history_file}") 