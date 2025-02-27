import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


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
        
        # Setup output directory
        self.output_dir = os.path.join(args.output_dir, args.language)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup best model tracking
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
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
        best_model_path = os.path.join(self.output_dir, "best_model.pt")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            
            # Evaluate on dev set
            dev_metrics = self.evaluate(dev_loader)
            precision, recall, f1 = dev_metrics['precision'], dev_metrics['recall'], dev_metrics['f1']
            
            # Print epoch summary
            print(f"Epoch {epoch}/{self.args.epochs} - "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
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
                
                print(f"New best model saved at epoch {epoch} with F1 score: {f1:.2f}%")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
                
                # Early stopping
                if self.args.early_stopping > 0 and self.patience_counter >= self.args.early_stopping:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(f1)
        
        print(f"Training completed. Best model at epoch {self.best_epoch} with F1 score: {self.best_f1:.2f}%")
        
        # Evaluate on test set if provided
        if test_loader and os.path.exists(best_model_path):
            print("Evaluating best model on test set...")
            self._load_checkpoint(best_model_path)
            test_metrics = self.evaluate(test_loader)
            precision, recall, f1 = test_metrics['precision'], test_metrics['recall'], test_metrics['f1']
            print(f"Test metrics - Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}%")
        
        return best_model_path
    
    def _train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for the epoch
            accuracy: Training accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
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
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = self.model.predict(word_ids, mask, seq_lengths, char_ids)
                for i, length in enumerate(seq_lengths):
                    pred_tags = predictions[i][:length]
                    true_tags = tag_ids[i][:length].cpu().numpy()
                    
                    # Count correct predictions
                    correct_preds += sum(p == t for p, t in zip(pred_tags, true_tags))
                    total_preds += length
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/(i+1), acc=f"{(correct_preds/max(1, total_preds))*100:.2f}%")
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        accuracy = (correct_preds / max(1, total_preds)) * 100
        
        return avg_loss, accuracy
    
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
        
        # Extract entities from tags
        for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
            true_entities.extend(self._extract_entities(true_tags))
            pred_entities.extend(self._extract_entities(pred_tags))
        
        # Calculate correct predictions
        correct_predictions = [entity for entity in pred_entities if entity in true_entities]
        
        # Calculate precision
        if not pred_entities:
            precision = 0.0
        else:
            precision = len(correct_predictions) / len(pred_entities)
        
        # Calculate recall
        if not true_entities:
            recall = 0.0
        else:
            recall = len(correct_predictions) / len(true_entities)
        
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