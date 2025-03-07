import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils import epoch_time, calculate_perplexity
import math
import torch.nn.functional as F
import copy
from collections import defaultdict
import random

# Import mixed precision training support
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    """
    Enhanced model trainer
    """
    def __init__(self, model, vocab_size, idx_to_char, args):
        """
        Initialize the trainer with advanced training strategies
        
        Args:
            model: Model to train
            vocab_size: Size of the vocabulary
            idx_to_char: Mapping from indices to characters
            args: Command line arguments
        """
        self.model = model
        self.vocab_size = vocab_size
        self.idx_to_char = idx_to_char
        self.args = args
        
        # Special token index - according to the definition in data_processor.py
        self.pad_idx = 0  # Index of <PAD> is 0
        self.unk_idx = 1  # Index of <UNK> is 1
        self.bos_idx = 2  # Index of <BOS> is 2
        self.eos_idx = 3  # Index of <EOS> is 3
        self.line_idx = 4  # Index of <LINE> is 4
        
        # Gradient centralization setting
        self.grad_centralization = False  # Default is off
        
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.output_dir = args.output_dir
        self.patience = args.patience if hasattr(args, 'patience') else 10
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize exponential moving average model if requested
        self.use_ema = getattr(args, 'use_ema', False)
        self.ema_decay = 0.9999 if hasattr(args, 'ema_decay') else 0.9999
        if self.use_ema:
            print(f"Using EMA with decay {self.ema_decay}")
            self.ema_model = copy.deepcopy(self.model)
        
        # Training setting
        self.epochs = args.epochs
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=getattr(args, 'label_smoothing', 0.0))
        
        # Optimizer setting
        optimizer_type = getattr(args, 'optimizer', 'adam').lower()
        lr = getattr(args, 'lr', 0.001)
        weight_decay = getattr(args, 'weight_decay', 0.0)
        
        # Create optimizer
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            print(f"Using AdamW optimizer with weight decay: {weight_decay}")
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler_type = getattr(args, 'scheduler', 'none').lower()
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        elif scheduler_type == 'cosine':
            warmup_epochs = getattr(args, 'warmup_epochs', 0)
            if warmup_epochs > 0:
                print(f"Using cosine scheduler with {warmup_epochs} warmup epochs")
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)
        else:
            self.scheduler = None
            
        if scheduler_type == 'cosine':
            print("Using cosine scheduler")
        
        self.patience = getattr(args, 'patience', 5)  # Default patience value is 5
        self.grad_clip = getattr(args, 'grad_clip', 0.0)  # Default is not to clip gradients
        if self.grad_clip > 0:
            print(f"Using gradient clipping with value: {self.grad_clip}")
        self.log_interval = getattr(args, 'log_interval', 10)  # Default is to output log every 10 batches
        self.grad_accum_steps = 1  # Default gradient accumulation steps is 1
        
        # Set loss function
        self.label_smoothing = 0.0
        if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
            self.label_smoothing = args.label_smoothing
            print(f"Using label smoothing with factor: {self.label_smoothing}")
            self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=self.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Set special token weight
        self.special_token_weight = 2.0
        
        # Enable mixed precision training
        self.use_amp = getattr(args, 'use_amp', False)
        self.scaler = None
        if self.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler()
            print("Using mixed precision training")
        
        # Create placeholders for tracking training statistics
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        
        # Print training setup
        print(f"Training on {self.device}")
        
        # Set random seed for reproducibility
        if hasattr(args, 'seed'):
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            import random
            import numpy as np
            random.seed(args.seed)
            np.random.seed(args.seed)
        
        # Set metrics tracking
        self.learning_rates = []
        
        # Advanced overfitting detection
        self.train_val_loss_ratio_history = []
        self.min_train_val_ratio = 1.0
        self.avg_train_val_ratio = 1.0
        
        # Create char_to_idx (reverse mapping from idx_to_char)
        self.char_to_idx = {char: idx for idx, char in idx_to_char.items()}
        
        # Focal loss gamma parameter (if using focal loss)
        self.focal_gamma = args.focal_gamma if hasattr(args, 'focal_gamma') else 0.0
    
    def train(self, train_loader, val_loader):
        """
        Train the model, including validation, early stopping mechanism, etc.
        
        Args:
            train_loader: Training set DataLoader
            val_loader: Validation set DataLoader
            
        Returns:
            str: Path to the best model
        """
        
        # Track best validation set performance
        best_val_loss = float('inf')
        best_val_ppl = float('inf')
        patience_counter = 0
        
        # Save training history for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.train_accs = []
        self.val_accs = []
        validation_ppls = []  # For calculating average growth rate
        
        # Create directory to save models
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Path to save the best model
        best_model_path = os.path.join(self.output_dir, f"{self.args.model_type}_best_model.pt")
        
        # Add overfitting detection variable
        consecutive_ppl_increases = 0
        
        # Add dropout dynamic strategy
        original_dropout = None
        
        # Adaptive gradient accumulation
        adaptive_grad_accum = True  # Whether to use adaptive gradient accumulation
        
        # Get warmup_epochs parameter
        warmup_epochs = getattr(self.args, 'warmup_epochs', 5)
        
        # Set flag to adjust learning rate every 10 epochs
        lr_adjust_epochs = set(range(10, self.args.epochs, 10))
        
        # Training loop
        for epoch in range(1, self.args.epochs + 1):
            # Dynamic learning rate adjustment
            if epoch in lr_adjust_epochs and patience_counter >= 2:
                # If two consecutive rounds have no improvement, reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.8
                print(f"Epoch {epoch}: Two consecutive rounds have no improvement, reducing learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Periodic dropout adjustment - Increase dropout every 15 epochs
            if epoch % 15 == 0 and epoch > 1:
                if not hasattr(self.model, 'increase_dropout'):
                    print("Model does not implement increase_dropout method")
                else:
                    print(f"Epoch {epoch}: Increase dropout to {self.args.dropout + 0.05} to reduce overfitting")
                    self.model.increase_dropout(0.05)
            
            # As training progresses, slightly increase dropout to reduce overfitting
            if original_dropout is not None and epoch > warmup_epochs + 5:
                # Increase 0.05 every 10 epochs, up to 0.8
                dropout_increase = min(0.2, 0.05 * ((epoch - warmup_epochs - 5) // 10))
                new_dropout = min(0.8, original_dropout + dropout_increase)
                
                # Apply to all dropout layers in the model
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Dropout):
                        module.p = new_dropout
                        
                print(f"Epoch {epoch}: Increase dropout to {new_dropout:.2f} to reduce overfitting")
            
            # Adaptive gradient accumulation - Increase gradient accumulation steps when PPL keeps growing
            if adaptive_grad_accum and consecutive_ppl_increases >= 2:
                old_grad_accum = self.grad_accum_steps
                self.grad_accum_steps = min(8, self.grad_accum_steps + 1)  # Accumulate up to 8 steps
                if old_grad_accum != self.grad_accum_steps:
                    print(f"Detected overfitting trend, increasing gradient accumulation steps to {self.grad_accum_steps}")
            
            # Record training start time
            start_time = time.time()
            
            # As training progresses, increase gradient clipping and weight decay to reduce overfitting
            if epoch > 10:
                # Adjust gradient clipping and weight decay based on relative growth rate of validation perplexity
                if len(validation_ppls) >= 3:
                    # Calculate relative growth rate
                    recent_ppls = validation_ppls[-3:]
                    relative_increase = (recent_ppls[-1] - recent_ppls[0]) / recent_ppls[0]
                    
                    # If perplexity starts to increase, enhance regularization
                    if relative_increase > 0.01:  # Trigger with only 1% growth
                        # Increase weight decay
                        for param_group in self.optimizer.param_groups:
                            if 'weight_decay' in param_group:
                                param_group['weight_decay'] = min(0.01, param_group['weight_decay'] * 1.2)
                                self.grad_clip = max(1.0, self.grad_clip * 0.9)  # Reduce gradient clipping, suppress large gradients
                                print(f"Detected perplexity growth ({relative_increase:.2%}), increasing weight decay to {param_group['weight_decay']:.5f}, adjusting gradient clipping to {self.grad_clip:.2f}")
                                break
            
            # Train for one epoch
            train_loss, train_ppl, train_acc = self._train_epoch(
                train_loader, 
                epoch, 
                grad_accum_steps=self.grad_accum_steps
            )
            
            # Update statistics
            self.train_losses.append(train_loss)
            self.train_ppls.append(train_ppl)
            self.train_accs.append(train_acc)
            
            # Evaluate on validation set
            val_loss, val_ppl, val_acc = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_ppls.append(val_ppl)
            self.val_accs.append(val_acc)
            
            # Record validation perplexity history
            validation_ppls.append(val_ppl)
            
            # Detect continuous PPL increase situation
            if len(validation_ppls) >= 2 and validation_ppls[-1] > validation_ppls[-2]:
                consecutive_ppl_increases += 1
            else:
                consecutive_ppl_increases = 0
                # If PPL has decreased, try to reduce gradient accumulation steps back to original state
                if adaptive_grad_accum and self.grad_accum_steps > 1:
                    self.grad_accum_steps = max(1, self.grad_accum_steps - 1)
                    print(f"Validation perplexity decreased, reducing gradient accumulation steps to {self.grad_accum_steps}")
            
            # If three consecutive PPL increases, reduce learning rate
            if consecutive_ppl_increases >= 3:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    # Also increase weight decay to reduce overfitting
                    param_group['weight_decay'] = min(0.01, param_group['weight_decay'] * 1.5)  # Increase L2 regularization strength
                
                print(f"Validation perplexity three consecutive increases, reducing learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")
                print(f"Also increasing weight decay to {self.optimizer.param_groups[0]['weight_decay']:.6f}")
                consecutive_ppl_increases = 0
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Use perplexity instead of loss to adjust learning rate
                    self.scheduler.step(val_ppl)
                elif isinstance(self.scheduler, torch.optim.lr_scheduler.LambdaLR):
                    self.scheduler.step()
                else:
                    self.scheduler.step()
            
            # Calculate moving average perplexity (average of last 3 epochs) to smooth short-term fluctuations
            if len(validation_ppls) >= 3:
                smoothed_val_ppl = sum(validation_ppls[-3:]) / 3
            else:
                smoothed_val_ppl = val_ppl
                
            # Print smoothed perplexity information
            if len(validation_ppls) >= 3:
                print(f"Current perplexity: {val_ppl:.4f}, Smoothed perplexity: {smoothed_val_ppl:.4f}")
            
            # Use validation perplexity instead of loss to judge performance (using smoothed perplexity)
            if val_ppl < best_val_ppl:
                # Calculate improvement percentage - Prevent nan at initial time
                if best_val_ppl == float('inf'):
                    improvement = 100.0  # Initial situation, set as 100% improvement
                else:
                    improvement = (best_val_ppl - val_ppl) / best_val_ppl * 100
                
                print(f"Validation perplexity improved from {best_val_ppl:.4f} to {val_ppl:.4f} ({improvement:.2f}%)")
                
                # Save best model
                best_val_ppl = val_ppl
                best_val_loss = val_loss  # Also update best loss value
                self._save_checkpoint(epoch, val_loss, val_ppl, best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Already {patience_counter} rounds have no improvement. Best perplexity: {best_val_ppl:.4f}")
                
                # Implement more aggressive early stopping strategy
                if patience_counter >= 5 and epoch > 10:
                    # Calculate average growth rate of recent 5 epochs
                    if len(validation_ppls) >= 5:
                        recent_ppls = validation_ppls[-5:]
                        avg_ppl_increase = (recent_ppls[-1] - recent_ppls[0]) / recent_ppls[0]
                        
                        # If average growth rate exceeds 10%, stop early (modified to stricter condition)
                        if avg_ppl_increase > 0.10:
                            print(f"Validation perplexity continuous increase ({avg_ppl_increase:.2%}), stopping early.")
                            break
            
            # Every 5 epochs or last epoch generate Tang poem sample - Regardless of model improvement
            if epoch % 5 == 0 or epoch == self.args.epochs:
                print(f"\n===== Tang poem generation example after {epoch} training rounds =====")
                # Use correct parameter form to call generation method
                self._generate_sample(epoch=epoch, num_samples=1, max_length=None, poem_types=['5', '7'])
                print(f"===== Generation ended =====\n")
            
            # Plot training curves
            if epoch % 5 == 0 or epoch == self.args.epochs:
                self._plot_training_curves()
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping after {epoch} epochs without improvement.")
                break
        
        # Restore original dropout value
        if original_dropout is not None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = original_dropout
        
        # Load best model
        self.load_checkpoint(best_model_path)
        
        # Plot final training curves
        self._plot_training_curves()
        
        # Generate final sample
        print(f"\n===== Final Tang poem generation example =====")
        # Use correct parameter form to call generation method
        self._generate_sample(epoch=epoch, num_samples=2, max_length=None, poem_types=['5', '7'])
        print(f"===== Generation ended =====\n")
        
        return best_model_path
    
    def _update_ema_model(self):
        """
        Update exponential moving average model parameters
        """
        if self.ema_model is not None:
            with torch.no_grad():
                for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
                    
    def _train_epoch(self, train_loader, epoch, grad_accum_steps=1):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader
            epoch: Current epoch number
            grad_accum_steps: Number of steps to accumulate gradients
            
        Returns:
            tuple: (train_loss, train_ppl, train_accuracy)
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        start_time = time.time()
        
        optimizer = self.optimizer
        criterion = self.criterion
        device = next(self.model.parameters()).device
        
        # Use tqdm progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.args.epochs} Training')
        
        # For calculating accuracy
        true_predictions = 0
        all_predictions = 0
        
        # Track loss, accuracy, and perplexity for each batch
        batch_losses = []
        batch_accs = []
        batch_ppls = []
        
        # Reset optimizer gradients
        optimizer.zero_grad()
        
        # Randomly drop 10% of batches, increase training noise
        drop_batch_prob = 0.1
        
        # Apply L2 regularization to recently updated parameters of larger magnitude every epoch start
        if hasattr(self, 'param_update_magnitudes') and epoch > 1:
            largest_updates = sorted(self.param_update_magnitudes.items(), key=lambda x: x[1], reverse=True)[:10]
            for name, _ in largest_updates:
                for n, p in self.model.named_parameters():
                    if n == name and p.requires_grad:
                        # Extra L2 regularization for these parameters
                        p.data *= (1 - 0.01)  # Mildly decay
            
        # Track update magnitude for each parameter
        self.param_update_magnitudes = defaultdict(float)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Randomly drop batch to increase training noise
            if random.random() < drop_batch_prob and epoch > 10:
                continue
                
            # Move data to device
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Apply different data augmentation strategies
            if hasattr(self.args, 'data_augmentation') and self.args.data_augmentation and random.random() < 0.3:
                # Randomly mask part of input (10-15%)
                mask_prob = random.uniform(0.1, 0.15)
                mask = (torch.rand_like(inputs, dtype=torch.float) < mask_prob).to(device)
                # Use a special index value as mask marker (usually 0 or for padding)
                inputs = inputs.masked_fill(mask, 0)
            
            with autocast(enabled=self.use_amp):
                # Forward pass
                outputs, hidden = self.model(inputs)
                
                # Calculate loss
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                # If label smoothing is used, no additional regularization loss is needed
                if not hasattr(self.args, 'label_smoothing') or self.args.label_smoothing <= 0:
                    # Add additional regularization loss
                    l2_reg = 0.0
                    for name, param in self.model.named_parameters():
                        if 'weight' in name and param.requires_grad:
                            l2_reg += torch.norm(param, p=2)
                    
                    alpha = 1e-4  # Regularization strength
                    loss += alpha * l2_reg
                
                # Scale loss to adapt to gradient accumulation
                loss = loss / grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Record unscaled loss
            unscaled_loss = loss.item() * grad_accum_steps
            total_loss += unscaled_loss
            
            # Calculate accuracy and perplexity
            _, predicted = outputs.max(dim=2)
            correct = (predicted == targets).sum().item()
            tokens = targets.ne(self.pad_idx).sum().item()
            
            # Accumulate statistics
            true_predictions += correct
            all_predictions += tokens
            total_correct += correct
            total_tokens += tokens
            
            # Calculate batch accuracy and perplexity
            accuracy = 100 * (true_predictions / max(1, all_predictions))
            ppl = calculate_perplexity(total_loss / (batch_idx + 1))
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Handle based on whether mixed precision training is used
                if self.use_amp:
                    # Handle gradient clipping and update for mixed precision training
                    if self.grad_clip > 0:
                        # First unscale, ensure only called once
                        self.scaler.unscale_(optimizer)
                        
                        # Apply progressive gradient centralization (if enabled)
                        if self.grad_centralization and (batch_idx + 1) % 3 == 0:
                            self._apply_gradient_centralization(self.model.parameters())
                        
                        # Calculate gradient norm and clip
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        
                        # Detect and handle unhealthy gradients
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"Warning: Bad gradient norm detected: {grad_norm}. Skipping this batch.")
                            optimizer.zero_grad()
                            continue
                    
                    # Execute optimizer step and update scaler
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Handle gradient clipping and update for non-mixed precision training
                    if self.grad_clip > 0:
                        # Apply gradient centralization (if enabled)
                        if self.grad_centralization and (batch_idx + 1) % 3 == 0:
                            self._apply_gradient_centralization(self.model.parameters())
                        
                        # Calculate gradient norm and clip
                        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        
                        # Detect unhealthy gradients
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"Warning: Bad gradient norm detected: {grad_norm}. Skipping this batch.")
                            optimizer.zero_grad()
                            continue
                    
                    # Execute optimizer step
                    optimizer.step()
                
                # Track update magnitude for each parameter
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            update_magnitude = torch.norm(param.grad * optimizer.param_groups[0]['lr'], p=2).item()
                            self.param_update_magnitudes[name] += update_magnitude
                
                # Update EMA model (if enabled)
                if self.use_ema:
                    self._update_ema_model()
                
                optimizer.zero_grad()  # Reset gradients
            
            # Collect current batch statistics
            batch_losses.append(unscaled_loss)
            batch_accs.append(accuracy)
            batch_ppls.append(ppl)
            
            # Update progress bar
            progress_bar.set_postfix({
                'acc': f"{accuracy:.2f}%", 
                'loss': f"{unscaled_loss:.2f}", 
                'ppl': f"{ppl:.2f}"
            })
            
            # Record interval
            if (batch_idx + 1) % self.log_interval == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                curr_wd = optimizer.param_groups[0]['weight_decay']
                print(f"\nEpoch {epoch}/{self.args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"LR: {curr_lr:.6f} | WD: {curr_wd:.6f} | "
                      f"Loss: {unscaled_loss:.4f} | PPL: {ppl:.2f} | Acc: {accuracy:.2f}%")
        
        # Calculate overall epoch statistics
        avg_loss = total_loss / len(train_loader)
        avg_ppl = calculate_perplexity(avg_loss)
        accuracy = 100 * (total_correct / max(1, total_tokens))
        
        # Close progress bar
        progress_bar.close()
        
        print(f"Epoch {epoch}/{self.args.epochs} | Time: {epoch_time(start_time, time.time())} | "
              f"Train Loss: {avg_loss:.4f} | Train PPL: {avg_ppl:.2f} | Train Acc: {accuracy:.2f}%")
        
        return avg_loss, avg_ppl, accuracy
    
    def evaluate(self, val_loader):
        """
        Evaluate the model on the validation set with enhanced metrics tracking
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            val_loss: Average validation loss
            val_ppl: Validation perplexity
            accuracy: Validation accuracy
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        start_time = time.time()
        
        # Track prediction confidence
        confidence_scores = []
        
        # Track detailed metrics by position in sequence
        position_correct = defaultdict(int)
        position_total = defaultdict(int)
        
        # Track confusion matrix for common errors (top 10 most frequent tokens)
        top_k = 10
        frequent_tokens = set(range(1, min(top_k + 1, self.vocab_size)))  # Start with 1 to skip padding
        confusion = torch.zeros(top_k + 1, top_k + 1, device=self.device)  # +1 for "other" category
        
        # Track batch-level metrics
        batch_losses = []
        batch_ppls = []
        batch_accs = []
        
        # Use mixed precision for evaluation if enabled
        use_amp = self.use_amp
        
        # Create progress bar
        pbar = tqdm(val_loader, desc="Validation", dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Get input and target
                data = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass with mixed precision if enabled
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        output, _ = self.model(data)
                        
                        # Reshape output and targets for loss calculation
                        batch_size, seq_len, vocab_size = output.shape
                        output_flat = output.view(-1, vocab_size)
                        targets_flat = targets.view(-1)
                        
                        # Calculate loss
                        if self.criterion:
                            loss = self.criterion(output_flat, targets_flat)
                        else:
                            loss = F.cross_entropy(output_flat, targets_flat, ignore_index=0)
                else:
                    output, _ = self.model(data)
                    
                    # Reshape output and targets for loss calculation
                    batch_size, seq_len, vocab_size = output.shape
                    output_flat = output.view(-1, vocab_size)
                    targets_flat = targets.view(-1)
                    
                    # Calculate loss
                    if self.criterion:
                        loss = self.criterion(output_flat, targets_flat)
                    else:
                        loss = F.cross_entropy(output_flat, targets_flat, ignore_index=0)
                
                # Update statistics
                total_loss += loss.item()
                
                # Calculate accuracy and detailed metrics
                probs = F.softmax(output_flat, dim=-1)
                _, predicted = torch.max(output_flat, 1)
                
                # Get confidence scores (probability of predicted class)
                confidence = torch.gather(probs, 1, predicted.unsqueeze(1)).squeeze(1)
                
                # Create mask to ignore padding tokens
                mask = targets_flat != 0
                
                # Calculate overall accuracy
                correct = (predicted == targets_flat) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                
                # Track confidence scores for correct and incorrect predictions
                confidence_scores.extend([(c.item(), corr.item()) for c, corr in zip(confidence[mask], correct[mask])])
                
                # Track position-wise accuracy
                if batch_size > 0 and seq_len > 0:
                    predicted_reshaped = predicted.view(batch_size, seq_len)
                    targets_reshaped = targets.view(batch_size, seq_len)
                    mask_reshaped = targets_reshaped != 0
                    
                    for pos in range(seq_len):
                        pos_mask = mask_reshaped[:, pos]
                        if pos_mask.sum() > 0:
                            pos_correct = (predicted_reshaped[:, pos] == targets_reshaped[:, pos]) & pos_mask
                            position_correct[pos] += pos_correct.sum().item()
                            position_total[pos] += pos_mask.sum().item()
                
                # Track confusion matrix for common errors
                for i, (pred, true, m) in enumerate(zip(predicted, targets_flat, mask)):
                    if m:
                        pred_idx = pred.item() if pred.item() in frequent_tokens else top_k
                        true_idx = true.item() if true.item() in frequent_tokens else top_k
                        confusion[true_idx, pred_idx] += 1
                
                # Track batch-level metrics
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                batch_ppl = math.exp(min(batch_loss, 20))  # Cap at 20 to avoid overflow
                batch_ppls.append(batch_ppl)
                
                batch_acc = 100 * (correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0)
                batch_accs.append(batch_acc)
                
                # Update progress bar with smoothed metrics
                smooth_loss = batch_losses[-1] if batch_idx == 0 else (smooth_loss * 0.9 + batch_losses[-1] * 0.1)
                smooth_ppl = batch_ppls[-1] if batch_idx == 0 else (smooth_ppl * 0.9 + batch_ppls[-1] * 0.1)
                smooth_acc = batch_accs[-1] if batch_idx == 0 else (smooth_acc * 0.9 + batch_accs[-1] * 0.1)
                
                pbar.set_postfix(loss=f"{smooth_loss:.2f}", ppl=f"{smooth_ppl:.2f}", acc=f"{smooth_acc:.2f}%")
        
        # Calculate average loss and metrics
        avg_loss = total_loss / max(1, len(val_loader))
        
        # Use improved perplexity calculation method to prevent extreme values
        try:
            if avg_loss > 20:
                # For losses greater than 20, use scaling method to avoid exponential explosion
                scaled_loss = 20.0 + math.log(1 + (avg_loss - 20.0))
                perplexity = math.exp(scaled_loss)
                print(f"Warning: High loss value ({avg_loss:.4f}), scaled perplexity calculation")
            else:
                perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = 1e4  # Set a large but finite value
            print("Warning: Perplexity calculation overflow, set to upper limit")
        
        # Calculate accuracy - Ensure denominator is not zero
        if total_tokens > 0:
            accuracy = 100 * (total_correct / total_tokens)
        else:
            accuracy = 0.0
            print("Warning: Evaluation sample count is zero")
        
        # Calculate accuracy for each position
        if position_total:
            positions = sorted(position_total.keys())
            pos_accs = []
            
            for p in positions:
                if position_total[p] > 0:
                    pos_acc = 100 * position_correct[p] / position_total[p]
                    pos_accs.append(pos_acc)
                else:
                    pos_accs.append(0)
                    
            avg_pos_acc = sum(pos_accs) / len(pos_accs) if pos_accs else 0
            min_pos_acc = min(pos_accs) if pos_accs else 0
            max_pos_acc = max(pos_accs) if pos_accs else 0
            
            # Check position-related accuracy degradation
            early_pos = [p for p in positions if p < len(positions) // 2]
            late_pos = [p for p in positions if p >= len(positions) // 2]
            
            # Ensure denominator is not zero
            early_acc = 0
            if early_pos:
                early_accs = []
                for p in early_pos:
                    if position_total[p] > 0:
                        early_accs.append(100 * position_correct[p] / position_total[p])
                    else:
                        early_accs.append(0)
                early_acc = sum(early_accs) / len(early_accs)
                
            late_acc = 0
            if late_pos:
                late_accs = []
                for p in late_pos:
                    if position_total[p] > 0:
                        late_accs.append(100 * position_correct[p] / position_total[p])
                    else:
                        late_accs.append(0)
                late_acc = sum(late_accs) / len(late_accs)
            
            pos_degradation = early_acc - late_acc
        else:
            avg_pos_acc = 0
            min_pos_acc = 0
            max_pos_acc = 0
            pos_degradation = 0
        
        # Analyze prediction confidence
        if confidence_scores:
            conf_correct = [conf for conf, corr in confidence_scores if corr == 1]
            conf_incorrect = [conf for conf, corr in confidence_scores if corr == 0]
            
            avg_conf_correct = sum(conf_correct) / max(1, len(conf_correct))
            avg_conf_incorrect = sum(conf_incorrect) / max(1, len(conf_incorrect))
            
            confidence_gap = avg_conf_correct - avg_conf_incorrect
        else:
            avg_conf_correct = 0
            avg_conf_incorrect = 0
            confidence_gap = 0
        
        # Calculate calibration error
        ece = 0  # Expected Calibration Error
        if confidence_scores:
            # Group predictions by confidence (10 bins)
            bin_size = 0.1
            bins = defaultdict(lambda: {'correct': 0, 'total': 0, 'conf_sum': 0})
            
            for conf, corr in confidence_scores:
                bin_idx = min(9, int(conf / bin_size))
                bins[bin_idx]['correct'] += corr
                bins[bin_idx]['total'] += 1
                bins[bin_idx]['conf_sum'] += conf
            
            # Calculate ECE
            total_samples = len(confidence_scores)
            for bin_idx, bin_data in bins.items():
                if bin_data['total'] > 0:
                    bin_acc = bin_data['correct'] / bin_data['total']
                    bin_conf = bin_data['conf_sum'] / bin_data['total']
                    bin_weight = bin_data['total'] / total_samples
                    ece += bin_weight * abs(bin_acc - bin_conf)
        
        # End timing
        end_time = time.time()
        eval_mins, eval_secs = epoch_time(start_time, end_time)
        
        # Print comprehensive validation summary
        print(f"Validation | Time: {eval_mins}m {eval_secs}s | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | Acc: {accuracy:.2f}%")
        print(f"Confidence | Correct: {avg_conf_correct:.4f} | Incorrect: {avg_conf_incorrect:.4f} | Gap: {confidence_gap:.4f} | ECE: {ece:.4f}")
        print(f"Position | Avg Acc: {avg_pos_acc:.2f}% | Min: {min_pos_acc:.2f}% | Max: {max_pos_acc:.2f}% | Degradation: {pos_degradation:.2f}%")
        
        # Record train vs. validation ratio for overfitting detection
        if hasattr(self, 'train_losses') and self.train_losses and hasattr(self, 'train_val_loss_ratio_history'):
            # Ensure safe calculation ratio
            if len(self.train_losses) > 0 and avg_loss > 0:
                train_val_ratio = self.train_losses[-1] / avg_loss
                # Limit abnormal values
                if train_val_ratio > 10.0:
                    train_val_ratio = 10.0
                    print("Warning: Training/Validation loss ratio abnormal large, limited to 10.0")
                elif train_val_ratio < 0.1:
                    train_val_ratio = 0.1
                    print("Warning: Training/Validation loss ratio abnormal small, limited to 0.1")
            else:
                train_val_ratio = 1.0  # Default value

            self.train_val_loss_ratio_history.append(train_val_ratio)
            
            # Update statistics
            self.min_train_val_ratio = min(self.min_train_val_ratio, train_val_ratio)
            if len(self.train_val_loss_ratio_history) >= 5:
                self.avg_train_val_ratio = sum(self.train_val_loss_ratio_history[-5:]) / 5
            else:
                self.avg_train_val_ratio = sum(self.train_val_loss_ratio_history) / len(self.train_val_loss_ratio_history)
                
            print(f"Train/Val Loss Ratio: {train_val_ratio:.4f} (Min: {self.min_train_val_ratio:.4f}, Avg: {self.avg_train_val_ratio:.4f})")
        
        return avg_loss, perplexity, accuracy
    
    def _plot_training_curves(self):
        """
        Plot training curves for loss, perplexity and accuracy
        """
        if not self.train_losses or not self.val_losses:
            return

        # Create output directory if not exists
        plots_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        epochs = list(range(1, len(self.train_losses) + 1))
        
        # Plot loss
        axs[0].plot(epochs, self.train_losses, 'b-', label='Training')
        axs[0].plot(epochs, self.val_losses, 'r-', label='Validation')
        axs[0].set_title(f'{self.args.model_type} Model - Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot perplexity
        axs[1].plot(epochs, self.train_ppls, 'b-', label='Training')
        axs[1].plot(epochs, self.val_ppls, 'r-', label='Validation')
        axs[1].set_title(f'{self.args.model_type} Model - Perplexity')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Perplexity')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot accuracy
        axs[2].plot(epochs, self.train_accs, 'b-', label='Training')
        axs[2].plot(epochs, self.val_accs, 'r-', label='Validation')
        axs[2].set_title(f'{self.args.model_type} Model - Accuracy')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Accuracy (%)')
        axs[2].legend()
        axs[2].grid(True)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{self.args.model_type}_training_curves.png'))
        plt.close()
    
    def _generate_sample(self, epoch, num_samples=1, max_length=None, poem_types=None):
        """
        Generate Tang poem sample for evaluation during training
        
        Args:
            epoch: Current training round
            num_samples: Number of samples to generate
            max_length: Maximum generation length
            poem_types: List of specified poem types, can be '5' or '7'
        """
        self.model.eval()
        
        # For Tang poem, usually shorter length is more suitable
        if max_length is None:
            max_length = min(150, self.args.generate_length)
            
        print("Generating Tang poem example:")
        
        # If no specified poem type, default to generating both five-character and seven-character poems
        if poem_types is None:
            poem_types = ['5', '7']
        elif isinstance(poem_types, str):
            poem_types = [poem_types]  # Ensure it's list type
            
        # Use <BOS> as starting marker
        if '<BOS>' in self.char_to_idx:
            start_idx = self.char_to_idx['<BOS>']
            
            # Set different temperature parameters, lower temperature more conservative, higher temperature more creative
            temperatures = [0.6, 0.8, 1.0]
            
            for i in range(num_samples):
                # Generate for each poem type
                for poem_type in poem_types:
                    print(f"\nSample {i+1} [{poem_type}言诗]:")
                    
                    # Use different temperatures to generate different styles of Tang poem
                    for temp_idx, temp in enumerate(temperatures):
                        try:
                            # Create initial sequence - Recreate each loop to avoid dimension problem
                            initial_seq = torch.tensor([[start_idx]], dtype=torch.long)
                            
                            # Generate text
                            generated_text = self.model.generate(
                                initial_seq,
                                max_length,
                                self.device,
                                temperature=temp,
                                idx_to_char=self.idx_to_char,
                                top_p=0.9,  # Use nucleus sampling
                                char_to_idx=self.char_to_idx,
                                poem_type=poem_type  # Specify poem type
                            )
                            
                            # Analyze poem format
                            lines = generated_text.strip().split('\n')
                            non_empty_lines = [line for line in lines if line.strip()]
                            
                            if non_empty_lines:
                                print(f"\nTemperature={temp:.1f}:")
                                
                                # Output according to original data format, no line numbers
                                for line in non_empty_lines:
                                    print(line)
                            else:
                                print(f"\nTemperature={temp:.1f}: (Model has not generated valid poem yet)")
                            
                            if temp_idx < len(temperatures) - 1:
                                print()  # Add blank line between temperatures
                                
                        except Exception as e:
                            print(f"\nTemperature={temp:.1f}: Generation process error ({str(e)})")
        else:
            print("Cannot generate sample: <BOS> token missing from vocabulary.")
    
    def generate_text(self, seed_text=None, max_length=None, temperature=1.0, top_k=0, top_p=0.9):
        """
        Generate text using the trained model with improved poetry generation
        
        Args:
            seed_text: Seed text (if not provided, randomly selected)
            max_length: Maximum generation length (if not provided, use args value)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            generated_text: Generated text
        """
        self.model.eval()
        
        # Set maximum length
        if max_length is None:
            max_length = self.args.generate_length
        
        # Determine initial sequence
        if seed_text:
            # Convert seed text to index sequence
            indices = []
            for char in seed_text:
                if char in self.char_to_idx:
                    indices.append(self.char_to_idx[char])
                else:
                    # Handle unknown characters
                    indices.append(self.char_to_idx['<UNK>'])
            
            # Create initial sequence
            initial_seq = torch.tensor([indices], dtype=torch.long)
        else:
            # Use <BOS> token as starting character
            if '<BOS>' in self.char_to_idx:
                start_idx = self.char_to_idx['<BOS>']
                initial_seq = torch.tensor([[start_idx]], dtype=torch.long)
                print(f"Starting with <BOS> token")
            else:
                # If no <BOS> token, randomly select a starting character
                start_char = None
                # Try to select a starting character from common Chinese characters
                common_chars = ['山', '水', '风', '云', '花', '月', '日', '天', '人', '心', '春', '秋', '冬', '夏']
                for char in common_chars:
                    if char in self.char_to_idx:
                        start_char = char
                        break
                
                # If no suitable common character found, randomly select
                if start_char is None:
                    valid_chars = []
                    for idx, char in self.idx_to_char.items():
                        if char not in ['<PAD>', '<UNK>', '<EOS>', '<LINE>', '<BOS>']:
                            valid_chars.append(char)
                    
                    if valid_chars:
                        start_char = np.random.choice(valid_chars)
                    else:
                        # If no valid characters, use the first non-special character
                        for idx in range(len(self.idx_to_char)):
                            char = self.idx_to_char[idx]
                            if char not in ['<PAD>', '<UNK>', '<EOS>', '<LINE>', '<BOS>']:
                                start_char = char
                                break
                
                # If still no suitable character found, use default
                if start_char is None:
                    print("Warning: Could not find suitable starting character, using default")
                    # Use character with index 1 (usually the first non-special character)
                    start_char = self.idx_to_char.get(1, '山')
                
                start_idx = self.char_to_idx[start_char]
                initial_seq = torch.tensor([[start_idx]], dtype=torch.long)
                
                # Print starting character
                print(f"Starting with character: {start_char}")
        
        # Generate text with improved sampling
        generated_text = self.model.generate(
            initial_seq,
            max_length,
            self.device,
            temperature=temperature,
            idx_to_char=self.idx_to_char,
            top_k=top_k,
            top_p=top_p,
            char_to_idx=self.char_to_idx
        )
        
        # Include seed text in result if used
        if seed_text:
            generated_text = seed_text + generated_text
        
        return generated_text
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            checkpoint: Checkpoint data
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            # First try loading with weights_only=True (safer)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print("Successfully loaded checkpoint with default settings")
        except Exception as e:
            print(f"Warning: Default loading failed with error: {e}")
            print("Attempting to load with weights_only=False (less secure but more compatible)")
            # Fall back to weights_only=False if needed
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            print("Successfully loaded checkpoint with weights_only=False")
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler if available
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
        
    def test(self, test_loader):
        """
        Test the model on test data
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            test_loss: Average test loss
            test_ppl: Test perplexity
            test_accuracy: Test accuracy
        """
        self.model.eval()
        total_loss = 0
        total_chars = 0
        correct_chars = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Testing")
            
            for batch in progress_bar:
                # Get input and target
                if isinstance(batch, dict):
                    source = batch['input']
                    target = batch['target']
                else:
                    source, target = batch
                
                source = source.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                logits, _ = self.model(source)
                
                # Reshape to match cross entropy loss expected format
                batch_size, seq_len, vocab_size = logits.shape
                logits = logits.reshape(-1, vocab_size)
                target = target.reshape(-1)
                
                # Calculate loss
                loss = self.criterion(logits, target)
                total_loss += loss.item()
                
                # Calculate accuracy
                preds = logits.argmax(dim=-1)
                mask = target != 0  # Ignore <PAD> tokens
                correct = (preds == target) & mask
                total_chars += mask.sum().item()
                correct_chars += correct.sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
        
        # Calculate average loss and perplexity
        test_loss = total_loss / len(test_loader)
        test_ppl = calculate_perplexity(test_loss)
        test_accuracy = 100 * (correct_chars / total_chars if total_chars > 0 else 0)
        
        # Print test results
        print(f"Test | Loss: {test_loss:.4f} | PPL: {test_ppl:.2f} | Acc: {test_accuracy:.2f}%")
        
        return test_loss, test_ppl, test_accuracy
    
    def _save_checkpoint(self, epoch, val_loss, val_ppl, checkpoint_path):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            val_ppl: Validation perplexity
            checkpoint_path: Path to save the checkpoint
        """
        # Save checkpoint with proper serialization
        # Store only necessary information to reduce file size and improve loading speed
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'args': vars(self.args) if hasattr(self.args, '__dict__') else self.args
        }, checkpoint_path)
    
    def _apply_gradient_centralization(self, parameters):
        """
        Apply gradient centralization to the gradients.
        This technique centers the gradient tensors, which can improve training stability.
        
        Args:
            parameters: Model parameters to apply gradient centralization to
        """
        for p in parameters:
            if p.grad is None or p.dim() <= 1:
                continue
                
            with torch.no_grad():
                p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.dim())), keepdim=True))
    
    def _calculate_focal_loss(self, outputs, targets, gamma):
        """
        Calculate focal loss, which gives less weight to well-classified examples.
        
        Args:
            outputs: Model predictions (logits)
            targets: Ground truth labels
            gamma: Focusing parameter
            
        Returns:
            Focal loss value
        """
        # Get standard cross entropy loss
        ce_loss = F.cross_entropy(
            outputs, targets, ignore_index=0, reduction='none'
        )
        
        # Convert logits to probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = (1 - pt) ** gamma * ce_loss
        
        # Return mean of focal loss
        return focal_loss.mean() 