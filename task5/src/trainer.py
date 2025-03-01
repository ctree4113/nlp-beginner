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

# 导入混合精度训练支持
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
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        self.output_dir = args.output_dir
        self.patience = args.patience if hasattr(args, 'patience') else 10
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize exponential moving average model if requested
        self.ema_model = None
        if hasattr(args, 'use_ema') and args.use_ema:
            self.ema_model = copy.deepcopy(model)
            self.ema_decay = 0.9999 if hasattr(args, 'ema_decay') else 0.9999
            print(f"Using EMA with decay {self.ema_decay}")
        
        # Set up optimizer with weight decay normalization
        if args.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self._get_optimizer_grouped_parameters(args.weight_decay),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print(f"Using Adam optimizer with weight decay: {args.weight_decay}")
        elif args.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self._get_optimizer_grouped_parameters(args.weight_decay),
                lr=args.lr,
                betas=(0.9, 0.95),
                eps=1e-8
            )
            print(f"Using AdamW optimizer with weight decay: {args.weight_decay}")
        elif args.optimizer.lower() == 'radam':
            try:
                from torch.optim import RAdam
                self.optimizer = RAdam(
                    self._get_optimizer_grouped_parameters(args.weight_decay),
                    lr=args.lr,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
                print(f"Using RAdam optimizer with weight decay: {args.weight_decay}")
            except ImportError:
                print("RAdam not available, falling back to AdamW")
                self.optimizer = optim.AdamW(
                    self._get_optimizer_grouped_parameters(args.weight_decay),
                    lr=args.lr
                )
        else:
            self.optimizer = optim.SGD(
                self._get_optimizer_grouped_parameters(args.weight_decay),
                lr=args.lr,
                momentum=0.9,
                nesterov=True
            )
            print(f"Using SGD optimizer with momentum and weight decay: {args.weight_decay}")
        
        # Set up advanced learning rate scheduler
        warmup_epochs = args.warmup_epochs if hasattr(args, 'warmup_epochs') else 0
        
        if args.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6)
            self.scheduler_type = 'plateau'
        elif args.scheduler == 'cosine':
            if warmup_epochs > 0:
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer, 
                    start_factor=0.1, 
                    end_factor=1.0, 
                    total_iters=warmup_epochs
                )
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=args.epochs - warmup_epochs, 
                    eta_min=1e-6
                )
                self.scheduler = optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=args.epochs, eta_min=1e-6)
            self.scheduler_type = 'cosine'
        elif args.scheduler == 'cosine_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
            self.scheduler_type = 'cosine_restarts'
        elif args.scheduler == 'onecycle':
            steps_per_epoch = 100  # This will be updated in train()
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=args.lr,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )
            self.scheduler_type = 'onecycle'
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.5)
            self.scheduler_type = 'step'
        
        # Set up loss function with label smoothing if enabled
        self.label_smoothing = args.label_smoothing if hasattr(args, 'label_smoothing') else 0.0
        if self.label_smoothing > 0:
            print(f"Using label smoothing with factor: {self.label_smoothing}")
            self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=self.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Initialize lists for tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []
        self.train_accs = []
        self.val_accs = []
        
        # Initialize training history
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Mixed precision training setup
        self.use_amp = hasattr(args, 'use_amp') and args.use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
            print("Using mixed precision training")
        
        # Print model information
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
        
        # Gradient accumulation steps
        self.grad_accum_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 1
        
        # Focal loss gamma parameter (if using focal loss)
        self.focal_gamma = args.focal_gamma if hasattr(args, 'focal_gamma') else 0.0
    
    def train(self, train_loader, val_loader):
        """
        Train the model with comprehensive optimization and monitoring strategies
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Path to the best model checkpoint
        """
        print("\n============================================================")
        print(f"Running {self.args.model_type} Language Model")
        print("============================================================")
        
        # Initialize variables for tracking training progress
        train_losses = []
        val_losses = []
        train_ppls = []
        val_ppls = []
        train_accs = []
        val_accs = []
        val_ppl_history = [float('inf')]  # Initialize with infinity
        learning_rates = []
        
        best_val_loss = float('inf')
        best_val_ppl = float('inf')
        best_epoch = 0
        no_improvement = 0
        best_model_path = os.path.join(self.output_dir, f"{self.args.model_type}_best_model.pt")
        
        # Initialize gradient accumulation steps from args or default
        grad_accum_steps = self.grad_accum_steps
        
        # Training loop
        for epoch in range(1, self.args.epochs + 1):
            # Update onecycle scheduler steps per epoch if needed
            if self.scheduler_type == 'onecycle' and hasattr(self.scheduler, 'total_steps'):
                self.scheduler.total_steps = len(train_loader) * self.args.epochs
            
            # Apply dynamic loss scaling based on epoch progress
            loss_scale = min(1.2, 0.8 + epoch / self.args.epochs * 0.4)  # Scale from 0.8 to 1.2
            
            # Set dynamic label smoothing if enabled
            if hasattr(self.args, 'dynamic_label_smoothing') and self.args.dynamic_label_smoothing:
                # Start with higher label smoothing, then gradually reduce
                if hasattr(self.criterion, 'label_smoothing') and hasattr(self.criterion, 'set_label_smoothing'):
                    dynamic_smoothing = self.label_smoothing * (1.0 - 0.5 * epoch / self.args.epochs)
                    self.criterion.set_label_smoothing(dynamic_smoothing)
                    print(f"Using dynamic label smoothing: {dynamic_smoothing:.3f}")
            
            # Train for one epoch
            train_loss, train_ppl, train_accuracy = self._train_epoch(
                train_loader, 
                epoch, 
                grad_accum_steps=grad_accum_steps
            )
            
            # Apply EMA if enabled
            if self.ema_model is not None:
                # Temporarily swap models for evaluation
                orig_model = self.model
                self.model = self.ema_model
                
                # Evaluate on validation set with EMA model
                val_loss, val_ppl, val_accuracy = self.evaluate(val_loader)
                
                # Swap back
                self.model = orig_model
            else:
                # Evaluate on validation set
                val_loss, val_ppl, val_accuracy = self.evaluate(val_loader)
            
            # Store metrics for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ppls.append(train_ppl)
            val_ppls.append(val_ppl)
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)
            val_ppl_history.append(val_ppl)
            
            # Store current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Update learning rate scheduler
            if self.scheduler_type == 'plateau':
                self.scheduler.step(val_loss)
            elif self.scheduler_type not in ['onecycle']:  # onecycle is updated per iteration
                self.scheduler.step()
            
            # Track the train/val loss ratio for overfitting detection
            train_val_ratio = train_loss / val_loss if val_loss > 0 else 1.0
            
            # Check for overfitting signals
            overfitting_detected = False
            severe_overfitting = False
            
            # Signal 1: Train-val loss ratio is too small (train doing much better than val)
            if epoch > 5 and train_val_ratio < 0.2:
                overfitting_detected = True
                print("Overfitting signal: Train loss much lower than validation loss")
                
                if train_val_ratio < 0.1:
                    severe_overfitting = True
                    print("WARNING: Severe overfitting detected!")
            
            # Signal 2: Validation performance is getting worse consistently
            if epoch > 3:
                recent_val_ppls = val_ppl_history[-4:]
                if all(recent_val_ppls[i] < recent_val_ppls[i+1] for i in range(len(recent_val_ppls)-1)):
                    overfitting_detected = True
                    print("Overfitting signal: Validation perplexity increasing consistently")
            
            # Signal 3: Train accuracy rising while val accuracy falling
            if epoch > 3 and len(train_accs) > 3 and len(val_accs) > 3:
                if (train_accs[-1] > train_accs[-2] > train_accs[-3]) and (val_accs[-1] < val_accs[-2] < val_accs[-3]):
                    overfitting_detected = True
                    print("Overfitting signal: Train accuracy increasing while validation accuracy decreasing")
            
            # Apply additional regularization if overfitting detected
            if overfitting_detected:
                # 1. Increase gradient accumulation steps (more stable gradients)
                if grad_accum_steps < 8:  # Cap at reasonable value
                    grad_accum_steps += 1
                    print(f"Increased gradient accumulation steps to {grad_accum_steps}")
                
                # 2. Increase weight decay temporarily
                for param_group in self.optimizer.param_groups:
                    if 'weight_decay' in param_group:
                        param_group['weight_decay'] = min(0.1, param_group['weight_decay'] * 1.2)
                        print(f"Increased weight decay to {param_group['weight_decay']:.6f}")
                
                # 3. Apply stronger dropout if available
                if hasattr(self.model, 'increase_dropout'):
                    increment = 0.1 if severe_overfitting else 0.05
                    old_dropout = self.model.dropout.p if hasattr(self.model, 'dropout') else 0.0
                    new_dropout = self.model.increase_dropout(increment)
                    print(f"Increased dropout from {old_dropout:.2f} to {new_dropout:.2f}")
                
                # 4. Apply learning rate reduction
                if severe_overfitting:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.7
                    print(f"Reduced learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping with model improvement tracking
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_val_loss = val_loss
                best_epoch = epoch
                no_improvement = 0
                
                # Save the best model (use EMA model if enabled)
                if self.ema_model is not None:
                    # Temporarily swap models for saving
                    orig_model = self.model
                    self.model = self.ema_model
                    self._save_checkpoint(epoch, val_loss, val_ppl, best_model_path)
                    self.model = orig_model
                    print(f"Saved best EMA model to {best_model_path}")
                else:
                    self._save_checkpoint(epoch, val_loss, val_ppl, best_model_path)
                    print(f"Saved best model to {best_model_path}")
                
                # Save checkpoint at each improvement if enabled
                if hasattr(self.args, 'save_all_improvements') and self.args.save_all_improvements:
                    improvement_path = os.path.join(self.output_dir, f"{self.args.model_type}_epoch{epoch}.pt")
                    self._save_checkpoint(epoch, val_loss, val_ppl, improvement_path)
            else:
                no_improvement += 1
                print(f"No improvement for {no_improvement} epochs")
            
            # Generate sample text periodically
            if epoch % 5 == 0 or epoch == self.args.epochs:
                self._generate_sample(epoch)
            
            # Plot training curves
            if epoch % 5 == 0 or epoch == self.args.epochs or no_improvement >= self.patience:
                self._plot_training_curves()
            
            # Check for early stopping with patience
            patience = self.patience
            if no_improvement >= patience:
                # Only stop if we've trained for at least 1/3 of the epochs
                min_epochs = max(10, self.args.epochs // 3)
                if epoch >= min_epochs:
                    print(f"Early stopping after {epoch} epochs")
                    break
                else:
                    print(f"No improvement for {no_improvement} epochs, but continuing training (minimum {min_epochs} epochs)")
                    
                    # Reset counter but with a penalty to avoid getting stuck in a loop
                    no_improvement = patience // 2
            
            # Print epoch divider for better logging
            print("------------------------------------------------------------")
        
        # Load the best model for final evaluation
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            self.load_checkpoint(best_model_path)
            print(f"Best validation perplexity: {best_val_ppl:.2f} at epoch {best_epoch}")
        else:
            print("Warning: Best model file not found. Using current model state.")
        
        # Provide training summary
        if epoch > 1:
            if best_epoch < epoch:
                epochs_after_best = epoch - best_epoch
                print(f"Model trained for {epochs_after_best} epochs after best validation performance.")
                if epochs_after_best > 10:
                    print("WARNING: Model might have started overfitting. Consider reducing epochs or increasing regularization.")
            
            # Calculate overall improvement
            initial_val_ppl = val_ppl_history[1] if len(val_ppl_history) > 1 else float('inf')
            ppl_improvement = initial_val_ppl - best_val_ppl
            ppl_improvement_pct = (ppl_improvement / initial_val_ppl) * 100 if initial_val_ppl != float('inf') else 0
            
            print(f"Initial validation PPL: {initial_val_ppl:.2f}")
            print(f"Best validation PPL: {best_val_ppl:.2f}")
            print(f"Overall improvement: {ppl_improvement:.2f} ({ppl_improvement_pct:.1f}%)")
        
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
        Train for one epoch with advanced techniques
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            grad_accum_steps: Number of steps to accumulate gradients before updating weights
            
        Returns:
            train_loss: Average training loss for this epoch
            train_ppl: Training perplexity for this epoch
            accuracy: Training accuracy for this epoch
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        start_time = time.time()
        
        # Initialize mixed precision training if available
        use_amp = hasattr(self.args, 'use_amp') and self.args.use_amp and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.args.epochs} Training")
        
        # Track batch-level metrics for better monitoring
        batch_losses = []
        batch_ppls = []
        batch_accs = []
        
        # Initialize gradient accumulation counter
        accum_step = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Get input and target
            data = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Only zero gradients when starting a new accumulation cycle
            if accum_step == 0:
                self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if use_amp:
                with torch.amp.autocast('cuda'):
                    output, _ = self.model(data)
                    
                    # Reshape output and targets for loss calculation
                    output = output.view(-1, self.vocab_size)
                    targets = targets.view(-1)
                    
                    # Calculate loss with label smoothing if enabled
                    if self.criterion:
                        loss = self.criterion(output, targets)
                    else:
                        loss = F.cross_entropy(output, targets, ignore_index=0)
                
                # Scale loss and perform backward pass
                scaled_loss = loss / grad_accum_steps  # Normalize loss
                scaler.scale(scaled_loss).backward()
            else:
                # Standard forward pass
                output, _ = self.model(data)
                
                # Reshape output and targets for loss calculation
                output = output.view(-1, self.vocab_size)
                targets = targets.view(-1)
                
                # Calculate loss with label smoothing if enabled
                if self.criterion:
                    loss = self.criterion(output, targets)
                else:
                    loss = F.cross_entropy(output, targets, ignore_index=0)
                
                # Normalize loss for gradient accumulation
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()
            
            # Update statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(output, 1)
                mask = targets != 0  # Ignore padding tokens
                correct = (predicted == targets) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
            
            # Track batch-level metrics
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            batch_ppl = math.exp(min(batch_loss, 20))  # Cap at 20 to avoid overflow
            batch_ppls.append(batch_ppl)
            
            batch_acc = 100 * (correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0)
            batch_accs.append(batch_acc)
            
            # Update progress bar with smoothed metrics
            recent_batches = 10  # Use last 10 batches for smoothing
            smooth_loss = sum(batch_losses[-recent_batches:]) / min(recent_batches, len(batch_losses))
            smooth_ppl = sum(batch_ppls[-recent_batches:]) / min(recent_batches, len(batch_ppls))
            smooth_acc = sum(batch_accs[-recent_batches:]) / min(recent_batches, len(batch_accs))
            
            pbar.set_postfix(loss=f"{smooth_loss:.2f}", ppl=f"{smooth_ppl:.2f}", acc=f"{smooth_acc:.2f}%")
            
            # Increment accumulation step
            accum_step += 1
            
            # Update weights if we've accumulated enough gradients
            if accum_step == grad_accum_steps or batch_idx == len(train_loader) - 1:
                # Gradient clipping
                if self.args.grad_clip > 0:
                    if use_amp:
                        # Unscale before clipping
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                # Update weights
                if use_amp:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                
                # Reset accumulation counter
                accum_step = 0
            
            # Log training progress at intervals
            if (batch_idx + 1) % self.args.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                ppl = math.exp(min(avg_loss, 20))  # Cap at 20 to avoid overflow
                accuracy = 100 * (total_correct / total_tokens if total_tokens > 0 else 0)
                
                print(f"Epoch {epoch}/{self.args.epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Acc: {accuracy:.2f}%")
        
        # Calculate final statistics
        avg_loss = total_loss / len(train_loader)
        ppl = math.exp(min(avg_loss, 20))  # Cap at 20 to avoid overflow
        accuracy = 100 * (total_correct / total_tokens if total_tokens > 0 else 0)
        
        # Update EMA model after each epoch
        self._update_ema_model()
        
        # Print epoch summary
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Epoch {epoch}/{self.args.epochs} | Time: {epoch_mins}m {epoch_secs}s | "
              f"Train Loss: {avg_loss:.4f} | Train PPL: {ppl:.2f} | Train Acc: {accuracy:.2f}%")
        
        return avg_loss, ppl, accuracy
    
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
        perplexity = math.exp(min(avg_loss, 20))  # Cap at 20 to avoid overflow
        accuracy = 100 * (total_correct / total_tokens if total_tokens > 0 else 0)
        
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
        
        # Analyze position-dependent accuracy if there is data
        if position_total:
            positions = sorted(position_total.keys())
            pos_accs = [100 * position_correct[p] / position_total[p] if position_total[p] > 0 else 0 for p in positions]
            avg_pos_acc = sum(pos_accs) / len(pos_accs) if pos_accs else 0
            min_pos_acc = min(pos_accs) if pos_accs else 0
            max_pos_acc = max(pos_accs) if pos_accs else 0
            
            # Check for position-dependent degradation
            early_pos = [p for p in positions if p < len(positions) // 2]
            late_pos = [p for p in positions if p >= len(positions) // 2]
            
            early_acc = sum([100 * position_correct[p] / position_total[p] if position_total[p] > 0 else 0 for p in early_pos]) / len(early_pos) if early_pos else 0
            late_acc = sum([100 * position_correct[p] / position_total[p] if position_total[p] > 0 else 0 for p in late_pos]) / len(late_pos) if late_pos else 0
            
            pos_degradation = early_acc - late_acc
        else:
            avg_pos_acc = 0
            min_pos_acc = 0
            max_pos_acc = 0
            pos_degradation = 0
        
        # Print comprehensive validation summary
        print(f"Validation | Time: {eval_mins}m {eval_secs}s | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | Acc: {accuracy:.2f}%")
        print(f"Confidence | Correct: {avg_conf_correct:.4f} | Incorrect: {avg_conf_incorrect:.4f} | Gap: {confidence_gap:.4f} | ECE: {ece:.4f}")
        print(f"Position | Avg Acc: {avg_pos_acc:.2f}% | Min: {min_pos_acc:.2f}% | Max: {max_pos_acc:.2f}% | Degradation: {pos_degradation:.2f}%")
        
        # Record train vs. validation ratio for overfitting detection
        if hasattr(self, 'train_losses') and self.train_losses and hasattr(self, 'train_val_loss_ratio_history'):
            train_val_ratio = self.train_losses[-1] / avg_loss if avg_loss > 0 else 1.0
            self.train_val_loss_ratio_history.append(train_val_ratio)
            
            # Update statistics
            self.min_train_val_ratio = min(self.min_train_val_ratio, train_val_ratio)
            if len(self.train_val_loss_ratio_history) >= 5:
                self.avg_train_val_ratio = sum(self.train_val_loss_ratio_history[-5:]) / 5
                
            print(f"Train/Val Loss Ratio: {train_val_ratio:.4f} (Min: {self.min_train_val_ratio:.4f}, Avg: {self.avg_train_val_ratio:.4f})")
        
        return avg_loss, perplexity, accuracy
    
    def _plot_training_curves(self):
        """
        Plot training curves
        """
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot perplexity
        plt.subplot(2, 2, 2)
        plt.plot(self.train_ppls, label='Train Perplexity')
        plt.plot(self.val_ppls, label='Validation Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Training and Validation Perplexity')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(self.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.args.model_type}_training_curves.png"))
        plt.close()
    
    def _generate_sample(self, epoch, num_samples=1, max_length=None):
        """
        Generate sample text during training
        
        Args:
            epoch: Current epoch number
            num_samples: Number of samples to generate
            max_length: Maximum generation length
        """
        self.model.eval()
        
        if max_length is None:
            max_length = min(100, self.args.generate_length)  # Generate shorter samples during training
            
        print("\nGenerating sample text:")
        
        # Use <BOS> token as starting character
        if '<BOS>' in self.char_to_idx:
            start_idx = self.char_to_idx['<BOS>']
            
            for i in range(num_samples):
                # Create initial sequence
                initial_seq = torch.tensor([[start_idx]], dtype=torch.long)
                
                # Generate text with different sampling strategies
                temps = [0.8, 1.2]  # Different sampling temperatures
                for temp in temps:
                    # Generate text
                    generated_text = self.model.generate(
                        initial_seq,
                        max_length,
                        self.device,
                        temperature=temp,
                        idx_to_char=self.idx_to_char,
                        top_p=0.9,  # Use nucleus sampling
                        char_to_idx=self.char_to_idx
                    )
                    
                    print(f"Sample {i+1} [temperature={temp}]:\n{generated_text}\n")
        else:
            print("Cannot generate samples: <BOS> token missing from vocabulary.")
    
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
    
    def _get_optimizer_grouped_parameters(self, weight_decay):
        """
        Get grouped parameters for optimizer with proper weight decay exclusions.
        Layernorm parameters and bias terms should not have weight decay applied.
        
        Args:
            weight_decay: Weight decay factor
            
        Returns:
            List of parameter groups with appropriate weight decay settings
        """
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        return optimizer_grouped_parameters
    
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