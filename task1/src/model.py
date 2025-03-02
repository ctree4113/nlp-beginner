import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
import time

class BaseModel:
    """Base model class"""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, 
                 batch_size: int = 32, random_state: int = 42, verbose: bool = True,
                 batch_strategy: str = 'mini-batch', optimizer: str = 'sgd', momentum: float = 0.9,
                 loss_function: str = 'cross_entropy', shuffle: bool = True):
        """
        Initialize base model
        
        Args:
            learning_rate: Learning rate
            num_iterations: Number of iterations
            batch_size: Batch size, if None use full batch
            random_state: Random seed
            verbose: Whether to print training progress
            batch_strategy: Batch processing strategy, options are 'full-batch', 'stochastic', 'mini-batch'
            optimizer: Optimization method, options are 'sgd' (Stochastic Gradient Descent), 
                      'momentum' (Momentum Gradient Descent)
            momentum: Momentum coefficient for momentum optimizer
            loss_function: Loss function to use, options are 'cross_entropy', 'squared_error', 'hinge'
            shuffle: Whether to shuffle data during training
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.batch_strategy = batch_strategy.lower()
        self.optimizer = optimizer.lower()
        self.momentum = momentum
        self.loss_function = loss_function.lower()
        self.shuffle = shuffle
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.train_accuracy_history = []  # Record training accuracy
        self.valid_accuracy_history = []  # Record validation accuracy
        self.test_accuracy_history = []   # Record test accuracy
        self.record_interval = 10         # Record metrics every record_interval iterations
        
        # Set random seed
        np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray = None, y_valid: np.ndarray = None, 
            X_test: np.ndarray = None, y_test: np.ndarray = None) -> 'BaseModel':
        """
        Train the model
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Label vector with shape (n_samples,)
            X_valid: Validation feature matrix
            y_valid: Validation label vector
            X_test: Test feature matrix
            y_test: Test label vector
            
        Returns:
            self
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted class labels with shape (n_samples,)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _initialize_parameters(self, n_features: int, n_classes: int) -> None:
        """
        Initialize model parameters
        
        Args:
            n_features: Number of features
            n_classes: Number of classes
        """
        if n_classes == 2:
            # Binary classification, weights shape is (n_features,)
            self.weights = np.zeros(n_features)
            self.bias = 0
        else:
            # Multi-class classification, weights shape is (n_classes, n_features)
            self.weights = np.zeros((n_classes, n_features))
            self.bias = np.zeros(n_classes)
    
    def _get_mini_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate mini-batches based on batch strategy
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            List of mini-batches, each element is (X_batch, y_batch)
        """
        n_samples = X.shape[0]
        
        # Handle different batch strategies
        if self.batch_strategy == 'full-batch' or (self.batch_size is None or self.batch_size >= n_samples):
            # Full batch
            return [(X, y)]
        elif self.batch_strategy == 'stochastic':
            # Stochastic (one sample at a time)
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
                
            # Return individual samples
            return [(X_shuffled[i:i+1], y_shuffled[i:i+1]) for i in range(n_samples)]
        else:  # mini-batch
            # Handle shuffling
            if self.shuffle:
                # Shuffle data
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                # No shuffle
                X_shuffled = X
                y_shuffled = y
            
            # Generate mini-batches
            mini_batches = []
            num_complete_batches = n_samples // self.batch_size
            
            for i in range(num_complete_batches):
                start_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size
                mini_batches.append((X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]))
            
            # Handle remaining samples
            if n_samples % self.batch_size != 0:
                start_idx = num_complete_batches * self.batch_size
                mini_batches.append((X_shuffled[start_idx:], y_shuffled[start_idx:]))
            
            return mini_batches


class LogisticRegression(BaseModel):
    """Logistic Regression model for multi-class classification using One-vs-Rest strategy"""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, 
                 batch_size: int = 32, random_state: int = 42, verbose: bool = True,
                 regularization: Optional[str] = None, lambda_param: float = 0.01,
                 batch_strategy: str = 'mini-batch', optimizer: str = 'sgd', momentum: float = 0.9,
                 loss_function: str = 'cross_entropy', shuffle: bool = True):
        """
        Initialize logistic regression model
        
        Args:
            learning_rate: Learning rate
            num_iterations: Number of iterations
            batch_size: Batch size, if None use full batch
            random_state: Random seed
            verbose: Whether to print training progress
            regularization: Regularization method, options are None, 'l1', 'l2'
            lambda_param: Regularization parameter
            batch_strategy: Batch processing strategy, options are 'full-batch', 'stochastic', 'mini-batch'
            optimizer: Optimization method, options are 'sgd', 'momentum'
            momentum: Momentum coefficient for momentum optimizer
            loss_function: Loss function to use, options are 'cross_entropy', 'squared_error', 'hinge'
            shuffle: Whether to shuffle data during training
        """
        super().__init__(learning_rate, num_iterations, batch_size, random_state, verbose,
                      batch_strategy, optimizer, momentum, loss_function, shuffle)
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.classes = None
        self.n_classes = None
        self.models = []  # For One-vs-Rest strategy
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray = None, y_valid: np.ndarray = None, 
            X_test: np.ndarray = None, y_test: np.ndarray = None) -> 'LogisticRegression':
        """
        Train logistic regression model using One-vs-Rest strategy for multi-class problems
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Label vector with shape (n_samples,)
            X_valid: Validation feature matrix
            y_valid: Validation label vector
            X_test: Test feature matrix
            y_test: Test label vector
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Get class information
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # For binary classification, use the standard approach
        if self.n_classes == 2:
            return self._fit_binary(X, y, X_valid, y_valid, X_test, y_test)
        
        # For multi-class, use One-vs-Rest strategy
        # 对于多分类，减少迭代次数以加快训练速度
        ovr_iterations = max(200, self.num_iterations // self.n_classes)
        
        if self.verbose:
            print(f"Using One-vs-Rest strategy with {self.n_classes} classes")
            print(f"Training each binary classifier for {ovr_iterations} iterations")
        
        self.models = []
        all_losses = []
        
        # Record start time
        start_time = time.time()
        
        # Train one binary classifier for each class
        for i, cls in enumerate(self.classes):
            if self.verbose:
                print(f"\nTraining classifier for class {cls} ({i+1}/{self.n_classes})")
            
            # Create binary labels (1 for current class, 0 for all other classes)
            y_binary = (y == cls).astype(int)
            
            # Create and train binary classifier
            binary_model = _BinaryLogisticRegression(
                learning_rate=self.learning_rate,
                num_iterations=ovr_iterations,  # 减少迭代次数
                batch_size=self.batch_size,
                random_state=self.random_state,
                verbose=self.verbose,  # 传递 verbose 参数
                regularization=self.regularization,
                lambda_param=self.lambda_param,
                batch_strategy=self.batch_strategy,
                optimizer=self.optimizer,
                momentum=self.momentum,
                loss_function=self.loss_function,
                shuffle=self.shuffle
            )
            
            # Track validation and test data for the binary model
            y_valid_binary = None if y_valid is None else (y_valid == cls).astype(int)
            y_test_binary = None if y_test is None else (y_test == cls).astype(int)
            
            # Train the binary model
            binary_model.fit(X, y_binary, X_valid, y_valid_binary, X_test, y_test_binary)
            
            # Save the trained model
            self.models.append(binary_model)
            
            # Collect losses for visualization
            if len(all_losses) == 0:
                all_losses = binary_model.loss_history
            else:
                # Average the losses
                # 确保长度匹配
                min_len = min(len(all_losses), len(binary_model.loss_history))
                all_losses = [(all_losses[i] + binary_model.loss_history[i]) / 2 for i in range(min_len)]
            
            # 打印每个分类器的训练时间
            if self.verbose:
                class_time = time.time() - start_time
                print(f"Classifier for class {cls} completed in {class_time:.2f}s")
                # 显示已用时间和预计总时间
                if i > 0:
                    avg_time_per_class = class_time / (i + 1)
                    remaining_time = avg_time_per_class * (self.n_classes - i - 1)
                    print(f"Estimated remaining time: {remaining_time:.2f}s")
        
        # Use the average loss across all binary classifiers
        self.loss_history = all_losses
        
        # 简化评估频率，不需要对每个记录点都计算
        self.train_accuracy_history = []
        self.valid_accuracy_history = []
        self.test_accuracy_history = []
        
        # 只计算最终准确率
        # Training accuracy
        train_pred = self.predict(X)
        train_acc = np.mean(train_pred == y)
        self.train_accuracy_history.append(train_acc)
        
        # Validation accuracy
        if X_valid is not None and y_valid is not None:
            valid_pred = self.predict(X_valid)
            valid_acc = np.mean(valid_pred == y_valid)
            self.valid_accuracy_history.append(valid_acc)
        
        # Test accuracy
        if X_test is not None and y_test is not None:
            test_pred = self.predict(X_test)
            test_acc = np.mean(test_pred == y_test)
            self.test_accuracy_history.append(test_acc)
        
        # Print final training information
        if self.verbose:
            elapsed_time = time.time() - start_time
            print(f"\nTraining completed in {elapsed_time:.2f}s")
            print(f"Final training accuracy: {train_acc:.4f}")
            if X_valid is not None:
                print(f"Final validation accuracy: {valid_acc:.4f}")
        
        return self
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray = None, y_valid: np.ndarray = None,
                   X_test: np.ndarray = None, y_test: np.ndarray = None) -> 'LogisticRegression':
        """Train a binary logistic regression classifier for a specific class"""
        # Create a binary classifier for this class
        binary_model = _BinaryLogisticRegression(
            learning_rate=self.learning_rate,
            num_iterations=self.num_iterations // self.n_classes,  # Reduce iterations per class
            batch_size=self.batch_size,
            random_state=self.random_state,
            verbose=self.verbose,
            regularization=self.regularization,
            lambda_param=self.lambda_param,
            batch_strategy=self.batch_strategy,
            optimizer=self.optimizer,
            momentum=self.momentum,
            loss_function=self.loss_function,
            shuffle=self.shuffle
        )
        
        # Train the binary classifier
        binary_model.fit(X, y, X_valid, y_valid, X_test, y_test)
        
        return binary_model
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability for each class
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities with shape (n_samples, n_classes)
        """
        # For binary classification
        if self.n_classes == 2:
            # Get probability of positive class
            pos_proba = self._sigmoid(np.dot(X, self.weights) + self.bias)
            # Return probabilities for both classes
            return np.vstack((1 - pos_proba, pos_proba)).T
        
        # For multi-class classification (One-vs-Rest)
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes))
        
        # Get probabilities from each binary classifier
        for i, model in enumerate(self.models):
            probas[:, i] = model.predict_proba(X)
        
        # Normalize probabilities to sum to 1
        # This is needed because OvR probabilities don't naturally sum to 1
        row_sums = probas.sum(axis=1)
        probas = probas / row_sums[:, np.newaxis]
        
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted class labels with shape (n_samples,)
        """
        if self.n_classes == 2:
            # For binary classification, use threshold 0.5
            binary_predictions = (self._sigmoid(np.dot(X, self.weights) + self.bias) >= 0.5).astype(int)
            # Map to actual class labels
            return self.classes[binary_predictions]
        
        # For multi-class, predict the class with highest probability
        y_pred_proba = self.predict_proba(X)
        y_pred_indices = np.argmax(y_pred_proba, axis=1)
        return self.classes[y_pred_indices]
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid function
        
        Args:
            z: Input values
            
        Returns:
            Sigmoid function values
        """
        # Prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute loss for binary classification
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Loss value
        """
        m = X.shape[0]
        y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
        
        # Calculate loss based on selected loss function
        if self.loss_function == 'cross_entropy':
            # Binary cross-entropy loss
            loss = -(1 / m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
        elif self.loss_function == 'squared_error':
            # Mean squared error loss
            loss = (1 / (2 * m)) * np.sum(np.square(y_pred - y))
        elif self.loss_function == 'hinge':
            # Hinge loss (for binary classification)
            # Convert predictions from [0,1] to [-1,1]
            y_pred_scaled = 2 * y_pred - 1
            # Convert labels from [0,1] to [-1,1]
            y_scaled = 2 * y - 1
            loss = (1 / m) * np.sum(np.maximum(0, 1 - y_scaled * y_pred_scaled))
        else:
            # Default to cross-entropy
            loss = -(1 / m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
        
        # Add regularization term
        if self.regularization == 'l1':
            loss += (self.lambda_param / (2 * m)) * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            loss += (self.lambda_param / (2 * m)) * np.sum(np.square(self.weights))
        
        return loss


# Helper class for One-vs-Rest implementation
class _BinaryLogisticRegression(LogisticRegression):
    """Helper class for binary logistic regression used in the One-vs-Rest strategy"""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, 
                 batch_size: int = 32, random_state: int = 42, verbose: bool = True,
                 regularization: Optional[str] = None, lambda_param: float = 0.01,
                 batch_strategy: str = 'mini-batch', optimizer: str = 'sgd', momentum: float = 0.9,
                 loss_function: str = 'cross_entropy', shuffle: bool = True):
        """Initialize binary logistic regression model"""
        super().__init__(learning_rate, num_iterations, batch_size, random_state, verbose,
                      regularization, lambda_param, batch_strategy, optimizer, momentum, loss_function, shuffle)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability for positive class
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities for positive class with shape (n_samples,)
        """
        # Get probability of positive class (class 1)
        pos_proba = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return pos_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted class labels with shape (n_samples,)
        """
        # For binary classification, use threshold 0.5
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def fit(self, X, y, X_valid=None, y_valid=None, X_test=None, y_test=None):
        """Train binary logistic regression model"""
        # Initialize parameters
        m, n = X.shape
        self._initialize_parameters(n, 2)
        
        # Initialize velocity for momentum optimizer
        vdw = np.zeros_like(self.weights)
        vdb = 0
        
        # Store the best weights (with lowest validation loss)
        best_weights = None
        best_bias = None
        best_val_loss = float('inf')
        
        # For recording metrics
        self.loss_history = []
        self.train_accuracy_history = []
        self.valid_accuracy_history = []
        self.test_accuracy_history = []
        
        # Gradient descent loop
        for i in range(self.num_iterations):
            if self.batch_strategy == 'full-batch':
                # Full batch gradient descent
                A = self._sigmoid(np.dot(X, self.weights) + self.bias)
                dw = (1 / m) * np.dot(X.T, (A - y))
                db = (1 / m) * np.sum(A - y)
                
                # Add regularization
                if self.regularization == 'l1':
                    dw += (self.lambda_param / m) * np.sign(self.weights)
                elif self.regularization == 'l2':
                    dw += (self.lambda_param / m) * self.weights
                
                # Update parameters
                if self.optimizer == 'momentum':
                    vdw = self.momentum * vdw + self.learning_rate * dw
                    vdb = self.momentum * vdb + self.learning_rate * db
                    self.weights -= vdw
                    self.bias -= vdb
                else:  # sgd
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
            else:
                # Mini-batch or stochastic gradient descent
                mini_batches = self._get_mini_batches(X, y)
                
                for mini_batch in mini_batches:
                    X_mini, y_mini = mini_batch
                    m_mini = X_mini.shape[0]
                    
                    # Forward and backward propagation
                    A = self._sigmoid(np.dot(X_mini, self.weights) + self.bias)
                    dw = (1 / m_mini) * np.dot(X_mini.T, (A - y_mini))
                    db = (1 / m_mini) * np.sum(A - y_mini)
                    
                    # Add regularization
                    if self.regularization == 'l1':
                        dw += (self.lambda_param / m_mini) * np.sign(self.weights)
                    elif self.regularization == 'l2':
                        dw += (self.lambda_param / m_mini) * self.weights
                    
                    # Update parameters
                    if self.optimizer == 'momentum':
                        vdw = self.momentum * vdw + self.learning_rate * dw
                        vdb = self.momentum * vdb + self.learning_rate * db
                        self.weights -= vdw
                        self.bias -= vdb
                    else:  # sgd
                        self.weights -= self.learning_rate * dw
                        self.bias -= self.learning_rate * db
            
            # Compute loss and accuracy every few iterations
            if i % self.record_interval == 0 or i == self.num_iterations - 1:
                # Compute training loss
                train_loss = self._compute_loss(X, y)
                self.loss_history.append(train_loss)
                
                # Compute accuracy
                y_pred_train = self.predict(X)
                train_accuracy = np.mean(y_pred_train == y)
                self.train_accuracy_history.append(train_accuracy)
                
                # Validation accuracy
                if X_valid is not None and y_valid is not None:
                    y_pred_valid = self.predict(X_valid)
                    valid_accuracy = np.mean(y_pred_valid == y_valid)
                    self.valid_accuracy_history.append(valid_accuracy)
                    
                    # Save best model based on validation loss
                    val_loss = self._compute_loss(X_valid, y_valid)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = self.weights.copy()
                        best_bias = self.bias
                
                # Test accuracy
                if X_test is not None and y_test is not None:
                    y_pred_test = self.predict(X_test)
                    test_accuracy = np.mean(y_pred_test == y_test)
                    self.test_accuracy_history.append(test_accuracy)
                
                if self.verbose and i % (self.record_interval * 10) == 0:
                    print(f"Iteration {i}: Train Loss = {train_loss:.6f}, Train Accuracy = {train_accuracy:.6f}")
                    if X_valid is not None and y_valid is not None:
                        print(f"Validation Accuracy = {valid_accuracy:.6f}")
        
        # Use the best model if validation was provided
        if best_weights is not None and best_bias is not None:
            self.weights = best_weights
            self.bias = best_bias
        
        return self


class SoftmaxRegression(BaseModel):
    """Softmax Regression model for multi-class classification"""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, 
                 batch_size: int = 32, random_state: int = 42, verbose: bool = True,
                 regularization: Optional[str] = None, lambda_param: float = 0.01,
                 batch_strategy: str = 'mini-batch', optimizer: str = 'sgd', momentum: float = 0.9,
                 loss_function: str = 'cross_entropy', shuffle: bool = True):
        """
        Initialize softmax regression model
        
        Args:
            learning_rate: Learning rate
            num_iterations: Number of iterations
            batch_size: Batch size, if None use full batch
            random_state: Random seed
            verbose: Whether to print training progress
            regularization: Regularization method, options are None, 'l1', 'l2'
            lambda_param: Regularization parameter
            batch_strategy: Batch processing strategy, options are 'full-batch', 'stochastic', 'mini-batch'
            optimizer: Optimization method, options are 'sgd', 'momentum'
            momentum: Momentum coefficient for momentum optimizer
            loss_function: Loss function to use, options are 'cross_entropy', 'squared_error'
            shuffle: Whether to shuffle data during training
        """
        super().__init__(learning_rate, num_iterations, batch_size, random_state, verbose,
                      batch_strategy, optimizer, momentum, loss_function, shuffle)
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.classes = None
        self.n_classes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray = None, y_valid: np.ndarray = None, 
            X_test: np.ndarray = None, y_test: np.ndarray = None) -> 'SoftmaxRegression':
        """
        Train softmax regression model
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Label vector with shape (n_samples,)
            X_valid: Validation feature matrix
            y_valid: Validation label vector
            X_test: Test feature matrix
            y_test: Test label vector
            
        Returns:
            Trained model
        """
        # Get unique classes and number of classes
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # Initialize parameters
        m, n = X.shape
        self._initialize_parameters(n, self.n_classes)
        
        # One-hot encode labels
        y_one_hot = self._one_hot_encode(y)
        
        # One-hot encode validation and test labels if provided
        y_valid_one_hot = None if y_valid is None else self._one_hot_encode(y_valid)
        y_test_one_hot = None if y_test is None else self._one_hot_encode(y_test)
        
        # Initialize velocity for momentum optimizer
        vdw = np.zeros_like(self.weights)
        vdb = np.zeros_like(self.bias)
        
        # Store the best weights (with lowest validation loss)
        best_weights = None
        best_bias = None
        best_val_loss = float('inf')
        
        # For recording metrics
        self.loss_history = []
        self.train_accuracy_history = []
        self.valid_accuracy_history = []
        self.test_accuracy_history = []
        
        # Start timer
        start_time = time.time()
        
        # Gradient descent loop
        for i in range(self.num_iterations):
            if self.batch_strategy == 'full-batch':
                # Full batch gradient descent
                # Forward propagation
                Z = np.dot(X, self.weights.T) + self.bias
                A = self._softmax(Z)
                
                # Compute gradients
                dZ = A - y_one_hot
                dw = (1 / m) * np.dot(dZ.T, X)
                db = (1 / m) * np.sum(dZ, axis=0)
                
                # Add regularization
                if self.regularization == 'l1':
                    dw += (self.lambda_param / m) * np.sign(self.weights)
                elif self.regularization == 'l2':
                    dw += (self.lambda_param / m) * self.weights
                
                # Update parameters
                if self.optimizer == 'momentum':
                    vdw = self.momentum * vdw + self.learning_rate * dw
                    vdb = self.momentum * vdb + self.learning_rate * db
                    self.weights -= vdw
                    self.bias -= vdb
                else:  # sgd
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
            else:
                # Mini-batch or stochastic gradient descent
                mini_batches = self._get_mini_batches(X, y_one_hot)
                
                for mini_batch in mini_batches:
                    X_mini, y_mini = mini_batch
                    m_mini = X_mini.shape[0]
                    
                    # Forward propagation
                    Z = np.dot(X_mini, self.weights.T) + self.bias
                    A = self._softmax(Z)
                    
                    # Compute gradients
                    dZ = A - y_mini
                    dw = (1 / m_mini) * np.dot(dZ.T, X_mini)
                    db = (1 / m_mini) * np.sum(dZ, axis=0)
                    
                    # Add regularization
                    if self.regularization == 'l1':
                        dw += (self.lambda_param / m_mini) * np.sign(self.weights)
                    elif self.regularization == 'l2':
                        dw += (self.lambda_param / m_mini) * self.weights
                    
                    # Update parameters
                    if self.optimizer == 'momentum':
                        vdw = self.momentum * vdw + self.learning_rate * dw
                        vdb = self.momentum * vdb + self.learning_rate * db
                        self.weights -= vdw
                        self.bias -= vdb
                    else:  # sgd
                        self.weights -= self.learning_rate * dw
                        self.bias -= self.learning_rate * db
            
            # Calculate loss
            loss = self._compute_loss(X, y_one_hot)
            self.loss_history.append(loss)
            
            # Record accuracy metrics at specified intervals
            if i % self.record_interval == 0:
                # Training accuracy
                train_pred = self.predict(X)
                train_acc = np.mean(train_pred == y)
                self.train_accuracy_history.append(train_acc)
                
                # Validation accuracy
                if X_valid is not None and y_valid is not None:
                    valid_pred = self.predict(X_valid)
                    valid_acc = np.mean(valid_pred == y_valid)
                    self.valid_accuracy_history.append(valid_acc)
                
                # Test accuracy
                if X_test is not None and y_test is not None:
                    test_pred = self.predict(X_test)
                    test_acc = np.mean(test_pred == y_test)
                    self.test_accuracy_history.append(test_acc)
            
            # Print training progress
            if self.verbose and (i + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration {i+1}/{self.num_iterations}, Loss: {abs(loss):.4f}, Time: {elapsed_time:.2f}s")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of each class
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities with shape (n_samples, n_classes)
        """
        return self._softmax(np.dot(X, self.weights.T) + self.bias)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted class labels with shape (n_samples,)
        """
        y_pred_proba = self.predict_proba(X)
        y_pred_indices = np.argmax(y_pred_proba, axis=1)
        return self.classes[y_pred_indices]
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax function
        
        Args:
            z: Input values with shape (n_samples, n_classes)
            
        Returns:
            Softmax function values with shape (n_samples, n_classes)
        """
        # Prevent overflow
        z = np.clip(z, -500, 500)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """
        Convert labels to one-hot encoding
        
        Args:
            y: Label vector with shape (n_samples,)
            
        Returns:
            One-hot encoding with shape (n_samples, n_classes)
        """
        n_samples = len(y)
        y_one_hot = np.zeros((n_samples, self.n_classes))
        
        for i, label in enumerate(y):
            class_idx = np.where(self.classes == label)[0][0]
            y_one_hot[i, class_idx] = 1
        
        return y_one_hot
    
    def _compute_loss(self, X: np.ndarray, y_one_hot: np.ndarray) -> float:
        """
        Compute loss for multi-class classification
        
        Args:
            X: Feature matrix
            y_one_hot: One-hot encoded labels
            
        Returns:
            Loss value
        """
        m = X.shape[0]
        y_pred = self.predict_proba(X)
        
        # Calculate loss based on selected loss function
        if self.loss_function == 'cross_entropy':
            # Cross-entropy loss
            loss = -(1 / m) * np.sum(y_one_hot * np.log(y_pred + 1e-15))
        elif self.loss_function == 'squared_error':
            # Mean squared error loss
            loss = (1 / (2 * m)) * np.sum(np.square(y_pred - y_one_hot))
        elif self.loss_function == 'hinge':
            # Multi-class hinge loss (one-vs-all approach)
            # For each sample, calculate max(0, 1 - (correct_class_score - other_class_score))
            y_true_idx = np.argmax(y_one_hot, axis=1)
            margins = np.zeros(m)
            
            for i in range(m):
                correct_class_score = y_pred[i, y_true_idx[i]]
                # Sum of hinge losses for all incorrect classes
                margin = np.sum(np.maximum(0, 1 - (correct_class_score - y_pred[i])))
                # Subtract 1 because we counted the correct class too
                margins[i] = margin - np.maximum(0, 1 - (correct_class_score - correct_class_score))
            
            loss = (1 / m) * np.sum(margins)
        else:
            # Default to cross-entropy
            loss = -(1 / m) * np.sum(y_one_hot * np.log(y_pred + 1e-15))
        
        # Add regularization term
        if self.regularization == 'l1':
            loss += (self.lambda_param / (2 * m)) * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            loss += (self.lambda_param / (2 * m)) * np.sum(np.square(self.weights))
        
        return loss 