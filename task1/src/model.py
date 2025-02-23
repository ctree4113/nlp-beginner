import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
import time

class BaseModel:
    """Base model class"""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, 
                 batch_size: int = 32, random_state: int = 42, verbose: bool = True,
                 optimizer: str = 'mini-batch', momentum: float = 0.9):
        """
        Initialize base model
        
        Args:
            learning_rate: Learning rate
            num_iterations: Number of iterations
            batch_size: Batch size, if None use full batch
            random_state: Random seed
            verbose: Whether to print training progress
            optimizer: Optimization method, options are 'gd' (Gradient Descent), 
                      'sgd' (Stochastic Gradient Descent), 'mini-batch' (Mini-batch Gradient Descent),
                      'momentum' (Momentum Gradient Descent)
            momentum: Momentum coefficient for momentum optimizer
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.optimizer = optimizer.lower()
        self.momentum = momentum
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
        Generate mini-batches
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            List of mini-batches, each element is (X_batch, y_batch)
        """
        n_samples = X.shape[0]
        
        if self.batch_size is None or self.batch_size >= n_samples:
            # Full batch
            return [(X, y)]
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
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
    """Binary logistic regression model"""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, 
                 batch_size: int = 32, random_state: int = 42, verbose: bool = True,
                 regularization: Optional[str] = None, lambda_param: float = 0.01,
                 optimizer: str = 'mini-batch', momentum: float = 0.9):
        """
        Initialize logistic regression model
        
        Args:
            learning_rate: Learning rate
            num_iterations: Number of iterations
            batch_size: Batch size
            random_state: Random seed
            verbose: Whether to print training progress
            regularization: Regularization type, options are 'l1', 'l2' or None
            lambda_param: Regularization parameter
            optimizer: Optimization method, options are 'gd', 'sgd', 'mini-batch', 'momentum'
            momentum: Momentum coefficient for momentum optimizer
        """
        super().__init__(learning_rate, num_iterations, batch_size, random_state, verbose, optimizer, momentum)
        self.regularization = regularization
        self.lambda_param = lambda_param
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray = None, y_valid: np.ndarray = None, 
            X_test: np.ndarray = None, y_test: np.ndarray = None) -> 'LogisticRegression':
        """
        Train logistic regression model
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Label vector with shape (n_samples,), values are 0 or 1
            X_valid: Validation feature matrix
            y_valid: Validation label vector
            X_test: Test feature matrix
            y_test: Test label vector
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self._initialize_parameters(n_features, 2)
        
        # Initialize velocity for momentum optimizer
        velocity_w = np.zeros_like(self.weights)
        velocity_b = 0
        
        # Record start time
        start_time = time.time()
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Get batches based on optimizer type
            if self.optimizer == 'gd':
                # Gradient Descent: use full batch
                batches = [(X, y)]
            elif self.optimizer == 'sgd':
                # Stochastic Gradient Descent: use single random sample
                idx = np.random.randint(n_samples)
                batches = [(X[idx:idx+1], y[idx:idx+1])]
            else:
                # Mini-batch or Momentum: use mini-batches
                batches = self._get_mini_batches(X, y)
            
            # Perform gradient descent on each batch
            for X_batch, y_batch in batches:
                # Calculate predictions
                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._sigmoid(z)
                
                # Calculate gradients
                dw = (1 / len(X_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1 / len(X_batch)) * np.sum(y_pred - y_batch)
                
                # Add regularization term
                if self.regularization == 'l1':
                    dw += (self.lambda_param / len(X_batch)) * np.sign(self.weights)
                elif self.regularization == 'l2':
                    dw += (self.lambda_param / len(X_batch)) * self.weights
                
                # Update parameters based on optimizer
                if self.optimizer == 'momentum':
                    # Momentum update
                    velocity_w = self.momentum * velocity_w - self.learning_rate * dw
                    velocity_b = self.momentum * velocity_b - self.learning_rate * db
                    self.weights += velocity_w
                    self.bias += velocity_b
                else:
                    # Standard update (GD, SGD, Mini-batch)
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
            
            # Calculate loss
            loss = self._compute_loss(X, y)
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
        Predict probability of positive class
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities with shape (n_samples,)
        """
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted class labels with shape (n_samples,), values are 0 or 1
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
    
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
        Compute binary cross-entropy loss
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Loss value
        """
        m = X.shape[0]
        y_pred = self.predict_proba(X)
        
        # Calculate cross-entropy loss
        loss = -(1 / m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
        
        # Add regularization term
        if self.regularization == 'l1':
            loss += (self.lambda_param / (2 * m)) * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            loss += (self.lambda_param / (2 * m)) * np.sum(np.square(self.weights))
        
        return loss


class SoftmaxRegression(BaseModel):
    """Multi-class Softmax regression model"""
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000, 
                 batch_size: int = 32, random_state: int = 42, verbose: bool = True,
                 regularization: Optional[str] = None, lambda_param: float = 0.01,
                 optimizer: str = 'mini-batch', momentum: float = 0.9):
        """
        Initialize Softmax regression model
        
        Args:
            learning_rate: Learning rate
            num_iterations: Number of iterations
            batch_size: Batch size
            random_state: Random seed
            verbose: Whether to print training progress
            regularization: Regularization type, options are 'l1', 'l2' or None
            lambda_param: Regularization parameter
            optimizer: Optimization method, options are 'gd', 'sgd', 'mini-batch', 'momentum'
            momentum: Momentum coefficient for momentum optimizer
        """
        super().__init__(learning_rate, num_iterations, batch_size, random_state, verbose, optimizer, momentum)
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.classes = None
        self.n_classes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray = None, y_valid: np.ndarray = None, 
            X_test: np.ndarray = None, y_test: np.ndarray = None) -> 'SoftmaxRegression':
        """
        Train Softmax regression model
        
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
        
        # Convert labels to one-hot encoding
        y_one_hot = self._one_hot_encode(y)
        
        # Initialize parameters
        self._initialize_parameters(n_features, self.n_classes)
        
        # Initialize velocity for momentum optimizer
        velocity_w = np.zeros_like(self.weights)
        velocity_b = np.zeros_like(self.bias)
        
        # Record start time
        start_time = time.time()
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Get batches based on optimizer type
            if self.optimizer == 'gd':
                # Gradient Descent: use full batch
                batches = [(X, y_one_hot)]
            elif self.optimizer == 'sgd':
                # Stochastic Gradient Descent: use single random sample
                idx = np.random.randint(n_samples)
                batches = [(X[idx:idx+1], y_one_hot[idx:idx+1])]
            else:
                # Mini-batch or Momentum: use mini-batches
                batches = self._get_mini_batches(X, y_one_hot)
            
            # Perform gradient descent on each batch
            for X_batch, y_batch in batches:
                # Calculate predictions
                y_pred = self._softmax(np.dot(X_batch, self.weights.T) + self.bias)
                
                # Calculate gradients
                dw = (1 / len(X_batch)) * np.dot((y_pred - y_batch).T, X_batch)
                db = (1 / len(X_batch)) * np.sum(y_pred - y_batch, axis=0)
                
                # Add regularization term
                if self.regularization == 'l1':
                    dw += (self.lambda_param / len(X_batch)) * np.sign(self.weights)
                elif self.regularization == 'l2':
                    dw += (self.lambda_param / len(X_batch)) * self.weights
                
                # Update parameters based on optimizer
                if self.optimizer == 'momentum':
                    # Momentum update
                    velocity_w = self.momentum * velocity_w - self.learning_rate * dw
                    velocity_b = self.momentum * velocity_b - self.learning_rate * db
                    self.weights += velocity_w
                    self.bias += velocity_b
                else:
                    # Standard update (GD, SGD, Mini-batch)
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
        Compute cross-entropy loss
        
        Args:
            X: Feature matrix
            y_one_hot: One-hot encoded labels
            
        Returns:
            Loss value
        """
        m = X.shape[0]
        y_pred = self.predict_proba(X)
        
        # Calculate cross-entropy loss
        loss = -(1 / m) * np.sum(y_one_hot * np.log(y_pred + 1e-15))
        
        # Add regularization term
        if self.regularization == 'l1':
            loss += (self.lambda_param / (2 * m)) * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            loss += (self.lambda_param / (2 * m)) * np.sum(np.square(self.weights))
        
        return loss 