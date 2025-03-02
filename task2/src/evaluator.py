import os
import json

class Evaluator:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def print_metrics(self, metrics):
        # Implementation of print_metrics method
        pass

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
        
        # Calculate confusion matrix
        trainer.plot_confusion_matrix(data_loader, class_names)
        
        # Save metrics
        with open(os.path.join(self.output_dir, f'{trainer.model_name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics 