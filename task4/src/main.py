import os
import time
import torch
from data_processor import DataProcessor
from models import BiLSTMCRF
from trainer import Trainer
from utils import get_args, set_seed, print_args, count_parameters, epoch_time


def main():
    # Get arguments
    args = get_args()
    
    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    # Print header
    print("\n" + "="*60)
    print(f"BiLSTM-CRF Model for Named Entity Recognition")
    print("="*60 + "\n")
    
    # Print arguments
    print_args(args)
    
    # Load data
    print("\nLoading data...")
    start_time = time.time()
    data_processor = DataProcessor(
        args.data_dir, 
        args.language, 
        args.max_seq_len,
        args.max_word_len
    )
    data_processor.load_data()
    
    # Create data loaders
    train_loader, dev_loader, test_loader = data_processor.create_dataloaders(args.batch_size)
    end_time = time.time()
    mins, secs = epoch_time(start_time, end_time)
    print(f"Data loaded in: {mins}m {secs}s")
    
    # Print dataset information
    print(f"\nDataset Information:")
    print(f"  Vocabulary size: {len(data_processor.word2idx)}")
    print(f"  Character vocabulary size: {len(data_processor.char2idx)}")
    print(f"  Tag set size: {len(data_processor.tag2idx)}")
    print(f"  Training samples: {len(data_processor.train_dataset)}")
    print(f"  Validation samples: {len(data_processor.valid_dataset)}")
    print(f"  Test samples: {len(data_processor.test_dataset)}")
    
    # Initialize model
    print("\nInitializing BiLSTM-CRF model...")
    model = BiLSTMCRF(
        vocab_size=len(data_processor.word2idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_tags=len(data_processor.tag2idx),
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_char_cnn=args.use_char_cnn,
        num_chars=len(data_processor.char2idx) if args.use_char_cnn else None,
        char_embedding_dim=args.char_embedding_dim,
        char_channel_size=args.char_channel_size
    )
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = Trainer(model, data_processor.tag2idx, args)
    
    # Train model
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    start_time = time.time()
    best_model_path = trainer.train(train_loader, dev_loader, test_loader)
    end_time = time.time()
    
    mins, secs = epoch_time(start_time, end_time)
    print(f"\nTotal training time: {mins}m {secs}s")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main() 