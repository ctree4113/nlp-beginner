#!/bin/bash

# Default option if none provided
OPTION=${1:-"help"}

# Base command with optimized parameters (using 50 epochs as default)
BASE_CMD="python main.py --embedding_dim 100 --hidden_dim 300 --dropout 0.5 --learning_rate 0.0004 --optimizer adam --clip_grad 10.0 --epochs 20 --batch_size 64 --pretrained_embedding glove --embedding_path ../glove/glove.6B.100d.txt"

# Output directory
OUTPUT_DIR="../output"

# Function to print section header
print_header() {
    echo "$1"
}

# Function to run command with header
run_with_header() {
    print_header "$1"
    echo "Running: $2"
    echo ""
    eval "$2"
    echo ""
}

case $OPTION in
    # Help option
    "help")
        print_header "TEXT MATCHING WITH ATTENTION MECHANISM EXPERIMENTS"
        echo "Usage: ./run.sh [OPTION]"
        echo ""
        echo "Available options:"
        echo "  help                - Display this help message"
        echo "  train               - Train the model with default settings"
        echo "  lstm_hidden         - Compare different LSTM hidden dimensions (100, 200, 300, 400)"
        echo "  max_seq_len         - Compare different maximum sequence lengths (50, 100, 150, 200)"
        echo "  embeddings          - Compare model performance with and without pre-trained GloVe embeddings"
        echo "  attention_analysis  - Analyze attention weights on examples from test set"
        echo "  example             - Run model on specific example sentences"
        echo "  clean               - Clean output directory"
        ;;
    
    # Train with default settings
    "train")
        run_with_header "TRAINING WITH DEFAULT SETTINGS" "$BASE_CMD"
        ;;
    
    # Compare different LSTM hidden dimensions
    "lstm_hidden")
        run_with_header "COMPARING LSTM HIDDEN DIMENSIONS" "$BASE_CMD --experiment hidden_dim --param_values 100 200 300 400"
        ;;
    
    # Compare different maximum sequence lengths
    "max_seq_len")
        run_with_header "COMPARING MAXIMUM SEQUENCE LENGTHS" "$BASE_CMD --experiment max_seq_len --param_values 50 100 150 200"
        ;;
    
    # Compare with and without pre-trained embeddings
    "embeddings")
        run_with_header "COMPARING PRE-TRAINED EMBEDDINGS" "$BASE_CMD --experiment pretrained_embedding --param_values none glove"
        ;;
    
    # Analyze attention weights
    "attention_analysis")
        run_with_header "ANALYZING ATTENTION WEIGHTS" "$BASE_CMD --analyze_attention"
        ;;
    
    # Run model on specific example sentences
    "example")
        PREMISE=${2:-"A person on a horse jumps over a broken down airplane."}
        HYPOTHESIS=${3:-"A person is training his horse for a competition."}
        run_with_header "RUNNING MODEL ON EXAMPLE" "$BASE_CMD --example --premise \"$PREMISE\" --hypothesis \"$HYPOTHESIS\""
        ;;
    
    # Clean output directory
    "clean")
        print_header "CLEANING OUTPUT DIRECTORY"
        echo "Removing all files in $OUTPUT_DIR"
        rm -rf $OUTPUT_DIR/*
        echo "Output directory cleaned."
        ;;
    
    # Invalid option
    *)
        echo "Error: Invalid option '$OPTION'"
        echo "Run './run.sh help' for usage information."
        exit 1
        ;;
esac 