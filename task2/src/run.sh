#!/bin/bash

# Default option if none provided
OPTION=${1:-"help"}

# Base command with optimized parameters
BASE_CMD="python main.py --embedding_type glove --embedding_dim 100 --dropout 0.5 --learning_rate 0.001 --optimizer adam --scheduler plateau --num_epochs 50 --patience 10 --weight_decay 1e-5"

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
        print_header "DEEP LEARNING TEXT CLASSIFICATION EXPERIMENTS"
        echo "Usage: ./run.sh [OPTION]"
        echo ""
        echo "Available options:"
        echo "  help                - Show this help message"
        echo "  single [model_type] - Run a single model with optimized settings (cnn, rnn, rcnn)"
        echo "  model_type          - Compare different model types (CNN, RNN, RCNN)"
        echo "  embedding_type      - Compare different embedding types (random, glove)"
        echo "  embedding_dim       - Compare different embedding dimensions (50, 100, 200, 300)"
        echo "  cnn_filters         - Compare different CNN filter configurations"
        echo "  rnn_architecture    - Compare different RNN architectures (LSTM vs GRU, uni vs bidirectional)"
        echo "  dropout             - Compare different dropout rates"
        echo "  optimizer           - Compare different optimization methods"
        echo "  lr_scheduler        - Compare different learning rate schedulers"
        echo "  optimal             - Run experiment with optimal parameters (best model, embedding, dropout)"
        echo "  all                 - Run all comparison experiments"
        echo "  clean               - Clean output directory"
        ;;
    
    # Run a single model with optimized settings
    "single")
        MODEL_TYPE=${2:-"cnn"}
        if [[ "$MODEL_TYPE" == "cnn" || "$MODEL_TYPE" == "rnn" || "$MODEL_TYPE" == "rcnn" ]]; then
            run_with_header "RUNNING $MODEL_TYPE MODEL WITH OPTIMIZED SETTINGS" "$BASE_CMD --model_type $MODEL_TYPE"
        else
            echo "Error: Invalid model type '$MODEL_TYPE'"
            echo "Available model types: cnn, rnn, rcnn"
            exit 1
        fi
        ;;
    
    # Compare different model types
    "model_type")
        run_with_header "COMPARING MODEL ARCHITECTURES (CNN, RNN, RCNN)" "$BASE_CMD --experiment model_type"
        ;;
    
    # Compare different embedding types
    "embedding_type")
        run_with_header "COMPARING EMBEDDING TYPES (RANDOM VS GLOVE)" "$BASE_CMD --experiment embedding_type"
        ;;
    
    # Compare different embedding dimensions
    "embedding_dim")
        run_with_header "COMPARING EMBEDDING DIMENSIONS" "$BASE_CMD --experiment embedding_dim"
        ;;
    
    # Compare different CNN filter configurations
    "cnn_filters")
        run_with_header "COMPARING CNN FILTER CONFIGURATIONS" "$BASE_CMD --model_type cnn --experiment filter_sizes"
        print_header "COMPARING CNN FILTER QUANTITIES"
        eval "$BASE_CMD --model_type cnn --experiment num_filters"
        ;;
    
    # Compare different RNN architectures
    "rnn_architecture")
        run_with_header "COMPARING RNN TYPES (LSTM VS GRU)" "$BASE_CMD --model_type rnn --experiment rnn_type"
        print_header "COMPARING RNN DIRECTIONALITY (UNI VS BIDIRECTIONAL)"
        eval "$BASE_CMD --model_type rnn --experiment bidirectional"
        print_header "COMPARING RNN HIDDEN DIMENSIONS"
        eval "$BASE_CMD --model_type rnn --experiment hidden_dim"
        print_header "COMPARING RNN LAYER DEPTHS"
        eval "$BASE_CMD --model_type rnn --experiment num_layers"
        ;;
    
    # Compare different dropout rates
    "dropout")
        run_with_header "COMPARING DROPOUT RATES" "$BASE_CMD --experiment dropout"
        ;;
    
    # Compare different optimization methods
    "optimizer")
        run_with_header "COMPARING OPTIMIZATION METHODS" "$BASE_CMD --experiment optimizer"
        ;;
    
    # Compare different learning rate schedulers
    "lr_scheduler")
        run_with_header "COMPARING LEARNING RATE SCHEDULERS" "$BASE_CMD --experiment scheduler"
        ;;
    
    # Run with optimal parameters
    "optimal")
        # Based on previous experiments, these are the optimal parameters
        # You may need to adjust these values based on your actual experimental results
        OPTIMAL_CMD="$BASE_CMD --model_type rcnn --embedding_type glove --embedding_dim 300 --dropout 0.5 --bidirectional --output_dir ../output/optimal_model"
        run_with_header "RUNNING WITH OPTIMAL PARAMETERS" "$OPTIMAL_CMD"
        ;;
    
    # Run all comparison experiments
    "all")
        print_header "RUNNING ALL COMPARISON EXPERIMENTS"
        
        # Run all experiments sequentially
        ./run.sh model_type
        ./run.sh embedding_type
        ./run.sh embedding_dim
        ./run.sh cnn_filters
        ./run.sh rnn_architecture
        ./run.sh dropout
        ./run.sh optimizer
        ./run.sh lr_scheduler
        
        print_header "ALL EXPERIMENTS COMPLETED"
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