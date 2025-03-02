#!/bin/bash

# Default option if none provided
OPTION=${1:-"help"}

# Base command with optimized parameters
BASE_CMD="python main.py --num_iterations 2000 --feature_type tfidf --ngram_max 2 --batch_strategy mini-batch --optimizer momentum"

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
        print_header "TEXT CLASSIFICATION EXPERIMENT RUNNER"
        echo "Usage: ./run.sh [OPTION]"
        echo ""
        echo "Available options:"
        echo "  help              - Show this help message"
        echo "  single            - Run a single model with optimized settings"
        echo "  model_type        - Compare different model types (logistic vs softmax)"
        echo "  feature_type      - Compare different feature types (bow, binary, tfidf)"
        echo "  ngram             - Compare different n-gram ranges"
        echo "  learning_rate     - Compare different learning rates"
        echo "  regularization    - Compare different regularization methods"
        echo "  batch_size        - Compare different batch sizes"
        echo "  batch_strategy    - Compare different batch strategies (full-batch, stochastic, mini-batch)"
        echo "  optimizer         - Compare different optimization methods (sgd, momentum)"
        echo "  loss_function     - Compare different loss functions"
        echo "  shuffle           - Compare shuffle vs no-shuffle"
        echo "  all               - Run all comparison experiments"
        echo "  clean             - Clean output directory"
        ;;
    
    # Run a single model with optimized settings
    "single")
        run_with_header "RUNNING SINGLE MODEL WITH OPTIMIZED SETTINGS" "$BASE_CMD"
        ;;
    
    # Compare different model types
    "model_type")
        run_with_header "COMPARING MODEL TYPES" "$BASE_CMD --experiment model_type"
        ;;
    
    # Compare different feature types
    "feature_type")
        run_with_header "COMPARING FEATURE TYPES" "$BASE_CMD --experiment feature_type"
        ;;
    
    # Compare different n-gram ranges
    "ngram")
        run_with_header "COMPARING N-GRAM RANGES" "$BASE_CMD --experiment ngram"
        ;;
    
    # Compare different learning rates
    "learning_rate")
        run_with_header "COMPARING LEARNING RATES" "$BASE_CMD --experiment learning_rate"
        ;;
    
    # Compare different regularization methods
    "regularization")
        run_with_header "COMPARING REGULARIZATION METHODS" "$BASE_CMD --experiment regularization"
        ;;
    
    # Compare different batch sizes
    "batch_size")
        run_with_header "COMPARING BATCH SIZES" "$BASE_CMD --experiment batch_size"
        ;;
    
    # Compare different batch strategies
    "batch_strategy")
        run_with_header "COMPARING BATCH STRATEGIES" "$BASE_CMD --experiment batch_strategy"
        ;;
    
    # Compare different optimization methods
    "optimizer")
        run_with_header "COMPARING OPTIMIZATION METHODS" "$BASE_CMD --experiment optimizer"
        ;;
    
    # Compare different loss functions
    "loss_function")
        run_with_header "COMPARING LOSS FUNCTIONS" "$BASE_CMD --experiment loss_function"
        ;;
    
    # Compare shuffle vs no-shuffle
    "shuffle")
        run_with_header "COMPARING SHUFFLE VS NO-SHUFFLE" "$BASE_CMD --experiment shuffle"
        ;;
    
    # Run all comparison experiments
    "all")
        print_header "RUNNING ALL COMPARISON EXPERIMENTS"
        
        # Run all experiments sequentially
        ./run.sh model_type
        ./run.sh feature_type
        ./run.sh ngram
        ./run.sh learning_rate
        ./run.sh regularization
        ./run.sh batch_size
        ./run.sh batch_strategy
        ./run.sh optimizer
        ./run.sh loss_function
        ./run.sh shuffle
        
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