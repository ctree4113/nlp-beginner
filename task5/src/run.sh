#!/bin/bash

# Set working directory
cd "$(dirname "$0")"

# Run option 
OPTION=${1:-"lstm"}

# Create output directory based on model type
OUTPUT_DIR="../output/${OPTION}"
mkdir -p ${OUTPUT_DIR}

case $OPTION in
    # Train LSTM model with optimized parameters for Tang poetry
    "lstm")
        echo "Training LSTM model with optimized parameters for Tang poetry..."
        echo "Results will be saved to ${OUTPUT_DIR}"
        python main.py \
            --model_type LSTM \
            --batch_size 64 \
            --epochs 100 \
            --embedding_dim 256 \
            --hidden_dim 256 \
            --num_layers 2 \
            --dropout 0.75 \
            --lr 0.0002 \
            --optimizer adamw \
            --scheduler cosine \
            --data_augmentation \
            --tie_weights \
            --use_layer_norm \
            --use_residual \
            --use_attention \
            --patience 20 \
            --warmup_epochs 5 \
            --top_p 0.9 \
            --generate_length 300 \
            --strict_chinese \
            --grad_clip 1.5 \
            --weight_decay 1e-2 \
            --label_smoothing 0.15 \
            --log_interval 10 \
            --use_ema \
            --seed 42 \
            --output_dir ${OUTPUT_DIR} \
            --cuda
        ;;
    
    # Train GRU model with optimized parameters for Tang poetry
    "gru")
        echo "Training GRU model with optimized parameters for Tang poetry..."
        echo "Results will be saved to ${OUTPUT_DIR}"
        python main.py \
            --model_type GRU \
            --batch_size 64 \
            --epochs 100 \
            --embedding_dim 256 \
            --hidden_dim 512 \
            --num_layers 3 \
            --dropout 0.6 \
            --lr 0.0005 \
            --optimizer adamw \
            --scheduler cosine \
            --data_augmentation \
            --tie_weights \
            --use_layer_norm \
            --use_residual \
            --use_attention \
            --patience 20 \
            --warmup_epochs 5 \
            --top_p 0.9 \
            --generate_length 300 \
            --strict_chinese \
            --grad_clip 3.0 \
            --weight_decay 1e-3 \
            --label_smoothing 0.1 \
            --log_interval 10 \
            --use_ema \
            --seed 42 \
            --output_dir ${OUTPUT_DIR} \
            --cuda
        ;;
    
    # Run model comparison experiment (LSTM vs GRU)
    "experiment")
        echo "Running model comparison experiment (LSTM vs GRU)..."
        echo "Results will be saved to ${OUTPUT_DIR}"
        python main.py \
            --experiment model_comparison \
            --epochs 100 \
            --batch_size 64 \
            --embedding_dim 256 \
            --hidden_dim 512 \
            --num_layers 3 \
            --dropout 0.6 \
            --lr 0.0005 \
            --optimizer adamw \
            --scheduler cosine \
            --data_augmentation \
            --tie_weights \
            --use_layer_norm \
            --use_residual \
            --use_attention \
            --patience 20 \
            --warmup_epochs 5 \
            --top_p 0.9 \
            --generate_length 300 \
            --strict_chinese \
            --grad_clip 3.0 \
            --weight_decay 1e-3 \
            --label_smoothing 0.1 \
            --log_interval 10 \
            --use_ema \
            --seed 42 \
            --output_dir ${OUTPUT_DIR} \
            --cuda
        ;;
    
    # Help information
    "help"|*)
        echo "Usage: ./run.sh [option]"
        echo "Options:"
        echo "  lstm       - Train LSTM model with optimized parameters for Tang poetry"
        echo "  gru        - Train GRU model with optimized parameters for Tang poetry"
        echo "  experiment - Run model comparison experiment (LSTM vs GRU)"
        echo "  help       - Show this help information"
        ;;
esac 