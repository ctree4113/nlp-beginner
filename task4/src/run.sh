#!/bin/bash

# Set default language to English, use provided parameter if available
OPTION=${1:-eng}

export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Handle the 'clean' command to remove output directory
if [ "$OPTION" = "clean" ]; then
    echo "Cleaning output directory..."
    rm -rf ../output
    mkdir -p ../output
    echo "Output directory cleaned successfully."
    exit 0
fi

# Handle the 'compare' command to run model comparison experiment
if [ "$OPTION" = "compare" ]; then
    echo "Running model comparison experiment..."
    
    # Create comparison directory
    mkdir -p ../output/comparison
    
    # Run Base BiLSTM model (no CRF, no char CNN)
    echo "Running Base BiLSTM model..."
    python ./main.py \
        --data_dir ../dataset \
        --output_dir ../output/comparison/base_bilstm \
        --language eng \
        --max_seq_len 128 \
        --embedding_dim 100 \
        --hidden_dim 300 \
        --num_layers 2 \
        --dropout 0.5 \
        --lr 0.001 \
        --batch_size 32 \
        --epochs 30 \
        --use_lr_scheduler \
        --scheduler_patience 3 \
        --scheduler_factor 0.5 \
        --grad_clip 5.0 \
        --weight_decay 1e-4 \
        --early_stopping 10 \
        --cuda \
        --seed 42 \
        --model_type bilstm \
        --comparison_mode
    
    # Run BiLSTM-CRF model (no char CNN)
    echo "Running BiLSTM-CRF model..."
    python ./main.py \
        --data_dir ../dataset \
        --output_dir ../output/comparison/bilstm_crf \
        --language eng \
        --max_seq_len 128 \
        --embedding_dim 100 \
        --hidden_dim 300 \
        --num_layers 2 \
        --dropout 0.5 \
        --lr 0.001 \
        --batch_size 32 \
        --epochs 30 \
        --use_lr_scheduler \
        --scheduler_patience 3 \
        --scheduler_factor 0.5 \
        --grad_clip 5.0 \
        --weight_decay 1e-4 \
        --early_stopping 10 \
        --cuda \
        --seed 42 \
        --model_type bilstm_crf \
        --comparison_mode
    
    # Run BiLSTM-CRF with Char CNN
    echo "Running BiLSTM-CRF with Char CNN model..."
    python ./main.py \
        --data_dir ../dataset \
        --output_dir ../output/comparison/bilstm_crf_char \
        --language eng \
        --max_seq_len 128 \
        --max_word_len 20 \
        --embedding_dim 100 \
        --hidden_dim 300 \
        --num_layers 2 \
        --dropout 0.5 \
        --use_char_cnn \
        --char_embedding_dim 30 \
        --char_channel_size 50 \
        --lr 0.001 \
        --batch_size 32 \
        --epochs 30 \
        --use_lr_scheduler \
        --scheduler_patience 3 \
        --scheduler_factor 0.5 \
        --grad_clip 5.0 \
        --weight_decay 1e-4 \
        --early_stopping 10 \
        --cuda \
        --seed 42 \
        --model_type bilstm_crf_char \
        --comparison_mode
    
    # Generate comparison visualizations
    echo "Generating comparison visualizations..."
    python ./main.py \
        --comparison_visualization \
        --output_dir ../output
    
    echo "Model comparison completed. Results saved to ../output/comparison"
    exit 0
fi

mkdir -p ../output

echo "Training model for language: $OPTION"

python ./main.py \
    --data_dir ../dataset \
    --output_dir ../output \
    --language $OPTION \
    --max_seq_len 128 \
    --max_word_len 20 \
    --embedding_dim 100 \
    --hidden_dim 300 \
    --num_layers 2 \
    --dropout 0.5 \
    --use_char_cnn \
    --char_embedding_dim 10 \
    --char_channel_size 25 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 30 \
    --use_lr_scheduler \
    --scheduler_patience 3 \
    --scheduler_factor 0.5 \
    --grad_clip 5.0 \
    --weight_decay 1e-4 \
    --early_stopping 10 \
    --cuda \
    --seed 42 \