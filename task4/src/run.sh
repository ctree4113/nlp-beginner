#!/bin/bash

# Set default language to English, use provided parameter if available
LANGUAGE=${1:-eng}

export PYTHONPATH="$PYTHONPATH:$(pwd)"

mkdir -p ../output

echo "Training model for language: $LANGUAGE"

python main.py \
    --data_dir ../dataset \
    --output_dir ../output \
    --language $LANGUAGE \
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
    --epochs 50 \
    --use_lr_scheduler \
    --scheduler_patience 3 \
    --scheduler_factor 0.5 \
    --grad_clip 5.0 \
    --weight_decay 1e-4 \
    --early_stopping 5 \
    --cuda \
    --seed 42 