#!/bin/bash

MODE_NAME = "BAAI/bge-base-en-v1.5"
TRAIN_DATA_PATH = "./data/triplets.jsonl"
OUTPUT_DIR = "/workspace/medical-bge-base-v0"

LEARNING_RATE = 2e-5

torchrun \
    -m FlagEmbedding.baai_general_embedding.finetune.run \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODE_NAME} \
    --train_data ${TRAIN_DATA_PATH} \
    --learning_rate ${LEARNING_RATE} \
    --bf16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 128 \
    --passage_max_len 256 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 500 \
    --query_instruction_for_retrieval "" \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine
