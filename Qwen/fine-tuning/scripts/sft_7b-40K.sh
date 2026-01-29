#!/bin/bash
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}

# DeepSpeed configuration
deepspeed=/fs04/scratch2/ml23/zlia0050/fine_tune/COMPLETELY_NEW_FROM_SCRATCH/pose-augmented-weapon-detection-using-machine-learning/qwen-vl-finetune/scripts/zero3.json
# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID
# Training hyperparameters
lr=5e-6
batch_size=3
grad_accum_steps=4
# Training entry point
entry_file=/fs04/scratch2/ml23/zlia0050/fine_tune/COMPLETELY_NEW_FROM_SCRATCH/pose-augmented-weapon-detection-using-machine-learning/qwen-vl-finetune/qwenvl/train/train_qwen.py
# Dataset configuration (replace with public dataset names)
datasets=40k_train
# Output configuration
run_name="qwen2vl-baseline"
output_dir=./output-40K
# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 462400 \
    --min_pixels 50176 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.1 \
    --max_grad_norm 0.5 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name ${run_name}"
# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
