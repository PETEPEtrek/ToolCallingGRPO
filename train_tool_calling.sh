#!/bin/bash

#SBATCH --job-name=grpo_qwen3
#SBATCH --error=cluster_logs/qwen3/logs/grpo_qwen3_3.err
#SBATCH --output=cluster_logs/qwen3/logs/grpo_qwen3_3.log
#SBATCH --partition=h100
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=8                 # хотя в вашем примере стояло 1, для multi-gpu лучше дать побольше CPU
#SBATCH --nodes=1                # Одна нода
#SBATCH --gres=gpu:1
#SBATCH --mem=0

export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=2000000 
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Запускаем Accelerate с тем же .yaml-конфигом, который мы сохранили ранее.
# Accelerate сам подхватит настройку из ~/.cache/huggingface/accelerate/default_config.yaml
#accelerate launch \
#  --num_processes 2 \
#  --mixed_precision bf16 \
#  advanced_grpo2.py
python grpo_tool_calling_new.py
