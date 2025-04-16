# INITIAL

uv run train.py \
  --train_data_path ../archive/tiny_train.npy \
  --val_data_path ../archive/tiny_valid.npy \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 20000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 200 \
  --cosine_iters 19800 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "checkpoint.pt" \
  --wandb_project "transformer-tuning"

# LR ABLATION
# Set min_lr=0.1*max_lr
# Use 10% warmup
# Try max_lr 1e-5 1e-4 1e-3 1e-2

uv run train.py \
  --train_data_path ../archive/tiny_train.npy \
  --val_data_path ../archive/tiny_valid.npy \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 20000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 2000 \
  --cosine_iters 18000 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "checkpoint.pt" \
  --wandb_project "transformer-tuning"

# BATCH ABLATION
# Tried batch_size 1, 16, 64, 128, TODO

uv run train.py \
  --train_data_path ../archive/tiny_train.bin \
  --val_data_path ../archive/tiny_valid.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 1 \
  --max_iters 1280000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 128000 \
  --cosine_iters 1152000 \
  --grad_clip 1.0 \
  --log_interval 6400 \
  --ckpt_interval 64000 \
  --ckpt_path "lr_1e-3_bs_1.pt" \
  --wandb_project "transformer-tuning"

uv run train.py \
  --train_data_path ../archive/tiny_train.bin \
  --val_data_path ../archive/tiny_valid.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 16 \
  --max_iters 80000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 8000 \
  --cosine_iters 72000 \
  --grad_clip 1.0 \
  --log_interval 400 \
  --ckpt_interval 4000 \
  --ckpt_path "lr_1e-3_bs_16.pt" \
  --wandb_project "transformer-tuning"

uv run train.py \
  --train_data_path ../archive/tiny_train.bin \
  --val_data_path ../archive/tiny_valid.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 128 \
  --max_iters 10000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 1000 \
  --cosine_iters 9000 \
  --grad_clip 1.0 \
  --log_interval 50 \
  --ckpt_interval 500 \
  --ckpt_path "lr_1e-3_bs_128.pt" \
  --wandb_project "transformer-tuning" \

# This experiment was terminated because of storage but resuming was possible, yay!
# --resume "lr_1e-3_bs_128.pt"

uv run train.py \
  --train_data_path ../archive/tiny_train.bin \
  --val_data_path ../archive/tiny_valid.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 512 \
  --max_iters 2500 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 250 \
  --cosine_iters 2250 \
  --grad_clip 1.0 \
  --log_interval 25 \
  --ckpt_interval 100 \
  --ckpt_path "lr_1e-3_bs_512.pt" \
  --wandb_project "transformer-tuning"

# RMS NORM ABLATION

uv run train.py \
  --train_data_path ../archive/tiny_train.bin \
  --val_data_path ../archive/tiny_valid.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 20000 \
  --max_lr 1e-2 \
  --min_lr 1e-3 \
  --warmup_iters 2000 \
  --cosine_iters 18000 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "no_rms_lr_1e-2.pt" \
  --wandb_project "transformer-tuning"

# The above one directly diverges (NaN after 500 steps)

uv run train.py \
  --train_data_path ../archive/tiny_train.bin \
  --val_data_path ../archive/tiny_valid.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 20000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 2000 \
  --cosine_iters 18000 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "no_rms_lr_1e-3.pt" \
  --wandb_project "transformer-tuning"

# NOPE ABLATION
uv run train.py \
  --train_data_path ../archive/tiny_train.bin \
  --val_data_path ../archive/tiny_valid.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 20000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 2000 \
  --cosine_iters 18000 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "nope_lr_1e-3.pt" \
  --wandb_project "transformer-tuning"

# POSTNORM ABLATION

CUDA_VISIBLE_DEVICES=1 uv run train.py \
  --train_data_path ../archive/tiny_train.bin \
  --val_data_path ../archive/tiny_valid.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 20000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 2000 \
  --cosine_iters 18000 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "postnorm_lr_1e-3.pt" \
  --wandb_project "transformer-tuning"

# SILU ABLATION
uv run train.py \
  --train_data_path ../archive/tiny_train.bin \
  --val_data_path ../archive/tiny_valid.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 2048 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 20000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 2000 \
  --cosine_iters 18000 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "silu_lr_1e-3.pt" \
  --wandb_project "transformer-tuning"

# OWT NAIVE
uv run train.py \
  --train_data_path ../archive/owt_train.bin \
  --val_data_path ../archive/owt_valid.bin \
  --vocab_size 32000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 20000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 2000 \
  --cosine_iters 18000 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "owt_1e-3.pt" \
  --wandb_project "transformer-tuning"

# LEADERBOARD

#!/bin/bash
#SBATCH --job-name=owt-radi
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH -c 1
#SBATCH --time=01:30:00
#SBATCH --output=owt-radi_%j.out
#SBATCH --error=owt-radi_%j.err

uv run train.py \
  --train_data_path ../archive/owt_train.bin \
  --val_data_path ../archive/owt_valid.bin \
  --vocab_size 32000 \
  --context_length 512 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 12000 \
  --max_lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 1200 \
  --cosine_iters 10800 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "owt_1e-3_context_512.pt" \
  --wandb_project "leaderboard"

# TIED WEIGHTS, BIGGER MODEL
uv run train.py \
  --train_data_path ../archive/owt_train.bin \
  --val_data_path ../archive/owt_valid.bin \
  --vocab_size 32000 \
  --context_length 512 \
  --d_model 768 \
  --d_ff 2048 \
  --num_layers 6 \
  --num_heads 16 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 8000 \
  --max_lr 8e-4 \
  --min_lr 8e-5 \
  --warmup_iters 800 \
  --cosine_iters 7200 \
  --grad_clip 1.0 \
  --log_interval 100 \
  --ckpt_interval 1000 \
  --ckpt_path "owt_8e-4_context_512_tie_d768.pt" \
  --wandb_project "leaderboard"
