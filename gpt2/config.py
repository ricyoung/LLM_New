# config.py

# General settings
WANDB_PROJECT = "gpt2-from-scratch"
WANDB_RUN_NAME = "run_1"

# Model configuration
MODEL_CONFIG = {
    "vocab_size": 50257,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "block_size": 1024,
    "dropout": 0.1,
}

# Training configuration
TRAIN_CONFIG = {
    "learning_rate": 6e-4,
    "batch_size": 12,
    "num_epochs": 1,
    "max_iters": 600000,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "decay_lr": True,
    "warmup_iters": 2000,
    "lr_decay_iters": 600000,
    "min_lr": 6e-5,
}

# Data configuration
DATA_CONFIG = {
    "dataset": "openwebtext",
    "data_dir": "data",
    "train_split": 0.9,
}

# Evaluation configuration
EVAL_CONFIG = {
    "eval_interval": 2000,
    "eval_iters": 200,
    "log_interval": 10,
}

# Generation configuration
GEN_CONFIG = {
    "num_samples": 10,
    "max_new_tokens": 500,
    "temperature": 0.8,
    "top_k": 200,
}
