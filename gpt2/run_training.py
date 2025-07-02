import subprocess
import sys

print("Starting GPT-2 training on Shakespeare dataset...")
print("This will train for a limited number of iterations for testing.")
print("-" * 50)

cmd = [
    sys.executable, "train.py",
    "--batch_size=4",
    "--block_size=64", 
    "--n_layer=4",
    "--n_head=4", 
    "--n_embd=128",
    "--max_iters=100",
    "--eval_interval=20",
    "--eval_iters=10",
    "--log_interval=10",
    "--compile=False"
]

subprocess.run(cmd)