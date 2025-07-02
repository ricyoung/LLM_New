import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from safetensors.torch import save_file, load_file

from model import GPT
from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, EVAL_CONFIG

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'])
parser.add_argument('--block_size', type=int, default=MODEL_CONFIG['block_size'])
parser.add_argument('--n_layer', type=int, default=MODEL_CONFIG['n_layer'])
parser.add_argument('--n_head', type=int, default=MODEL_CONFIG['n_head'])
parser.add_argument('--n_embd', type=int, default=MODEL_CONFIG['n_embd'])
parser.add_argument('--max_iters', type=int, default=TRAIN_CONFIG['max_iters'])
parser.add_argument('--eval_interval', type=int, default=EVAL_CONFIG['eval_interval'])
parser.add_argument('--eval_iters', type=int, default=EVAL_CONFIG['eval_iters'])
parser.add_argument('--log_interval', type=int, default=EVAL_CONFIG['log_interval'])
parser.add_argument('--compile', type=str, default='True')
parser.add_argument('--learning_rate', type=float, default=TRAIN_CONFIG['learning_rate'])
parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'], help='Precision to use for training')
args = parser.parse_args()

out_dir = 'out'
eval_interval = args.eval_interval
log_interval = args.log_interval
eval_iters = args.eval_iters
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

wandb_log = os.environ.get('WANDB_API_KEY') is not None
wandb_project = 'gpt2-safety-testing'
wandb_run_name = f'gpt2-{args.n_layer}L-{args.n_head}H-{args.n_embd}D'

dataset = DATA_CONFIG['dataset']
batch_size = args.batch_size
block_size = args.block_size
bias = False
learning_rate = args.learning_rate
max_iters = args.max_iters
weight_decay = TRAIN_CONFIG['weight_decay']
beta1 = TRAIN_CONFIG['beta1']
beta2 = TRAIN_CONFIG['beta2']
grad_clip = TRAIN_CONFIG['grad_clip']
decay_lr = TRAIN_CONFIG['decay_lr']
warmup_iters = TRAIN_CONFIG['warmup_iters']
lr_decay_iters = TRAIN_CONFIG['lr_decay_iters']
min_lr = TRAIN_CONFIG['min_lr']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Determine dtype based on argument or auto-detect
if args.dtype == 'auto':
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
else:
    dtype = args.dtype
compile = args.compile.lower() == 'true'

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert batch_size % ddp_world_size == 0
    batch_size //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

gradient_accumulation_steps = 5 * 8
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size * ddp_world_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = 'data'
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    block_size=args.block_size,
    bias=bias,
    vocab_size=None,
    dropout=MODEL_CONFIG['dropout'],
)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = model_args
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    
    # Check for safetensors format first
    safetensors_path = os.path.join(out_dir, 'model.safetensors')
    training_state_path = os.path.join(out_dir, 'training_state.pt')
    legacy_path = os.path.join(out_dir, 'ckpt.pt')
    
    if os.path.exists(safetensors_path) and os.path.exists(training_state_path):
        print("Loading from safetensors format...")
        # Load training state
        training_state = torch.load(training_state_path, map_location=device)
        checkpoint_model_args = training_state['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = model_args
        model = GPT(gptconf)
        
        # Load model weights from safetensors
        state_dict = load_file(safetensors_path, device=str(device))
        model.load_state_dict(state_dict)
        
        iter_num = training_state['iter_num']
        best_val_loss = training_state['best_val_loss']
        optimizer_state = training_state['optimizer']
    elif os.path.exists(legacy_path):
        print("Loading from legacy .pt format...")
        checkpoint = torch.load(legacy_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = model_args
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        optimizer_state = checkpoint['optimizer']
    else:
        raise FileNotFoundError(f"No checkpoint found in {out_dir}")

model.to(device)

# GradScaler is only needed for float16, not for bfloat16 or float32
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16')) if device_type == 'cuda' else None

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(optimizer_state)
checkpoint = None
optimizer_state = None

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

if wandb_log and master_process:
    import wandb
    config_wandb = {
        'batch_size': batch_size * ddp_world_size,
        'block_size': block_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
        'learning_rate': learning_rate,
        'max_iters': max_iters,
        'weight_decay': weight_decay,
        'grad_accumulation_steps': gradient_accumulation_steps,
        'model_params': raw_model.get_num_params(),
        'dataset': dataset,
        'device': device,
        'compile': compile,
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=config_wandb)
    wandb.watch(raw_model, log='all', log_freq=100)
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            try:
                train_perplexity = torch.exp(torch.tensor(losses['train'])).item()
                val_perplexity = torch.exp(torch.tensor(losses['val'])).item()
            except:
                train_perplexity = float('inf')
                val_perplexity = float('inf')
            
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/perplexity": train_perplexity,
                "val/perplexity": val_perplexity,
                "train/accuracy": 1.0 / train_perplexity if train_perplexity != float('inf') else 0.0,
                "val/accuracy": 1.0 / val_perplexity if val_perplexity != float('inf') else 0.0,
                "learning_rate": lr,
                "mfu": running_mfu * 100,
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                # Save model with safetensors
                model_path = os.path.join(out_dir, 'model.safetensors')
                save_file(raw_model.state_dict(), model_path)
                
                # Save training state with pickle (optimizer, metadata)
                training_state = {
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(training_state, os.path.join(out_dir, 'training_state.pt'))
                
                # Also save legacy format for compatibility
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                
                print(f"saving checkpoint to {out_dir} (safetensors + legacy format)")
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    if grad_clip != 0.0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        # Log metrics to W&B at every log_interval (more frequent)
        if wandb_log:
            try:
                perplexity = torch.exp(torch.tensor(lossf)).item() if lossf < 50 else float('inf')
            except:
                perplexity = float('inf')
            
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "train/perplexity": perplexity,
                "train/accuracy": 1.0 / perplexity if perplexity != float('inf') else 0.0,
                "learning_rate": lr,
                "performance/iter_time_ms": dt * 1000,
                "performance/tokens_per_sec": tokens_per_iter / dt,
                "performance/mfu": running_mfu * 100,
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
            })
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()