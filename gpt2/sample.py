import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPT
from config import MODEL_CONFIG, GEN_CONFIG

init_from = 'resume'
out_dir = 'out'
start = "\n"
num_samples = GEN_CONFIG['num_samples']
max_new_tokens = GEN_CONFIG['max_new_tokens']
temperature = GEN_CONFIG['temperature']
top_k = GEN_CONFIG['top_k']
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = checkpoint['model_args']
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from == 'scratch':
    print("Initializing from scratch")
    gptconf = MODEL_CONFIG
    model = GPT(gptconf)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

load_meta = False
if init_from == 'resume' and 'data' in checkpoint and 'dataset' in checkpoint['data']:
    meta_path = os.path.join('data', checkpoint['data']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
else:
    meta_path = os.path.join('data', 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("Using GPT-2 tiktoken encoding")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={""})
    decode = lambda l: enc.decode(l)

if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')