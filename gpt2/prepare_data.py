import os
import pickle
import requests
import numpy as np
import tiktoken
from tqdm import tqdm

input_file_path = os.path.join('data', 'input.txt')
if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists(input_file_path):
    print("Downloading shakespeare text...")
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(data)
print(f"train has {len(train_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)

train_bin = 'data/train.bin'
val_bin = 'data/val.bin'
train_ids[:int(len(train_ids)*0.9)].tofile(train_bin)
train_ids[int(len(train_ids)*0.9):].tofile(val_bin)

meta = {
    'vocab_size': 50257,
    'itos': {i: enc.decode([i]) for i in range(50257)},
    'stoi': {enc.decode([i]): i for i in range(50257)},
}
with open('data/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved {len(train_ids[:int(len(train_ids)*0.9)]):,} tokens to {train_bin}")
print(f"Saved {len(train_ids[int(len(train_ids)*0.9):]):,} tokens to {val_bin}")