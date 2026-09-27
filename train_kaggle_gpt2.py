import os
import urllib.request
import torch
import tiktoken
from gpt2_scratch import GPT2, GPT2Config

# =============================================================================
# 1. Dataset Downloader (Kaggle / Open Dataset Loader)
# =============================================================================
DATASET_PATH = "input_dataset.txt"

def download_sample_dataset():
    """Downloads TinyShakespeare (standard LLM benchmark dataset) if dataset isn't present."""
    if not os.path.exists(DATASET_PATH):
        print("Downloading dataset (TinyShakespeare)...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, DATASET_PATH)
        print(f"Dataset downloaded and saved to: {DATASET_PATH}")

download_sample_dataset()

# Read the dataset
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    text_data = f.read()

print(f"\n[1] Dataset Statistics:")
print(f"    - Total Characters: {len(text_data):,}")

# Tokenize using GPT-2 BPE Tokenizer
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text_data)
print(f"    - Total BPE Tokens: {len(tokens):,}")

# Convert tokens to PyTorch tensor
data_tensor = torch.tensor(tokens, dtype=torch.long)

# =============================================================================
# 2. Model Configuration (Scalable GPT-2 Architecture)
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[2] Training Setup:")
print(f"    - Compute Device: {device.upper()}")

# Lightweight GPT-2 configuration tuned for fast local training
config = GPT2Config(
    vocab_size=50257,
    block_size=128,   # Sequence length / Context window
    n_layer=6,        # Transformer decoder blocks
    n_head=8,         # Multi-head attention heads
    n_embd=256,       # Embedding dimension (d_model)
    dropout=0.1
)

model = GPT2(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"    - Model Parameters: {num_params / 1e6:.2f} Million")

# =============================================================================
# 3. Batch Sampler
# =============================================================================
def get_batch(data, batch_size, block_size):
    """Samples random sequences (X) and shifted next-token targets (Y)."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# =============================================================================
# 4. Training Loop with Learning Rate Scheduler
# =============================================================================
batch_size = 8
max_steps = 300
learning_rate = 1e-3

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

print(f"\n[3] Starting Training ({max_steps} steps)...")
model.train()

for step in range(1, max_steps + 1):
    x, y = get_batch(data_tensor, batch_size, config.block_size)
    
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if step % 50 == 0 or step == 1:
        print(f"    Step {step:4d} / {max_steps} | Training Loss: {loss.item():.4f}")

# Save the trained model checkpoint
checkpoint_file = "gpt2_kaggle_trained.pt"
torch.save(model.state_dict(), checkpoint_file)
print(f"\n[4] Trained Weights Saved: {checkpoint_file}")

# =============================================================================
# 5. Text Generation Test (Autoregressive Sampling)
# =============================================================================
model.eval()
prompt_text = "To be, or not to be"
prompt_tokens = torch.tensor([enc.encode(prompt_text)], dtype=torch.long).to(device)

print(f"\n[5] Generating Text After Training:")
print(f"    Prompt: \"{prompt_text}\"")

with torch.no_grad():
    generated = model.generate(prompt_tokens, max_new_tokens=40, temperature=0.7, top_k=20)

generated_text = enc.decode(generated[0].tolist())
print(f"\nGenerated Output:\n{'-'*50}\n{generated_text}\n{'-'*50}")
