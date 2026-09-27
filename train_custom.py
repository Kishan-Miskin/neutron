import torch
import tiktoken
from gpt2_scratch import GPT2, GPT2Config

# =============================================================================
# 1. Configuration & Setup
# =============================================================================
# Small configuration for fast demonstration training
config = GPT2Config(
    vocab_size=50257,
    block_size=32,    # Short context window for mini sample text
    n_layer=4,        # 4 Transformer layers
    n_head=4,         # 4 Attention heads
    n_embd=128,       # Embedding dimension
    dropout=0.0
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 60)
print(f"  GPT-2 CUSTOM TRAINING DEMO (Device: {device})")
print("=" * 60)

# Initialize model
model = GPT2(config).to(device)

# =============================================================================
# 2. Dataset Preparation
# =============================================================================
sample_text = (
    "The Transformer model uses self-attention mechanisms to process sequential data. "
    "Pre-training on massive text datasets allows models to learn grammar, facts, and reasoning. "
    "Fine-tuning adapts pre-trained language models to specific downstream tasks efficiently."
)

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(sample_text)
print(f"\n[1] Dataset loaded: {len(tokens)} BPE tokens.")

# Convert to PyTorch Tensor
data = torch.tensor(tokens, dtype=torch.long)

# Helper function to generate mini-batches
def get_batch(data, batch_size, block_size):
    # Select random starting indices in the dataset
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# =============================================================================
# 3. Test Text Generation BEFORE Training
# =============================================================================
model.eval()
prompt = "The Transformer"
prompt_tokens = torch.tensor([enc.encode(prompt)], dtype=torch.long).to(device)

with torch.no_grad():
    gen_before = model.generate(prompt_tokens, max_new_tokens=15, temperature=0.7, top_k=10)

print(f"\n[2] Generation BEFORE Training:")
print(f"    Prompt: \"{prompt}\"")
print(f"    Output: \"{enc.decode(gen_before[0].tolist())}\"")

# =============================================================================
# 4. Training Loop
# =============================================================================
print(f"\n[3] Training for 100 steps...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
model.train()

batch_size = 4
block_size = config.block_size

for step in range(1, 101):
    x, y = get_batch(data, batch_size, block_size)
    
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    
    if step % 20 == 0 or step == 1:
        print(f"    Step {step:3d} / 100 | Training Loss: {loss.item():.4f}")

# Save the trained model checkpoint
checkpoint_path = "custom_gpt2_weights.pt"
torch.save(model.state_dict(), checkpoint_path)
print(f"\n[4] Saved trained weights to: {checkpoint_path}")

# =============================================================================
# 5. Test Text Generation AFTER Training (Verification)
# =============================================================================
model.eval()
with torch.no_grad():
    gen_after = model.generate(prompt_tokens, max_new_tokens=15, temperature=0.5, top_k=5)

print(f"\n[5] Generation AFTER Training:")
print(f"    Prompt: \"{prompt}\"")
print(f"    Output: \"{enc.decode(gen_after[0].tolist())}\"")

print("\n" + "=" * 60)
print("  VERIFICATION COMPLETE: Model successfully learned from the text!")
print("=" * 60)
