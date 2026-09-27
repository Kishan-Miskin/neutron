import torch
import tiktoken
from datasets import load_dataset
from gpt2_scratch import GPT2

# =============================================================================
# 1. Load SlimOrca Dataset (Streaming for maximum speed and efficiency)
# =============================================================================
print("=" * 65)
print("  FINE-TUNING GPT-2 ON SLIM-ORCA INSTRUCTION DATASET")
print("=" * 65)

print("\n[1] Loading 'Open-Orca/SlimOrca' dataset...")
# Load in streaming mode so we don't need to download tens of gigabytes
ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
enc = tiktoken.get_encoding("gpt2")

print("    Dataset loaded successfully!")

# Parse SlimOrca dialogues into Q: ... A: ... instruction pairs
formatted_texts = []
print("\n[2] Processing SlimOrca instruction dialogues...")

for idx, item in enumerate(ds):
    if idx >= 500: # Take first 500 high-quality conversation pairs for training
        break
    
    conversations = item.get("conversations", [])
    user_msg = ""
    gpt_msg = ""
    
    for turn in conversations:
        from_role = turn.get("from", "")
        val = turn.get("value", "")
        if from_role in ["human", "user"]:
            user_msg = val
        elif from_role in ["gpt", "assistant"]:
            gpt_msg = val
            
    if user_msg and gpt_msg:
        # Format into Q: ... A: ... instruction prompt structure
        formatted_str = f"Q: {user_msg.strip()}\nA: {gpt_msg.strip()}\n<|endoftext|>"
        formatted_texts.append(formatted_str)

print(f"    Extracted {len(formatted_texts)} instruction dialogue samples.")

# Tokenize formatted dataset
full_text = "\n".join(formatted_texts)
data_tokens = torch.tensor(enc.encode(full_text, allowed_special="all"), dtype=torch.long)
print(f"    Total Training Tokens: {len(data_tokens):,}")

# =============================================================================
# 2. Model Setup & Device
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[3] Model Setup:")
print(f"    - Compute Device: {device.upper()}")

# Initialize model with OpenAI pretrained weights
print("    - Loading OpenAI GPT-2 Pre-trained Weights (124M)...")
model = GPT2.from_pretrained("gpt2").to(device)

# =============================================================================
# 3. Batch Sampler
# =============================================================================
batch_size = 4
block_size = 128

def get_batch():
    ix = torch.randint(0, len(data_tokens) - block_size - 1, (batch_size,))
    x = torch.stack([data_tokens[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data_tokens[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

# =============================================================================
# 4. Fine-Tuning Loop
# =============================================================================
max_steps = 150
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

print(f"\n[4] Starting Instruction Fine-Tuning ({max_steps} steps)...")
model.train()

for step in range(1, max_steps + 1):
    x, y = get_batch()
    
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if step % 25 == 0 or step == 1:
        print(f"    Step {step:3d} / {max_steps} | Loss: {loss.item():.4f}")

# Save fine-tuned checkpoint
checkpoint_path = "gpt2_slimorca_trained.pt"
torch.save(model.state_dict(), checkpoint_path)
print(f"\n[5] Saved SlimOrca fine-tuned weights: {checkpoint_path}")

# =============================================================================
# 5. Verification Test
# =============================================================================
model.eval()
test_prompt = "Q: Who is Leonardo da Vinci?\nA:"
tokens = torch.tensor([enc.encode(test_prompt)], dtype=torch.long).to(device)

print(f"\n[6] Evaluation after SlimOrca Fine-Tuning:")
print(f"    Prompt: \"{test_prompt}\"")

with torch.no_grad():
    gen = model.generate(tokens, max_new_tokens=45, temperature=0.3, top_k=20)

out_text = enc.decode(gen[0].tolist())
print(f"\nResponse:\n{'-'*50}\n{out_text}\n{'-'*50}")
