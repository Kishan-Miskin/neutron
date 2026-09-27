import torch
import tiktoken
from gpt2_scratch import GPT2, GPT2Config

# 1. Match the configuration used during training
config = GPT2Config(
    vocab_size=50257,
    block_size=32,
    n_layer=4,
    n_head=4,
    n_embd=128
)

# 2. Initialize architecture & load saved weights
model = GPT2(config)
model.load_state_dict(torch.load("custom_gpt2_weights.pt"))
model.eval()

# 3. Test prompt
enc = tiktoken.get_encoding("gpt2")
prompt = "Fine-tuning adapts"
tokens = torch.tensor([enc.encode(prompt)], dtype=torch.long)

with torch.no_grad():
    output_tokens = model.generate(tokens, max_new_tokens=20, temperature=0.5, top_k=5)

print("=== Output from Saved Checkpoint ===")
print(enc.decode(output_tokens[0].tolist()))
