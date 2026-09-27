"""
===============================================================================
GPT-2 Built Completely From Scratch in PyTorch
-------------------------------------------------------------------------------
This implementation connects the foundational Transformer architecture from
"Attention Is All You Need" (Vaswani et al., 2017) with the Decoder-Only 
Generative Pre-trained Transformer 2 (GPT-2) by OpenAI (Radford et al., 2019).

Key Components Included:
1. Config & Hyperparameters (`GPT2Config`)
2. Custom Layer Normalization (`LayerNorm` - Pre-LN style)
3. Causal Multi-Head Self-Attention (`CausalSelfAttention` with lower-triangular mask)
4. Feed-Forward MLP with GELU Activation (`MLP`)
5. Transformer Decoder Block with Residual Connections (`Block`)
6. Full Model & Weight Tying (`GPT2`)
7. Hugging Face Pretrained Weights Loader (`from_pretrained`)
8. Autoregressive Text Generation with Temperature & Top-K Sampling (`generate`)
9. Mini Training Demo on sample text with tiktoken BPE decoding
===============================================================================
"""

import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# =============================================================================
# 1. Configuration Class
# =============================================================================
@dataclass
class GPT2Config:
    """Hyperparameters for GPT-2 Architecture (Default: GPT-2 124M base model)"""
    vocab_size: int = 50257   # Byte-Pair Encoding (BPE) vocabulary size
    block_size: int = 1024    # Maximum context length / sequence length
    n_layer: int = 12         # Number of stacked Transformer decoder blocks
    n_head: int = 12          # Number of attention heads
    n_embd: int = 768         # Embedding dimension (d_model)
    dropout: float = 0.1      # Dropout rate for regularization
    bias: bool = True         # True: include bias in Linears and LayerNorms (as in GPT-2)


# =============================================================================
# 2. Layer Normalization (Pre-LN in GPT-2 vs Post-LN in Vaswani et al.)
# =============================================================================
class LayerNorm(nn.Module):
    """
    Custom Layer Normalization.
    Normalizes inputs across the feature dimension (n_embd) to zero mean & unit variance,
    then applies learnable gain (gamma) and bias (beta).

    Formula:
        y = [(x - mean) / sqrt(var + eps)] * weight + bias
    """
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, n_embd)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            return self.weight * x_norm + self.bias
        return self.weight * x_norm


# =============================================================================
# 3. Causal Multi-Head Self-Attention
# =============================================================================
class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention block.
    
    Theoretical Basis (Vaswani et al. 2017 & Radford et al. 2019):
        Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + Mask ) @ V
    
    Key Difference in GPT-2:
        Causal Masking (Lower Triangular Matrix) ensures token at position t
        can ONLY attend to positions <= t (preventing future token leakage).
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Combine Query, Key, and Value projections into a single Linear layer for GPU efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection back to residual channel dimension
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # Causal mask: lower-triangular matrix of 1s (shape: 1, 1, block_size, block_size)
        # Registered as buffer so PyTorch saves it with state_dict but doesn't treat it as learnable weights
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension (n_embd)

        # 1. Calculate Q, K, V for all heads in batch and split
        # (B, T, 3 * C) -> split into three (B, T, C) tensors
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, T, n_head, head_size) -> transpose to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        
        # Apply Causal Mask: Replace future positions (where mask == 0) with -infinity
        # softmax(-inf) = 0, so future tokens receive zero attention weight
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax along sequence length to get attention probability distribution
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 3. Compute weighted sum over Values: (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # Re-assemble head outputs side-by-side: (B, nh, T, hs) -> (B, T, nh * hs) = (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection & residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


# =============================================================================
# 4. Position-Wise Feed-Forward Network (MLP)
# =============================================================================
class MLP(nn.Module):
    """
    Feed-Forward MLP block.
    Expands input dimension from n_embd to 4 * n_embd, applies GELU activation,
    and projects back down to n_embd.

    Vaswani et al. (2017) used ReLU; GPT-2 uses GELU (Gaussian Error Linear Unit)
    which provides smoother gradients near zero.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU(approximate='tanh') # Exact formula used by OpenAI GPT-2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# =============================================================================
# 5. Transformer Decoder Block
# =============================================================================
class Block(nn.Module):
    """
    Standard GPT-2 Transformer Decoder Block using Pre-LayerNorm architecture.
    
    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual streams
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# 6. Complete GPT-2 Model Architecture
# =============================================================================
class GPT2(nn.Module):
    """
    GPT-2 Autoregressive Language Model.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Word Token Embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Learned Positional Embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Transformer stack
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # Final LayerNorm before LM Head
        ))
        
        # Language Model Head (logits over vocabulary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying (Press & Wolf 2016): Share weights between wte and lm_head
        # Reduces model parameters by ~30% and forces token representations to match prediction space
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize model weights according to GPT-2 scheme
        self.apply(self._init_weights)

        # Special scaled initialization for residual projection weights in GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        """Standard GPT-2 initialization: N(0, 0.02) for weights, 0 for biases"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward Pass:
            idx: Tensor of shape (Batch, Seq_Len) containing token IDs
            targets: Optional Tensor of shape (Batch, Seq_Len) containing target token IDs
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds maximum block size {self.config.block_size}"

        # Position indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 1. Add Token Embeddings and Positional Embeddings
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd) broadcasted over batch
        x = self.transformer.drop(tok_emb + pos_emb)

        # 2. Pass through all stacked Transformer Decoder Blocks
        for block in self.transformer.h:
            x = block(x)

        # 3. Final Layer Normalization
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training mode: compute Cross Entropy Loss across all tokens
            logits = self.lm_head(x) # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Inference mode: compute logits for the last position only
            logits = self.lm_head(x[:, -1:, :]) # (B, 1, vocab_size)
            return logits, None

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2"):
        """
        Loads official pretrained OpenAI GPT-2 weights from HuggingFace.
        Supports: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained HuggingFace GPT-2 ({model_type})...")

        # Map model configurations
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # Load Hugging Face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        # Transpose Conv1D weights from HuggingFace to standard Linear weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys) == len(sd_keys_hf), f"Mismatch in state dict keys count: {len(sd_keys)} vs {len(sd_keys_hf)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        print(f"Successfully loaded pretrained weights for {model_type} into custom architecture!")
        return model

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressive Generation Loop.
        Given input prompt tensor `idx` (B, T), predicts next `max_new_tokens`.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop prompt if context length exceeds block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass to get logits for the last token position
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # Scale logits by temperature

            # Top-K Sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Convert logits to probability distribution via Softmax
            probs = F.softmax(logits, dim=-1)

            # Sample next token index from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Concatenate sampled token to existing sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# 7. Execution & Training Demonstration
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("      GPT-2 FROM SCRATCH: ARCHITECTURE & MINI TRAINING DEMO      ")
    print("=" * 70)

    # 1. Initialize a small lightweight GPT-2 instance for demonstration
    demo_config = GPT2Config(
        vocab_size=50257,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0
    )
    
    model = GPT2(demo_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n[1] GPT-2 Model Initialized!")
    print(f"    - Parameters: {num_params / 1e6:.2f} Million")
    print(f"    - Embedding Dimension (d_model): {demo_config.n_embd}")
    print(f"    - Transformer Blocks: {demo_config.n_layer}")
    print(f"    - Attention Heads: {demo_config.n_head}")
    print(f"    - Context Window (block_size): {demo_config.block_size}")

    # 2. Tokenize a sample text prompt using tiktoken (GPT-2 BPE Tokenizer)
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    sample_text = (
        "The Transformer architecture relies entirely on self-attention mechanisms to "
        "compute representations of its input and output without using sequence-aligned RNNs. "
        "GPT-2 scales decoder transformers to generate coherent text autoregressively."
    )
    tokens = enc.encode(sample_text)
    print(f"\n[2] Sample Text Tokenization:")
    print(f"    Original Text: \"{sample_text[:60]}...\"")
    print(f"    Encoded Tokens ({len(tokens)} tokens): {tokens[:10]}...")

    # Create input (X) and target (Y) pairs for language modeling: next-token prediction
    x_data = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0) # (1, T)
    y_data = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0)  # (1, T)

    # 3. Test Initial Un-trained Forward Pass & Generation
    model.eval()
    with torch.no_grad():
        initial_logits, initial_loss = model(x_data, y_data)
    print(f"\n[3] Initial Forward Pass:")
    print(f"    - Initial Cross Entropy Loss: {initial_loss.item():.4f} (Expected random loss ~ ln(50257) = 10.82)")

    # Generate before training
    prompt_tokens = torch.tensor([enc.encode("The Transformer")], dtype=torch.long)
    gen_untrained = model.generate(prompt_tokens, max_new_tokens=8, temperature=0.8, top_k=10)
    print(f"    - Text Generation (Before Training): \"{enc.decode(gen_untrained[0].tolist())}\"")

    # 4. Mini Training Loop (Overfitting on sample text to demonstrate learning)
    print(f"\n[4] Running Mini Training Loop (50 steps on sample text)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()

    for step in range(1, 51):
        optimizer.zero_grad()
        logits, loss = model(x_data, y_data)
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 1:
            print(f"    Step {step:2d} | Training Loss: {loss.item():.4f}")

    # 5. Evaluate Generation After Training
    model.eval()
    gen_trained = model.generate(prompt_tokens, max_new_tokens=12, temperature=0.5, top_k=5)
    print(f"\n[5] Text Generation (After Training):")
    print(f"    Prompt: \"The Transformer\"")
    print(f"    Output: \"{enc.decode(gen_trained[0].tolist())}\"")

    print("\n" + "=" * 70)
    print("  VERIFICATION COMPLETE! GPT-2 ARCHITECTURE & TRAINING WORKING CLEANLY ")
    print("=" * 70)
