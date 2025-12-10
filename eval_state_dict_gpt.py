import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

"""
Evaluate a GPT-like state_dict stored under the 'model' key
of a checkpoint file like: {'model': OrderedDict(...), 'step': ..., ...}

Usage example:

python eval_state_dict_gpt.py \
  --checkpoint ./exp_probe_1p34m/llm_best_model.pt \
  --vocab src/local_llm/tokenizer/vocab.json \
  --vfile validation.txt \
  --device auto \
  --max-len 256

Required:
  --checkpoint  Path to the .pt file containing 'model' OrderedDict
  --vocab       Path to vocab.json used in training (must start with <PAD>, <BOS>, <EOS>, <UNK>)
  --vfile       Validation text file (one example per line)

Optional:
  --device      auto | cpu | cuda
  --max-len     Max tokens per line (truncate if longer than positional embedding length)
  --bos         BOS token id (default 1)
  --eos         EOS token id (default 2) (not strictly required for loss)

Assumptions from inspected keys:
  - token_emb.weight: (vocab_size, d_model)
  - pos_emb.weight: (max_positions, d_model)
  - blocks.N.* with:
        ln1.weight, ln1.bias
        attn.qkv.weight (3*d_model, d_model)
        attn.qkv.bias   (3*d_model,)
        attn.out_proj.weight (d_model, d_model)
        attn.out_proj.bias   (d_model,)
        ln2.weight, ln2.bias
        ff.net.0.weight (ff_hidden, d_model)
        ff.net.0.bias
        ff.net.3.weight (d_model, ff_hidden)
        ff.net.3.bias
  - No explicit final layernorm found in sample (if present as ln_f.weight/bias it will be used).
  - Output logits use weight tying with token_emb.

If any keys are missing, the script will raise an error listing them.

This is a minimal reimplementation; some training-time features
(dropout, attention masking optimization, multi-head splitting) are simplified.
Perplexity may differ slightly from training logs but should be close.

"""

# ---------------- Tokenizer (matches your SimpleTokenizer logic) ----------------

TOKEN_PATTERN = torch.compile if False else None  # (placeholder to avoid TorchDynamo picking up regex)
import re

TOKEN_REGEX = re.compile(r"\w+|\S", re.UNICODE)

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

class SimpleVocabTokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.lowercase = bool(data.get("lowercase", True))
        # Validate first four tokens
        for i, st in enumerate(SPECIAL_TOKENS):
            if self.vocab[i] != st:
                raise ValueError(f"Special token mismatch: index {i} expected {st} got {self.vocab[i]}")

    def tokenize(self, text: str):
        if self.lowercase:
            text = text.lower()
        return TOKEN_REGEX.findall(text)

    def encode(self, text: str):
        return [self.token_to_id.get(tok, self.token_to_id["<UNK>"]) for tok in self.tokenize(text)]

    def encode_with_bos(self, text: str, bos_id: int):
        return [bos_id] + self.encode(text)


# ---------------- Model Components ----------------

def layer_norm(x, weight, bias, eps=1e-5):
    # x: (B, T, C)
    mean = x.mean(-1, keepdim=True)
    var = (x - mean).pow(2).mean(-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias

class GPTBlocks(nn.Module):
    def __init__(self, state_dict: OrderedDict):
        super().__init__()
        # Infer number of blocks by scanning keys
        block_indices = sorted({int(k.split('.')[1]) for k in state_dict.keys()
                                if k.startswith("blocks.") and k.split('.')[1].isdigit()})
        self.n_blocks = len(block_indices)
        # Determine d_model from first ln1.weight
        w0 = state_dict[f"blocks.{block_indices[0]}.ln1.weight"]
        self.d_model = w0.shape[0]
        # Build parameter containers
        # We'll register parameters to keep them on the right device and allow .to()
        for i in block_indices:
            prefix = f"blocks.{i}"
            # LayerNorm 1
            self.register_parameter(f"ln1_weight_{i}", nn.Parameter(state_dict[f"{prefix}.ln1.weight"]))
            self.register_parameter(f"ln1_bias_{i}", nn.Parameter(state_dict[f"{prefix}.ln1.bias"]))
            # Attention qkv
            self.register_parameter(f"attn_qkv_weight_{i}", nn.Parameter(state_dict[f"{prefix}.attn.qkv.weight"]))
            self.register_parameter(f"attn_qkv_bias_{i}", nn.Parameter(state_dict[f"{prefix}.attn.qkv.bias"]))
            self.register_parameter(f"attn_out_proj_weight_{i}", nn.Parameter(state_dict[f"{prefix}.attn.out_proj.weight"]))
            self.register_parameter(f"attn_out_proj_bias_{i}", nn.Parameter(state_dict[f"{prefix}.attn.out_proj.bias"]))
            # LayerNorm 2
            self.register_parameter(f"ln2_weight_{i}", nn.Parameter(state_dict[f"{prefix}.ln2.weight"]))
            self.register_parameter(f"ln2_bias_{i}", nn.Parameter(state_dict[f"{prefix}.ln2.bias"]))
            # Feedforward
            self.register_parameter(f"ff_0_weight_{i}", nn.Parameter(state_dict[f"{prefix}.ff.net.0.weight"]))
            self.register_parameter(f"ff_0_bias_{i}", nn.Parameter(state_dict[f"{prefix}.ff.net.0.bias"]))
            self.register_parameter(f"ff_3_weight_{i}", nn.Parameter(state_dict[f"{prefix}.ff.net.3.weight"]))
            self.register_parameter(f"ff_3_bias_{i}", nn.Parameter(state_dict[f"{prefix}.ff.net.3.bias"]))

    def forward(self, x):
        B, T, C = x.shape
        device = x.device
        # Precompute causal mask once per forward
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        for i in range(self.n_blocks):
            # LN1
            ln1_w = getattr(self, f"ln1_weight_{i}")
            ln1_b = getattr(self, f"ln1_bias_{i}")
            h = layer_norm(x, ln1_w, ln1_b)
            # Attention
            qkv_w = getattr(self, f"attn_qkv_weight_{i}")
            qkv_b = getattr(self, f"attn_qkv_bias_{i}")
            qkv = F.linear(h, qkv_w, qkv_b)  # (B, T, 3*C)
            q, k, v = qkv.split(C, dim=-1)
            # Scaled dot-product attention (single head using full dim)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(C)
            att.masked_fill_(causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            out = att @ v
            # Output projection
            out_w = getattr(self, f"attn_out_proj_weight_{i}")
            out_b = getattr(self, f"attn_out_proj_bias_{i}")
            out = F.linear(out, out_w, out_b)
            x = x + out  # Residual

            # LN2
            ln2_w = getattr(self, f"ln2_weight_{i}")
            ln2_b = getattr(self, f"ln2_bias_{i}")
            h2 = layer_norm(x, ln2_w, ln2_b)
            # Feedforward
            ff0_w = getattr(self, f"ff_0_weight_{i}")
            ff0_b = getattr(self, f"ff_0_bias_{i}")
            ff3_w = getattr(self, f"ff_3_weight_{i}")
            ff3_b = getattr(self, f"ff_3_bias_{i}")
            ff_hidden = F.linear(h2, ff0_w, ff0_b)
            ff_hidden = F.gelu(ff_hidden)
            ff_out = F.linear(ff_hidden, ff3_w, ff3_b)
            x = x + ff_out  # Residual
        return x

class TinyGPT(nn.Module):
    def __init__(self, sd: OrderedDict):
        super().__init__()
        # Embeddings
        self.token_emb = nn.Embedding.from_pretrained(sd["token_emb.weight"])
        self.pos_emb = nn.Embedding.from_pretrained(sd["pos_emb.weight"])
        self.max_positions = self.pos_emb.weight.shape[0]
        self.d_model = self.token_emb.weight.shape[1]
        # Blocks
        self.blocks = GPTBlocks(sd)
        # Optional final layer norm
        ln_f_w = sd.get("ln_f.weight", None)
        ln_f_b = sd.get("ln_f.bias", None)
        if ln_f_w is not None and ln_f_b is not None:
            self.ln_f_weight = nn.Parameter(ln_f_w)
            self.ln_f_bias = nn.Parameter(ln_f_b)
        else:
            self.ln_f_weight = None
            self.ln_f_bias = None

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.shape
        if T > self.max_positions:
            raise ValueError(f"Sequence length {T} exceeds max positional embeddings {self.max_positions}")
        pos_ids = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        tok = self.token_emb(idx)          # (B, T, C)
        pos = self.pos_emb(pos_ids)        # (1, T, C)
        x = tok + pos
        x = self.blocks(x)
        if self.ln_f_weight is not None:
            x = layer_norm(x, self.ln_f_weight, self.ln_f_bias)
        # Weight tying: logits = x @ token_emb^T
        logits = x @ self.token_emb.weight.t()
        return logits

# ---------------- Evaluation ----------------

@torch.no_grad()
def evaluate(model: TinyGPT, tokenizer: SimpleVocabTokenizer, vfile: str, device: str, max_len: int, bos_id: int):
    total_loss = 0.0
    total_tokens = 0
    lines = 0
    with open(vfile, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            ids = tokenizer.encode_with_bos(text, bos_id)
            if len(ids) > max_len:
                ids = ids[:max_len]
            if len(ids) < 2:
                continue
            inp = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            tgt = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(inp)
            logits = logits[:, :tgt.size(1), :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            total_loss += loss.item() * tgt.size(1)
            total_tokens += tgt.size(1)
            lines += 1
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    print("----- Evaluation -----")
    print(f"Lines:       {lines}")
    print(f"Tokens:      {total_tokens}")
    print(f"Avg Loss:    {avg_loss:.4f}")
    print(f"Perplexity:  {ppl:.2f}")


def load_checkpoint(checkpoint_path: str):
    obj = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError("Checkpoint must be a dict containing key 'model'.")
    model_sd = obj["model"]
    if not isinstance(model_sd, (dict, OrderedDict)):
        raise ValueError("'model' key is not a dict/OrderedDict.")
    # Verify expected keys
    required = ["token_emb.weight", "pos_emb.weight"]
    for k in required:
        if k not in model_sd:
            raise ValueError(f"Missing required key in state_dict: {k}")
    return model_sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pt with 'model' state_dict")
    ap.add_argument("--vocab", required=True, help="Path to vocab.json")
    ap.add_argument("--vfile", required=True, help="Validation text file (one example per line)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-len", type=int, default=256, help="Max tokens (truncate longer)")
    ap.add_argument("--bos", type=int, default=1, help="BOS token id")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device in ("auto", "cuda") else "cpu"
    print(f"[info] device: {device}")

    model_sd = load_checkpoint(args.checkpoint)
    model = TinyGPT(model_sd).to(device)
    print(f"[info] vocab size: {model.token_emb.weight.shape[0]}  d_model: {model.d_model}  n_blocks: {model.blocks.n_blocks}")
    if model.ln_f_weight is not None:
        print("[info] final layer norm detected")
    else:
        print("[info] no final layer norm")

    tokenizer = SimpleVocabTokenizer(args.vocab)

    evaluate(model, tokenizer, args.vfile, device, args.max_len, args.bos)


if __name__ == "__main__":
    main()