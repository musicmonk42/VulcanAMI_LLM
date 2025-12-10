import argparse
import importlib
import math
import os
import sys
from pathlib import Path

# If you use torch for loss calculation; uncomment if required:
import torch
import torch.nn.functional as F

# Add src/ to sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def _dynamic_import(path):
    """
    Import a class given its full dotted path, e.g., llm_core.graphix_transformer.GraphixTransformer.
    """
    parts = path.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls

def load_tokenizer(tokenizer_class_path, vocab_path):
    """
    Load a tokenizer either via .load() classmethod or via its constructor, as needed.
    """
    TokClass = _dynamic_import(tokenizer_class_path)
    if hasattr(TokClass, "load"):
        tokenizer = TokClass.load(vocab_path)
    else:
        tokenizer = TokClass(vocab_path)
    return tokenizer

def _load_model(model_class_path, checkpoint_path, device):
    """
    Load custom model using its class-based loader.
    This implementation uses .load(path) classmethod if available.
    """
    ModelClass = _dynamic_import(model_class_path)
    if hasattr(ModelClass, "load"):
        model = ModelClass.load(checkpoint_path)
    else:
        model = ModelClass(checkpoint_path)
    # Optionally set to eval mode
    if hasattr(model, 'eval'):
        model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path for the model weights/state")
    parser.add_argument("--model-class", required=True, help="Dotted path to model class e.g. llm_core.graphix_transformer.GraphixTransformer")
    parser.add_argument("--tokenizer-class", required=True, help="Dotted path to tokenizer class e.g. local_llm.tokenizer.simple_tokenizer.SimpleTokenizer")
    parser.add_argument("--vocab", required=True, help="Path to vocab JSON or tokenizer weights")
    parser.add_argument("--vfile", required=True, help="Validation file (one text per line)")
    parser.add_argument("--device", default="cpu", help="Device type (default cpu)")
    parser.add_argument("--max-len", type=int, default=None, help="Max tokens per line")
    args = parser.parse_args()

    device = args.device

    print(f"[info] device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_class, args.vocab)
    try:
        vocab_sz = len(tokenizer.word_to_id)
        lower = getattr(tokenizer, 'lowercase', 'n/a')
        print(f"[info] tokenizer vocab size: {vocab_sz} (lowercase={lower})")
    except Exception:
        vocab_sz = getattr(tokenizer, 'vocab_size', 'n/a')
        print(f"[info] tokenizer vocab size: {vocab_sz}")

    # Load model
    model = _load_model(args.model_class, args.checkpoint, device)

    total_loss = 0.0
    total_tokens = 0
    line_count = 0
    stats = {"lines": 0, "tokens": 0, "avg_loss": 0.0, "perplexity": 0.0}

    with open(args.vfile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Get token ids using .encode() if present, else .tokenize()
            ids = None
            if hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(line)
            elif hasattr(tokenizer, "tokenize"):
                ids = tokenizer.tokenize(line)
            else:
                raise ValueError("Tokenizer missing encode/tokenize method")

            if args.max_len:
                ids = ids[:args.max_len]
            if len(ids) < 2:
                continue

            # Forward pass for all except last token for next-token prediction
            # This expects model.forward and model.executor.get_logits per your implementation
            out = None
            if hasattr(model, "forward"):
                out = model.forward(ids[:-1])
            elif callable(model):
                out = model(ids[:-1])
            elif hasattr(model, "encode"):
                out = model.encode(ids[:-1])
            else:
                raise ValueError("Model missing forward/encode/callable method for inference")

            if hasattr(model, "executor") and hasattr(model.executor, "get_logits"):
                logits = model.executor.get_logits(out.get("hidden_states", []), ids[:-1])
            elif "logits" in out:
                logits = out["logits"]
            else:
                raise ValueError("Model output missing 'logits' or model.executor.get_logits()")

            # Convert to torch tensor as needed for cross-entropy loss
            logits = torch.tensor(logits)
            target = torch.tensor(ids[1:])

            # If logits shape is wrong, fix shape (should be [seq_len, vocab_size])
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.shape[0] != target.shape[0]:
                # Try to fix by length cropping; you may want to adapt as per your model output
                logits = logits[:target.shape[0], :]

            ce_loss = F.cross_entropy(logits, target, reduction="mean")
            total_loss += ce_loss.item() * target.shape[0]
            total_tokens += target.shape[0]
            line_count += 1

    stats["lines"] = line_count
    stats["tokens"] = total_tokens
    stats["avg_loss"] = total_loss / total_tokens if total_tokens > 0 else float("nan")
    stats["perplexity"] = math.exp(stats["avg_loss"]) if total_tokens > 0 else float("nan")

    print("----- Evaluation -----")
    print(f"Lines:       {stats['lines']}")
    print(f"Tokens:      {stats['tokens']}")
    print(f"Avg Loss:    {stats['avg_loss']:.4f}")
    print(f"Perplexity:  {stats['perplexity']:.2f}")

if __name__ == "__main__":
    main()
