import math
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

"""
Evaluate a pickled checkpoint (.pkl) IF:
- It contains a full model object with .forward
OR
- It contains a dict with 'state_dict' and you can provide --model-class (NOT implemented yet)
OR
- It contains a raw state_dict AND you have a trivial model wrapper (NOT implemented yet)

This script ONLY handles the easiest case: the .pkl holds a full model object.

If inspection shows it's just a state_dict, STOP and tell me; I'll give you a different version.

Usage (full model object case):
  python simple_eval_pkl.py --checkpoint checkpoints/checkpoint_final_0.pkl --vfile validation.txt --max-len 256

Validation file: one sample per line.

Tokenizer:
If the loaded model has model.tokenizer with .encode_with_bos or .encode, we use it.
Otherwise we error (need to supply a tokenizer separately in a future version).

Steps:
1. Load the .pkl via torch.load.
2. Expect a model object (has .forward).
3. For each line: get token ids, do next-token loss.

If your tokenizer only has .encode(text) add BOS manually (id=1) if that matches your special tokens.
Adjust BOS_ID if different.
"""

BOS_ID = 1  # Change if your tokenizer uses a different BOS id

@torch.no_grad()
def evaluate(model, tokenizer, vfile, device, max_len):
    total_loss = 0.0
    total_tokens = 0
    lines = 0

    path = Path(vfile)
    if not path.exists():
        print(f"[error] validation file not found: {vfile}")
        sys.exit(1)

    with open(vfile, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue

            # Get ids
            if hasattr(tokenizer, "encode_with_bos"):
                ids = tokenizer.encode_with_bos(text)
            elif hasattr(tokenizer, "encode"):
                base = tokenizer.encode(text)
                ids = [BOS_ID] + base
            else:
                print("[error] tokenizer lacks encode methods.")
                return

            if len(ids) > max_len:
                ids = ids[:max_len]

            if len(ids) < 2:
                continue

            inp = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
            tgt = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)

            logits = model(inp)  # (1, seq, vocab)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pkl file (full model object)")
    ap.add_argument("--vfile", required=True, help="Validation text file")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device in ("auto","cuda") else "cpu"
    print(f"[info] device: {device}")

    # Security: Use weights_only=True to prevent arbitrary code execution (CWE-502)
    # This prevents malicious pickle files from executing code during deserialization
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not hasattr(ckpt, "forward"):
        print("[error] This checkpoint does NOT contain a full model object.")
        print("        Run inspect_pkl_checkpoint.py and paste output here.")
        sys.exit(1)

    model = ckpt.to(device)
    model.eval()

    # Try to find tokenizer
    tokenizer = None
    for attr_name in ("tokenizer", "tok", "tokenizer_", "tokenizer_obj"):
        if hasattr(model, attr_name):
            tokenizer = getattr(model, attr_name)
            break

    if tokenizer is None:
        print("[error] Model object does not have an attached tokenizer attribute.")
        print("        We would need to load tokenizer separately. Tell me and I'll modify the script.")
        sys.exit(1)

    print(f"[info] Found tokenizer on model: {tokenizer.__class__.__name__}")

    evaluate(model, tokenizer, args.vfile, device, args.max_len)


if __name__ == "__main__":
    main()