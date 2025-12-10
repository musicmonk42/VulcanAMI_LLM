import argparse
import sys
from pathlib import Path

import torch

"""
Inspect a .pt file to see if it is:
- A full model object (has .forward)
- A dict with 'state_dict'
- A raw state_dict

Usage:
  python inspect_pt.py --checkpoint ./logs/run1/llm_last_model.pt
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    p = Path(args.checkpoint)
    if not p.exists():
        print(f"[error] File not found: {p}")
        sys.exit(1)

    obj = torch.load(str(p), map_location="cpu")
    print(f"[info] Loaded object type: {type(obj)}")

    # Case 1: full model object
    if hasattr(obj, "forward"):
        print("[info] This looks like a full model object (has .forward).")
        print("[info] You can evaluate directly once we know how to tokenize.")
        # Try a quick attribute listing
        public = [a for a in dir(obj) if not a.startswith("_")]
        print("[info] Sample public attributes:", public[:20])
        if hasattr(obj, "tokenizer"):
            print("[info] Found tokenizer attribute on model.")
        return

    # Case 2: dict with 'state_dict'
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
        keys = list(sd.keys())
        print(f"[info] Dict with 'state_dict' containing {len(keys)} keys.")
        for k in keys[:args.limit]:
            print("  ", k)
        return

    # Case 3: raw state_dict
    if isinstance(obj, dict):
        # Heuristic: many tensor values
        tensor_keys = [k for k, v in obj.items() if isinstance(v, torch.Tensor)]
        if len(tensor_keys) > 10:
            print(f"[info] Raw state_dict with {len(tensor_keys)} tensor entries.")
            for k in tensor_keys[:args.limit]:
                print("  ", k)
            return
        else:
            print("[warn] Dict but not a typical state_dict. Keys:", list(obj.keys())[:20])
            return

    print("[warn] Unrecognized format; may need original training script.")

if __name__ == "__main__":
    main()
