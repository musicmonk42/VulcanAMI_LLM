import argparse, pickle, torch, json, sys
from pathlib import Path

def load(p):
    """
    Load checkpoint with security restrictions.
    
    Security: Uses weights_only=True for torch.load to prevent code execution.
    For pickle files, uses restricted unpickler (if available).
    """
    try:
        # Use weights_only=True to prevent arbitrary code execution (CWE-502)
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception as e:
        # If torch.load fails, try pickle with safety checks
        print(f"[warn] torch.load failed: {e}, trying pickle...")
        try:
            from src.vulcan.security_fixes import safe_pickle_load
            with open(p, "rb") as f:
                return safe_pickle_load(f)
        except ImportError:
            print("[SECURITY WARNING] safe_pickle_load not available, using unsafe pickle.load")
            print("[SECURITY WARNING] This could allow arbitrary code execution!")
            with open(p, "rb") as f:
                return pickle.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="checkpoint_final_0.pkl or auto")
    ap.add_argument("--dump-json", action="store_true", help="Dump system_state as JSON if possible")
    args = ap.parse_args()

    obj = load(args.checkpoint)
    if not isinstance(obj, dict):
        print("[error] top-level not dict")
        sys.exit(1)

    print("[info] top-level keys:", list(obj.keys()))
    ss = obj.get("system_state")
    if ss is None:
        print("[warn] no system_state key")
        sys.exit(0)

    if isinstance(ss, dict):
        print("[info] system_state subkeys:", list(ss.keys())[:50])
        # Look for config-like structures
        def search(d, path=()):
            if isinstance(d, dict):
                for k,v in d.items():
                    np = path+(k,)
                    if isinstance(v, (dict,list,tuple)):
                        search(v, np)
                    else:
                        ks = ".".join(np)
                        if any(s in k.lower() for s in ["head","n_head","num_heads","d_model","hidden","emb","layers"]):
                            print(f"[cfg] {ks} = {v}")
            elif isinstance(d, (list,tuple)):
                for i,v in enumerate(d):
                    search(v, path+(str(i),))
        search(ss)

        if args.dump_json:
            try:
                out = "system_state_dump.json"
                with open(out,"w",encoding="utf-8") as f:
                    json.dump(ss,f,indent=2)
                print(f"[info] wrote {out}")
            except Exception as e:
                print("[warn] could not dump system_state:", e)
    else:
        print(f"[warn] system_state not a dict: {type(ss)}")

if __name__ == "__main__":
    main()