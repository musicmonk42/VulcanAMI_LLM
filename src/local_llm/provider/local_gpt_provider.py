from __future__ import annotations

import json
import os
import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Iterable, Tuple, Generator, Union

import torch

# Reuse your training model
from src.training.gpt_model import GPTModel, GPTConfig
from src.local_llm.tokenizer.simple_tokenizer import SimpleTokenizer, BOS_ID, EOS_ID, UNK_ID

log = logging.getLogger("local_gpt_provider")


@dataclass
class ProviderInitConfig:
    model_path: str
    vocab_path: str
    device: str = "cpu"
    seq_len: int = 256
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    dtype: str = "float32"  # one of: float32, float16, bfloat16
    use_autocast: bool = False
    calibration_path: Optional[str] = None
    # generation defaults
    temperature: float = 0.9
    top_k: int = 64
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    eos_token: Optional[str] = None  # e.g., "</s>" if present in vocab


class OptionalCalibrator:
    """
    Placeholder confidence calibrator.
    If a calibration map is present at path, it can be used to remap confidences.
    Supports:
      - kind="scale" with params {"scale": s, "bias": b}
      - kind="temperature" with params {"T": float} for simple temperature scaling
      - kind="isotonic" (placeholder for future piecewise mapping)
    """
    def __init__(self, calib_path: Optional[str] = None):
        self.ready = False
        self.kind = None
        self.params: Dict[str, Any] = {}
        if calib_path and os.path.exists(calib_path):
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.kind = data.get("kind")
                self.params = data.get("params", {})
                self.ready = True
            except Exception as e:
                log.warning(f"Failed to load calibration map: {e}")
                self.ready = False

    def calibrate_prob(self, p: float) -> float:
        if not self.ready:
            return p
        if self.kind == "scale":
            s = float(self.params.get("scale", 1.0))
            b = float(self.params.get("bias", 0.0))
            return max(0.0, min(1.0, s * p + b))
        if self.kind == "temperature":
            # Simple temperature scaling in logit space: p^ (1/T) normalized;
            # but for a single prob we approximate by p' = sigmoid(logit(p)/T)
            T = max(1e-6, float(self.params.get("T", 1.0)))
            p = max(1e-12, min(1.0 - 1e-12, p))
            logit = math.log(p / (1 - p))
            p2 = 1.0 / (1.0 + math.exp(-logit / T))
            return max(0.0, min(1.0, p2))
        # TODO: isotonic mapping with piecewise linear interpolation
        return p


class LocalGPTProvider:
    """
    Adapter that loads a fine-tuned GPTModel and serves text generation, both single and batch,
    with optional streaming and perplexity/scoring utilities.

    Artifacts:
      - model_path: path to llm_best_model.pt (trainer output)
          Expected dict keys: {"model": state_dict, "step": int, "sched_state": dict} or a raw state_dict
      - vocab_path: path to vocab.json as exported by the exporter script

    Main APIs:
      - generate(prompt, ...)
      - generate_batch(prompts, ...)
      - generate_stream(prompt, ...) -> yields (partial_text, step_meta)
      - perplexity(text) / batch_perplexity(texts)
      - score_next_token_probs(prompt, continuation_len=k)  # debugging

    Initialization:
      - Prefer using LocalGPTProvider.from_config_file(path) to load ProviderInitConfig from provider_config.json
    """

    def __init__(self, cfg: ProviderInitConfig):
        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(f"model_path not found: {cfg.model_path}")
        if not os.path.exists(cfg.vocab_path):
            raise FileNotFoundError(f"vocab_path not found: {cfg.vocab_path}")

        self.cfg = cfg
        self.device = cfg.device
        self.dtype = self._resolve_dtype(cfg.dtype)

        self.tokenizer = SimpleTokenizer(cfg.vocab_path)
        vocab_size = len(self.tokenizer.vocab)

        model_cfg = GPTConfig(
            vocab_size=vocab_size,
            seq_len=cfg.seq_len,
            dim=cfg.dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
            tied_embeddings=True,
            device=cfg.device
        )
        self.model = GPTModel(model_cfg).to(cfg.device).eval()
        if self.dtype != torch.float32:
            self.model = self.model.to(self.dtype)

        ckpt = torch.load(cfg.model_path, map_location=cfg.device)
        state_dict = ckpt.get("model", ckpt)
        self.model.load_state_dict(state_dict)

        self.autocast_dtype = self._autocast_dtype(self.dtype)
        self.use_autocast = cfg.use_autocast and self.autocast_dtype is not None

        self.calibrator = OptionalCalibrator(cfg.calibration_path)

        # Prepare EOS ID resolution
        self.eos_id = None
        if cfg.eos_token and cfg.eos_token in self.tokenizer.token_to_id:
            self.eos_id = self.tokenizer.token_to_id[cfg.eos_token]
        # Fall back to training EOS
        if self.eos_id is None and "<EOS>" in self.tokenizer.token_to_id:
            self.eos_id = self.tokenizer.token_to_id["<EOS>"]

        log.info(f"LocalGPTProvider initialized (device={self.device}, dtype={self.dtype}, autocast={self.use_autocast})")

    # ----------------------
    # Helpers
    # ----------------------
    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        n = (name or "float32").lower()
        if n in ("fp32", "float32", "f32"):
            return torch.float32
        if n in ("fp16", "float16", "f16"):
            return torch.float16
        if n in ("bf16", "bfloat16"):
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _autocast_dtype(dtype: torch.dtype) -> Optional[torch.dtype]:
        if dtype == torch.float16:
            return torch.float16
        if dtype == torch.bfloat16:
            return torch.bfloat16
        return None

    @classmethod
    def from_config_file(cls, provider_config_path: str) -> "LocalGPTProvider":
        with open(provider_config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = ProviderInitConfig(
            model_path=raw["model_path"],
            vocab_path=raw["vocab_path"],
            device=raw.get("device", "cpu"),
            seq_len=int(raw.get("seq_len", 256)),
            dim=int(raw.get("dim", 384)),
            n_layers=int(raw.get("n_layers", 6)),
            n_heads=int(raw.get("n_heads", 8)),
            ff_mult=int(raw.get("ff_mult", 4)),
            dropout=float(raw.get("dropout", 0.0)),
            dtype=str(raw.get("dtype", "float32")),
            use_autocast=bool(raw.get("use_autocast", False)),
            calibration_path=raw.get("calibration_path"),
            temperature=float(raw.get("temperature", 0.9)),
            top_k=int(raw.get("top_k", 64)),
            top_p=float(raw.get("top_p", 0.95)),
            repetition_penalty=float(raw.get("repetition_penalty", 1.05)),
            eos_token=raw.get("eos_token")
        )
        return cls(cfg)

    def _gen_context_ids(self, prompt: str) -> List[int]:
        # Aligns with training input: prepend BOS
        return self.tokenizer.encode_with_bos(prompt)

    def _guard_sampling(self, temperature: float, top_k: int, top_p: float):
        if temperature <= 0:
            raise ValueError("temperature must be > 0 for sampling; set a tiny value for greedy fallback.")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

    # ----------------------
    # Public APIs
    # ----------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        eos_token: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from prompt. Returns (text, metadata).
        """
        if seed is not None:
            torch.manual_seed(seed)

        temperature = self.cfg.temperature if temperature is None else temperature
        top_k = self.cfg.top_k if top_k is None else top_k
        top_p = self.cfg.top_p if top_p is None else top_p
        repetition_penalty = self.cfg.repetition_penalty if repetition_penalty is None else repetition_penalty
        eos_id = self.eos_id
        if eos_token and eos_token in self.tokenizer.token_to_id:
            eos_id = self.tokenizer.token_to_id[eos_token]

        self._guard_sampling(temperature, top_k, top_p)

        start_ids = self._gen_context_ids(prompt)
        if self.use_autocast:
            with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):  # type: ignore[arg-type]
                gen_ids = self.model.generate(
                    start_ids=start_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_id=eos_id
                )
        else:
            gen_ids = self.model.generate(
                start_ids=start_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_id=eos_id
            )

        text = self.tokenizer.decode(gen_ids, strip_special=True)
        meta: Dict[str, Any] = {
            "prompt": prompt,
            "prompt_ids": start_ids,
            "generated_ids": gen_ids,
            "params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty
            }
        }
        return text, meta

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: Iterable[str],
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Simple batch wrapper that calls generate() per prompt.
        For true batched decoding, adapt GPTModel.generate to accept batched contexts.
        """
        outs: List[str] = []
        metas: List[Dict[str, Any]] = []
        for i, p in enumerate(prompts):
            s = None if seed is None else (seed + i)
            text, meta = self.generate(
                prompt=p,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=s
            )
            outs.append(text)
            metas.append(meta)
        return outs, metas

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        eos_token: Optional[str] = None,
        seed: Optional[int] = None,
        chunk_size: int = 1
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        Streaming generator that yields (partial_text, meta) every chunk_size new tokens.
        This wraps repeated calls to model.generate with incremental context (simple baseline).
        """
        if seed is not None:
            torch.manual_seed(seed)

        temperature = self.cfg.temperature if temperature is None else temperature
        top_k = self.cfg.top_k if top_k is None else top_k
        top_p = self.cfg.top_p if top_p is None else top_p
        repetition_penalty = self.cfg.repetition_penalty if repetition_penalty is None else repetition_penalty
        eos_id = self.eos_id
        if eos_token and eos_token in self.tokenizer.token_to_id:
            eos_id = self.tokenizer.token_to_id[eos_token]

        self._guard_sampling(temperature, top_k, top_p)

        ctx_ids = self._gen_context_ids(prompt)
        generated: List[int] = ctx_ids[:]
        emitted = 0

        for _ in range(max_new_tokens // max(1, chunk_size) + 1):
            step_ids = self.model.generate(
                start_ids=generated,
                max_new_tokens=max(1, chunk_size),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_id=eos_id
            )
            # model.generate returns full sequence; slice new tail:
            new_ids = step_ids[len(generated):]
            if not new_ids:
                break
            generated = step_ids
            emitted += len(new_ids)
            text = self.tokenizer.decode(generated, strip_special=True)
            meta = {
                "new_ids": new_ids,
                "total_len": len(generated),
                "emitted": emitted
            }
            yield text, meta
            if eos_id is not None and eos_id in new_ids:
                break
            if emitted >= max_new_tokens:
                break

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        """
        Compute perplexity by running one forward pass over the prompt (BOS + text).
        """
        ids = self.tokenizer.encode_with_bos(text)
        if len(ids) < 2:
            return float("inf")
        import torch.nn.functional as F
        x = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=self.device).unsqueeze(0)
        logits = self.model.forward(x)
        B, T, V = logits.size()
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T), reduction="mean")
        ppl = math.exp(min(100.0, float(loss.item())))
        return ppl

    @torch.no_grad()
    def batch_perplexity(self, texts: Iterable[str]) -> List[float]:
        return [self.perplexity(t) for t in texts]

    def shutdown(self):
        """
        Best-effort resource cleanup (esp. for CUDA).
        """
        try:
            del self.model
        except Exception:
            pass
        if self.device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# Convenience factory for external runtimes
def build_provider_from_artifacts(artifacts_dir: str) -> LocalGPTProvider:
    cfg_path = os.path.join(artifacts_dir, "provider_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"provider_config.json not found in {artifacts_dir}")
    return LocalGPTProvider.from_config_file(cfg_path)