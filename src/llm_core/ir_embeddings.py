from __future__ import annotations

"""
IREmbeddings (Enhanced for Rotary Positional Embeddings - RoPE)

Creates the IR subgraph for token embeddings. Positional encoding is signaled
via parameters for RoPE application during Attention Q/K projection.
"""

from typing import Any, Dict


class IREmbeddings:
    def build_ir(
        self,
        vocab_size: int,
        hidden_size: int,
        max_positions: int,
        dropout_p: float = 0.1,
    ) -> Dict[str, Any]:
        return {
            "type": "embedding_subgraph",
            "params": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "max_positions": max_positions,
                # CRITICAL ENHANCEMENT: Mode hint for RoPE (not additive lookup)
                "pos_mode": "rotary",
                "pos_base": 10000,  # Base frequency for RoPE calculation
                # HIGH ENHANCEMENT: Hint for dynamic vocab pruning
                "prune_vocab_below_freq": 5,
            },
            "nodes": [
                # Keeping detailed nodes for maximum trace/debugging capability:
                {
                    "id": "emb_tokens",
                    "type": "lookup",
                    "params": {"table": "token", "size": hidden_size},
                },
                # Positional embedding node removed as RoPE is applied externally (in Attention)
                # The token embedding now goes directly to combination/dropout.
                {
                    "id": "emb_combine",
                    "type": "combine",
                    "params": {"mode": "none"},
                },  # No combination needed if no position vector
                # New Node: Dropout for Regularization
                {"id": "emb_dropout", "type": "dropout", "params": {"p": dropout_p}},
                {
                    "id": "emb_norm",
                    "type": "transform",
                    "params": {"operation": "norm_scale"},
                },
            ],
            "edges": [
                # Connect token lookup directly to dropout/normalization path
                {"src": "emb_tokens", "dst": "emb_combine"},
                {"src": "emb_combine", "dst": "emb_dropout"},
                {
                    "src": "emb_dropout",
                    "dst": "emb_norm",
                },  # Connect dropout to normalization
            ],
            "metadata": {
                "semantic": "token_embedding_rope_configured",
                "execution_hint": "memory_bound_lookup_and_fuse_with_norm_scale",
            },
        }
