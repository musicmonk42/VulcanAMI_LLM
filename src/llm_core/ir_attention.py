from __future__ import annotations

"""
IRAttention (Enhanced for Hybrid and Sparse Attention)

Multi-head attention IR structure:
- Supports Multi-Head Attention (MHA) and Grouped Query Attention (GQA).
- Integrates KV Caching, Causal Masking, Softmax, and Dropout.
- Adds nodes/hints for Hybrid Attention and Sparse Pattern optimization.
"""

from typing import Any, Dict, List


class IRAttention:
    def build_ir(
        self, num_heads: int, hidden_size: int, num_kv_heads: int | None = None
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # Determine if we're using Grouped Query Attention (GQA)
        is_gqa = num_kv_heads is not None and num_kv_heads < num_heads

        # Calculate size for K/V projections, which is based on num_kv_heads for GQA
        kv_hidden_size = (
            hidden_size if not is_gqa else hidden_size // num_heads * num_kv_heads
        )

        # Define the number of head loops for K/V projections
        num_kv_proj = num_heads if not is_gqa else num_kv_heads

        for h in range(num_heads):
            prefix = f"h{h}"

            # K/V projections only run up to num_kv_heads in GQA mode
            kv_prefix = f"kv{h % num_kv_proj}"

            # Q projection (Always per-head)
            nodes.append(
                {
                    "id": f"{prefix}_q",
                    "type": "transform",
                    "params": {"operation": "linear_q", "size": hidden_size},
                }
            )

            # K projection and KV cache (Only define these once per kv_head if using GQA)
            if h < num_kv_proj:
                nodes.extend(
                    [
                        {
                            "id": f"{kv_prefix}_k",
                            "type": "transform",
                            "params": {"operation": "linear_k", "size": hidden_size},
                        },
                        {
                            "id": f"{kv_prefix}_v",
                            "type": "transform",
                            "params": {"operation": "linear_v", "size": hidden_size},
                        },
                        # Enhancement 2: KV cache nodes
                        {
                            "id": f"{kv_prefix}_cache_k",
                            "type": "cache",
                            "params": {"axis": 0},
                        },
                        {
                            "id": f"{kv_prefix}_cache_v",
                            "type": "cache",
                            "params": {"axis": 0},
                        },
                    ]
                )
                edges.extend(
                    [
                        {"src": f"{kv_prefix}_k", "dst": f"{kv_prefix}_cache_k"},
                        {"src": f"{kv_prefix}_v", "dst": f"{kv_prefix}_cache_v"},
                    ]
                )

            # Enhancement: Hybrid Attention Node (Replaces scaled_dot)
            nodes.append(
                {
                    "id": f"{prefix}_scores",
                    "type": "generative",
                    "params": {
                        "operation": "hybrid_attn",
                        "causal": True,
                        "kv_head_ref": kv_prefix,
                        "linear_ratio": 0.5,  # Critical Enhancement: Hybrid Attention ratio
                    },
                }
            )

            # Enhancement 3 & 6: Add Softmax and Dropout
            nodes.extend(
                [
                    {
                        "id": f"{prefix}_softmax",
                        "type": "softmax",
                        "params": {"dim": -1},
                    },
                    {"id": f"{prefix}_attn_drop", "type": "dropout"},
                    {
                        "id": f"{prefix}_weighted",
                        "type": "transform",
                        "params": {"operation": "weight_values"},
                    },
                    {
                        "id": f"{prefix}_out",
                        "type": "transform",
                        "params": {"operation": "linear_o", "size": hidden_size},
                    },
                    # Enhancement: Spiking Gate
                    {
                        "id": f"{prefix}_spike_gate",
                        "type": "spike",
                        "params": {"threshold": 0.3},
                    },
                ]
            )

            # Edges for attention head calculation (using KV Cache outputs)
            edges.extend(
                [
                    {"src": f"{prefix}_q", "dst": f"{prefix}_scores"},
                    {
                        "src": f"{kv_prefix}_cache_k",
                        "dst": f"{prefix}_scores",
                    },  # From KV Cache K
                    {"src": f"{prefix}_scores", "dst": f"{prefix}_softmax"},
                    {
                        "src": f"{prefix}_softmax",
                        "dst": f"{prefix}_attn_drop",
                    },  # Dropout after Softmax
                    {
                        "src": f"{kv_prefix}_cache_v",
                        "dst": f"{prefix}_weighted",
                    },  # From KV Cache V
                    {
                        "src": f"{prefix}_attn_drop",
                        "dst": f"{prefix}_weighted",
                    },  # Use dropped scores for weighting
                    {
                        "src": f"{prefix}_weighted",
                        "dst": f"{prefix}_spike_gate",
                    },  # Output goes into spike gate
                    {
                        "src": f"{prefix}_spike_gate",
                        "dst": f"{prefix}_out",
                    },  # Spiked output goes into final projection
                ]
            )

        # Head Merge (Combine)
        nodes.append(
            {
                "id": "attn_merge",
                "type": "combine",
                "params": {"mode": "mean", "heads": num_heads},
            }
        )
        for h in range(num_heads):
            edges.append({"src": f"h{h}_out", "dst": "attn_merge"})

        return {
            "type": "attention_subgraph",
            "params": {
                "num_heads": num_heads,
                "hidden_size": hidden_size,
                "num_kv_heads": num_kv_heads,
                "causal_masking": True,
                "sparse": "windowed",  # Enhancement: Sparse Pattern hint
                "window_size": 128,
            },
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "semantic": "hybrid_grouped_attention",
                "inference": "kv_cache_supported",
                "sampling": {"type": "top_p", "p": 0.9},
            },
        }
