from __future__ import annotations
"""
IRFeedForward (Enhanced for SwiGLU Architecture with Dynamic Gating)

Builds FFN subgraph using a SwiGLU structure, including nodes for:
Linear (gate) & Linear (expand) -> SwiGLU -> Linear (project) -> Dropout -> Residual (scaled).
The new ffn_dynamic_gate node signals sparse/MoE functionality.
"""

from typing import Any, Dict


class IRFeedForward:
    def build_ir(self, hidden_size: int, intermediate_size: int, dropout_p: float = 0.1) -> Dict[str, Any]:
        
        # SwiGLU architectures typically use three parallel linear layers before the activation/gating.

        return {
            "type": "ffn_subgraph",
            "params": {
                "hidden_size": hidden_size, 
                "intermediate": intermediate_size,
                "dropout_p": dropout_p,
                "residual_scale": 0.5,
                # CRITICAL ENHANCEMENT: Dynamic Gating/MoE configuration
                "moe_experts": 4, 
                "moe_top_k": 2
            },
            "nodes": [
                {"id": "ffn_input", "type": "input", "params": {"size": hidden_size}},
                
                # CRITICAL ENHANCEMENT: Dynamic Gating Node
                # This node takes the input and determines which FFN expert(s) (the SwiGLU path) to activate.
                {"id": "ffn_dynamic_gate", "type": "moe_gate", "params": {"experts": 4, "mode": "top_k_select"}},
                
                {"id": "ffn_gate", "type": "transform", "params": {"operation": "linear_gate", "to": intermediate_size}},
                
                {"id": "ffn_expand", "type": "transform", "params": {"operation": "linear_expand", "to": intermediate_size}},
                
                {"id": "ffn_swiglu", "type": "filter", "params": {"operation": "swiglu"}},
                
                {"id": "ffn_project", "type": "transform", "params": {"operation": "linear_project", "to": hidden_size}},
                
                {"id": "ffn_dropout", "type": "dropout", "params": {"p": dropout_p}},

                {"id": "ffn_residual", "type": "combine", "params": {"mode": "add_scale", "scale": 0.5}}
                
            ],
            "edges": [
                # Input flows into the Dynamic Gate
                {"src": "ffn_input", "dst": "ffn_dynamic_gate"},
                
                # Dynamic Gate output flows into the parallel input projections (conceptually activating them)
                {"src": "ffn_dynamic_gate", "dst": "ffn_gate"},
                {"src": "ffn_dynamic_gate", "dst": "ffn_expand"},
                
                # Feed into SwiGLU activation
                {"src": "ffn_gate", "dst": "ffn_swiglu"},
                {"src": "ffn_expand", "dst": "ffn_swiglu"},
                
                # Project back
                {"src": "ffn_swiglu", "dst": "ffn_project"},
                
                # Apply dropout
                {"src": "ffn_project", "dst": "ffn_dropout"},
                
                # Combine dropout output with original input (residual)
                {"src": "ffn_dropout", "dst": "ffn_residual"},
                {"src": "ffn_input", "dst": "ffn_residual"}
            ],
            "metadata": {
                "semantic": "feedforward_swiglu_moe",
                "optimization_hint": "fuse_all_linear_and_swiglu_into_ffn_block",
                "sparse_activation": True
            }
        }