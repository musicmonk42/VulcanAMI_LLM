from __future__ import annotations
"""
IRLayerNorm (Enhanced for RMSNorm and GroupNorm Support)

Builds a small IR subgraph for normalization, supporting RMSNorm and standard LayerNorm.
Integrates GroupNorm as an optional node and adds hints for learned epsilon scheduling.
"""

from typing import Any, Dict, Optional  # ✅ FIXED: Added Optional import


class IRLayerNorm:
    def build_ir(self, hidden_size: int, eps: float, norm_type: str = "rmsnorm", num_groups: Optional[int] = None) -> Dict[str, Any]:
        
        # Determine the normalization mode and positional hint
        is_rmsnorm = (norm_type == "rmsnorm")
        
        nodes = [
            {"id": "ln_input", "type": "input", "params": {"size": hidden_size}},
            # RMSNorm uses variance/square mean, standard LayerNorm uses mean and variance
            {"id": "ln_var", "type": "reduce_var_sq", "params": {"axis": -1}}, # Using reduce_var_sq for RMS
            {"id": "ln_norm", "type": "transform", "params": {"operation": "normalize", "eps": eps}},
            {"id": "ln_scale", "type": "transform", "params": {"operation": "scale"}},
            # RMSNorm typically omits the 'shift' (beta/bias) parameter
        ]

        edges = [
            {"src": "ln_input", "dst": "ln_var"},
            {"src": "ln_var", "dst": "ln_norm"},
            {"src": "ln_norm", "dst": "ln_scale"},
        ]

        if not is_rmsnorm:
            # Revert to standard LayerNorm nodes if not RMSNorm
            nodes.insert(1, {"id": "ln_mean", "type": "reduce_mean", "params": {"axis": -1}})
            nodes.append({"id": "ln_shift", "type": "transform", "params": {"operation": "shift"}})
            
            # Revert edges
            edges = [
                {"src": "ln_input", "dst": "ln_mean"},
                {"src": "ln_input", "dst": "ln_var"},
                {"src": "ln_mean", "dst": "ln_norm"},
                {"src": "ln_var", "dst": "ln_norm"},
                {"src": "ln_norm", "dst": "ln_scale"},
                {"src": "ln_scale", "dst": "ln_shift"},
            ]
            
            final_norm_output_id = "ln_shift"
        else:
            final_norm_output_id = "ln_scale"


        # Enhancement: GroupNorm Integration (applied after base normalization)
        if num_groups is not None and num_groups > 0:
            nodes.append({"id": "ln_group", "type": "transform", "params": {"operation": "group_norm", "groups": num_groups}})
            edges.append({"src": final_norm_output_id, "dst": "ln_group"})
            final_norm_output_id = "ln_group"


        return {
            "type": "layer_norm_subgraph",
            "params": {
                "hidden_size": hidden_size, 
                "eps": eps,
                "type": norm_type,
                "groups": num_groups,
                # Enhancement: Learned Eps Scheduling hint
                "eps_schedule": "learned", 
                # Pre-Norm Flag hint
                "position": "pre" 
            },
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "semantic": norm_type,
                "output_node": final_norm_output_id, # Explicitly state the final node
                "note": (
                    f"{norm_type} uses {num_groups} groups. RMSNorm omits mean and shift." 
                    if num_groups else f"Base {norm_type} implementation."
                )
            }
        }
