"""
Graph-Aware Evolution Engine

Integrates metaprogramming handlers with evolution engine to enable
autonomous graph self-modification through Graph IR execution rather
than Python dict manipulation.

Evolution flow:
1. Load mutator.json metaprogramming pipeline
2. For each mutation: Execute PATTERN_COMPILE → FIND → SPLICE → COMMIT
3. Evaluate fitness through graph execution
4. Apply safety gates (NSO, ethical labels)
"""

import asyncio
import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.evolution_engine import EvolutionEngine, Individual

logger = logging.getLogger(__name__)


class GraphAwareEvolutionEngine(EvolutionEngine):
    """
    Evolution engine that operates via Graph IR execution.
    
    Instead of Python dict manipulation, mutations are applied
    through PATTERN_COMPILE → FIND_SUBGRAPH → GRAPH_SPLICE → GRAPH_COMMIT chain.
    """
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        max_generations: int = 100,
        runtime=None,
        mutator_graph_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize graph-aware evolution engine.
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation (0-1)
            crossover_rate: Probability of crossover (0-1)
            max_generations: Maximum generations to evolve
            runtime: UnifiedRuntime instance for graph execution
            mutator_graph_path: Path to mutator.json graph
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            max_generations=max_generations,
            **kwargs
        )
        
        self.runtime = runtime
        self.mutator_graph = None
        self.metaprogramming_enabled = False
        
        # Load mutator graph if path provided
        if mutator_graph_path:
            self._load_mutator_graph(mutator_graph_path)
        
        # Track metaprogramming statistics
        self.meta_stats = {
            "mutations_via_metaprog": 0,
            "mutations_via_dict": 0,
            "authorization_denials": 0,
            "ethical_blocks": 0,
        }
    
    def _load_mutator_graph(self, path: str):
        """Load mutator.json graph definition."""
        try:
            mutator_path = Path(path)
            if not mutator_path.exists():
                # Try relative to project root
                mutator_path = Path(__file__).parent.parent / path
            
            if mutator_path.exists():
                with open(mutator_path, 'r') as f:
                    self.mutator_graph = json.load(f)
                logger.info(f"Loaded mutator graph from {mutator_path}")
                self.metaprogramming_enabled = bool(self.runtime)
            else:
                logger.warning(f"Mutator graph not found at {path}")
        except Exception as e:
            logger.error(f"Failed to load mutator graph: {e}")
    
    def _apply_random_mutation(self, graph: Dict) -> Dict:
        """
        Apply mutation using metaprogramming pipeline if available,
        fallback to dict manipulation otherwise.
        """
        if self.metaprogramming_enabled and self.runtime and self.mutator_graph:
            try:
                # Try graph IR mutation
                mutated = asyncio.run(self._apply_metaprogramming_mutation(graph))
                self.meta_stats["mutations_via_metaprog"] += 1
                return mutated
            except Exception as e:
                logger.warning(f"Metaprogramming mutation failed: {e}, falling back")
                self.meta_stats["mutations_via_dict"] += 1
                return super()._apply_random_mutation(graph)
        else:
            # Fallback to traditional dict mutation
            self.meta_stats["mutations_via_dict"] += 1
            return super()._apply_random_mutation(graph)
    
    async def _apply_metaprogramming_mutation(self, graph: Dict) -> Dict:
        """
        Apply mutation through Graph IR metaprogramming pipeline.
        
        Executes: PATTERN_COMPILE → FIND → SPLICE → NSO → ETHICAL → COMMIT
        
        Args:
            graph: Target graph to mutate
            
        Returns:
            Modified graph (or original if blocked by safety)
        """
        # Import metaprogramming handlers
        from src.unified_runtime.metaprogramming_handlers import (
            pattern_compile_node,
            find_subgraph_node,
            graph_splice_node,
            nso_modify_node,
            ethical_label_node,
            graph_commit_node,
        )
        
        # Create execution context
        context = {
            "runtime": self.runtime,
            "graph": graph,
            "node_map": {},
            "outputs": {},
            "recursion_depth": 0,
            "audit_log": [],
            "agent_id": "evolution_engine"
        }
        
        # Generate mutation pattern and template
        pattern, template = self._generate_mutation_pattern_and_template(graph)
        
        # Step 1: Compile pattern
        compile_result = await pattern_compile_node(
            {"id": "pat", "type": "PATTERN_COMPILE"},
            context,
            {"pattern_in": pattern}
        )
        
        if compile_result.get("status") != "success":
            logger.debug("Pattern compilation failed")
            return graph
        
        # Step 2: Find subgraph matches
        find_result = await find_subgraph_node(
            {"id": "find", "type": "FIND_SUBGRAPH", "params": {"start_idx": 0}},
            context,
            {
                "pattern_in": compile_result["pattern_out"],
                "graph_ref": graph
            }
        )
        
        if find_result.get("status") != "success" or find_result.get("match_out", {}).get("match_count", 0) == 0:
            logger.debug("No pattern matches found")
            return graph
        
        # Step 3: Splice with template
        splice_result = await graph_splice_node(
            {"id": "splice", "type": "GRAPH_SPLICE"},
            context,
            {
                "match_in": find_result["match_out"],
                "template_in": template
            }
        )
        
        if splice_result.get("status") != "success":
            logger.debug("Graph splice failed")
            return graph
        
        # Step 4: Get NSO authorization
        nso_result = await nso_modify_node(
            {"id": "nso", "type": "NSO_MODIFY", "params": {"target": "graph_structure"}},
            context,
            {}
        )
        
        if not nso_result.get("nso_out", {}).get("authorized"):
            logger.info("Mutation blocked by NSO authorization")
            self.meta_stats["authorization_denials"] += 1
            return graph
        
        # Step 5: Get ethical label (use 'safe' for autonomous evolution)
        label_result = await ethical_label_node(
            {"id": "label", "type": "ETHICAL_LABEL", "params": {"label": "safe"}},
            context,
            {}
        )
        
        if not label_result.get("label_out", {}).get("approved"):
            logger.info("Mutation blocked by ethical label")
            self.meta_stats["ethical_blocks"] += 1
            return graph
        
        # Step 6: Commit modified graph
        commit_result = await graph_commit_node(
            {"id": "commit", "type": "GRAPH_COMMIT"},
            context,
            {
                "graph_in": splice_result["graph_out"],
                "nso_in": nso_result["nso_out"],
                "label_in": label_result["label_out"]
            }
        )
        
        if commit_result.get("status") == "success":
            logger.debug(f"Metaprogramming mutation committed: version={commit_result['version']['hash']}")
            return commit_result["committed_graph"]
        else:
            logger.debug("Commit failed")
            return graph
    
    def _generate_mutation_pattern_and_template(self, graph: Dict) -> tuple[Dict, Dict]:
        """
        Generate pattern to find and template to replace for mutation.
        
        Strategies:
        1. Find ADD nodes and enhance them
        2. Find single nodes and duplicate
        3. Find simple operations and replace with complex ones
        
        Args:
            graph: Target graph
            
        Returns:
            Tuple of (pattern, template)
        """
        import random
        
        # Strategy 1: Enhance ADD nodes with parameters
        if random.random() < 0.4:
            pattern = {
                "nodes": [{"id": "?target", "type": "ADD"}],
                "edges": []
            }
            template = {
                "nodes": [{
                    "id": "?target",
                    "type": "ADD",
                    "params": {"optimized": True, "method": "fast"}
                }],
                "edges": []
            }
            return pattern, template
        
        # Strategy 2: Enhance MULTIPLY nodes
        if random.random() < 0.4:
            pattern = {
                "nodes": [{"id": "?target", "type": "MULTIPLY"}],
                "edges": []
            }
            template = {
                "nodes": [{
                    "id": "?target",
                    "type": "MULTIPLY",
                    "params": {"algorithm": "karatsuba"}
                }],
                "edges": []
            }
            return pattern, template
        
        # Strategy 3: Enhance CONST nodes with metadata
        if random.random() < 0.4:
            pattern = {
                "nodes": [{"id": "?target", "type": "CONST"}],
                "edges": []
            }
            template = {
                "nodes": [{
                    "id": "?target",
                    "type": "CONST",
                    "params": {"cached": True, "memoize": True}
                }],
                "edges": []
            }
            return pattern, template
        
        # Default: Try to find any node type present in graph
        if graph.get("nodes"):
            node_types = list({n.get("type") for n in graph["nodes"] if n.get("type")})
            if node_types:
                target_type = random.choice(node_types)
                pattern = {
                    "nodes": [{"id": "?target", "type": target_type}],
                    "edges": []
                }
                template = {
                    "nodes": [{
                        "id": "?target",
                        "type": target_type,
                        "params": {"enhanced": True, "generation": self.generation}
                    }],
                    "edges": []
                }
                return pattern, template
        
        # Fallback: No-op pattern (won't match anything)
        return {"nodes": [], "edges": []}, {"nodes": [], "edges": []}
    
    def get_metaprogramming_stats(self) -> Dict[str, Any]:
        """Get statistics about metaprogramming usage."""
        total_mutations = (
            self.meta_stats["mutations_via_metaprog"] +
            self.meta_stats["mutations_via_dict"]
        )
        
        return {
            **self.meta_stats,
            "total_mutations": total_mutations,
            "metaprog_percentage": (
                self.meta_stats["mutations_via_metaprog"] / total_mutations * 100
                if total_mutations > 0 else 0
            ),
            "metaprogramming_enabled": self.metaprogramming_enabled,
            "safety_block_rate": (
                (self.meta_stats["authorization_denials"] + self.meta_stats["ethical_blocks"]) /
                max(1, self.meta_stats["mutations_via_metaprog"]) * 100
            )
        }
    
    def print_stats(self):
        """Print evolution and metaprogramming statistics."""
        super().print_stats()
        
        meta_stats = self.get_metaprogramming_stats()
        print("\n=== Metaprogramming Statistics ===")
        print(f"Mutations via metaprogramming: {meta_stats['mutations_via_metaprog']}")
        print(f"Mutations via dict manipulation: {meta_stats['mutations_via_dict']}")
        print(f"Metaprogramming percentage: {meta_stats['metaprog_percentage']:.1f}%")
        print(f"NSO authorization denials: {meta_stats['authorization_denials']}")
        print(f"Ethical label blocks: {meta_stats['ethical_blocks']}")
        print(f"Safety block rate: {meta_stats['safety_block_rate']:.1f}%")


def create_graph_aware_engine(
    runtime=None,
    mutator_graph_path: str = "graphs/mutator.json",
    **kwargs
) -> GraphAwareEvolutionEngine:
    """
    Factory function to create graph-aware evolution engine.
    
    Args:
        runtime: UnifiedRuntime instance
        mutator_graph_path: Path to mutator.json
        **kwargs: Additional arguments for engine
        
    Returns:
        Configured GraphAwareEvolutionEngine
    """
    return GraphAwareEvolutionEngine(
        runtime=runtime,
        mutator_graph_path=mutator_graph_path,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Example: Create engine without runtime (falls back to dict mutation)
    engine = GraphAwareEvolutionEngine(
        population_size=10,
        max_generations=5,
        mutator_graph_path="graphs/mutator.json"
    )
    
    # Define simple fitness function
    def fitness_fn(graph: Dict) -> float:
        # Simple fitness: reward graphs with more nodes
        return len(graph.get("nodes", [])) / 20.0
    
    # Evolve
    print("Evolving graphs...")
    best = engine.evolve(fitness_fn, generations=5)
    
    print(f"\nBest fitness: {best.fitness:.4f}")
    print(f"Best graph nodes: {len(best.graph.get('nodes', []))}")
    
    # Print statistics
    engine.print_stats()
