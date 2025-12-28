"""
Bridge between query analysis and problem decomposition.

Converts query metadata into ProblemGraph for decomposition by the
ProblemDecomposer module. This enables complex queries to be broken
down into subproblems before tool selection.

Architecture:
    Query → QueryToProblemBridge → ProblemGraph → ProblemDecomposer
                                                       ↓
                                              DecompositionPlan
                                                       ↓
                                         ToolSelector (per subproblem)
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import problem decomposer types with graceful fallback
try:
    from vulcan.problem_decomposer.problem_decomposer_core import (
        DomainDataCategory,
        ProblemComplexity,
        ProblemGraph,
    )
    DECOMPOSER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Problem decomposer not available: {e}")
    DECOMPOSER_AVAILABLE = False
    ProblemGraph = None
    ProblemComplexity = None
    DomainDataCategory = None


@dataclass
class SubproblemResult:
    """Result from executing a subproblem."""
    
    step_id: str
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated result from multiple subproblem executions."""
    
    status: str  # 'complete', 'partial', 'empty', 'error'
    total_subproblems: int
    successful: int
    failed: int
    content: Optional[str] = None
    results: List[SubproblemResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    overall_confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryToProblemBridge:
    """
    Converts query analysis results into ProblemGraph for decomposition.
    
    This bridge handles the translation between:
    - Query analysis (type, complexity, uncertainty)
    - ProblemGraph (nodes, edges, metadata for decomposition)
    
    It detects structural patterns in queries that indicate decomposability:
    - Multiple questions (? marks)
    - Conditional structures (if/then)
    - Sequential requirements (first/then/finally)
    - Comparative structures (vs, compared to)
    """
    
    # Map query types to domain categories
    QUERY_TYPE_TO_DOMAIN: Dict[str, str] = {
        'reasoning': 'logical',
        'causal': 'causal_analysis',
        'perception': 'perceptual',
        'execution': 'procedural',
        'planning': 'planning',
        'learning': 'educational',
        'general': 'mixed',
    }
    
    # Map complexity scores to complexity levels
    COMPLEXITY_THRESHOLDS: Dict[float, str] = {
        0.0: 'trivial',
        0.2: 'simple',
        0.4: 'moderate',
        0.6: 'complex',
        0.8: 'expert',
    }
    
    # Keywords for detecting subproblem patterns
    CONDITIONAL_KEYWORDS = ['if ', 'when ', 'assuming ', 'given that ', 'provided ']
    SEQUENTIAL_KEYWORDS = ['first ', 'then ', 'finally ', 'next ', 'after ', 'before ']
    COMPARATIVE_KEYWORDS = [' vs ', ' versus ', ' compared to ', ' or ', ' and ']
    
    def __init__(self):
        """Initialize the query-to-problem bridge."""
        self.logger = logging.getLogger(f"{__name__}.QueryToProblemBridge")
        self._conversion_count = 0
        self._aggregation_count = 0
        
        if not DECOMPOSER_AVAILABLE:
            self.logger.warning(
                "[QueryToProblemBridge] ProblemDecomposer not available, "
                "conversions will return None"
            )
    
    def convert_to_problem_graph(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        tool_selection: Optional[Dict[str, Any]] = None,
    ) -> Optional['ProblemGraph']:
        """
        Convert query and analysis into a ProblemGraph.
        
        Args:
            query: The raw query string
            query_analysis: Results from query analyzer containing:
                - type: Query type (reasoning, execution, etc.)
                - complexity: Complexity score (0.0 to 1.0)
                - uncertainty: Uncertainty score
                - requires_reasoning: Whether reasoning is needed
            tool_selection: Optional tool selection results containing:
                - tools: List of selected tools
                - strategy: Selection strategy used
                - confidence: Selection confidence
                
        Returns:
            ProblemGraph ready for decomposition, or None if unavailable
        """
        if not DECOMPOSER_AVAILABLE:
            self.logger.debug("[QueryToProblemBridge] Decomposer not available")
            return None
        
        self._conversion_count += 1
        start_time = time.perf_counter()
        
        # Extract complexity
        complexity_score = query_analysis.get('complexity', 0.3)
        if isinstance(complexity_score, str):
            try:
                complexity_score = float(complexity_score)
            except ValueError:
                complexity_score = 0.3
        
        complexity_level = self._score_to_complexity_name(complexity_score)
        
        # Extract domain from query type
        query_type = query_analysis.get('type', 'general')
        if hasattr(query_type, 'value'):  # Handle enum
            query_type = query_type.value
        domain = self.QUERY_TYPE_TO_DOMAIN.get(query_type, 'mixed')
        
        # Generate problem ID
        problem_id = self._generate_problem_id(query)
        
        # Build problem graph
        problem_graph = ProblemGraph(
            nodes={},
            edges=[],
            root="root",
            metadata={
                'domain': domain,
                'complexity_level': complexity_level,
                'query_type': query_type,
                'original_query': query,
                'uncertainty': query_analysis.get('uncertainty', 0.0),
                'requires_reasoning': query_analysis.get('requires_reasoning', False),
            },
            complexity_score=complexity_score,
        )
        
        # Add root node
        problem_graph.nodes["root"] = {
            'type': 'query',
            'content': query,
            'complexity': complexity_score,
            'query_type': query_type,
            'level': 0,
        }
        
        # If we have tool selection hints, add them as constraints
        if tool_selection:
            selected_tools = tool_selection.get('tools', [])
            strategy = tool_selection.get('strategy', 'direct')
            selection_confidence = tool_selection.get('confidence', 0.5)
            
            problem_graph.nodes["tool_hints"] = {
                'type': 'constraint',
                'tools': selected_tools,
                'strategy': strategy,
                'confidence': selection_confidence,
                'level': 1,
            }
            problem_graph.edges.append(("root", "tool_hints", {'edge_type': 'requires'}))
        
        # Detect and add subproblems from query structure
        subproblems = self._detect_subproblems(query, query_analysis)
        for i, subproblem in enumerate(subproblems):
            node_id = f"subproblem_{i}"
            problem_graph.nodes[node_id] = {
                'type': 'subproblem',
                'content': subproblem['content'],
                'complexity': subproblem.get('complexity', complexity_score * 0.5),
                'subproblem_type': subproblem.get('type', 'general'),
                'level': 1,
            }
            problem_graph.edges.append(("root", node_id, {'edge_type': 'decomposes_to'}))
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        self.logger.debug(
            f"[QueryToProblemBridge] Created ProblemGraph: id={problem_id}, "
            f"complexity={complexity_level}, domain={domain}, "
            f"nodes={len(problem_graph.nodes)}, time={elapsed_ms:.1f}ms"
        )
        
        return problem_graph
    
    def _generate_problem_id(self, query: str) -> str:
        """
        Generate a unique problem ID from the query.
        
        Uses SHA-256 hash for better collision resistance and security.
        Full 64 hex chars ensures collision probability is negligible.
        """
        # Use SHA-256 for better collision resistance
        hash_val = hashlib.sha256(query.encode()).hexdigest()
        return f"query_{hash_val}"
    
    def _score_to_complexity_name(self, score: float) -> str:
        """Convert numeric complexity score to complexity level name."""
        for threshold in sorted(self.COMPLEXITY_THRESHOLDS.keys(), reverse=True):
            if score >= threshold:
                return self.COMPLEXITY_THRESHOLDS[threshold]
        return 'trivial'
    
    def _detect_subproblems(
        self,
        query: str,
        query_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect subproblems in query structure.
        
        Looks for structural patterns that indicate decomposability:
        - Multiple questions (? marks)
        - Conditional structures (if/then)
        - Sequential requirements (first/then/finally)
        - Comparative structures (vs, compared to)
        
        Args:
            query: The raw query string
            query_analysis: Query analysis results
            
        Returns:
            List of detected subproblems with content and metadata
        """
        subproblems: List[Dict[str, Any]] = []
        query_lower = query.lower()
        base_complexity = query_analysis.get('complexity', 0.3)
        
        # Pattern 1: Multiple questions (? marks)
        if query.count('?') > 1:
            questions = [q.strip() + '?' for q in query.split('?') if q.strip()]
            for idx, q in enumerate(questions):
                if len(q) > 5:  # Skip very short fragments
                    subproblems.append({
                        'content': q,
                        'type': 'question',
                        'complexity': base_complexity / len(questions),
                        'index': idx,
                    })
            if subproblems:
                self.logger.debug(
                    f"[QueryToProblemBridge] Detected {len(subproblems)} questions"
                )
                return subproblems  # Return early if we found multiple questions
        
        # Pattern 2: Conditional structures (if/then/when)
        for cond in self.CONDITIONAL_KEYWORDS:
            if cond in query_lower:
                parts = query_lower.split(cond, 1)
                if len(parts) > 1 and len(parts[1]) > 10:
                    # Extract the condition
                    condition_part = parts[1].split(',')[0].split('.')[0].strip()
                    if len(condition_part) > 5:
                        subproblems.append({
                            'content': f"Evaluate condition: {condition_part}",
                            'type': 'condition',
                            'complexity': 0.2,
                        })
                        # Extract the consequent (what happens if condition is true)
                        remaining = parts[1][len(condition_part):].strip()
                        if remaining and len(remaining) > 10:
                            subproblems.append({
                                'content': f"Process: {remaining[:100]}",
                                'type': 'consequent',
                                'complexity': base_complexity * 0.6,
                            })
                        break
        
        # Pattern 3: Sequential requirements (first/then/finally)
        for seq in self.SEQUENTIAL_KEYWORDS:
            if seq in query_lower:
                # This query has sequential structure
                subproblems.append({
                    'content': f"Identify sequence: {query[:100]}",
                    'type': 'sequence_analysis',
                    'complexity': base_complexity * 0.4,
                })
                break
        
        # Pattern 4: Comparative structures (vs, compared to)
        for comp in self.COMPARATIVE_KEYWORDS:
            if comp in query_lower:
                parts = query_lower.split(comp)
                if len(parts) >= 2:
                    left_part = parts[0].strip()[-50:]  # Last 50 chars before comparison
                    right_part = parts[1].strip()[:50]  # First 50 chars after comparison
                    
                    if len(left_part) > 5 and len(right_part) > 5:
                        subproblems.append({
                            'content': f"Analyze: {left_part}",
                            'type': 'comparison_left',
                            'complexity': base_complexity / 2,
                        })
                        subproblems.append({
                            'content': f"Analyze: {right_part}",
                            'type': 'comparison_right',
                            'complexity': base_complexity / 2,
                        })
                        subproblems.append({
                            'content': f"Compare results",
                            'type': 'comparison_synthesis',
                            'complexity': base_complexity * 0.3,
                        })
                        break
        
        return subproblems
    
    def aggregate_subproblem_results(
        self,
        results: List[SubproblemResult],
        problem_graph: Optional['ProblemGraph'] = None,
    ) -> AggregatedResult:
        """
        Aggregate results from subproblem executions.
        
        Combines individual subproblem results into a unified response,
        handling partial failures gracefully.
        
        Args:
            results: List of SubproblemResult from executing subproblems
            problem_graph: Original problem graph (optional, for context)
            
        Returns:
            AggregatedResult with combined content and statistics
        """
        self._aggregation_count += 1
        
        if not results:
            return AggregatedResult(
                status='empty',
                total_subproblems=0,
                successful=0,
                failed=0,
                content=None,
            )
        
        # Categorize results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Determine status
        if len(failed) == 0:
            status = 'complete'
        elif len(successful) == 0:
            status = 'error'
        else:
            status = 'partial'
        
        # Combine content from successful results
        content_parts = []
        for r in successful:
            if r.content:
                content_parts.append(r.content)
        
        combined_content = '\n\n'.join(content_parts) if content_parts else None
        
        # Calculate overall confidence (weighted average)
        if successful:
            total_weight = sum(r.execution_time_ms + 1 for r in successful)
            weighted_confidence = sum(
                r.confidence * (r.execution_time_ms + 1) for r in successful
            ) / total_weight
        else:
            weighted_confidence = 0.0
        
        # Calculate total execution time
        total_time = sum(r.execution_time_ms for r in results)
        
        # Collect all tools used
        all_tools = set()
        for r in successful:
            all_tools.update(r.tools_used)
        
        # Collect errors
        errors = [r.error for r in failed if r.error]
        
        aggregated = AggregatedResult(
            status=status,
            total_subproblems=len(results),
            successful=len(successful),
            failed=len(failed),
            content=combined_content,
            results=results,
            errors=errors,
            total_execution_time_ms=total_time,
            overall_confidence=weighted_confidence,
            metadata={
                'tools_used': list(all_tools),
                'aggregation_count': self._aggregation_count,
            },
        )
        
        self.logger.debug(
            f"[QueryToProblemBridge] Aggregated {len(results)} results: "
            f"status={status}, success={len(successful)}/{len(results)}, "
            f"confidence={weighted_confidence:.2f}"
        )
        
        return aggregated
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge usage statistics."""
        return {
            'conversion_count': self._conversion_count,
            'aggregation_count': self._aggregation_count,
            'decomposer_available': DECOMPOSER_AVAILABLE,
        }


# Module-level singleton for convenience
_bridge_instance: Optional[QueryToProblemBridge] = None


def get_query_to_problem_bridge() -> QueryToProblemBridge:
    """Get the singleton QueryToProblemBridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = QueryToProblemBridge()
    return _bridge_instance
