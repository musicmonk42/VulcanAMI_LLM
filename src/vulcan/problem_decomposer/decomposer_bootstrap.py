"""
decomposer_bootstrap.py - Bootstrap and initialization for problem decomposer
Part of the VULCAN-AGI system

This module wires up all components of the decomposition system:
- Instantiates all decomposition strategies
- Registers strategies in the library
- Populates the fallback chain
- Provides factory function for fully-initialized ProblemDecomposer
"""

import logging
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import all decomposition components
try:
    from .problem_decomposer_core import ProblemDecomposer, ProblemGraph
    from .decomposition_strategies import (
        ExactDecomposition,
        SemanticDecomposition,
        StructuralDecomposition,
        SyntheticBridging,
        AnalogicalDecomposition,
        BruteForceSearch,
        DecompositionStrategy
    )
    from .decomposition_library import (
        StratifiedDecompositionLibrary,
        Pattern,
        Context,
        DecompositionPrinciple
    )
    from .fallback_chain import FallbackChain
    from .adaptive_thresholds import AdaptiveThresholds
    from .problem_executor import ProblemExecutor
except ImportError as e:
    logging.error(f"Failed to import decomposer components: {e}")
    raise

logger = logging.getLogger(__name__)


class DecomposerBootstrap:
    """Handles initialization and wiring of decomposition system"""
    
    def __init__(self):
        """Initialize bootstrap"""
        self.strategy_registry = {}
        self.strategy_instances = []
        self.library = None
        self.fallback_chain = None
        self._lock = threading.RLock()
        
        logger.info("DecomposerBootstrap initialized")
    
    def create_strategy_instances(self) -> Dict[str, DecompositionStrategy]:
        """
        Create instances of all decomposition strategies
        
        Returns:
            Dictionary mapping strategy names to instances
        """
        with self._lock:
            logger.info("Creating strategy instances")
            
            strategies = {}
            
            # Create exact decomposition
            exact = ExactDecomposition()
            strategies['exact'] = exact
            strategies['exact_decomposition'] = exact
            strategies['pattern_match'] = exact
            
            # Create semantic decomposition
            semantic = SemanticDecomposition()
            strategies['semantic'] = semantic
            strategies['semantic_decomposition'] = semantic
            strategies['concept_based'] = semantic
            
            # Create structural decomposition (most versatile)
            structural = StructuralDecomposition()
            strategies['structural'] = structural
            strategies['structural_decomposition'] = structural
            
            # Map predicted types to structural (it handles hierarchical, modular, etc.)
            strategies['hierarchical_decomposition'] = structural
            strategies['hierarchical'] = structural
            strategies['modular_decomposition'] = structural
            strategies['modular'] = structural
            strategies['pipeline_decomposition'] = structural
            strategies['pipeline'] = structural
            strategies['parallel_decomposition'] = structural
            strategies['parallel'] = structural
            strategies['recursive_decomposition'] = structural
            strategies['recursive'] = structural
            
            # Create synthetic bridging
            synthetic = SyntheticBridging()
            strategies['synthetic'] = synthetic
            strategies['synthetic_bridging'] = synthetic
            strategies['bridge'] = synthetic
            
            # Create analogical decomposition
            analogical = AnalogicalDecomposition()
            strategies['analogical'] = analogical
            strategies['analogical_decomposition'] = analogical
            strategies['analogy'] = analogical
            
            # Create brute force search (last resort)
            brute_force = BruteForceSearch()
            strategies['brute_force'] = brute_force
            strategies['brute_force_search'] = brute_force
            strategies['exhaustive'] = brute_force
            
            # Additional type mappings for predicted strategy types
            strategies['temporal_decomposition'] = structural  # Temporal is structural
            strategies['temporal'] = structural
            strategies['constraint_based_decomposition'] = structural  # Constraints handled structurally
            strategies['constraint_based'] = structural
            strategies['direct_decomposition'] = exact  # Direct uses exact matching
            strategies['direct'] = exact
            strategies['iterative_decomposition'] = structural  # Iterative is structural
            strategies['iterative'] = structural
            strategies['hybrid_decomposition'] = structural  # Hybrid uses structural
            strategies['hybrid'] = structural
            strategies['simple'] = structural  # Simple uses structural
            
            # Store unique instances
            self.strategy_instances = [
                exact,
                semantic,
                structural,
                synthetic,
                analogical,
                brute_force
            ]
            
            logger.info("Created %d strategy instances with %d name mappings",
                       len(self.strategy_instances), len(strategies))
            
            return strategies
    
    def register_strategies_in_library(self, library: StratifiedDecompositionLibrary,
                                      strategies: Dict[str, DecompositionStrategy]):
        """
        Register all strategies in the library
        
        Args:
            library: Library to register strategies in
            strategies: Dictionary of strategy instances
        """
        with self._lock:
            logger.info("Registering strategies in library")
            
            # Store strategy registry in library for get_strategy_by_type and get_strategy
            if not hasattr(library, 'strategy_registry'):
                library.strategy_registry = {}
            
            library.strategy_registry.update(strategies)
            
            # Store original methods before overriding
            if hasattr(library, 'get_strategy_by_type'):
                library._original_get_strategy_by_type = library.get_strategy_by_type
            if hasattr(library, 'get_strategy'):
                library._original_get_strategy = library.get_strategy
            
            # Override get_strategy_by_type to use registry
            def get_strategy_by_type(strategy_type: str):
                if not isinstance(strategy_type, str):
                    logger.warning("get_strategy_by_type called with non-string type: %s", type(strategy_type))
                    return None
                return library.strategy_registry.get(strategy_type)
            
            library.get_strategy_by_type = get_strategy_by_type
            
            # Override get_strategy to use registry
            def get_strategy(strategy_name: str):
                if not isinstance(strategy_name, str):
                    logger.warning("get_strategy called with non-string name: %s", type(strategy_name))
                    return None
                return library.strategy_registry.get(strategy_name)
            
            library.get_strategy = get_strategy
            
            logger.info("Registered %d strategies in library", len(strategies))
    
    def populate_fallback_chain(self, fallback_chain: FallbackChain,
                                strategies: List[DecompositionStrategy]):
        """
        Populate fallback chain with ordered strategies
        
        FIXED: Now ensures ALL 6 strategies are added to the chain
        
        Args:
            fallback_chain: Chain to populate
            strategies: List of strategy instances
        """
        with self._lock:
            logger.info("Populating fallback chain with %d strategies", len(strategies))
            
            # Order strategies by cost-effectiveness
            # Fast and reliable first, expensive last
            ordered_strategy_names = [
                ('exact', 1.0, 'Fast pattern matching'),
                ('structural', 2.0, 'Structural analysis'),
                ('semantic', 3.0, 'Semantic matching'),
                ('analogical', 4.0, 'Analogy-based'),
                ('synthetic', 5.0, 'Synthetic bridging'),
                ('brute_force', 10.0, 'Exhaustive search')
            ]
            
            # Build mapping of strategy instances by multiple keys
            strategy_map = {}
            for s in strategies:
                if hasattr(s, 'name'):
                    name_lower = s.name.lower()
                    strategy_map[name_lower] = s
                    
                    # Map by common keywords in the name
                    if 'exact' in name_lower:
                        strategy_map['exact'] = s
                    if 'semantic' in name_lower:
                        strategy_map['semantic'] = s
                    if 'structural' in name_lower:
                        strategy_map['structural'] = s
                    if 'synthetic' in name_lower or 'bridging' in name_lower:
                        strategy_map['synthetic'] = s
                    if 'analogical' in name_lower:
                        strategy_map['analogical'] = s
                    if 'brute' in name_lower or 'force' in name_lower:
                        strategy_map['brute_force'] = s
            
            added_count = 0
            added_strategies = set()
            
            # Add strategies to chain in order
            for strategy_key, cost, description in ordered_strategy_names:
                strategy = strategy_map.get(strategy_key)
                
                if strategy and id(strategy) not in added_strategies:
                    fallback_chain.add_strategy(strategy, cost=cost)
                    added_strategies.add(id(strategy))
                    added_count += 1
                    logger.debug("Added %s to fallback chain (cost=%.1f)", 
                               strategy.name if hasattr(strategy, 'name') else strategy_key, 
                               cost)
                else:
                    if not strategy:
                        logger.warning("Could not find strategy for key: %s", strategy_key)
                    else:
                        logger.debug("Strategy %s already added", strategy_key)
            
            # Verify all strategies were added
            if added_count != len(strategies):
                logger.warning("Strategy count mismatch: added %d, expected %d", 
                             added_count, len(strategies))
                logger.warning("Available strategies: %s", 
                             [s.name for s in strategies if hasattr(s, 'name')])
                logger.warning("Strategy map keys: %s", list(strategy_map.keys()))
                
                # Add any missing strategies
                for s in strategies:
                    if id(s) not in added_strategies:
                        fallback_chain.add_strategy(s, cost=15.0)  # High cost for unmatched
                        added_strategies.add(id(s))
                        added_count += 1
                        logger.warning("Added missing strategy %s with high cost", 
                                     s.name if hasattr(s, 'name') else 'unknown')
            
            logger.info("Populated fallback chain with %d strategies", added_count)
    
    def initialize_library_with_base_principles(self, library: StratifiedDecompositionLibrary):
        """
        Initialize library with base decomposition principles
        
        Args:
            library: Library to initialize
        """
        with self._lock:
            logger.info("Initializing library with base principles")
            
            # Create base contexts
            contexts = {
                'optimization': Context(
                    domain='optimization',
                    problem_type='continuous',
                    constraints={'bounded': True}
                ),
                'classification': Context(
                    domain='classification',
                    problem_type='supervised',
                    constraints={'labeled_data': True}
                ),
                'planning': Context(
                    domain='planning',
                    problem_type='sequential',
                    constraints={'goal_directed': True}
                ),
                'analysis': Context(
                    domain='analysis',
                    problem_type='exploratory',
                    constraints={}
                ),
                'generation': Context(
                    domain='generation',
                    problem_type='creative',
                    constraints={}
                )
            }
            
            # Create base principles for each strategy type
            principle_configs = [
                {
                    'id': 'hierarchical_principle',
                    'name': 'Hierarchical Decomposition',
                    'contexts': [contexts['planning'], contexts['analysis']],
                    'contraindications': ['flat_structure', 'no_hierarchy']
                },
                {
                    'id': 'modular_principle',
                    'name': 'Modular Decomposition',
                    'contexts': [contexts['classification'], contexts['optimization']],
                    'contraindications': ['tightly_coupled', 'monolithic']
                },
                {
                    'id': 'sequential_principle',
                    'name': 'Sequential Decomposition',
                    'contexts': [contexts['planning']],
                    'contraindications': ['parallel_required', 'no_ordering']
                },
                {
                    'id': 'parallel_principle',
                    'name': 'Parallel Decomposition',
                    'contexts': [contexts['optimization'], contexts['generation']],
                    'contraindications': ['sequential_dependencies', 'ordered']
                },
                {
                    'id': 'iterative_principle',
                    'name': 'Iterative Refinement',
                    'contexts': [contexts['optimization'], contexts['generation']],
                    'contraindications': ['one_shot_only', 'no_feedback']
                }
            ]
            
            # Create and add principles
            for config in principle_configs:
                # Create dummy pattern (would be real pattern in production)
                pattern = Pattern(
                    pattern_id=config['id'] + '_pattern',
                    structure=None,
                    features={'type': config['name']},
                    metadata={'auto_generated': True}
                )
                
                principle = DecompositionPrinciple(
                    principle_id=config['id'],
                    name=config['name'],
                    pattern=pattern,
                    applicable_contexts=config['contexts'],
                    contraindications=config['contraindications'],
                    success_rate=0.6  # Initial moderate success rate
                )
                
                library.add_principle(principle)
            
            logger.info("Initialized library with %d base principles", len(principle_configs))
    
    def configure_adaptive_thresholds(self, thresholds: AdaptiveThresholds) -> AdaptiveThresholds:
        """
        Configure adaptive thresholds with sensible defaults
        
        Args:
            thresholds: Thresholds to configure
            
        Returns:
            Configured thresholds
        """
        with self._lock:
            logger.info("Configuring adaptive thresholds")
            
            # Set initial thresholds based on experience
            initial_values = {
                'confidence': 0.6,      # Moderate confidence threshold
                'complexity': 3.0,      # Medium complexity threshold
                'performance': 0.5,     # 50% success rate threshold
                'timeout': 60.0,        # 60 second timeout
                'resource': 0.7         # 70% resource usage threshold
            }
            
            # Update thresholds
            for threshold_name, value in initial_values.items():
                if hasattr(thresholds, 'thresholds') and threshold_name in thresholds.thresholds:
                    thresholds.thresholds[threshold_name].value = value
            
            logger.info("Configured thresholds: %s", initial_values)
            
            return thresholds
    
    def create_initialized_decomposer(self, semantic_bridge=None, vulcan_memory=None,
                                     validator=None, storage_path: Optional[Path] = None,
                                     config: Optional[Dict[str, Any]] = None) -> ProblemDecomposer:
        """
        Create fully initialized and wired ProblemDecomposer
        
        Args:
            semantic_bridge: Optional semantic bridge component
            vulcan_memory: Optional VULCAN memory system
            validator: Optional validator for solution validation
            storage_path: Optional path for persistent storage
            config: Optional configuration dictionary (including test_mode)
            
        Returns:
            Fully initialized ProblemDecomposer
        """
        with self._lock:
            logger.info("Creating fully initialized ProblemDecomposer")
            
            # Step 1: Create strategy instances
            strategies = self.create_strategy_instances()
            self.strategy_registry = strategies
            
            # Step 2: Create decomposer with components - pass config as safety_config
            decomposer = ProblemDecomposer(
                semantic_bridge=semantic_bridge,
                vulcan_memory=vulcan_memory,
                validator=validator,
                safety_config=config  # Pass config to enable test_mode
            )
            
            # Step 3: Register strategies in library
            self.register_strategies_in_library(decomposer.library, strategies)
            
            # Step 4: Initialize library with base principles
            self.initialize_library_with_base_principles(decomposer.library)
            
            # Step 5: Populate fallback chain
            self.populate_fallback_chain(decomposer.fallback_chain, self.strategy_instances)
            
            # Step 6: Configure adaptive thresholds
            self.configure_adaptive_thresholds(decomposer.thresholds)
            
            # Step 7: Profile all strategies
            for strategy in self.strategy_instances:
                decomposer.strategy_profiler.profile_strategy(strategy)
            
            logger.info("ProblemDecomposer fully initialized and ready")
            
            return decomposer


# Global bootstrap instance
_bootstrap = None
_bootstrap_lock = threading.RLock()


def get_bootstrap() -> DecomposerBootstrap:
    """
    Get singleton bootstrap instance
    
    Returns:
        DecomposerBootstrap instance
    """
    global _bootstrap
    if _bootstrap is None:
        with _bootstrap_lock:
            if _bootstrap is None:  # Double-check locking pattern
                _bootstrap = DecomposerBootstrap()
    return _bootstrap


def create_decomposer(semantic_bridge=None, vulcan_memory=None, validator=None,
                     storage_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None) -> ProblemDecomposer:
    """
    Factory function to create fully initialized ProblemDecomposer
    
    This is the recommended way to create a ProblemDecomposer instance.
    All strategies are registered, fallback chain is populated, and
    the system is ready to use.
    
    Args:
        semantic_bridge: Optional semantic bridge component
        vulcan_memory: Optional VULCAN memory system
        validator: Optional validator for solution validation
        storage_path: Optional path for persistent storage
        config: Optional configuration dictionary (including test_mode)
        
    Returns:
        Fully initialized and wired ProblemDecomposer
        
    Example:
        >>> from vulcan.problem_decomposer import create_decomposer, ProblemGraph
        >>> 
        >>> # Create decomposer
        >>> decomposer = create_decomposer()
        >>> 
        >>> # Create decomposer in test mode (fast, no storage)
        >>> decomposer = create_decomposer(config={'test_mode': True})
        >>> 
        >>> # Create problem
        >>> problem = ProblemGraph(
        ...     nodes={'A': {}, 'B': {}, 'C': {}},
        ...     edges=[('A', 'B', {}), ('B', 'C', {})],
        ...     metadata={'domain': 'planning'}
        ... )
        >>> 
        >>> # Decompose and execute
        >>> plan, outcome = decomposer.decompose_and_execute(problem)
        >>> 
        >>> print(f"Success: {outcome.success}")
        >>> print(f"Execution time: {outcome.execution_time:.2f}s")
    """
    bootstrap = get_bootstrap()
    return bootstrap.create_initialized_decomposer(
        semantic_bridge=semantic_bridge,
        vulcan_memory=vulcan_memory,
        validator=validator,
        storage_path=storage_path,
        config=config
    )


def create_test_problem(problem_type: str = 'hierarchical') -> ProblemGraph:
    """
    Create test problem for validation
    
    Args:
        problem_type: Type of problem ('hierarchical', 'sequential', 'parallel', 'cyclic', 'simple')
        
    Returns:
        Test ProblemGraph
    """
    if problem_type == 'hierarchical':
        # Hierarchical problem with tree structure
        return ProblemGraph(
            nodes={
                'root': {'type': 'decision', 'level': 0},
                'branch1': {'type': 'operation', 'level': 1},
                'branch2': {'type': 'operation', 'level': 1},
                'leaf1': {'type': 'transform', 'level': 2},
                'leaf2': {'type': 'transform', 'level': 2},
                'leaf3': {'type': 'transform', 'level': 2}
            },
            edges=[
                ('root', 'branch1', {'weight': 1.0}),
                ('root', 'branch2', {'weight': 1.0}),
                ('branch1', 'leaf1', {'weight': 0.5}),
                ('branch1', 'leaf2', {'weight': 0.5}),
                ('branch2', 'leaf3', {'weight': 1.0})
            ],
            root='root',
            metadata={'domain': 'planning', 'type': 'hierarchical'}
        )
    
    elif problem_type == 'sequential':
        # Sequential pipeline problem
        return ProblemGraph(
            nodes={
                'input': {'type': 'operation', 'operation': 'read'},
                'process1': {'type': 'transform', 'transform': 'filter'},
                'process2': {'type': 'transform', 'transform': 'aggregate'},
                'process3': {'type': 'transform', 'transform': 'normalize'},
                'output': {'type': 'operation', 'operation': 'write'}
            },
            edges=[
                ('input', 'process1', {}),
                ('process1', 'process2', {}),
                ('process2', 'process3', {}),
                ('process3', 'output', {})
            ],
            root='input',
            metadata={'domain': 'analysis', 'type': 'sequential'}
        )
    
    elif problem_type == 'parallel':
        # Parallel processing problem
        return ProblemGraph(
            nodes={
                'start': {'type': 'decision'},
                'task1': {'type': 'operation', 'independent': True},
                'task2': {'type': 'operation', 'independent': True},
                'task3': {'type': 'operation', 'independent': True},
                'merge': {'type': 'operation', 'operation': 'combine'}
            },
            edges=[
                ('start', 'task1', {}),
                ('start', 'task2', {}),
                ('start', 'task3', {}),
                ('task1', 'merge', {}),
                ('task2', 'merge', {}),
                ('task3', 'merge', {})
            ],
            root='start',
            metadata={'domain': 'optimization', 'type': 'parallel'}
        )
    
    elif problem_type == 'cyclic':
        # Cyclic/iterative problem
        return ProblemGraph(
            nodes={
                'init': {'type': 'operation'},
                'evaluate': {'type': 'decision'},
                'refine': {'type': 'transform'},
                'output': {'type': 'operation'}
            },
            edges=[
                ('init', 'evaluate', {}),
                ('evaluate', 'refine', {'condition': 'not_satisfied'}),
                ('refine', 'evaluate', {}),  # Cycle
                ('evaluate', 'output', {'condition': 'satisfied'})
            ],
            root='init',
            metadata={'domain': 'optimization', 'type': 'iterative'}
        )
    
    else:  # simple
        # Simple linear problem
        return ProblemGraph(
            nodes={
                'A': {'type': 'operation'},
                'B': {'type': 'operation'},
                'C': {'type': 'operation'}
            },
            edges=[
                ('A', 'B', {}),
                ('B', 'C', {})
            ],
            root='A',
            metadata={'domain': 'general', 'type': 'simple'}
        )


def validate_decomposer_setup(decomposer: ProblemDecomposer) -> Dict[str, Any]:
    """
    Validate that decomposer is properly set up
    
    Args:
        decomposer: Decomposer to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'checks': {}
    }
    
    # Check library has strategies
    if hasattr(decomposer.library, 'strategy_registry'):
        strategy_count = len(decomposer.library.strategy_registry)
        results['checks']['strategy_count'] = strategy_count
        if strategy_count == 0:
            results['valid'] = False
            results['errors'].append("No strategies registered in library")
    else:
        results['valid'] = False
        results['errors'].append("Library missing strategy_registry")
    
    # Check fallback chain has strategies
    fallback_count = len(decomposer.fallback_chain.strategies)
    results['checks']['fallback_chain_count'] = fallback_count
    if fallback_count == 0:
        results['valid'] = False
        results['errors'].append("Fallback chain has no strategies")
    
    # Check executor is initialized
    if decomposer.executor is None:
        results['valid'] = False
        results['errors'].append("Executor not initialized")
    else:
        results['checks']['executor_initialized'] = True
    
    # Check thresholds are configured
    if decomposer.thresholds is None:
        results['valid'] = False
        results['errors'].append("Thresholds not initialized")
    else:
        confidence_threshold = decomposer.thresholds.get_confidence_threshold()
        results['checks']['confidence_threshold'] = confidence_threshold
        if confidence_threshold is None:
            results['warnings'].append("Confidence threshold not set")
    
    # Check library has principles
    if hasattr(decomposer.library, 'principles'):
        principle_count = len(decomposer.library.principles)
        results['checks']['principle_count'] = principle_count
        if principle_count == 0:
            results['warnings'].append("No principles in library")
    
    # Test strategy retrieval
    test_types = ['hierarchical_decomposition', 'structural', 'exact']
    missing_types = []
    for strategy_type in test_types:
        strategy = decomposer.library.get_strategy_by_type(strategy_type)
        if strategy is None:
            missing_types.append(strategy_type)
    
    if missing_types:
        results['valid'] = False
        results['errors'].append(f"Cannot retrieve strategies: {missing_types}")
    else:
        results['checks']['strategy_retrieval'] = 'success'
    
    return results


def run_bootstrap_test():
    """
    Run complete bootstrap test
    
    Returns:
        True if test passes, False otherwise
    """
    logger.info("Running bootstrap test")
    
    try:
        # Create decomposer
        decomposer = create_decomposer()
        
        # Validate setup
        validation = validate_decomposer_setup(decomposer)
        
        if not validation['valid']:
            logger.error("Bootstrap validation failed:")
            for error in validation['errors']:
                logger.error("  - %s", error)
            return False
        
        if validation['warnings']:
            logger.warning("Bootstrap validation warnings:")
            for warning in validation['warnings']:
                logger.warning("  - %s", warning)
        
        # Test with simple problem
        problem = create_test_problem('simple')
        
        # Test decomposition
        plan = decomposer.decompose_novel_problem(problem)
        
        if plan is None:
            logger.error("Decomposition returned None")
            return False
        
        if len(plan.steps) == 0:
            logger.error("Decomposition produced empty plan")
            return False
        
        logger.info("Bootstrap test passed!")
        logger.info("  - Strategy count: %d", validation['checks'].get('strategy_count', 0))
        logger.info("  - Fallback chain: %d strategies", validation['checks'].get('fallback_chain_count', 0))
        logger.info("  - Plan steps: %d", len(plan.steps))
        logger.info("  - Plan confidence: %.2f", plan.confidence)
        
        return True
        
    except Exception as e:
        logger.error("Bootstrap test failed with exception: %s", e)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Run test when module executed directly
    logging.basicConfig(level=logging.INFO)
    success = run_bootstrap_test()
    exit(0 if success else 1)
