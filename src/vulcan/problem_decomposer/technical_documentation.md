\# VULCAN Problem Decomposer - Technical Documentation

\## Overview

The VULCAN Problem Decomposer is an adaptive system for breaking down complex problems into manageable subproblems. It uses multiple decomposition strategies, learns from experience, and automatically adjusts its approach based on performance.

\## Architecture

\### Core Components

\#### 1. ProblemDecomposer (`problem\_decomposer\_core.py`)

The main orchestrator that coordinates all decomposition activities.

\*\*Key Responsibilities:\*\*

\- Coordinates strategy selection and execution

\- Manages the problem decomposition lifecycle

\- Integrates with semantic bridge and memory systems

\- Tracks performance and adapts strategies

\*\*Main Methods:\*\*

```python

decompose\_novel\_problem(problem: ProblemGraph) -> ExecutionPlan

decompose\_and\_execute(problem: ProblemGraph) -> Tuple\[ExecutionPlan, ExecutionOutcome]

```

\#### 2. Decomposition Strategies (`decomposition\_strategies.py`)

Six specialized strategies for different problem types:

\*\*ExactDecomposition\*\*

\- Fast pattern-based matching

\- Uses known decomposition templates

\- Best for: Previously seen problem patterns

\- Cost: Low (1.0)

\- Includes 4 built-in patterns: linear, tree, cycle, star

\- Uses subgraph isomorphism for matching

\- Deterministic and parallelizable

\- Match threshold: 0.95

\*\*StructuralDecomposition\*\*

\- Analyzes problem structure (hierarchy, dependencies)

\- Most versatile strategy

\- Handles: hierarchical, modular, pipeline, parallel, recursive patterns

\- Cost: Medium (2.0)

\- Detects 5 structural patterns:

&nbsp; - \*\*Hierarchical\*\*: DAG with clear root and depth ≥ 2

&nbsp; - \*\*Modular\*\*: Multiple strongly connected components

&nbsp; - \*\*Pipeline\*\*: Linear chain with 60%+ nodes having degree 2

&nbsp; - \*\*Recursive\*\*: Contains cycles

&nbsp; - \*\*Parallel\*\*: Node groups with same predecessors/successors

\- Returns confidence scores based on pattern strength

\*\*SemanticDecomposition\*\*

\- Uses semantic similarity and concept matching

\- Leverages semantic bridge for understanding

\- Best for: Problems with rich semantic content

\- Cost: Medium-High (3.0)

\- Features:

&nbsp; - 128-dim embeddings for nodes

&nbsp; - Cosine similarity clustering (threshold: 0.7)

&nbsp; - Embedding cache (1000 entries with LRU eviction)

&nbsp; - Cluster cohesion calculation

&nbsp; - Handles up to 1000 nodes efficiently

\*\*AnalogicalDecomposition\*\*

\- Finds analogies to known problems

\- Adapts solutions from similar contexts

\- Best for: Novel problems with familiar aspects

\- Cost: High (4.0)

\- Includes analogy database with domains:

&nbsp; - Sorting (merge\_sort, quick\_sort)

&nbsp; - Optimization (gradient\_descent)

&nbsp; - Search (breadth\_first)

\- Similarity threshold: 0.6

\- Returns top 5 analogies with mappings

\- Creates feature correspondences and transformations

\*\*SyntheticBridging\*\*

\- Creates intermediate steps to bridge gaps

\- Generates novel decomposition approaches

\- Best for: Complex problems requiring creative solutions

\- Cost: High (5.0)

\- Features:

&nbsp; - Pattern mutation (10% mutation rate)

&nbsp; - Bridge templates: simple, chain, hub, mesh

&nbsp; - Alternative generation (linear, parallel bridges)

&nbsp; - Identifies unknown subgraphs (disconnected components)

&nbsp; - Confidence typically 0.3-0.5 for synthetic solutions

\*\*BruteForceSearch\*\*

\- Exhaustive search through solution space

\- Last resort when other strategies fail

\- Best for: When correctness matters more than efficiency

\- Cost: Very High (10.0)

\- Configuration:

&nbsp; - Max depth: 3

&nbsp; - Max iterations: 1000

&nbsp; - Non-deterministic due to cutoffs

&nbsp; - Simple partitioning strategy

&nbsp; - Returns trivial decomposition as fallback

\#### 3. Stratified Decomposition Library (`decomposition\_library.py`)

Knowledge repository managing decomposition patterns and principles.

\*\*Key Features:\*\*

\- Stores successful decomposition patterns

\- Manages decomposition principles with context

\- Provides pattern retrieval by similarity

\- Tracks pattern success rates

\*\*Data Structures:\*\*

```python

Pattern: Reusable decomposition template

Context: Problem domain and constraints

DecompositionPrinciple: General decomposition guideline

```

\#### 4. Fallback Chain (`fallback\_chain.py`)

Manages strategy execution order with automatic fallback.

\*\*Behavior:\*\*

\- Tries strategies in cost-effectiveness order

\- Falls back to next strategy on failure

\- Tracks which strategies work for which problems

\- Caches successful strategy choices

\*\*Strategy Ordering (default):\*\*

1\. Exact (cost: 1.0)

2\. Structural (cost: 2.0)

3\. Semantic (cost: 3.0)

4\. Analogical (cost: 4.0)

5\. Synthetic (cost: 5.0)

6\. BruteForce (cost: 10.0)

\*\*Failure Handling:\*\*

The chain provides comprehensive failure tracking and recovery:

```python

\# Execute with automatic fallback

plan, failure = fallback\_chain.execute\_with\_fallbacks(problem\_graph)

if plan is None:

&nbsp; # Access detailed failure information

&nbsp; print(f"Failed component: {failure.missing\_component}")

&nbsp; print(f"Attempted strategies: {failure.attempted\_strategies}")

&nbsp; print(f"Failure reasons: {failure.failure\_reasons}")

&nbsp; print(f"Suggested fallbacks: {failure.suggested\_fallbacks}")

```

\*\*Failure Types:\*\*

\- `EXCEPTION`: Unexpected error

\- `TIMEOUT`: Exceeded time limit

\- `INVALID\_OUTPUT`: Result validation failed

\- `INCOMPLETE`: Partial decomposition only

\- `RESOURCE\_EXCEEDED`: Memory/resource limits

\- `UNSUPPORTED`: Strategy can't handle problem type

\*\*Recovery Strategies:\*\*

The chain automatically suggests recovery approaches:

\- Timeout → increase timeout or simplify problem

\- Resource exceeded → reduce problem size

\- Unsupported → use alternative strategy

\- Invalid output → validate and retry

\- Exception → fallback to simpler approach

\*\*Dynamic Reordering:\*\*

```python

\# Reorder by cost-effectiveness

fallback\_chain.reorder\_by\_cost()

\# Reorder for specific problem class

fallback\_chain.reorder\_by\_cost(problem\_class='optimization')

\# Success rates are adjusted per problem class:

\# - optimization: prefers gradient methods

\# - planning: prefers hierarchical approaches

\# - classification: prefers tree/ensemble methods

```

\*\*Strategy Management:\*\*

```python

\# Add strategy to chain

fallback\_chain.add\_strategy(new\_strategy, cost=2.5)

\# Remove underperforming strategy

fallback\_chain.remove\_strategy('problematic\_strategy')

\# Generate multiple fallback plans

plans = fallback\_chain.generate\_fallback\_plans(problem\_graph)

\# Returns: \[standard\_plan, simplified\_plan, hierarchical\_plan]

```

\*\*Execution Tracking:\*\*

\- Maintains execution history (1000 most recent)

\- Tracks failure counts per strategy

\- Updates success rates with exponential decay

\- Auto-demotes strategies after 5+ consecutive failures

\#### 5. Adaptive Thresholds (`adaptive\_thresholds.py`)

Self-adjusting system parameters that learn from performance.

\*\*Threshold Types:\*\*

\- `CONFIDENCE`: Minimum confidence for accepting decomposition (default: 0.7)

\- `COMPLEXITY`: Maximum problem complexity to attempt (default: 3.0)

\- `PERFORMANCE`: Minimum success rate threshold (default: 0.6)

\- `TIMEOUT`: Maximum execution time in seconds (default: 60.0)

\- `RESOURCE`: Maximum resource usage (default: 0.8)

\*\*Auto-Calibration:\*\*

\- Monitors success rates and execution times

\- Adjusts thresholds based on performance windows

\- Lowers confidence threshold if success rate < 40%

\- Raises confidence threshold if success rate > 90%

\- Adjusts timeouts based on actual execution patterns

\#### 6. Performance Tracking

\*\*PerformanceTracker:\*\*

\- Records all decomposition attempts

\- Tracks success/failure rates per strategy

\- Analyzes failure reasons

\- Maintains windowed performance history (default: 100 records)

\*\*StrategyProfiler:\*\*

\- Profiles each strategy's performance

\- Estimates execution costs per problem class

\- Recommends optimal strategy ordering

\- Caches cost-effectiveness calculations

\#### 7. Problem Executor (`problem\_executor.py`)

Executes decomposed plans with monitoring and validation.

\*\*Features:\*\*

\- Step-by-step execution with dependency tracking

\- Timeout management

\- Result validation

\- Execution state tracking

\- Error handling and recovery

\## Initialization and Bootstrap

\### Using the Factory Function

The recommended way to create a decomposer:

```python

from vulcan.problem\_decomposer import create\_decomposer, ProblemGraph

\# Create fully initialized decomposer

decomposer = create\_decomposer(

&nbsp; semantic\_bridge=None, # Optional: semantic understanding

&nbsp; vulcan\_memory=None, # Optional: memory system

&nbsp; validator=None, # Optional: solution validator

&nbsp; storage\_path=None # Optional: persistent storage

)

```

\### Bootstrap Process (`decomposer\_bootstrap.py`)

The `DecomposerBootstrap` class handles complete system initialization:

1\. \*\*Create Strategy Instances\*\*: Instantiates all six strategies

2\. \*\*Register Strategies\*\*: Maps strategy names to instances (30+ name mappings)

3\. \*\*Initialize Library\*\*: Populates with base decomposition principles

4\. \*\*Configure Fallback Chain\*\*: Orders strategies by cost-effectiveness

5\. \*\*Set Thresholds\*\*: Configures adaptive thresholds with sensible defaults

6\. \*\*Profile Strategies\*\*: Initial profiling of all strategies

\### Strategy Name Mappings

The system recognizes multiple aliases for each strategy:

```python

\# Exact decomposition

'exact', 'exact\_decomposition', 'pattern\_match', 'direct', 'direct\_decomposition'

\# Structural decomposition (most aliases)

'structural', 'hierarchical', 'modular', 'pipeline', 'parallel', 

'recursive', 'temporal', 'constraint\_based', 'iterative', 'hybrid', 'simple'

\# Semantic decomposition

'semantic', 'semantic\_decomposition', 'concept\_based'

\# Others

'synthetic', 'analogical', 'brute\_force'

```

\## Problem Representation

\### ProblemGraph

Problems are represented as directed graphs:

```python

problem = ProblemGraph(

&nbsp; nodes={

&nbsp; 'node\_id': {

&nbsp; 'type': 'operation',

&nbsp; 'complexity': 2.0,

&nbsp; # ... other attributes

&nbsp; }

&nbsp; },

&nbsp; edges=\[

&nbsp; ('source', 'target', {'weight': 1.0}),

&nbsp; # ... more edges

&nbsp; ],

&nbsp; root='starting\_node',

&nbsp; metadata={

&nbsp; 'domain': 'planning',

&nbsp; 'type': 'hierarchical',

&nbsp; # ... other metadata

&nbsp; }

)

```

\*\*Node Types:\*\*

\- `operation`: Computational operation

\- `decision`: Branching/conditional logic

\- `transform`: Data transformation

\- `terminal`: End state

\*\*Metadata Fields:\*\*

\- `domain`: Problem domain (planning, optimization, classification, etc.)

\- `type`: Problem structure type (hierarchical, sequential, parallel, etc.)

\- `complexity`: Estimated problem complexity

\- `constraints`: Problem-specific constraints

\### ExecutionPlan

The result of decomposition is an ExecutionPlan:

```python

plan = ExecutionPlan(plan\_id="unique\_id")

\# Components are DecompositionComponent objects

component = DecompositionComponent(

&nbsp; component\_id="comp\_1",

&nbsp; component\_type=ComponentType.ATOMIC,

&nbsp; description="Process data",

&nbsp; dependencies=\["comp\_0"],

&nbsp; estimated\_cost=1.0,

&nbsp; confidence=0.8

)

plan.add\_components(\[component])

\# Get execution order (respects dependencies)

order = plan.get\_execution\_order() # Topological sort

\# Check plan validity

is\_complete, issues = plan.validate\_completeness()

\# Get overall confidence

confidence = plan.overall\_confidence() # Weighted by cost

```

\*\*Component Types:\*\*

\- `ATOMIC`: Indivisible component

\- `COMPOSITE`: Contains sub-components

\- `SEQUENTIAL`: Must execute in order

\- `PARALLEL`: Can execute concurrently

\- `CONDITIONAL`: Depends on runtime conditions

\## Usage Examples

\### Basic Decomposition

```python

\# Create decomposer

decomposer = create\_decomposer()

\# Create problem

problem = ProblemGraph(

&nbsp; nodes={'A': {}, 'B': {}, 'C': {}},

&nbsp; edges=\[('A', 'B', {}), ('B', 'C', {})],

&nbsp; metadata={'domain': 'planning'}

)

\# Decompose problem

plan = decomposer.decompose\_novel\_problem(problem)

print(f"Strategy: {plan.strategy\_used}")

print(f"Steps: {len(plan.steps)}")

print(f"Confidence: {plan.confidence:.2f}")

```

\### Decompose and Execute

```python

\# Decompose and execute in one call

plan, outcome = decomposer.decompose\_and\_execute(problem)

print(f"Success: {outcome.success}")

print(f"Execution time: {outcome.execution\_time:.2f}s")

print(f"Steps completed: {len(outcome.step\_results)}")

```

\### Working with ExecutionPlan

```python

\# Create custom execution plan

plan = ExecutionPlan(plan\_id="custom\_plan")

\# Add components

components = \[

&nbsp; DecompositionComponent(

&nbsp; component\_id="step1",

&nbsp; component\_type=ComponentType.ATOMIC,

&nbsp; description="Load data",

&nbsp; confidence=0.9

&nbsp; ),

&nbsp; DecompositionComponent(

&nbsp; component\_id="step2",

&nbsp; component\_type=ComponentType.ATOMIC,

&nbsp; description="Process data",

&nbsp; dependencies=\["step1"],

&nbsp; confidence=0.8

&nbsp; )

]

plan.add\_components(components)

\# Validate plan

is\_complete, issues = plan.validate\_completeness()

if not is\_complete:

&nbsp; print(f"Issues: {issues}")

\# Get execution order (respects dependencies)

execution\_order = plan.get\_execution\_order()

\# Check overall confidence

overall\_conf = plan.overall\_confidence()

\# Execute components in order

for component\_id in execution\_order:

&nbsp; component = plan.component\_map\[component\_id]

&nbsp; # Execute component...

&nbsp; plan.update\_component\_status(component\_id, StrategyStatus.SUCCESS)

\# Get summary

summary = plan.get\_execution\_summary()

```

\### Fallback Chain with Error Handling

```python

\# Create fallback chain

chain = FallbackChain()

\# Add strategies

chain.add\_strategy(ExactDecomposition(), cost=1.0)

chain.add\_strategy(StructuralDecomposition(), cost=2.0)

chain.add\_strategy(SemanticDecomposition(), cost=3.0)

\# Execute with fallbacks

plan, failure = chain.execute\_with\_fallbacks(problem)

if plan is None:

&nbsp; print(f"All strategies failed!")

&nbsp; print(f"Attempted: {failure.attempted\_strategies}")

&nbsp; print(f"Reasons: {failure.failure\_reasons}")

&nbsp; print(f"Suggestions: {failure.suggested\_fallbacks}")

&nbsp; 

&nbsp; # Try alternative approaches

&nbsp; fallback\_plans = chain.generate\_fallback\_plans(problem)

&nbsp; for alt\_plan in fallback\_plans:

&nbsp; print(f"Alternative plan: {alt\_plan.metadata.get('type')}")

else:

&nbsp; print(f"Success with strategy: {plan.metadata\['strategy']}")

```

\### Library Pattern Management

```python

\# Create library

library = StratifiedDecompositionLibrary()

\# Add pattern

pattern = Pattern(

&nbsp; pattern\_id="custom\_pattern",

&nbsp; structure=graph,

&nbsp; features={'type': 'hierarchical', 'depth': 3}

)

\# Create principle

principle = DecompositionPrinciple(

&nbsp; principle\_id="custom\_principle",

&nbsp; name="Custom Decomposition",

&nbsp; pattern=pattern,

&nbsp; applicable\_contexts=\[

&nbsp; Context(domain='planning', problem\_type='sequential')

&nbsp; ]

)

library.add\_principle(principle)

\# Find similar patterns

similar = library.find\_similar(new\_graph, top\_k=5)

for pattern\_id, similarity in similar:

&nbsp; print(f"Pattern {pattern\_id}: {similarity:.2f}")

\# Get applicable principles for context

context = Context(domain='planning', problem\_type='hierarchical')

applicable = library.get\_applicable\_principles(context)

\# Reinforce successful pattern

library.reinforce\_pattern(pattern\_signature, decomposition, performance=0.85)

\# Mark failed pattern

library.mark\_failed\_pattern(pattern\_signature, decomposition, reason="timeout")

\# Get domain statistics

stats = library.get\_domain\_statistics()

```

\### Working with Test Problems

```python

from vulcan.problem\_decomposer import create\_test\_problem

\# Create different test problem types

hierarchical = create\_test\_problem('hierarchical')

sequential = create\_test\_problem('sequential')

parallel = create\_test\_problem('parallel')

cyclic = create\_test\_problem('cyclic')

simple = create\_test\_problem('simple')

\# Test decomposition

plan = decomposer.decompose\_novel\_problem(hierarchical)

```

\## Adaptive Learning

\### How the System Learns

The decomposer continuously improves through:

1\. \*\*Performance Tracking\*\*: Records every decomposition attempt with outcomes

2\. \*\*Strategy Profiling\*\*: Updates success rates and costs per strategy

3\. \*\*Threshold Adjustment\*\*: Auto-calibrates based on recent performance

4\. \*\*Pattern Recognition\*\*: Stores successful decomposition patterns

5\. \*\*Cache Building\*\*: Remembers which strategies work for which problems

\### Performance Windows

The system maintains sliding windows of recent performance:

\- \*\*Threshold auto-calibration\*\*: Last 50 attempts

\- \*\*Performance tracking\*\*: Last 100 attempts

\- \*\*Problem-specific history\*\*: Last 50 attempts per problem

\### When Thresholds Adjust

\*\*Confidence Threshold:\*\*

\- Decreases if success rate < 40% (more attempts allowed)

\- Increases if success rate > 90% (higher bar for quality)

\*\*Timeout Threshold:\*\*

\- Increases if execution time > 80% of current timeout

\- Decreases if execution time < 30% of current timeout

\*\*Complexity Threshold:\*\*

\- Adjusts based on average complexity of handled problems

\### Pattern and Principle Library

The system builds a knowledge base through the `StratifiedDecompositionLibrary`:

\*\*Pattern Storage:\*\*

\- Stores reusable decomposition patterns with embeddings

\- Indexes patterns by signature for fast lookup

\- Tracks pattern performance across domains

\- Uses cosine similarity for pattern matching

\*\*Pattern Components:\*\*

```python

pattern = Pattern(

&nbsp; pattern\_id="pattern\_123",

&nbsp; structure=graph, # NetworkX graph or equivalent

&nbsp; features={'node\_count': 5, 'density': 0.4},

&nbsp; embedding=np.array(\[...]), # 64-dim vector

&nbsp; metadata={'status': 'proven', 'success\_rate': 0.85}

)

```

\*\*Decomposition Principles:\*\*

```python

principle = DecompositionPrinciple(

&nbsp; principle\_id="hierarchical\_principle",

&nbsp; name="Hierarchical Decomposition",

&nbsp; pattern=pattern,

&nbsp; applicable\_contexts=\[context1, context2],

&nbsp; success\_rate=0.75,

&nbsp; contraindications=\['flat\_structure', 'no\_hierarchy'],

&nbsp; usage\_count=150,

&nbsp; success\_count=112

)

```

\*\*Context Matching:\*\*

\- Principles match contexts with scores \[0, 1]

\- Domain match: +0.5

\- Problem type match: +0.3

\- Constraint overlap: +0.2

\- Contraindications: automatic rejection

\*\*Pattern Performance Tracking:\*\*

```python

performance = PatternPerformance(

&nbsp; pattern\_signature="abc123...",

&nbsp; total\_uses=100,

&nbsp; successful\_uses=85,

&nbsp; failed\_uses=15,

&nbsp; avg\_execution\_time=1.5,

&nbsp; domains\_used={'planning', 'optimization'},

&nbsp; failure\_reasons=\['timeout', 'incomplete']

)

```

\*\*Domain Stratification:\*\*

The library categorizes domains by usage frequency:

\- `FREQUENT`: 100+ patterns

\- `COMMON`: 20-99 patterns

\- `RARE`: 5-19 patterns

\- `NOVEL`: <5 patterns

\*\*Cross-Domain Patterns:\*\*

Patterns used in 3+ domains are identified as cross-domain, indicating general applicability.

\*\*Pattern Reinforcement:\*\*

\- Successful patterns are reinforced with performance scores

\- Status promotes to `PROVEN` after 10+ uses with 80%+ success

\- Failed patterns downgrade to `EXPERIMENTAL` or `FAILED`

\- Patterns maintain failure reason lists (bounded to 100 entries)

\## Monitoring and Statistics

\### Get Current Statistics

```python

\# Threshold statistics

threshold\_stats = decomposer.thresholds.get\_statistics()

print(f"Current thresholds: {threshold\_stats\['current\_thresholds']}")

print(f"Auto-calibrations: {threshold\_stats\['auto\_calibrations']}")

\# Performance statistics

perf\_stats = decomposer.performance\_tracker.get\_statistics()

print(f"Success rate: {perf\_stats\['overall\_success\_rate']:.2%}")

print(f"Unique problems: {perf\_stats\['unique\_problems']}")

\# Strategy profiler statistics

profiler\_stats = decomposer.strategy\_profiler.get\_statistics()

for name, summary in profiler\_stats\['strategy\_summaries'].items():

&nbsp; print(f"{name}: {summary\['success\_rate']:.2%} success")

```

\### Failure Analysis

```python

\# Analyze why decompositions fail

failure\_analysis = decomposer.performance\_tracker.get\_failure\_analysis()

print(f"Total failures: {failure\_analysis\['total\_failures']}")

print(f"Top reason: {failure\_analysis\['top\_reason']}")

for reason, percentage in failure\_analysis\['percentages'].items():

&nbsp; print(f" {reason}: {percentage:.1%}")

```

\## Thread Safety

All core components are thread-safe:

\- Use `threading.RLock()` for recursive locking

\- Lock acquisition in all public methods

\- Safe for concurrent decomposition requests

\- Protected shared state (thresholds, performance data, caches)

\## Performance Considerations

\### Memory Management

The system bounds all unbounded structures:

\- Performance records: 100-item deque (not unbounded list)

\- Problem history: 50 records per problem (bounded deque)

\- Adjustment history: 100-item deque

\- Strategy counters: Use `Counter` (efficient for sparse data)

\- Cache limits: 100 items max in ordering cache

\- Embedding cache: 1000 entries with LRU eviction (20% cleared when full)

\- Similarity cache: 1000 entries with FIFO eviction (10% cleared when full)

\- Recovery strategies: 100 max entries

\- Execution history: 1000 most recent executions

\- Failure reasons: 100 most recent per pattern

\### Optimization Strategies

1\. \*\*Fast Path\*\*: Try cheapest strategies first

2\. \*\*Caching\*\*: Remember successful strategy choices

3\. \*\*Early Exit\*\*: Stop on first successful decomposition

4\. \*\*Lazy Evaluation\*\*: Don't profile until needed

5\. \*\*Windowed Data\*\*: Only keep recent performance data

6\. \*\*LRU/FIFO Eviction\*\*: Automatic cache management

7\. \*\*Bounded Collections\*\*: All collections have size limits

8\. \*\*Signature-based Indexing\*\*: Fast pattern lookup via MD5 hashes

\### Typical Performance

\- \*\*Simple problems\*\*: < 0.1s (exact/structural)

\- \*\*Medium problems\*\*: 0.1-1s (semantic/analogical)

\- \*\*Complex problems\*\*: 1-10s (synthetic/brute force)

\### Pattern Matching Performance

\- \*\*Exact matching\*\*: O(n) where n = number of patterns

\- \*\*Structural matching\*\*: O(m) where m = number of nodes

\- \*\*Semantic matching\*\*: O(n²) for clustering, limited to 1000 nodes

\- \*\*Similarity search\*\*: O(k) where k = patterns in library (with caching)

\### Scalability Limits

\*\*Hard Limits:\*\*

\- Semantic clustering: 1000 nodes per operation

\- Pattern clustering: 100 clusters maximum

\- Library storage files: 100MB size limit

\- Recovery strategies: 100 unique strategies

\*\*Soft Limits (configurable):\*\*

\- Brute force iterations: 1000 (max\_iterations)

\- Brute force depth: 3 (max\_depth)

\- Fallback retries: 3 (max\_retries)

\- Strategy timeout: 60 seconds

\## Error Handling

\### Strategy Fallback

When a strategy fails, the system:

1\. Logs the failure reason

2\. Records performance data

3\. Automatically tries next strategy in chain

4\. Adjusts thresholds based on outcome

\### Comprehensive Failure Information

The `DecompositionFailure` class provides detailed diagnostics:

```python

failure = DecompositionFailure(

&nbsp; problem\_signature="abc123...",

&nbsp; missing\_component="hierarchical\_decomp",

&nbsp; attempted\_strategies=\['exact', 'structural', 'semantic'],

&nbsp; failure\_reasons={

&nbsp; 'exact': 'no matching pattern',

&nbsp; 'structural': 'timeout',

&nbsp; 'semantic': 'resource\_exceeded'

&nbsp; },

&nbsp; suggested\_fallbacks=\['simplified', 'hierarchical']

)

```

\### Failure Types and Recovery

\*\*EXCEPTION\*\* → fallback\_to\_simple

\- Unexpected errors

\- Stack traces logged for debugging

\*\*TIMEOUT\*\* → increase\_timeout\_or\_simplify

\- Strategy exceeded time limit

\- Consider simpler decomposition

\*\*INVALID\_OUTPUT\*\* → validate\_and\_retry

\- Result didn't pass validation

\- Retry with adjusted parameters

\*\*INCOMPLETE\*\* → use\_next\_strategy

\- Partial decomposition only

\- Need complementary approach

\*\*RESOURCE\_EXCEEDED\*\* → reduce\_problem\_size

\- Memory/CPU limits hit

\- Simplify or partition problem

\*\*UNSUPPORTED\*\* → use\_alternative\_strategy

\- Strategy can't handle problem type

\- Try different approach

\### Validation

```python

from vulcan.problem\_decomposer import validate\_decomposer\_setup

\# Validate decomposer is properly configured

validation = validate\_decomposer\_setup(decomposer)

if not validation\['valid']:

&nbsp; print("Errors:", validation\['errors'])

&nbsp; # Examples:

&nbsp; # - "No strategies registered in library"

&nbsp; # - "Fallback chain has no strategies"

&nbsp; # - "Executor not initialized"

&nbsp; 

if validation\['warnings']:

&nbsp; print("Warnings:", validation\['warnings'])

&nbsp; # Examples:

&nbsp; # - "No principles in library"

&nbsp; # - "Confidence threshold not set"

print("Checks passed:", validation\['checks'])

\# - strategy\_count: 30+

\# - fallback\_chain\_count: 6

\# - executor\_initialized: True

\# - confidence\_threshold: 0.6

```

\### Strategy Lifecycle Management

```python

\# Strategies track their own statistics

strategy = StructuralDecomposition()

print(f"Success rate: {strategy.get\_success\_rate():.2%}")

print(f"Avg time: {strategy.get\_average\_execution\_time():.2f}s")

\# Chain manages strategy health

chain.handle\_strategy\_failure(strategy, exception)

\# After 5+ failures, strategy is demoted

\# (moved to end of chain, not removed)

if chain.failure\_counts\[strategy.name] > 5:

&nbsp; # Strategy moved to end automatically

&nbsp; pass

```

\## Extension Points

\### Adding New Strategies

```python

from vulcan.problem\_decomposer.decomposition\_strategies import DecompositionStrategy

class CustomStrategy(DecompositionStrategy):

&nbsp; def \_\_init\_\_(self):

&nbsp; super().\_\_init\_\_("custom\_strategy")

&nbsp; 

&nbsp; def decompose(self, problem: ProblemGraph) -> ExecutionPlan:

&nbsp; # Implement decomposition logic

&nbsp; pass

&nbsp; 

&nbsp; def can\_handle(self, problem: ProblemGraph) -> bool:

&nbsp; # Check if strategy can handle this problem

&nbsp; pass

\# Register in bootstrap

\# Add to create\_strategy\_instances() in decomposer\_bootstrap.py

```

\### Custom Thresholds

```python

\# Create decomposer with custom thresholds

decomposer = create\_decomposer()

\# Set custom values

decomposer.thresholds.thresholds\['confidence'].value = 0.8

decomposer.thresholds.thresholds\['timeout'].value = 120.0

\# Or initialize with custom values

custom\_thresholds = AdaptiveThresholds({

&nbsp; 'confidence': 0.8,

&nbsp; 'complexity': 5.0,

&nbsp; 'timeout': 120.0

})

```

\## Best Practices

1\. \*\*Use the Factory\*\*: Always use `create\_decomposer()` for initialization

2\. \*\*Validate Setup\*\*: Run `validate\_decomposer\_setup()` in production

3\. \*\*Monitor Performance\*\*: Regularly check statistics and failure analysis

4\. \*\*Let It Learn\*\*: Allow auto-calibration to run (20+ attempts needed)

5\. \*\*Set Metadata\*\*: Provide rich metadata in ProblemGraph for better strategy selection

6\. \*\*Handle Outcomes\*\*: Check `ExecutionOutcome.success` and handle failures

7\. \*\*Use Test Problems\*\*: Validate with `create\_test\_problem()` during development

\## Testing

\### Run Bootstrap Test

```python

from vulcan.problem\_decomposer.decomposer\_bootstrap import run\_bootstrap\_test

\# Runs complete system test

success = run\_bootstrap\_test()

```

\### Create Test Problems

```python

from vulcan.problem\_decomposer import create\_test\_problem

\# Available types: 'hierarchical', 'sequential', 'parallel', 'cyclic', 'simple'

problem = create\_test\_problem('hierarchical')

```

\## Configuration Files

The system doesn't require configuration files but supports:

\- Custom storage paths for persistence

\- Integration with external semantic bridges

\- Connection to VULCAN memory systems

\- Custom validators for solution verification

\## Logging

All components use Python's logging system:

```python

import logging

\# Configure logging level

logging.basicConfig(level=logging.INFO)

\# Component-specific loggers

logger = logging.getLogger('vulcan.problem\_decomposer')

logger.setLevel(logging.DEBUG)

```

\*\*Log Levels:\*\*

\- `DEBUG`: Strategy selection, threshold adjustments

\- `INFO`: Decomposition starts/completions, initialization

\- `WARNING`: Missing strategies, validation issues

\- `ERROR`: Decomposition failures, system errors

\## Common Patterns

\### Retry with Different Strategy

```python

max\_attempts = 3

for attempt in range(max\_attempts):

&nbsp; plan, outcome = decomposer.decompose\_and\_execute(problem)

&nbsp; if outcome.success:

&nbsp; break

&nbsp; print(f"Attempt {attempt + 1} failed, retrying...")

```

\### Strategy-Specific Decomposition

```python

\# Force a specific strategy

from vulcan.problem\_decomposer.decomposition\_strategies import StructuralDecomposition

strategy = StructuralDecomposition()

result = strategy.apply(problem)

\# Check result

if result.is\_complete():

&nbsp; print(f"Confidence: {result.confidence}")

&nbsp; print(f"Components: {len(result.components)}")

```

\### Performance Monitoring

```python

\# Monitor success rate in real-time

recent\_rate = decomposer.performance\_tracker.get\_success\_rate(window=20)

if recent\_rate < 0.5:

&nbsp; print("Warning: Success rate dropped below 50%")

&nbsp; # Maybe adjust thresholds manually

&nbsp; decomposer.thresholds.adjust\_down(0.1, 'confidence')

```

\### Pattern-Based Decomposition

```python

\# Find and use similar patterns

library = decomposer.library

\# Find similar patterns

similar\_patterns = library.find\_similar(problem\_graph, top\_k=3)

for pattern\_id, similarity in similar\_patterns:

&nbsp; if similarity > 0.8:

&nbsp; # Use this pattern

&nbsp; pattern = library.patterns\[pattern\_id]

&nbsp; print(f"Using pattern: {pattern\_id} (similarity: {similarity:.2f})")

```

\### Context-Aware Decomposition

```python

\# Define problem context

context = Context(

&nbsp; domain='optimization',

&nbsp; problem\_type='continuous',

&nbsp; constraints={'bounded': True, 'convex': True}

)

\# Get applicable principles

principles = library.get\_applicable\_principles(context)

for principle in principles:

&nbsp; is\_applicable, match\_score = principle.is\_applicable(context)

&nbsp; if is\_applicable:

&nbsp; print(f"{principle.name}: {match\_score:.2f} match")

&nbsp; print(f"Success rate: {principle.success\_rate:.2%}")

```

\### Fallback Plan Generation

```python

\# Generate multiple fallback plans

chain = decomposer.fallback\_chain

plans = chain.generate\_fallback\_plans(problem\_graph)

for i, plan in enumerate(plans):

&nbsp; print(f"Plan {i}: {plan.metadata.get('type')}")

&nbsp; print(f" Components: {len(plan.components)}")

&nbsp; print(f" Confidence: {plan.overall\_confidence():.2f}")

&nbsp; print(f" Cost: {plan.total\_cost:.2f}")

\# Try plans in order of confidence

plans.sort(key=lambda p: p.overall\_confidence(), reverse=True)

for plan in plans:

&nbsp; # Try to execute plan

&nbsp; pass

```

\### Library Statistics and Reporting

```python

\# Get comprehensive statistics

domain\_stats = library.get\_domain\_statistics()

for domain, stats in domain\_stats.items():

&nbsp; print(f"\\n{domain}:")

&nbsp; print(f" Patterns: {stats\['pattern\_count']}")

&nbsp; print(f" Success rate: {stats\['avg\_success\_rate']:.2%}")

&nbsp; print(f" Category: {stats\['category']}")

\# Get cross-domain patterns

cross\_domain = library.get\_cross\_domain\_patterns(min\_domains=3)

print(f"\\nCross-domain patterns: {len(cross\_domain)}")

\# Get frequent patterns

frequent = library.get\_patterns\_by\_frequency(min\_count=10)

print(f"Frequent patterns: {len(frequent)}")

```

\## Troubleshooting

\### Low Success Rates

\- Check problem metadata is properly set

\- Verify problem graph structure is valid

\- Review failure analysis for common reasons

\- Consider lowering confidence threshold

\- Ensure validator (if used) isn't too strict

\- Check if problem type is supported by strategies

\*\*Diagnosis:\*\*

```python

\# Get failure analysis

failure\_analysis = decomposer.performance\_tracker.get\_failure\_analysis()

print(f"Top failure reason: {failure\_analysis\['top\_reason']}")

for reason, percentage in failure\_analysis\['percentages'].items():

&nbsp; print(f" {reason}: {percentage:.1%}")

\# Check strategy performance

for strategy\_name in \['exact', 'structural', 'semantic']:

&nbsp; perf = decomposer.performance\_tracker.get\_strategy\_performance(strategy\_name)

&nbsp; print(f"{strategy\_name}: {perf\['success\_rate']:.2%} success")

```

\### High Latency

\- Check which strategies are being used

\- Consider setting lower timeout threshold

\- Profile strategy performance

\- Look for expensive operations in custom code

\- Check if semantic clustering hitting 1000 node limit

\*\*Optimization:\*\*

```python

\# Get strategy statistics

for strategy in decomposer.fallback\_chain.strategies:

&nbsp; print(f"{strategy.name}:")

&nbsp; print(f" Avg time: {strategy.get\_average\_execution\_time():.2f}s")

&nbsp; print(f" Success rate: {strategy.get\_success\_rate():.2%}")

\# Reorder by cost-effectiveness

decomposer.fallback\_chain.reorder\_by\_cost()

```

\### Memory Issues

\- All structures are bounded by default

\- Check if custom extensions leak memory

\- Review window sizes (default: 100 items)

\- Monitor problem\_history size

\- Check library storage file sizes

\*\*Memory Checks:\*\*

```python

\# Check cache sizes

print(f"Embedding cache: {len(decomposer.semantic\_strategy.embedding\_cache)}")

print(f"Similarity cache: {len(decomposer.library.similarity\_cache)}")

print(f"Pattern count: {len(decomposer.library.patterns)}")

\# Check bounded collections

print(f"Performance records: {len(decomposer.performance\_tracker.records)}")

print(f"Execution history: {len(decomposer.fallback\_chain.execution\_history)}")

```

\### Strategy Not Found

\- Verify strategy name spelling

\- Check strategy\_registry contents

\- Ensure bootstrap completed successfully

\- Review name mapping in decomposer\_bootstrap.py

\*\*Debugging:\*\*

```python

\# List all registered strategies

if hasattr(decomposer.library, 'strategy\_registry'):

&nbsp; print("Registered strategies:")

&nbsp; for name in decomposer.library.strategy\_registry.keys():

&nbsp; print(f" - {name}")

\# Test strategy retrieval

test\_names = \['hierarchical', 'structural', 'exact']

for name in test\_names:

&nbsp; strategy = decomposer.library.get\_strategy\_by\_type(name)

&nbsp; print(f"{name}: {'✓' if strategy else '✗'}")

```

\### Pattern Not Matching

\- Check pattern signatures are generated correctly

\- Verify graph structure conversion

\- Review similarity thresholds

\- Check if pattern exists in library

\*\*Pattern Debugging:\*\*

```python

\# Get pattern signature

if hasattr(problem\_graph, 'get\_signature'):

&nbsp; sig = problem\_graph.get\_signature()

&nbsp; print(f"Problem signature: {sig}")

\# Check if pattern exists

pattern = library.patterns.get(pattern\_id)

if pattern:

&nbsp; print(f"Pattern signature: {pattern.get\_signature()}")

&nbsp; print(f"Features: {pattern.features}")

\# Test similarity calculation

similar = library.find\_similar(problem\_graph, top\_k=5)

print(f"Similar patterns: {len(similar)}")

for pid, sim in similar:

&nbsp; print(f" {pid}: {sim:.3f}")

```

\### Validation Failures

Run comprehensive validation:

```python

validation = validate\_decomposer\_setup(decomposer)

\# Check each validation result

checks = validation\['checks']

print(f"Strategy count: {checks.get('strategy\_count', 0)}")

print(f"Fallback chain: {checks.get('fallback\_chain\_count', 0)}")

print(f"Executor: {'✓' if checks.get('executor\_initialized') else '✗'}")

\# Review errors

if validation\['errors']:

&nbsp; print("\\nErrors to fix:")

&nbsp; for error in validation\['errors']:

&nbsp; print(f" ❌ {error}")

\# Review warnings

if validation\['warnings']:

&nbsp; print("\\nWarnings:")

&nbsp; for warning in validation\['warnings']:

&nbsp; print(f" ⚠️ {warning}")

```

\### Library Loading Issues

Check for corrupted or oversized storage files:

```python

storage\_path = Path("decomposition\_library")

if storage\_path.exists():

&nbsp; # Check file sizes

&nbsp; for file in storage\_path.iterdir():

&nbsp; size\_mb = file.stat().st\_size / 1\_000\_000

&nbsp; print(f"{file.name}: {size\_mb:.1f} MB")

&nbsp; 

&nbsp; if size\_mb > 100:

&nbsp; print(f" ⚠️ File too large, may fail to load")

&nbsp; # Try loading manually

&nbsp; try:

&nbsp; library = StratifiedDecompositionLibrary(storage\_path)

&nbsp; print(f"✓ Library loaded: {len(library.patterns)} patterns")

&nbsp; except Exception as e:

&nbsp; print(f"✗ Library loading failed: {e}")

```

\## Future Enhancements

Potential areas for extension:

1\. \*\*Parallel Execution\*\*: Execute independent subproblems concurrently

2\. \*\*Distributed Decomposition\*\*: Spread decomposition across multiple nodes

3\. \*\*Learning Integration\*\*: Connect to principle learner for deeper learning

4\. \*\*Visualization\*\*: Add decomposition tree visualization

5\. \*\*Persistence\*\*: Save/load decomposer state and learned patterns

6\. \*\*Metrics Dashboard\*\*: Real-time monitoring interface

7\. \*\*A/B Testing\*\*: Compare strategy effectiveness systematically

\## Learning Integration System

The decomposer includes a comprehensive learning system that closes the loop from execution to knowledge extraction and reuse.

\### UnifiedDecomposerLearner

The main interface combining decomposition with all learning systems:

```python

from vulcan.problem\_decomposer.learning\_integration import create\_unified\_decomposer

\# Create unified system with all learning capabilities

learner = create\_unified\_decomposer(

&nbsp; enable\_all=True, # Enable all systems

&nbsp; enable\_principle\_learning=True # Enable automatic principle extraction

)

\# Decompose and execute with full learning

plan, outcome = learner.decompose\_and\_execute(problem)

\# Get comprehensive statistics

stats = learner.get\_comprehensive\_statistics()

print(f"Principles extracted: {stats\['integration\_stats']\['principles\_extracted']}")

print(f"Principles promoted: {stats\['integration\_stats']\['principles\_promoted']}")

\# Get learning recommendations

recommendations = learner.get\_learning\_recommendations()

```

\### Learning Systems Integration

\*\*Seven Integrated Learning Systems:\*\*

1\. \*\*Continual Learning\*\* - Task detection, catastrophic forgetting prevention (EWC), experience replay

2\. \*\*Curriculum Learning\*\* - Difficulty-based problem ordering, adaptive pacing

3\. \*\*Meta-Learning\*\* - Learning to learn, fast adaptation to new tasks

4\. \*\*Metacognition\*\* - Self-awareness, confidence calibration, strategy selection

5\. \*\*RLHF (Reinforcement Learning from Human Feedback)\*\* - Human preference integration

6\. \*\*Principle Learning\*\* - Automatic extraction and promotion of decomposition principles

7\. \*\*Adaptive Thresholds\*\* - Dynamic parameter adjustment based on performance

\### Principle Learning Pipeline

The complete learning loop from execution to knowledge reuse:

```

ExecutionOutcome → Crystallization → Validation → Promotion → Library → Reuse

```

\*\*Step 1: Conversion to ExecutionTrace\*\*

```python

\# Automatic conversion in learn\_integrated()

trace = converter.convert(problem, plan, outcome)

\# Contains: actions, outcomes, context, metrics, patterns

```

\*\*Step 2: Principle Extraction (Crystallization)\*\*

```python

\# Extracts reusable principles from execution

crystallization\_result = crystallizer.crystallize(trace)

principles = crystallization\_result.principles

\# Extracted principles include:

\# - Core patterns (sequential, hierarchical, iterative, etc.)

\# - Success conditions

\# - Applicable contexts

\# - Confidence scores

```

\*\*Step 3: Cross-Domain Validation\*\*

```python

\# Validates principles across multiple domains

validation\_results = validator.validate\_across\_domains(

&nbsp; principle, applicable\_domains

)

\# Validation provides:

\# - Success rate per domain

\# - Overall confidence

\# - Validation level (basic, standard, comprehensive, rigorous)

\# - Successful/failed domains

```

\*\*Step 4: Promotion to Library\*\*

```python

\# Evaluates for promotion

candidate = promoter.evaluate\_for\_promotion(principle, validation\_results)

\# Promotion score based on:

\# - Validation success rate (35% weight)

\# - Confidence level (25% weight) 

\# - Domain breadth (20% weight)

\# - Evidence strength (15% weight)

\# - Overall validation confidence (5% weight)

\# Promotes if score >= threshold (default 0.7)

if candidate.promotion\_score >= 0.7:

&nbsp; promoter.promote(candidate)

&nbsp; # Principle added to library for future reuse

```

\*\*Step 5: Reuse in Future Problems\*\*

```python

\# Find applicable principles for new problem

applicable = learner.get\_applicable\_principles(problem)

\# Principles are automatically considered in decomposition

\# Improves success rate and reduces exploration

```

\### Principle Learner Components

\*\*PrincipleLearner Class:\*\*

```python

learner = PrincipleLearner(

&nbsp; library=decomposition\_library,

&nbsp; min\_promotion\_score=0.7,

&nbsp; enable\_auto\_promotion=True

)

\# Main entry point - called automatically in integrated learning

results = learner.extract\_and\_promote(problem, plan, outcome)

\# Results include:

\# - principles\_extracted: Number extracted from execution

\# - principles\_validated: Number that passed validation

\# - principles\_promoted: Number added to library

\# - extraction\_time: Time spent on extraction

\# - validation\_time: Time spent on validation

\# - promotion\_time: Time spent on promotion decisions

```

\*\*Knowledge Base Management:\*\*

```python

\# Versioned storage with history

knowledge\_base = VersionedKnowledgeBase()

\# Store principle with version control

knowledge\_base.store(principle, author='system', message='Extracted from decomposition')

\# Index for fast retrieval

knowledge\_index = KnowledgeIndex()

knowledge\_index.index\_principle(principle)

\# Find relevant principles

relevant = knowledge\_index.find\_relevant({

&nbsp; 'domain': 'planning',

&nbsp; 'patterns': \['hierarchical'],

&nbsp; 'keywords': \['optimization', 'sequential']

})

```

\*\*Principle Quality Management:\*\*

```python

\# Prune low-quality or outdated principles

pruned = learner.prune\_low\_quality\_principles(

&nbsp; age\_threshold\_days=90,

&nbsp; confidence\_threshold=0.3

)

\# Export learned knowledge

learner.export\_principles(Path('principles.json'), format='json')

\# Import from file

learner.import\_principles(Path('principles.json'))

```

\### Experience Conversion

\*\*Problem to Experience Format:\*\*

```python

converter = ProblemToExperienceConverter(embedding\_dim=512)

\# Converts decomposition artifacts to learning format

experience = converter.convert\_to\_experience(problem, plan, outcome)

\# Experience includes:

\# - embedding: 512-dim problem representation

\# - reward: Performance-based reward signal

\# - loss: Error metric

\# - modality: 'problem\_decomposition'

\# - metadata: Problem characteristics, strategy used, metrics

\# - context: Problem structure, constraints, resources

```

\*\*Difficulty Estimation:\*\*

```python

estimator = DecompositionDifficultyEstimator()

\# Estimates problem difficulty \[0, 1]

difficulty = estimator.estimate(problem)

\# Based on:

\# - Complexity score

\# - Node/edge counts

\# - Constraint count

\# - Historical domain difficulty

\# - Structural features

```

\### RLHF Integration

\*\*Feedback Routing:\*\*

```python

router = RLHFFeedbackRouter(rlhf\_manager)

\# Route execution outcomes to RLHF system

feedback = router.route\_outcome\_to\_feedback(problem, plan, outcome)

\# Provide explicit human feedback

learner.provide\_human\_feedback(

&nbsp; problem\_signature="abc123...",

&nbsp; rating=0.85,

&nbsp; comments="Good decomposition but could be more efficient"

)

```

\### Curriculum Learning

\*\*Adaptive Problem Ordering:\*\*

```python

\# Get next problems according to curriculum

next\_batch = learner.get\_next\_curriculum\_problems(batch\_size=10)

\# Curriculum considers:

\# - Current performance level

\# - Problem difficulty progression

\# - Domain coverage

\# - Learning objectives

```

\### Learning Statistics and Monitoring

\*\*Comprehensive Statistics:\*\*

```python

stats = learner.get\_comprehensive\_statistics()

\# Returns nested dictionary with:

stats = {

&nbsp; 'integration\_stats': {

&nbsp; 'total\_learning\_calls': 150,

&nbsp; 'continual\_learning\_updates': 148,

&nbsp; 'curriculum\_batches': 15,

&nbsp; 'meta\_updates': 145,

&nbsp; 'rlhf\_feedback': 150,

&nbsp; 'metacognition\_introspections': 150,

&nbsp; 'principles\_extracted': 45,

&nbsp; 'principles\_validated': 38,

&nbsp; 'principles\_promoted': 12,

&nbsp; 'learning\_errors': 2

&nbsp; },

&nbsp; 'decomposer\_stats': {...},

&nbsp; 'continual\_learning': {...},

&nbsp; 'curriculum\_learning': {...},

&nbsp; 'meta\_learning': {...},

&nbsp; 'metacognition': {...},

&nbsp; 'rlhf': {...},

&nbsp; 'principle\_learning': {

&nbsp; 'extraction': {...},

&nbsp; 'validation': {...},

&nbsp; 'promotion': {

&nbsp; 'promoter\_stats': {

&nbsp; 'promoted\_count': 12,

&nbsp; 'rejected\_count': 26,

&nbsp; 'promotion\_rate': 0.32

&nbsp; }

&nbsp; },

&nbsp; 'knowledge\_base': {

&nbsp; 'total\_principles': 12,

&nbsp; 'total\_versions': 38,

&nbsp; 'storage\_size': 124567

&nbsp; },

&nbsp; 'domain\_coverage': {...},

&nbsp; 'pattern\_usage': {...}

&nbsp; }

}

```

\*\*Learning Recommendations:\*\*

```python

recommendations = learner.get\_learning\_recommendations()

\# Returns actionable suggestions like:

\# - "High forgetting detected - increase EWC lambda"

\# - "Low principle promotion rate - consider lowering threshold"

\# - "Large principle library - consider pruning low-quality principles"

\# - "Low decomposition success rate - consider adding more strategies"

```

\### State Persistence

\*\*Save and Load Complete State:\*\*

```python

\# Save all learning system states

saved\_paths = learner.save\_state(path='unified\_learner\_states')

\# Returns paths to saved components:

saved\_paths = {

&nbsp; 'continual\_learner': 'path/to/continual\_learner\_123456.pkl',

&nbsp; 'curriculum\_learner': 'path/to/curriculum\_learner\_123456.pkl',

&nbsp; 'metacognition': 'path/to/metacognition\_123456.pkl',

&nbsp; 'principle\_learner': 'path/to/principle\_learner\_123456',

&nbsp; 'coordinator': 'path/to/coordinator\_123456.json'

}

\# Load from saved state

learner.load\_state(saved\_paths)

\# Graceful shutdown with automatic save

learner.shutdown()

```

\### Memory Management

\*\*Bounded Learning Collections:\*\*

\- Learning history: 1000 most recent

\- Conversion cache: 100 entries (FIFO)

\- Embedding cache: 1000 entries (LRU)

\- Prediction history: 500 entries

\- Domain coverage: 10,000 max domains

\- Pattern usage: Counter (auto-bounded)

\### Safety Integration

The decomposer includes comprehensive safety validation:

```python

\# Automatic safety checks at each stage

plan, outcome = learner.decompose\_and\_execute(problem)

\# Safety validation occurs at:

\# 1. Problem validation (before decomposition)

\# 2. Plan validation (before execution)

\# 3. Outcome validation (before learning)

\# Safety statistics

stats = decomposer.get\_statistics()

safety\_stats = stats\['safety']

print(f"Blocks: {safety\_stats\['total\_blocks']}")

print(f"Corrections: {safety\_stats\['total\_corrections']}")

```

\### Advanced Usage Examples

\*\*Complete Learning Pipeline:\*\*

```python

from vulcan.problem\_decomposer.learning\_integration import create\_unified\_decomposer

\# Create unified learner

learner = create\_unified\_decomposer()

\# Process multiple problems

for problem in problem\_set:

&nbsp; # Decompose and execute with full learning

&nbsp; plan, outcome = learner.decompose\_and\_execute(problem)

&nbsp; 

&nbsp; # Automatic learning happens:

&nbsp; # 1. Experience recorded in continual learner

&nbsp; # 2. Principles extracted and validated

&nbsp; # 3. Successful principles promoted to library

&nbsp; # 4. Feedback sent to RLHF system

&nbsp; # 5. Metacognition updated

&nbsp; # 6. Thresholds adapted

\# Check what was learned

stats = learner.get\_comprehensive\_statistics()

principle\_stats = stats\['principle\_learning']

print(f"Total principles: {principle\_stats\['knowledge\_base']\['total\_principles']}")

print(f"Domain coverage: {len(principle\_stats\['domain\_coverage'])} domains")

print(f"Most used patterns: {principle\_stats\['pattern\_usage']}")

\# Get recommendations for improvement

for rec in learner.get\_learning\_recommendations():

&nbsp; print(f" → {rec}")

\# Prune low-quality knowledge

pruned = learner.prune\_principles(

&nbsp; age\_threshold\_days=90,

&nbsp; confidence\_threshold=0.3

)

print(f"Pruned {pruned} low-quality principles")

\# Export learned knowledge

learner.export\_principles(Path('learned\_principles.json'))

```

\*\*Principle Inspection:\*\*

```python

\# Find principles for a specific problem

problem = create\_test\_problem('hierarchical')

applicable = learner.get\_applicable\_principles(problem)

for principle in applicable:

&nbsp; print(f"Principle: {principle.id}")

&nbsp; print(f" Confidence: {principle.confidence:.2f}")

&nbsp; print(f" Success rate: {principle.get\_success\_rate():.2%}")

&nbsp; print(f" Pattern: {principle.core\_pattern.pattern\_type.value}")

&nbsp; print(f" Applicable domains: {principle.applicable\_domains}")

```

\## Problem Executor

The executor converts abstract decomposition plans into executable code and produces actual solutions with comprehensive safety validation.

\### Core Execution Pipeline

\*\*Main Entry Point:\*\*

```python

executor = ProblemExecutor(

&nbsp; validator=solution\_validator,

&nbsp; semantic\_bridge=semantic\_bridge,

&nbsp; safety\_config={'max\_execution\_time': 3600}

)

\# Execute plan to get solution

outcome = executor.execute\_plan(problem\_graph, plan)

\# Execute with validation

outcome, validation = executor.execute\_and\_validate(problem\_graph, plan)

```

\*\*Safety-First Design:\*\*

The executor implements three-stage safety validation:

1\. \*\*Plan Validation\*\* (before execution)

&nbsp; - Validates plan confidence bounds

&nbsp; - Checks step count limits (max 100)

&nbsp; - Detects unsafe step types (shell\_command, system\_call, etc.)

&nbsp; - Validates problem graph structure

2\. \*\*Principle Validation\*\* (before execution)

&nbsp; - Verifies principle count limits (max 50)

&nbsp; - Checks confidence bounds \[0, 1]

&nbsp; - Ensures all principles have execution logic

3\. \*\*Outcome Validation\*\* (after execution)

&nbsp; - Validates execution time limits

&nbsp; - Checks metric bounds (must be finite)

&nbsp; - Limits error counts (max 100)

&nbsp; - Applies corrections if needed

\*\*Safety Blocks vs Corrections:\*\*

\- \*\*Blocks\*\*: Prevent execution entirely (plan/principle validation failures)

\- \*\*Corrections\*\*: Allow execution but fix unsafe outputs (outcome validation)

\### Execution Strategies

\*\*Strategy Selection:\*\*

The executor automatically determines the best execution approach:

```python

\# Sequential (default)

ExecutionStrategy.SEQUENTIAL

\- Steps execute in order

\- Each step gets previous results

\- Stops on first failure

\# Parallel (for independent steps)

ExecutionStrategy.PARALLEL

\- All steps execute concurrently (simulated)

\- Results aggregated at end

\- Continues even if some fail

\# Iterative (for recursive/cyclic patterns)

ExecutionStrategy.ITERATIVE

\- Steps repeat until convergence

\- Max 10 iterations default

\- Checks convergence after each iteration

```

\### Step-to-Principle Conversion

The executor converts decomposition steps into executable principles:

```python

\# Conversion happens automatically in execute\_plan()

principles = executor.\_convert\_steps\_to\_principles(plan.steps, problem\_graph)

\# Each step type gets specialized solver:

step\_types = {

&nbsp; 'structural\_match': create\_structural\_solver,

&nbsp; 'semantic\_match': create\_semantic\_solver,

&nbsp; 'exact\_match': create\_exact\_solver,

&nbsp; 'synthetic\_bridge': create\_synthetic\_solver,

&nbsp; 'analogical': create\_analogical\_solver,

&nbsp; 'brute\_force': create\_brute\_force\_solver

}

```

\### Solver Types

\*\*1. Structural Solver\*\*

Handles five structural patterns:

```python

\# Hierarchical - processes from root to leaves

outcome = executor.\_solve\_hierarchical(nodes, inputs, problem\_graph)

\# Returns: root\_nodes, processed\_count, level-by-level results

\# Modular - identifies and solves independent modules

outcome = executor.\_solve\_modular(nodes, inputs, problem\_graph)

\# Returns: module\_count, per-module solutions

\# Pipeline - sequential transformation chain

outcome = executor.\_solve\_pipeline(nodes, inputs, problem\_graph)

\# Returns: pipeline stages, final output

\# Recursive - stack-protected recursive solving

outcome = executor.\_solve\_recursive(nodes, inputs, problem\_graph)

\# Returns: base\_case, recursive\_step, final result

\# Protection: Max depth 50, recursion counter

\# Parallel - concurrent execution (simulated)

outcome = executor.\_solve\_parallel(nodes, inputs, problem\_graph)

\# Returns: parallel\_results, aggregated output

```

\*\*2. Semantic Solver\*\*

Applies semantic concepts to nodes:

```python

\# Clusters semantically similar nodes

\# Applies concept-based transformations

\# Aggregates semantic results

result = solver({

&nbsp; 'concept': 'semantic\_cluster\_1',

&nbsp; 'nodes': \['A', 'B', 'C'],

&nbsp; 'similarity': 0.85

})

```

\*\*3. Exact Solver\*\*

Uses known pattern solutions:

```python

\# Pattern library includes:

patterns = {

&nbsp; 'linear': {'type': 'sequential', 'steps': \[...]},

&nbsp; 'tree': {'type': 'hierarchical', 'root': 'start'},

&nbsp; 'cycle': {'type': 'iterative', 'condition': 'convergence'},

&nbsp; 'star': {'type': 'centralized', 'hub': 'center'}

}

\# Applies pattern solution to problem nodes

\# Falls back to heuristic if pattern not found

```

\*\*4. Synthetic Solver\*\*

Generates novel solutions:

```python

\# Template-based generation:

templates = \['linear', 'parallel', 'simple', 'generic']

\# Each template generates appropriate solution structure

result = executor.\_generate\_linear\_solution(inputs, problem\_graph)

\# Returns: {'solution\_type': 'linear', 'steps': \[...]}

```

\*\*5. Analogical Solver\*\*

Uses cross-domain analogies:

```python

\# Analogy database:

analogies = {

&nbsp; 'sorting': {'method': 'merge\_sort', 'complexity': 'O(n log n)'},

&nbsp; 'optimization': {'method': 'gradient\_descent', 'iterations': 100},

&nbsp; 'search': {'method': 'breadth\_first', 'queue': True}

}

\# Maps source domain solution to target

mapped = executor.\_map\_solution(source\_solution, target\_mapping, inputs)

```

\*\*6. Brute Force Solver\*\*

Exhaustive search for last resort:

```python

\# Domain-specific implementations:

\# - Optimization: Grid search over parameter space

\# - Search: Exhaustive item-by-item search

\# - Generic: Fallback brute force approach

result = executor.\_brute\_force\_optimization(content, inputs)

\# Returns: best\_point, best\_value from grid search

```

\### Node Processing

The executor handles three node types:

\*\*Operation Nodes:\*\*

```python

operations = {

&nbsp; 'sum': sum(data),

&nbsp; 'product': np.prod(data),

&nbsp; 'filter': \[x for x in data if x > threshold],

&nbsp; 'map': \[func(x) for x in data] # square, sqrt, etc.

}

result = executor.\_execute\_operation(node\_data, inputs)

```

\*\*Decision Nodes:\*\*

```python

operators = \['>', '<', '>=', '<=', '==']

\# Evaluates condition

decision = executor.\_execute\_decision(node\_data, inputs)

\# Returns: {'decision': True/False, 'condition': {...}}

```

\*\*Transform Nodes:\*\*

```python

transforms = {

&nbsp; 'normalize': (x - min) / (max - min),

&nbsp; 'scale': x \* factor,

&nbsp; 'identity': x

}

result = executor.\_execute\_transform(node\_data, inputs)

```

\### Solution Caching

\*\*Automatic Caching:\*\*

```python

\# Cache key = problem\_signature + plan\_signature

cache\_key = f"{problem\_sig}\_{plan\_sig}"

\# LRU eviction when cache full (max 100 entries)

if len(self.solution\_cache) >= 100:

&nbsp; # Remove oldest entry

&nbsp; oldest = next(iter(self.solution\_cache))

&nbsp; del self.solution\_cache\[oldest]

\# Only successful solutions are cached

if outcome.success:

&nbsp; self.solution\_cache\[cache\_key] = outcome

```

\### Recursion Protection

\*\*Stack Overflow Prevention:\*\*

```python

\# Global recursion depth counter

self.\_recursion\_depth = 0

self.max\_recursion\_depth = 50

\# Protected recursive solving:

def recursive\_solve(data, depth=0):

&nbsp; self.\_recursion\_depth += 1

&nbsp; if self.\_recursion\_depth > self.max\_recursion\_depth:

&nbsp; raise RuntimeError("Maximum recursion depth exceeded")

&nbsp; 

&nbsp; try:

&nbsp; # Recursive logic...

&nbsp; pass

&nbsp; finally:

&nbsp; self.\_recursion\_depth -= 1

```

\### Domain-Specific Configuration

\*\*Domain Configs:\*\*

```python

domain\_configs = {

&nbsp; 'optimization': {

&nbsp; 'default\_solver': custom\_optimizer,

&nbsp; 'timeout': 60,

&nbsp; 'max\_iterations': 100

&nbsp; },

&nbsp; 'classification': {

&nbsp; 'default\_solver': classifier,

&nbsp; 'timeout': 30

&nbsp; },

&nbsp; 'general': {

&nbsp; 'default\_solver': generic\_solver,

&nbsp; 'timeout': 30

&nbsp; }

}

\# Applied automatically based on problem domain

config = executor.domain\_configs.get(problem\_domain, default)

```

\### Execution Outcomes

\*\*ExecutionOutcome Structure:\*\*

```python

outcome = ExecutionOutcome(

&nbsp; success=True,

&nbsp; execution\_time=1.25,

&nbsp; sub\_results=\[

&nbsp; {'principle\_id': 'step\_0', 'success': True, 'result': {...}},

&nbsp; {'principle\_id': 'step\_1', 'success': True, 'result': {...}}

&nbsp; ],

&nbsp; errors=\[],

&nbsp; metrics={

&nbsp; 'plan\_confidence': 0.85,

&nbsp; 'num\_steps': 5,

&nbsp; 'num\_principles': 5,

&nbsp; 'execution\_strategy': 'sequential',

&nbsp; 'safety\_validated': True

&nbsp; },

&nbsp; metadata={

&nbsp; 'from\_cache': False,

&nbsp; 'validation': {...}

&nbsp; }

)

```

\### Solution Validation

\*\*With Validator:\*\*

```python

\# Automatic validation if validator provided

outcome, validation = executor.execute\_and\_validate(problem\_graph, plan)

validation = {

&nbsp; 'validated': True,

&nbsp; 'confidence': 0.7,

&nbsp; 'passed': True,

&nbsp; 'test\_case': {...}

}

```

\### Statistics and Monitoring

\*\*Execution Statistics:\*\*

```python

stats = executor.get\_statistics()

stats = {

&nbsp; 'total\_executions': 150,

&nbsp; 'successful\_executions': 142,

&nbsp; 'success\_rate': 0.947,

&nbsp; 'cached\_solutions': 85,

&nbsp; 'solver\_types': 7,

&nbsp; 'safety': {

&nbsp; 'enabled': True,

&nbsp; 'blocks': {

&nbsp; 'plan': 3,

&nbsp; 'principles': 1

&nbsp; },

&nbsp; 'corrections': {

&nbsp; 'outcome': 2

&nbsp; },

&nbsp; 'total\_blocks': 4,

&nbsp; 'total\_corrections': 2

&nbsp; }

}

```

\### Error Handling

\*\*Graceful Degradation:\*\*

```python

\# Each principle execution is try-catch protected

try:

&nbsp; result = principle.execute(current\_inputs)

&nbsp; sub\_results.append({'success': True, 'result': result})

except Exception as e:

&nbsp; sub\_results.append({'success': False, 'error': str(e)})

&nbsp; # Sequential: stops on failure

&nbsp; # Parallel: continues with other principles

\# Fallback principles created for conversion failures

fallback\_logic = lambda inputs: {'error': str(e), 'fallback': True}

```

\### Memory Management

\*\*Bounded Collections:\*\*

\- Solution cache: 100 entries (FIFO)

\- Execution history: 1000 entries (deque)

\- Safety blocks/corrections: Counter (auto-bounded)

\### Advanced Usage

\*\*Custom Solver Registration:\*\*

```python

\# Add custom solver type

executor.solvers\['custom\_type'] = custom\_solver\_factory

\# Add domain config

executor.domain\_configs\['custom\_domain'] = {

&nbsp; 'default\_solver': custom\_default,

&nbsp; 'timeout': 45,

&nbsp; 'max\_iterations': 50

}

```

\*\*Safety Configuration:\*\*

```python

\# Custom safety limits

safety\_config = {

&nbsp; 'max\_execution\_time': 1800, # 30 minutes

&nbsp; 'max\_steps': 200,

&nbsp; 'max\_principles': 100

}

executor = ProblemExecutor(safety\_config=safety\_config)

```

\*\*Execution Monitoring:\*\*

```python

\# Monitor execution progress

for record in executor.execution\_history:

&nbsp; print(f"Problem: {record\['problem\_signature']\[:8]}")

&nbsp; print(f"Success: {record\['success']}")

&nbsp; print(f"Time: {record\['execution\_time']:.2f}s")

&nbsp; print(f"Safety validated: {record\['safety\_validated']}")

```

\### Integration with Decomposer

\*\*Automatic Integration:\*\*

```python

\# Executor is automatically created by decomposer

decomposer = create\_decomposer(

&nbsp; validator=solution\_validator,

&nbsp; safety\_config=safety\_config

)

\# Executor receives same safety config

\# Used automatically in decompose\_and\_execute()

plan, outcome = decomposer.decompose\_and\_execute(problem)

\# Outcome includes execution details:

\# - Which solver was used

\# - Which nodes were processed

\# - Intermediate results

\# - Safety validation status

```

\### Performance Characteristics

\*\*Typical Execution Times:\*\*

\- Structural solving: 10-100ms

\- Semantic solving: 50-200ms

\- Exact pattern matching: 5-50ms

\- Synthetic generation: 100-500ms

\- Analogical mapping: 50-150ms

\- Brute force: 1-10s (limited search space)

\*\*Optimization Techniques:\*\*

\- Solution caching (100x speedup for repeated problems)

\- Early termination (sequential strategy)

\- Lazy evaluation (only create needed solvers)

\- Bounded iterations (recursion, brute force)

\### Troubleshooting

\*\*Common Issues:\*\*

1\. \*\*Safety Blocks:\*\*

```python

\# Check safety statistics

stats = executor.get\_statistics()

if stats\['safety']\['total\_blocks'] > 0:

&nbsp; print("Blocks by type:", stats\['safety']\['blocks'])

&nbsp; # Adjust safety config if blocks are too aggressive

```

2\. \*\*Execution Failures:\*\*

```python

\# Examine sub\_results for failure points

for result in outcome.sub\_results:

&nbsp; if not result.get('success'):

&nbsp; print(f"Failed: {result\['principle\_id']}")

&nbsp; print(f"Error: {result.get('error')}")

```

3\. \*\*Recursion Depth:\*\*

```python

\# If hitting recursion limit:

executor.max\_recursion\_depth = 100 # Increase from default 50

\# Or redesign problem to avoid deep recursion

```

4\. \*\*Timeout Issues:\*\*

```python

\# Check execution times

if outcome.execution\_time > 60:

&nbsp; # Problem may be too complex

&nbsp; # Consider decomposing further or using simpler strategy

```

\## References

\### Core Components

\- `problem\_decomposer\_core.py`: Main orchestrator with safety validation

\- `decomposition\_strategies.py`: Strategy implementations (6 strategies)

\- `decomposer\_bootstrap.py`: Initialization and factory functions

\- `adaptive\_thresholds.py`: Self-adjusting parameters

\- `fallback\_chain.py`: Strategy ordering and fallback logic

\- `decomposition\_library.py`: Pattern and principle storage

\- \*\*`problem\_executor.py`: Plan execution engine with safety\*\* ← NEW

\### Learning Integration

\- `learning\_integration.py`: Unified learning system coordinator

\- `principle\_learner.py`: Principle extraction and promotion pipeline

