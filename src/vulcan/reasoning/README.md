# VULCAN Reasoning Module
## Comprehensive Technical Documentation

**Version:** 1.0.0  
**Component:** `vulcan.reasoning`  
**Lines of Code:** ~95,000+  
**Status:** Production-Ready

---

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [Architecture](#architecture)
3. [Core Reasoning Engines](#core-reasoning-engines)
4. [Symbolic Logic Foundation](#symbolic-logic-foundation)
5. [Intelligent Tool Selection](#intelligent-tool-selection)
6. [Safety & Validation](#safety--validation)
7. [Installation & Setup](#installation--setup)
8. [Quick Start Guide](#quick-start-guide)
9. [Advanced Usage](#advanced-usage)
10. [Configuration](#configuration)
11. [Performance Optimization](#performance-optimization)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)
14. [Appendix](#appendix)
    - [Recent Bug Fixes](#recent-bug-fixes-and-improvements-january-2026)
    - [Glossary](#glossary)
    - [License](#license)

---

## Executive Overview

### What Is VULCAN Reasoning?

The VULCAN Reasoning Module is a **production-grade, multi-paradigm reasoning system** that combines:

- **18 specialized reasoning engines** for different problem types
- **Formal logic foundation** with theorem proving
- **Intelligent tool selection** using multi-armed bandits
- **Safety validation** at every step
- **Transparent explanations** for all reasoning steps
- **Cryptographic engine** for deterministic hash/encoding computations

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Code** | ~100,000 lines |
| **Reasoning Engines** | 19 specialized types |
| **Theorem Provers** | 6 different methods |
| **Selection Components** | 12 orchestration modules |
| **Safety Layers** | 5 validation systems |
| **Hash Algorithms** | 15 (SHA-2, SHA-3, BLAKE2, etc.) |

### Architecture at a Glance

```
User Query
    ↓
┌─────────────────────────────────────────────┐
│   INTELLIGENT TOOL SELECTION (~12K lines)   │
│                                             │
│ ToolSelector decides which engine(s) to use│
│ • Multi-armed bandits                       │
│ • Cost/utility optimization                 │
│ • Portfolio execution (parallel/sequential) │
│ • Safety governor validation                │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│    REASONING ENGINES (~83K lines)           │
│                                             │
│ 18 Specialized Engines:                    │
│ • Symbolic, Causal, Probabilistic          │
│ • Analogical, Mathematical, Philosophical  │
│ • Multimodal, Language, and more           │
│                                             │
│ Each engine validated for safety           │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  FORMAL FOUNDATION (~7K lines)              │
│                                             │
│ First-Order Logic + Theorem Proving        │
│ • Provably correct reasoning               │
│ • Multiple proof methods                   │
│ • Formal verification                      │
└─────────────────────────────────────────────┘
```

---

## Architecture

### Three-Layer Design

#### Layer 1: Formal Foundation (Symbolic Logic)

The bedrock of the system - **provably correct** reasoning when applicable.

```python
from vulcan.reasoning.symbolic import SymbolicReasoner

reasoner = SymbolicReasoner()
reasoner.add_rule("∀X (Human(X) → Mortal(X))")
reasoner.add_fact("Human(Socrates)")
result = reasoner.query("Mortal(Socrates)")
# result.proven = True (with formal proof tree)
```

**Features:**
- First-Order Logic (FOL) theorem proving
- 6 different proof methods (Tableau, Resolution, Model Elimination, etc.)
- CNF conversion, Skolemization, Unification
- Bayesian networks, CSP solving
- Fuzzy logic, Temporal reasoning

#### Layer 2: Reasoning Engines (18 Specialized Types)

Practical reasoning for real-world problems that don't have formal proofs.

```python
from vulcan.reasoning import apply_reasoning

# Causal reasoning
result = apply_reasoning(
    query="What causes X to affect Y?",
    query_type="causal",
    complexity=0.7
)

# Probabilistic reasoning
result = apply_reasoning(
    query="What's the probability of event E given evidence D?",
    query_type="probabilistic"
)
```

**Available Engines:**
1. **Symbolic** - Language understanding, logical deduction
2. **Causal** - Cause-effect relationships, interventions
3. **Probabilistic** - Uncertainty quantification, Bayesian inference
4. **Analogical** - Reasoning by analogy, transfer learning
5. **Mathematical** - Symbolic math (SymPy integration)
6. **Philosophical** - Ethical reasoning, deontic logic
7. **Multimodal** - Combining text, vision, audio
8. **Abductive** - Best explanation inference
9. **Inductive** - Pattern generalization
10. **Deductive** - Logical consequences
11. **Counterfactual** - "What if" scenarios
12. **Abstract** - Concept abstraction
13. **Ensemble** - Combining multiple models
14. **Hybrid** - Mixed paradigms
15. **Bayesian** - Prior-posterior updating
16. **Contextual** - Context-aware reasoning
17. **Meta** - Reasoning about reasoning
18. **Creative** - Novel solution generation

#### Layer 3: Intelligent Tool Selection

Automatically selects the best reasoning engine(s) for each query.

```python
from vulcan.reasoning.selection import ToolSelector

selector = ToolSelector()
result = selector.select(
    query="Explain the causal relationship between A and B",
    context={
        'domain': 'science',
        'complexity': 0.8,
        'time_budget_ms': 5000
    }
)
# Automatically selects: ['causal', 'symbolic']
# Executes in portfolio, combines results
```

**Selection Features:**
- Multi-armed bandit optimization
- Cost/utility trade-offs
- Bayesian priors from history
- Safety validation
- Semantic understanding
- Circuit breakers for failures

---

## Core Reasoning Engines

### 1. Symbolic Reasoning (Language Understanding)

**Purpose:** Logical deduction, formal reasoning, language understanding

**When to Use:**
- Logical puzzles
- Formal proofs
- Rule-based inference
- Entailment checking

**Example:**
```python
from vulcan.reasoning import language_reasoning

engine = language_reasoning.LanguageReasoningEngine()
result = engine.reason({
    'query': 'If all humans are mortal, and Socrates is human, is Socrates mortal?',
    'context': {}
})

print(result.conclusion)  # "Yes, Socrates is mortal"
print(result.confidence)  # 0.95
print(result.reasoning_chain)  # Step-by-step explanation
```

**Key Features:**
- FOL theorem proving integration
- Natural language parsing
- Proof tree generation
- Confidence scoring

---

### 2. Causal Reasoning

**Purpose:** Understanding cause-effect relationships, planning interventions

**When to Use:**
- "What causes X?"
- "What would happen if we intervene on Y?"
- Counterfactual analysis
- Root cause analysis

**Example:**
```python
from vulcan.reasoning import causal_reasoning

engine = causal_reasoning.CausalReasoningEngine()

# Discover causal structure from data
result = engine.discover_structure(data={
    'temperature': [20, 25, 30, 35],
    'ice_cream_sales': [100, 150, 200, 250],
    'drowning_incidents': [5, 7, 9, 11]
})

# Infer causality
causal_result = engine.infer_causality(
    cause='temperature',
    effect='ice_cream_sales',
    data=result.graph
)

print(causal_result.strength)  # 0.87 (strong causal link)
print(causal_result.mechanism)  # "Temperature → Demand"

# Plan intervention
intervention = engine.plan_intervention(
    target='drowning_incidents',
    desired_change=-5,
    graph=result.graph
)
```

**Key Features:**
- Causal discovery (PC, GES, FCI, LiNGAM)
- Do-calculus for interventions
- Counterfactual inference
- Granger causality tests
- Structural equation models

---

### 3. Probabilistic Reasoning

**Purpose:** Reasoning under uncertainty, Bayesian inference

**When to Use:**
- Probabilistic queries
- Bayesian updating
- Risk assessment
- Uncertainty quantification

**Example:**
```python
from vulcan.reasoning import probabilistic_reasoning

engine = probabilistic_reasoning.ProbabilisticReasoningEngine()

# Bayesian inference
result = engine.bayesian_inference(
    hypothesis='disease',
    evidence={'symptom_1': True, 'symptom_2': False},
    priors={'disease': 0.01},
    likelihoods={
        'symptom_1': {'disease': 0.9, 'no_disease': 0.1},
        'symptom_2': {'disease': 0.3, 'no_disease': 0.05}
    }
)

print(result.posterior['disease'])  # Updated probability
print(result.confidence)  # Confidence in estimate
```

**Key Features:**
- Bayesian networks
- MCMC sampling
- Variational inference
- Probabilistic graphical models
- Uncertainty propagation

---

### 4. Analogical Reasoning

**Purpose:** Reasoning by analogy, transfer learning

**When to Use:**
- "X is like Y, so..."
- Novel problem solving
- Knowledge transfer
- Creative analogies

**Example:**
```python
from vulcan.reasoning import analogical_reasoning

engine = analogical_reasoning.AnalogicalReasoningEngine()

result = engine.reason_by_analogy(
    source={
        'domain': 'atom',
        'structure': {
            'nucleus': 'center',
            'electrons': 'orbit around nucleus'
        }
    },
    target={
        'domain': 'solar_system',
        'structure': {
            'sun': 'center',
            'planets': '?'
        }
    }
)

print(result.mapping)  # {'nucleus': 'sun', 'electrons': 'planets'}
print(result.conclusion)  # "Planets orbit around sun"
```

**Key Features:**
- Structure mapping
- Analogical transfer
- Similarity metrics
- Analogy generation

---

### 5. Mathematical Computation

**Purpose:** Symbolic mathematics, numerical computation

**When to Use:**
- Symbolic math (calculus, algebra)
- Equation solving
- Mathematical proofs
- Numerical analysis

**Example:**
```python
from vulcan.reasoning import mathematical_computation

engine = mathematical_computation.MathematicalEngine()

# Symbolic computation
result = engine.compute("""
x = Symbol('x')
result = integrate(x**2, x)
""")

print(result.result)  # x**3/3
print(result.latex)  # LaTeX representation

# Numerical computation
numeric_result = engine.evaluate(
    expression='sin(pi/2) + cos(0)',
    precision=10
)
```

**Key Features:**
- SymPy integration
- Safe code execution (RestrictedPython)
- Numerical methods
- LaTeX output
- Verification of results

---

### 6. Philosophical Reasoning

**Purpose:** Ethical reasoning, normative logic, deontic reasoning

**When to Use:**
- Ethical dilemmas
- Permissibility questions
- Obligation analysis
- Value reasoning

**Example:**
```python
from vulcan.reasoning import philosophical_reasoning

engine = philosophical_reasoning.PhilosophicalReasoningEngine()

result = engine.reason({
    'query': 'Is it permissible to lie to save a life?',
    'framework': 'consequentialism',
    'context': {
        'situation': 'medical emergency',
        'stakeholders': ['patient', 'family']
    }
})

print(result.conclusion)  # Ethical analysis
print(result.framework_analysis)  # Different ethical perspectives
print(result.confidence)  # Confidence in reasoning
```

**Key Features:**
- Deontic logic
- Ethical frameworks (consequentialism, deontology, virtue ethics)
- Multi-perspective analysis
- Value alignment reasoning

---

### 7. Multimodal Reasoning

**Purpose:** Combining information from multiple modalities

**When to Use:**
- Image + text analysis
- Audio + transcript reasoning
- Cross-modal inference
- Sensor fusion

**Example:**
```python
from vulcan.reasoning import multimodal_reasoning

engine = multimodal_reasoning.MultimodalReasoningEngine()

result = engine.reason({
    'modalities': {
        'vision': image_data,
        'language': text_description,
        'audio': audio_features
    },
    'query': 'What is happening in this scene?',
    'fusion_strategy': 'attention'
})

print(result.conclusion)  # Integrated understanding
print(result.modality_contributions)  # How each modality contributed
```

**Key Features:**
- Cross-modal attention
- Multimodal fusion
- Modality alignment
- Uncertainty across modalities

---

## Symbolic Logic Foundation

### Overview

The symbolic reasoning layer provides **formally provable** reasoning using First-Order Logic (FOL).

### Components

#### 1. Core Data Structures

```python
from vulcan.reasoning.symbolic import Term, Variable, Constant, Function, Literal, Clause

# Create terms
x = Variable('x')
socrates = Constant('socrates')
human = Function('Human', [socrates])

# Create literals
lit = Literal(predicate='Mortal', terms=[x], negated=False)

# Create clauses
clause = Clause([lit])
```

#### 2. Parsing & Formula Building

```python
from vulcan.reasoning.symbolic import FormulaParser

parser = FormulaParser()

# Parse FOL formula
formula = parser.parse("∀X (Human(X) → Mortal(X))")

# Convert to CNF
cnf = formula.to_cnf()

# Skolemize
skolem = formula.skolemize()
```

#### 3. Theorem Provers

**Available Provers:**

| Prover | Method | Best For |
|--------|--------|----------|
| **TableauProver** | Semantic tableau | General proofs |
| **ResolutionProver** | Resolution refutation | Efficient for large KBs |
| **ModelEliminationProver** | Model elimination | Constructive proofs |
| **ConnectionMethodProver** | Connection calculus | Fast proof search |
| **NaturalDeductionProver** | Natural deduction | Human-readable proofs |
| **ParallelProver** | Runs multiple provers | Best overall (default) |

**Example:**
```python
from vulcan.reasoning.symbolic import SymbolicReasoner

# Create reasoner with parallel prover
reasoner = SymbolicReasoner(prover_type='parallel')

# Add knowledge
reasoner.add_rule("∀X (Human(X) → Mortal(X))")
reasoner.add_rule("∀X (Greek(X) → Human(X))")
reasoner.add_fact("Greek(Socrates)")

# Prove theorem
result = reasoner.query("Mortal(Socrates)", timeout=5.0)

print(result['proven'])  # True
print(result['confidence'])  # 0.95
print(result['proof_tree'])  # Full proof structure
print(result['steps'])  # Step-by-step proof
```

#### 4. Constraint Satisfaction

```python
from vulcan.reasoning.symbolic import CSPSolver

solver = CSPSolver()

# Define variables and domains
solver.add_variable('X', domain=[1, 2, 3])
solver.add_variable('Y', domain=[1, 2, 3])
solver.add_variable('Z', domain=[1, 2, 3])

# Add constraints
solver.add_constraint('X', 'Y', lambda x, y: x != y)
solver.add_constraint('Y', 'Z', lambda y, z: y < z)

# Solve
solution = solver.solve()
print(solution)  # {'X': 1, 'Y': 2, 'Z': 3}
```

#### 5. Bayesian Networks

```python
from vulcan.reasoning.symbolic import BayesianNetworkReasoner

bn = BayesianNetworkReasoner()

# Add nodes and edges
bn.add_node('Rain', states=['yes', 'no'])
bn.add_node('Sprinkler', states=['on', 'off'])
bn.add_node('GrassWet', states=['yes', 'no'])

bn.add_edge('Rain', 'GrassWet')
bn.add_edge('Sprinkler', 'GrassWet')

# Define CPDs
bn.set_cpd('Rain', {(): {'yes': 0.2, 'no': 0.8}})
# ... more CPDs

# Inference
result = bn.query('GrassWet', evidence={'Rain': 'yes'})
print(result)  # Probability distribution
```

---

## Intelligent Tool Selection

### Overview

The selection subsystem uses **multi-armed bandits** and **portfolio optimization** to intelligently choose which reasoning engine(s) to use for each query.

### Architecture

```
Query → PreProcessor → ToolSelector → Portfolio → Result
           ↓              ↓              ↓
     Extract formal   Bandit +      Parallel/
     syntax          utility       sequential
                     optimization   execution
```

### Components

#### 1. ToolSelector (Main Orchestrator)

```python
from vulcan.reasoning.selection import ToolSelector, SelectionMode

selector = ToolSelector()

# Automatic selection
result = selector.select(
    query="What causes temperature to affect ice cream sales?",
    context={
        'domain': 'economics',
        'complexity': 0.7,
        'time_budget_ms': 5000,
        'min_confidence': 0.7
    },
    mode=SelectionMode.BALANCED  # FAST, BALANCED, or ACCURATE
)

print(result.selected_tools)  # ['causal', 'probabilistic']
print(result.reasoning_strategy)  # 'causal_reasoning'
print(result.confidence)  # 0.85
```

**Selection Modes:**
- **FAST:** Single best tool, minimal overhead
- **BALANCED:** 2-3 tools, portfolio execution
- **ACCURATE:** Multiple tools, ensemble methods

#### 2. Portfolio Execution

```python
from vulcan.reasoning.selection import PortfolioExecutor, ExecutionStrategy

executor = PortfolioExecutor()

result = executor.execute(
    tools=['causal', 'symbolic', 'probabilistic'],
    query="Complex multi-step problem",
    strategy=ExecutionStrategy.PARALLEL,  # or SEQUENTIAL, PIPELINE
    timeout=10.0
)

print(result.outputs)  # Results from each tool
print(result.ensemble_result)  # Combined result
print(result.confidence)  # Aggregated confidence
```

**Execution Strategies:**
- **PARALLEL:** Run all tools simultaneously, combine results
- **SEQUENTIAL:** Run tools one by one, stop when confident
- **PIPELINE:** Output of one tool feeds into next

#### 3. Cost & Utility Optimization

```python
from vulcan.reasoning.selection import StochasticCostModel, UtilityModel

# Cost estimation
cost_model = StochasticCostModel()
cost_estimate = cost_model.estimate(
    tool='causal',
    query_complexity=0.7,
    data_size=1000
)

print(cost_estimate.time_ms)  # Estimated time
print(cost_estimate.energy_mj)  # Estimated energy
print(cost_estimate.confidence)  # Estimate confidence

# Utility calculation
utility_model = UtilityModel()
utility = utility_model.calculate(
    tool='causal',
    context={
        'accuracy_weight': 0.7,
        'speed_weight': 0.2,
        'cost_weight': 0.1
    }
)
```

#### 4. Bayesian Memory Prior

```python
from vulcan.reasoning.selection import BayesianMemoryPrior

prior = BayesianMemoryPrior()

# Update with outcomes
prior.update(
    tool='causal',
    query_signature='cause-effect-relationship',
    success=True,
    confidence=0.9
)

# Get prior for new query
prior_prob = prior.get_prior(
    query_signature='similar-cause-effect',
    candidate_tools=['causal', 'symbolic', 'probabilistic']
)

print(prior_prob)  # {'causal': 0.7, 'symbolic': 0.2, 'probabilistic': 0.1}
```

#### 5. Safety Governor

```python
from vulcan.reasoning.selection import SafetyGovernor, ToolContract

governor = SafetyGovernor()

# Define tool contract
contract = ToolContract(
    tool_name='causal',
    required_inputs=['data', 'variables'],
    forbidden_inputs=['passwords', 'pii'],
    output_constraints={
        'max_size': 1000000,
        'allowed_types': ['graph', 'dict']
    }
)

# Validate selection
validation = governor.validate_selection(
    tool='causal',
    query="What causes X to affect Y?",
    context={},
    contract=contract
)

if not validation.approved:
    print(validation.veto_reason)  # Why it was vetoed
    print(validation.alternative_tools)  # Suggested alternatives
```

**Safety Features:**
- Tool contract enforcement
- Input/output validation
- Critical violation blocking
- Semantic keyword understanding
- PII detection
- Harmful content filtering

#### 6. Admission Control

```python
from vulcan.reasoning.selection import AdmissionControlIntegration, RequestPriority

admission = AdmissionControlIntegration()

# Check if request can be admitted
admitted = admission.admit_request(
    request_id='req-123',
    priority=RequestPriority.HIGH,
    estimated_cost_ms=5000
)

if admitted:
    # Process request
    result = process_query(...)
    admission.release_request('req-123')
else:
    print("System overloaded, request queued")
```

#### 7. Selection Cache

```python
from vulcan.reasoning.selection import SelectionCache

cache = SelectionCache(
    max_size=1000,
    ttl_seconds=3600
)

# Check cache
cached = cache.get(query_signature='hash-123')
if cached:
    return cached

# Store result
cache.put(
    query_signature='hash-123',
    result=result,
    metadata={'tools': ['causal'], 'confidence': 0.9}
)
```

#### 8. Warm Start Pool

```python
from vulcan.reasoning.selection import WarmStartPool

pool = WarmStartPool(
    tools=['causal', 'symbolic', 'probabilistic'],
    pool_size=3
)

# Get pre-initialized tool
tool = pool.acquire('causal')

# Use tool
result = tool.reason(query)

# Return to pool
pool.release('causal', tool)
```

---

## Safety & Validation

### Multi-Layer Safety Architecture

```
Layer 1: ReasoningExplainer
    ↓ Validates each reasoning step
Layer 2: SafetyGovernor
    ↓ Enforces tool contracts
Layer 3: SafetyValidator (from safety module)
    ↓ System-wide safety checks
Layer 4: EthicalBoundaryMonitor (from meta-reasoning)
    ↓ Ethical constraints
Layer 5: Human Oversight
    ↓ Final approval for critical decisions
```

### ReasoningExplainer

```python
from vulcan.reasoning import ReasoningExplainer

explainer = ReasoningExplainer()

# Explain reasoning step
explanation = explainer.explain_step(step)
print(explanation)  # Human-readable explanation

# Explain entire chain
chain_explanation = explainer.explain_chain(reasoning_chain)

# Validate safety
safety_check = explainer.validate_safety(reasoning_result)
if not safety_check.safe:
    print(safety_check.violations)  # What safety rules were violated
```

### Safety-Aware Reasoning

```python
from vulcan.reasoning import SafetyAwareReasoning

# All reasoning engines support safety validation
result = engine.reason(query, safety_checks=True)

if result.safety_violations:
    print(result.safety_violations)  # Specific violations
    print(result.safety_level)  # Overall safety level
    # Handle violation appropriately
```

---

## Installation & Setup

### Requirements

**Python:** 3.10+

**Required Dependencies:**
```bash
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
```

**Optional Dependencies:**
```bash
# For symbolic reasoning
sympy>=1.12
RestrictedPython>=6.0

# For causal reasoning
networkx>=3.0
pandas>=2.0.0
causallearn>=0.1.3.3
lingam>=1.8.0

# For probabilistic reasoning
statsmodels>=0.14.0

# For embeddings (semantic matching)
sentence-transformers>=2.2.0
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd vulcan-ami

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install optional dependencies (recommended)
pip install -r requirements-reasoning.txt

# Install in development mode
pip install -e .
```

### Verification

```bash
# Run tests
pytest tests/reasoning/

# Check imports
python -c "from vulcan.reasoning import apply_reasoning; print('✓ Reasoning module loaded')"
```

---

## Quick Start Guide

### Example 1: Simple Reasoning

```python
from vulcan.reasoning import apply_reasoning

# Let the system automatically select the best reasoning engine
result = apply_reasoning(
    query="If all birds can fly, and penguins are birds, can penguins fly?",
    query_type="symbolic"
)

print(f"Conclusion: {result.conclusion}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reasoning: {result.explanation}")
```

### Example 2: Causal Analysis

```python
from vulcan.reasoning import apply_reasoning

result = apply_reasoning(
    query="What is the causal relationship between smoking and lung cancer?",
    query_type="causal",
    context={
        'domain': 'medical',
        'data_available': True
    }
)

print(f"Causal strength: {result.metadata['causal_strength']}")
print(f"Mechanism: {result.metadata['mechanism']}")
print(f"Confounders: {result.metadata['confounders']}")
```

### Example 3: Portfolio Reasoning

```python
from vulcan.reasoning import run_portfolio_reasoning

# Use multiple reasoning engines for complex problems
result = run_portfolio_reasoning(
    query="Analyze the economic impact of climate change on agriculture",
    tools=['causal', 'probabilistic', 'analogical'],
    strategy='parallel',
    min_confidence=0.8
)

print(f"Ensemble result: {result.conclusion}")
print(f"Tool contributions: {result.tool_contributions}")
print(f"Overall confidence: {result.confidence:.2f}")
```

### Example 4: Mathematical Computation

```python
from vulcan.reasoning.mathematical_computation import MathematicalEngine

engine = MathematicalEngine()

result = engine.compute("""
from sympy import Symbol, integrate, diff, solve

x = Symbol('x')
# Solve differential equation: y'' + y = 0
result = solve(diff(diff(y(x), x), x) + y(x), y(x))
""")

print(result.result)  # Solution to differential equation
```

### Example 5: With Explicit Tool Selection

```python
from vulcan.reasoning.selection import ToolSelector

selector = ToolSelector()

# Get recommended tools
selection = selector.select(
    query="Complex scientific problem requiring multiple approaches",
    context={
        'complexity': 0.9,
        'accuracy_critical': True,
        'time_budget_ms': 30000
    }
)

print(f"Recommended tools: {selection.selected_tools}")
print(f"Estimated cost: {selection.estimated_cost} ms")

# Execute with selected tools
result = selection.execute()
```

---

## Advanced Usage

### Custom Reasoning Engines

```python
from vulcan.reasoning import BaseReasoningEngine, ReasoningResult

class CustomReasoningEngine(BaseReasoningEngine):
    """Custom reasoning engine for domain-specific logic"""
    
    def __init__(self):
        super().__init__()
        self.reasoning_type = 'custom'
    
    def reason(self, query: dict) -> ReasoningResult:
        """Implement custom reasoning logic"""
        # Your logic here
        
        return ReasoningResult(
            conclusion="Custom conclusion",
            confidence=0.9,
            reasoning_type='custom',
            explanation="How we arrived at this conclusion",
            metadata={}
        )

# Register custom engine
from vulcan.reasoning import register_engine
register_engine('custom', CustomReasoningEngine)

# Use custom engine
result = apply_reasoning(
    query="Domain-specific problem",
    query_type="custom"
)
```

### Advanced Portfolio Configuration

```python
from vulcan.reasoning.selection import PortfolioExecutor, ExecutionStrategy

executor = PortfolioExecutor(
    timeout=30.0,
    max_workers=4,
    fallback_on_failure=True
)

# Pipeline execution (sequential with data flow)
result = executor.execute(
    tools=['causal', 'symbolic', 'probabilistic'],
    query="Multi-stage reasoning problem",
    strategy=ExecutionStrategy.PIPELINE,
    pipeline_config={
        'causal': {
            'output_to': 'symbolic',
            'extract': ['graph_structure']
        },
        'symbolic': {
            'output_to': 'probabilistic',
            'extract': ['logical_constraints']
        }
    }
)
```

### Contextual Bandits for Adaptive Selection

```python
from vulcan.reasoning.contextual_bandit import AdaptiveBanditOrchestrator

bandit = AdaptiveBanditOrchestrator()

# Select tool based on context
selection = bandit.select_tool(
    context={
        'query_type': 'causal',
        'domain': 'economics',
        'complexity': 0.7,
        'user_preference': 'accuracy'
    },
    candidate_tools=['causal', 'symbolic', 'probabilistic']
)

# Execute
result = execute_tool(selection.tool, query)

# Update bandit with outcome
bandit.update(
    context=selection.context,
    tool=selection.tool,
    reward=compute_reward(result),  # Based on accuracy, speed, cost
    outcome=result
)
```

### Integration with World Model

```python
from vulcan.reasoning import apply_reasoning
from vulcan.world_model import WorldModel

# Initialize world model
world_model = WorldModel()

# Use reasoning to discover causal relationships
causal_result = apply_reasoning(
    query="Discover causal structure in this data",
    query_type="causal",
    data=observations
)

# Integrate into world model
world_model.update_causal_structure(
    causal_result.metadata['causal_graph']
)

# Use world model for predictions
prediction = world_model.predict(
    target='variable_y',
    evidence={'variable_x': 10}
)
```

---

## Configuration

### Environment Variables

```bash
# Tool selection
export VULCAN_SELECTION_MODE=balanced  # fast, balanced, accurate
export VULCAN_DEFAULT_TIMEOUT=10000  # milliseconds
export VULCAN_MAX_WORKERS=4

# Cost optimization
export VULCAN_TIME_BUDGET_MS=5000
export VULCAN_ENERGY_BUDGET_MJ=1000

# Safety
export VULCAN_SAFETY_LEVEL=strict  # permissive, normal, strict
export VULCAN_ENABLE_SAFETY_GOVERNOR=true

# Caching
export VULCAN_CACHE_SIZE=1000
export VULCAN_CACHE_TTL=3600

# Logging
export VULCAN_LOG_LEVEL=INFO
export VULCAN_LOG_REASONING_STEPS=true
```

### Configuration Files

**config/reasoning.yaml:**
```yaml
reasoning:
  default_mode: balanced
  timeout_ms: 10000
  
  engines:
    symbolic:
      enabled: true
      prover: parallel
      timeout: 5.0
    
    causal:
      enabled: true
      discovery_method: pc
      max_lag: 5
    
    probabilistic:
      enabled: true
      inference_method: variational
      num_samples: 1000

selection:
  bandit:
    algorithm: thompson_sampling
    exploration_rate: 0.1
  
  portfolio:
    max_tools: 3
    default_strategy: parallel
  
  safety:
    level: strict
    enable_contracts: true
    enable_validation: true

performance:
  cache_enabled: true
  warm_pool_size: 3
  max_concurrent_requests: 10
```

### Loading Configuration

```python
from vulcan.reasoning import load_config

config = load_config('config/reasoning.yaml')

# Use configuration
selector = ToolSelector(config=config.selection)
```

---

## Performance Optimization

### 1. Caching

```python
from vulcan.reasoning.selection import SelectionCache

# Enable aggressive caching
cache = SelectionCache(
    max_size=10000,
    ttl_seconds=7200,
    enable_semantic_matching=True
)

# Cache hit rates
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Avg retrieval time: {stats.avg_retrieval_ms:.1f}ms")
```

### 2. Warm Start Pools

```python
from vulcan.reasoning.selection import WarmStartPool

# Pre-initialize expensive engines
pool = WarmStartPool(
    tools=['causal', 'symbolic', 'probabilistic'],
    pool_size=5,
    initialization_config={
        'causal': {'preload_algorithms': True},
        'symbolic': {'precompile_rules': True}
    }
)
```

### 3. Parallel Execution

```python
from vulcan.reasoning.selection import PortfolioExecutor

# Maximize parallelism
executor = PortfolioExecutor(
    max_workers=8,  # Use more workers
    enable_async=True,
    timeout=30.0
)

result = executor.execute(
    tools=['causal', 'symbolic', 'probabilistic', 'analogical'],
    query=query,
    strategy=ExecutionStrategy.PARALLEL
)
```

### 4. Fast Path Optimization

```python
from vulcan.reasoning import apply_reasoning

# For simple queries, use fast mode
result = apply_reasoning(
    query="Simple factual question",
    mode='fast',  # Single tool, minimal overhead
    max_complexity=0.3
)
```

### 5. Batch Processing

```python
from vulcan.reasoning import batch_reasoning

# Process multiple queries efficiently
queries = [
    "Query 1: Causal question",
    "Query 2: Logical puzzle",
    "Query 3: Probabilistic inference"
]

results = batch_reasoning(
    queries=queries,
    batch_size=10,
    parallel=True
)
```

### Performance Benchmarks

| Operation | Latency (p50) | Latency (p95) | Throughput |
|-----------|---------------|---------------|------------|
| Simple query (fast mode) | 50ms | 150ms | 200 qps |
| Medium query (balanced) | 500ms | 2s | 20 qps |
| Complex query (accurate) | 5s | 15s | 2 qps |
| Batch (10 queries) | 2s | 8s | 5 batches/s |

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:**
```
ImportError: cannot import name 'apply_reasoning'
```

**Solution:**
```bash
# Verify installation
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall if needed
pip uninstall vulcan-ami
pip install -e .
```

#### 2. Tool Selection Defaults to 'general'

**Problem:** ToolSelector always selects 'general' instead of specialized tools.

**Solution:**
```python
# Enable semantic matcher
from vulcan.reasoning.selection import ToolSelector

selector = ToolSelector(enable_semantic_matching=True)

# Or install sentence-transformers
pip install sentence-transformers
```

#### 3. Low Confidence Results

**Problem:** Reasoning results have unexpectedly low confidence.

**Solution:**
```python
# Check if engines are properly trained
from vulcan.reasoning import get_reasoning_statistics

stats = get_reasoning_statistics()
print(stats['tool_success_rates'])  # Should be > 0.5

# Use multiple tools for higher confidence
result = run_portfolio_reasoning(
    query=query,
    tools=['causal', 'symbolic', 'probabilistic'],
    strategy='parallel'
)
```

#### 4. Safety Violations

**Problem:** Queries are being blocked by safety governor.

**Solution:**
```python
# Check violation reason
result = selector.select(query)
if not result.approved:
    print(result.veto_reason)
    print(result.alternative_tools)

# Adjust safety level if appropriate
from vulcan.reasoning.selection import SafetyGovernor, SafetyLevel

governor = SafetyGovernor(safety_level=SafetyLevel.NORMAL)
```

#### 5. Performance Issues

**Problem:** Queries are taking too long.

**Solution:**
```python
# Use fast mode for simple queries
result = apply_reasoning(query, mode='fast')

# Enable caching
from vulcan.reasoning.selection import SelectionCache
cache = SelectionCache(max_size=1000)

# Reduce time budget
result = selector.select(
    query=query,
    context={'time_budget_ms': 2000}
)

# Check system load
from vulcan.reasoning.selection import get_system_metrics
metrics = get_system_metrics()
print(metrics)
```

---

## API Reference

### Core Functions

#### apply_reasoning()

```python
def apply_reasoning(
    query: str,
    query_type: Optional[str] = None,
    context: Optional[Dict] = None,
    mode: str = 'balanced',
    complexity: Optional[float] = None,
    **kwargs
) -> ReasoningResult:
    """
    Main entry point for reasoning.
    
    Args:
        query: The query string
        query_type: Optional reasoning type hint
        context: Additional context
        mode: Selection mode ('fast', 'balanced', 'accurate')
        complexity: Query complexity (0-1)
        
    Returns:
        ReasoningResult with conclusion, confidence, explanation
    """
```

#### run_portfolio_reasoning()

```python
def run_portfolio_reasoning(
    query: str,
    tools: List[str],
    strategy: str = 'parallel',
    min_confidence: float = 0.5,
    timeout: float = 30.0
) -> PortfolioResult:
    """
    Execute multiple reasoning tools and combine results.
    
    Args:
        query: The query string
        tools: List of tool names to use
        strategy: Execution strategy
        min_confidence: Minimum acceptable confidence
        timeout: Maximum execution time
        
    Returns:
        PortfolioResult with ensemble result
    """
```

### Classes

#### ToolSelector

```python
class ToolSelector:
    def __init__(
        self,
        config: Optional[Dict] = None,
        enable_semantic_matching: bool = True,
        enable_safety: bool = True
    ):
        """Initialize tool selector"""
        
    def select(
        self,
        query: str,
        context: Optional[Dict] = None,
        mode: SelectionMode = SelectionMode.BALANCED
    ) -> SelectionResult:
        """Select best tools for query"""
```

#### ReasoningResult

```python
@dataclass
class ReasoningResult:
    conclusion: str
    confidence: float
    reasoning_type: str
    explanation: str
    reasoning_chain: List[ReasoningStep]
    metadata: Dict[str, Any]
    safety_checks: List[SafetyCheck]
```

---

## Appendix


### Recent Bug Fixes and Improvements (January 2026)

#### BUG #5: Symbolic Parser Cannot Handle Natural Language

**Problem:** The parser expected formal logic notation but received natural language.

**Solution:** Created `nl_converter.py` with pattern-based NL to logic conversion.

```python
# Before (failed):
"Every engineer reviewed a document"
# Parse error: Unexpected token 'Every'

# After (works):
from vulcan.reasoning.symbolic import SymbolicReasoner
reasoner = SymbolicReasoner()
result = reasoner.reason("Every engineer reviewed a document")
# Converts to: ∀e ∃d Review(e, d)
```

#### BUG #13: Probabilistic Non-Determinism

**Problem:** Same query produced different results across sessions.

**Solution:** Added `reset_state()` method with deterministic seeding.

```python
from vulcan.reasoning import ProbabilisticReasoner

reasoner = ProbabilisticReasoner()
reasoner.reset_state(seed=42)  # Deterministic results
result = reasoner.reason(query)
```

#### BUG #14: No Cryptographic Engine

**Problem:** System fell back to OpenAI for hash computations, which hallucinated incorrect values.

**Solution:** Created `cryptographic_engine.py` with deterministic hash/encoding support.

```python
from vulcan.reasoning import compute_crypto

# SHA-256
result = compute_crypto("Calculate SHA-256 of 'Hello, World!'")
# result['result'] = 'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'

# Supported algorithms:
# - SHA-2: SHA-256, SHA-1, SHA-512, SHA-384, SHA-224
# - SHA-3: SHA3-256, SHA3-512, SHA3-384, SHA3-224
# - BLAKE2: BLAKE2b, BLAKE2s
# - Legacy: MD5, RIPEMD-160
# - Encoding: Base64, Hex, URL/Percent
# - HMAC: HMAC-SHA256, HMAC-SHA512
# - Checksums: CRC32
```

#### BUG #15: Learning Rewards Wrong Answers

**Problem:** System learned from incorrect but confident answers.

**Solution:** Penalize unverified high-confidence results in learning.

```python
# Constants in tool_selector.py:
UNVERIFIED_QUALITY_PENALTY = 0.7  # Reduce to 70% of claimed confidence
FALLBACK_QUALITY_PENALTY = 0.3   # Reduce fallback rewards to 30%
```

#### BUG #7: Fallback Reporting Lies

**Problem:** System reported success when using fallback, with misleading confidence.

**Solution:** Fallback results now receive heavily reduced rewards and are tracked in metadata.

#### BUG #9: Router Over-Weights Keywords

**Problem:** "Write a sonnet about quantum" routed to Math due to "quantum" keyword.

**Solution:** Task type detection (verbs like "write", "compose") now takes priority over domain keywords.

```python
# Now correctly routes:
# "Write a sonnet about quantum" → general (creative task)
# "Calculate quantum entanglement probability" → symbolic/mathematical
```

#### BUG #10: Ethical Override Blocks Math

**Problem:** "Optimize welfare function" routed to Philosophy due to "welfare" keyword.

**Solution:** Mathematical task verbs (optimize, maximize, minimize) override ethical keywords.

```python
# Now correctly routes:
# "Optimize welfare function" → symbolic (math task)
# "Discuss welfare ethics" → philosophical
```


### Glossary

- **Reasoning Engine:** Specialized module for a particular type of reasoning (causal, symbolic, etc.)
- **Tool Selection:** Process of choosing which reasoning engine(s) to use
- **Portfolio Execution:** Running multiple engines and combining results
- **Multi-Armed Bandit:** Algorithm for balancing exploration/exploitation in tool selection
- **Safety Governor:** Component that enforces safety constraints
- **Contextual Bandit:** Bandit algorithm that considers context when selecting
- **Warm Start Pool:** Pre-initialized instances for fast access
- **Circuit Breaker:** Failure handling mechanism
- **Cryptographic Engine:** Deterministic hash/encoding computation engine (BUG #14 fix)
- **NL Converter:** Natural Language to Logic converter (BUG #5 fix)

### License

See `LICENSE.txt` for complete terms.

---

**Maintainers:** Brian Anderson  
**Last Updated:** January 2026  
**Version:** 1.1.0

---

END OF DOCUMENTATION
