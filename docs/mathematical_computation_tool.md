# Mathematical Computation Tool

## Overview

The MathematicalComputationTool provides state-of-the-art symbolic mathematical computation for VULCAN-AGI. It generates and executes SymPy code to solve calculus, algebra, and symbolic mathematics problems.

**Key Difference from MathematicalVerificationEngine:**
- `MathematicalComputationTool` **generates** solutions (computes results)
- `MathematicalVerificationEngine` **verifies** existing solutions (checks correctness)

## Features

- **Problem Classification**: Automatic detection of problem type (calculus, algebra, etc.)
- **Multi-Strategy Solving**: LLM-based, template-based, or hybrid code generation
- **Safe Execution**: Sandboxed code execution via RestrictedPython
- **Learning**: Learns from successful solutions to improve over time
- **Comprehensive Coverage**: Calculus, algebra, differential equations, linear algebra

## Supported Problem Types

| Type | Examples |
|------|----------|
| **Calculus** | Integration, differentiation, limits, series |
| **Algebra** | Equation solving, factoring, simplification |
| **Differential Equations** | ODEs, systems of ODEs |
| **Linear Algebra** | Matrix operations, eigenvalues, determinants |
| **Statistics** | Probability distributions, expected values |
| **Number Theory** | Primes, factorization, modular arithmetic |

## Installation

The tool requires these dependencies (already in `requirements.txt`):

```
sympy>=1.14.0
RestrictedPython>=8.0
asteval>=1.0.6
```

## Usage

### Basic Usage

```python
from vulcan.reasoning.mathematical_computation import MathematicalComputationTool

# Create tool
tool = MathematicalComputationTool()

# Solve a problem
result = tool.execute("Integrate x^2 with respect to x")

print(f"Success: {result.success}")
print(f"Result: {result.result}")
print(f"Code: {result.code}")
```

**Output:**
```
Success: True
Result: x**3/3
Code:
# Indefinite Integration
x = Symbol('x')
f = x**2
result = integrate(f, x)
```

### With LLM Provider

```python
from vulcan.reasoning.mathematical_computation import MathematicalComputationTool

# Create tool with LLM for complex problems
tool = MathematicalComputationTool(llm=my_llm_provider)

# Solve complex problem
result = tool.execute(
    "Find the derivative of sin(x^2) with respect to x"
)

print(result.result)  # 2*x*cos(x**2)
```

### Via UnifiedReasoner

```python
from vulcan.reasoning.unified_reasoning import UnifiedReasoner
from vulcan.reasoning.reasoning_types import ReasoningType

reasoner = UnifiedReasoner()

result = reasoner.reason(
    input_data="Integrate x^2",
    reasoning_type=ReasoningType.MATHEMATICAL
)

print(result.conclusion)
```

### Using Factory Function

```python
from vulcan.reasoning.mathematical_computation import create_mathematical_computation_tool

tool = create_mathematical_computation_tool(
    llm=None,              # Optional LLM provider
    max_tokens=500,        # Max tokens for LLM generation
    enable_learning=True,  # Learn from successful solutions
    prefer_templates=False # Prefer LLM over templates
)
```

## Supported Operations

### Calculus

```python
# Integration (indefinite)
tool.execute("Integrate x^3 with respect to x")
# Result: x**4/4

# Integration (definite)
tool.execute("Integrate x^2 from 0 to 1")
# Result: 1/3

# Differentiation
tool.execute("Differentiate sin(x)*cos(x)")
# Result: cos(x)**2 - sin(x)**2

# Limits
tool.execute("Find the limit as x approaches 0 of sin(x)/x")
# Result: 1

# Series expansions
tool.execute("Taylor series of e^x around 0")
# Result: 1 + x + x**2/2 + x**3/6 + ...
```

### Algebra

```python
# Equation solving
tool.execute("Solve x^2 - 4 = 0")
# Result: [-2, 2]

# Factoring
tool.execute("Factor x^2 + 2x + 1")
# Result: (x + 1)**2

# Simplification
tool.execute("Simplify (x^2 - 1)/(x - 1)")
# Result: x + 1

# Expansion
tool.execute("Expand (x + 1)^3")
# Result: x**3 + 3*x**2 + 3*x + 1
```

### Linear Algebra

```python
# Determinant
tool.execute("Find the determinant of [[1, 2], [3, 4]]")
# Result: -2

# Inverse
tool.execute("Find the inverse of matrix [[1, 2], [3, 4]]")

# Eigenvalues
tool.execute("Find the eigenvalues of [[1, 2], [2, 1]]")
```

### Differential Equations

```python
# Simple ODE
tool.execute("Solve the differential equation dy/dx = y")
# Result: f(x) = C1*exp(x)
```

## Configuration

```python
tool = MathematicalComputationTool(
    llm=my_llm,              # Optional LLM for code generation
    max_tokens=500,          # Max tokens for LLM generation
    enable_learning=True,    # Enable learning from solutions
    prefer_templates=False   # Prefer templates over LLM
)
```

### Strategy Override

You can override the solving strategy per-request:

```python
from vulcan.reasoning.mathematical_computation import SolutionStrategy

result = tool.execute(
    "Integrate x^2",
    strategy=SolutionStrategy.TEMPLATE  # Force template-based
)
```

Available strategies:
- `SYMBOLIC`: Pure SymPy symbolic computation
- `NUMERIC`: NumPy numerical computation
- `HYBRID`: Combined symbolic + numeric
- `TEMPLATE`: Template-based code generation
- `LLM_GENERATED`: LLM-generated code

## Response Format

The tool returns a `ComputationResult` dataclass:

```python
@dataclass
class ComputationResult:
    success: bool                      # Whether computation succeeded
    code: str                          # Generated SymPy code
    result: Optional[str]              # Computed result
    explanation: str                   # Human-readable explanation
    error: Optional[str]               # Error message if failed
    tool: str                          # Tool name
    problem_type: ProblemType          # Detected problem type
    strategy: SolutionStrategy         # Strategy used
    execution_time: float              # Time in seconds
    metadata: Dict[str, Any]           # Additional metadata
```

### Formatted Response

Use `format_response()` for display:

```python
result = tool.execute("Integrate x^2")
print(tool.format_response(result))
```

**Output:**
```
**Mathematical Computation**

**Code:**
```python
x = Symbol('x')
f = x**2
result = integrate(f, x)
```

**Result:** x**3/3

**Explanation:** The computation was performed using SymPy. The result is: x**3/3

---
*Problem Type: calculus | Strategy: template | Time: 0.015s*
```

## Performance

Typical performance metrics:
- Simple problems (e.g., polynomial integration): 10-50ms
- Medium problems (e.g., trigonometric integration): 50-200ms
- Complex problems (e.g., differential equations): 200-1000ms

## Error Handling

The tool provides graceful degradation:

1. Try LLM code generation (if LLM available)
2. Fall back to template-based generation
3. Return informative error if both fail

```python
result = tool.execute("unsolvable problem")
if not result.success:
    print(f"Error: {result.error}")
    print(f"Explanation: {result.explanation}")
```

## Statistics and Monitoring

```python
stats = tool.get_statistics()
print(f"Cache size: {stats['cache_size']}")
print(f"Safe execution: {stats['safe_execution_available']}")
print(f"LLM available: {stats['llm_available']}")
```

## Security

The tool uses RestrictedPython to sandbox code execution:

- ✅ SymPy functions (integrate, diff, solve, etc.)
- ✅ NumPy array operations
- ✅ Basic Python math
- ❌ File system access
- ❌ Network operations
- ❌ System commands
- ❌ Arbitrary imports

## Troubleshooting

### Issue: Tool not selected by UnifiedReasoner

**Solution:** Ensure you're using `ReasoningType.MATHEMATICAL`:
```python
result = reasoner.reason(
    input_data="...",
    reasoning_type=ReasoningType.MATHEMATICAL
)
```

### Issue: LLM generates incorrect code

**Solution:** Use `prefer_templates=True` for more reliable results:
```python
tool = MathematicalComputationTool(prefer_templates=True)
```

### Issue: Execution timeout

**Solution:** Simplify the problem or increase timeout in safe_execution module.

### Issue: No result returned

**Solution:** Ensure your code assigns to the `result` variable. The tool looks for a variable named `result` in the execution namespace.

## API Reference

### MathematicalComputationTool

```python
class MathematicalComputationTool:
    def __init__(
        self,
        llm=None,
        max_tokens: int = 500,
        enable_learning: bool = True,
        prefer_templates: bool = False
    ): ...
    
    def execute(self, query: str, **kwargs) -> ComputationResult: ...
    def format_response(self, result: ComputationResult) -> str: ...
    def get_statistics(self) -> Dict[str, Any]: ...
```

### Factory Function

```python
def create_mathematical_computation_tool(
    llm=None,
    max_tokens: int = 500,
    enable_learning: bool = True,
    prefer_templates: bool = False
) -> MathematicalComputationTool: ...
```

## See Also

- [Safe Execution Module](../utils/safe_execution.py) - Sandboxed code execution
- [Mathematical Verification](mathematical_verification.py) - Result verification
- [Unified Reasoning](unified_reasoning.py) - Main reasoning orchestrator
