Curiosity Engine

Overview

The Curiosity Engine is a core module of the VULCAN-AGI system, designed to drive curiosity-based learning and exploration. It identifies knowledge gaps in the system's understanding, analyzes dependencies between these gaps, generates targeted experiments to address them, and manages resource budgets for efficient exploration. The engine follows a structured EXAMINE → SELECT → APPLY → REMEMBER (ESA-R) pattern to ensure systematic and adaptive learning.

This module enables AI systems to proactively seek new knowledge, detect anomalies, and iteratively improve through simulated or real-world experiments, all while respecting resource constraints.

Key Features



Knowledge Gap Detection: Analyzes failures, predictions, and patterns to identify gaps (e.g., causal, latent, decomposition).

Dependency Management: Builds and analyzes graphs of gap dependencies to prioritize learning paths and avoid cycles.

Experiment Generation: Designs iterative, adaptive experiments tailored to specific gaps, with safety constraints and pivoting on failures.

Resource Budgeting: Dynamically allocates and recovers budgets based on system load, historical costs, and efficiency metrics.

Orchestration: Coordinates the full curiosity loop, from gap prioritization to knowledge integration.



Architecture and Components

The module is composed of several interconnected Python files, each handling a separated concern:



curiosity\_engine\_core.py: The main orchestrator. Manages the learning cycle, integrates components, and runs experiments in sandboxed environments. Key classes: CuriosityEngine, RegionManager, ExplorationValueEstimator.

gap\_analyzer.py: Identifies and analyzes knowledge gaps from failures and patterns. Supports types like decomposition, causal, latent, and transfer. Key classes: GapAnalyzer, KnowledgeGap, LatentGap, AnomalyAnalyzer.

dependency\_graph.py: Models dependencies between gaps as a directed graph, detects cycles, and calculates adjusted ROI (Return on Investment) for prioritization. Key classes: CycleAwareDependencyGraph, DependencyAnalyzer, ROICalculator.

experiment\_generator.py: Generates and iterates on experiments to fill gaps, with failure analysis and adaptive pivoting. Supports types like causal, transfer, and exploratory. Key classes: ExperimentGenerator, Experiment, IterativeExperimentDesigner.

exploration\_budget.py: Manages dynamic budgets, monitors system resources (CPU, memory, etc.), and calibrates cost estimates from historical data. Key classes: DynamicBudget, ResourceMonitor, CostEstimator.

\_\_init\_\_.py: Empty module initializer (for package structure).



The system uses thread-safe locks for concurrency, caching for performance, and optional libraries like NumPy, SciPy, and scikit-learn for advanced analytics (with fallbacks if unavailable).

Installation and Dependencies

This module is part of the larger VULCAN-AGI project. To use it:



Clone the repository (or integrate into your project).

Install required dependencies:

textpip install numpy scipy scikit-learn networkx psutil



Core: numpy, logging, typing, dataclasses, collections, queue, threading.

Optional: scipy (stats), sklearn (anomaly detection), networkx (graph algorithms), psutil (resource monitoring).

Fallbacks are provided for missing optional libraries.





Import the module:

pythonfrom vulcan.curiosity\_engine import CuriosityEngine





Usage Example

pythonimport logging

from vulcan.curiosity\_engine import CuriosityEngine



\# Set up logging

logging.basicConfig(level=logging.INFO)



\# Initialize the engine with custom parameters

engine = CuriosityEngine(

&nbsp;   base\_exploration\_budget=100.0,

&nbsp;   max\_experiments\_per\_cycle=5,

&nbsp;   anomaly\_threshold=0.2

)



\# Run a learning cycle

cycle\_summary = engine.run\_learning\_cycle(max\_experiments=3)

print("Learning Cycle Summary:", cycle\_summary)



\# Manually record a failure for analysis

engine.record\_failure(

&nbsp;   failure\_type="prediction",

&nbsp;   failure\_data={"domain": "physics", "error": 0.45}

)



\# Get identified gaps

gaps = engine.get\_all\_gaps()

print("Identified Gaps:", \[gap.type for gap in gaps])

Configuration



Budget Parameters: Adjust base\_allocation, recovery\_rate, and adjustment\_rate in DynamicBudget.

Thresholds: Set anomaly\_threshold in GapAnalyzer or min\_frequency for pattern detection.

Graph Limits: Configure max\_nodes and max\_edges in GraphStorage for large-scale graphs.

Caching: TTLs and sizes are tunable in classes like CostEstimator and ROICalculator.



Notes



Thread Safety: All major operations are locked for multi-threaded environments.

Error Handling: Robust fallbacks for missing libraries and edge cases (e.g., empty sets, division by zero).

Performance: Uses caching (lru\_cache, deques with maxlen) and efficient algorithms (e.g., BFS for paths, NetworkX for graph ops).

Extensibility: Add custom gap types, experiment strategies, or resource types by extending enums and classes.

Limitations: No direct internet access or package installation in code execution tools (as per design).



For contributions or issues, refer to the VULCAN-AGI project repository.

