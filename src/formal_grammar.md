markdown# Graphix IR: AI-Native Language Formal Grammar Specification

**Version**: 2.0.0  
**Last Updated**: 2025-10-01  
**Status**: Production-Ready

This document specifies the formal grammar for Graphix IR's JSON-based computational graphs, designed for AI agents to build, optimize, secure, and explain autonomously. Version 2.0.0 fixes grammar inconsistencies, adds validation constraints, enhances error handling, and introduces AI-native features for autonomous operation.

---

## Formal Grammar (EBNF, v2.0.0)
```ebnf
<graph> ::= "{"
    "grammar_version" : <version_string>,
    "id" : <identifier>,
    "type" : "Graph",
    "nodes" : [ <node> ("," <node>)* ],
    "edges" : [ <edge> ("," <edge>)* ]
    [ "," "metadata" : <graph_metadata> ]
    [ "," "ontology_version" : <version_string> ]
    [ "," "custom_types" : <custom_type_registry> ]
    [ "," "constraints" : <constraint_set> ]
    [ "," "capabilities" : <capability_set> ]
"}"

<node> ::= "{"
    "id" : <identifier>,
    "type" : <nodetype>,
    "params" : <params>
    [ "," "metadata" : <node_metadata> ]
    [ "," "status" : <status> ]
    [ "," "ports" : <port_spec> ]
    [ "," "resource_requirements" : <resource_spec> ]
"}"

<nodetype> ::= 
    "InputNode" | "OutputNode" | "ContractNode" | "PhotonicMVMNode" |
    "MetaGraphNode" | "GenerativeNode" | "FusedPhotonicNode" |
    "ShardedComputationNode" | "OperatorNode" | "ExplainabilityNode" |
    "FeedbackNode" | "DriftDetectorNode" | "EvolutionNode" |
    <custom_nodetype>

<custom_nodetype> ::= <string>
    // CONSTRAINT: Must match ^[A-Z][a-zA-Z0-9]*Node$ pattern
    // CONSTRAINT: Must be registered in custom_types registry

<edge> ::= "{"
    "id" : <identifier>,
    "source" : <portref>,
    "target" : <portref>,
    "type" : <edgetype>
    [ "," "metadata" : <edge_metadata> ]
    [ "," "weight" : <number> ]
    [ "," "conditional" : <condition> ]
"}"

<portref> ::= "{"
    "node" : <identifier>,
    "port" : <port_name>
"}"

<port_name> ::= <string>
    // CONSTRAINT: Must match ^[a-z_][a-z0-9_]*$ pattern

<edgetype> ::= "data" | "control" | "contract" | "stream" | "feedback"

<params> ::= "{"
    ( <param_entry> ("," <param_entry>)* )?
"}"

<param_entry> ::=
    "value" : <value> |
    "subgraph" : <graph> |
    "meta_graph" : <graph> |
    "contract" : <contract> |
    "photonic_params" : <photonic_params> |
    "rlhf_params" : <rlhf_params> |
    "explanation_params" : <explanation_params> |
    "drift_params" : <drift_params> |
    "evolution_params" : <evolution_params> |
    <custom_param_key> : <value>

<contract> ::= "{"
    "latency_ms" : <positive_number>,
    "accuracy" : <probability>,
    "ethical_label" : <ethical_label>
    [ "," "energy_budget_nj" : <positive_number> ]
    [ "," "privacy_level" : <privacy_level> ]
    [ "," "auditability" : <auditability_level> ]
"}"

<ethical_label> ::= "safe" | "risky" | "unknown" | "requires_review"

<privacy_level> ::= "public" | "private" | "sensitive" | "pii"

<auditability_level> ::= "full" | "partial" | "minimal" | "none"

<photonic_params> ::= "{"
    "noise_std" : <non_negative_number>,
    "multiplexing" : <multiplexing_mode>,
    "compression" : <compression_mode>,
    "bandwidth_ghz" : <positive_number>,
    "latency_ps" : <positive_number>
    [ "," "device_type" : <string> ]
    [ "," "wavelength_nm" : <positive_number> ]
"}"

<multiplexing_mode> ::= 
    "microwave-lightwave" | "wavelength" | "space-time-wavelength"

<compression_mode> ::= 
    "ITU-F.748-quantized" | "ITU-F.748-sparse" | "ITU-F.748" | "none"

<rlhf_params> ::= "{"
    "temperature" : <positive_number>,
    "max_tokens" : <positive_integer>,
    "rlhf_train" : <boolean>
    [ "," "feedback_weight" : <number> ]
    [ "," "reward_model" : <string> ]
"}"

<explanation_params> ::= "{"
    "method" : <explanation_method>,
    [ "baseline" : <value> ]
    [ "," "coverage_threshold" : <probability> ]
    [ "," "require_validation" : <boolean> ]
"}"

<explanation_method> ::= 
    "integrated_gradients" | "saliency" | "shap" | 
    "lime" | "attention" | "counterfactual"

<drift_params> ::= "{"
    "threshold" : <probability>,
    "history_size" : <positive_integer>,
    "realignment_method" : <realignment_method>
    [ "," "window_size" : <positive_integer> ]
"}"

<realignment_method> ::= "center" | "pca" | "procrustes"

<evolution_params> ::= "{"
    "population_size" : <positive_integer>,
    "mutation_rate" : <probability>,
    "crossover_rate" : <probability>,
    "max_generations" : <positive_integer>
    [ "," "fitness_function" : <string> ]
    [ "," "diversity_threshold" : <probability> ]
"}"

<graph_metadata> ::= "{"
    ( <metadata_field> ("," <metadata_field>)* )?
"}"

<node_metadata> ::= "{"
    ( <metadata_field> ("," <metadata_field>)* )?
"}"

<edge_metadata> ::= "{"
    ( <metadata_field> ("," <metadata_field>)* )?
"}"

<metadata_field> ::=
    "explanation" : <string> |
    "provenance" : <provenance_info> |
    "ethical_score" : <probability> |
    "audit_trail" : [ <audit_event> ("," <audit_event>)* ] |
    "diagnostic_info" : <string> |
    "recursion_depth" : <non_negative_integer> |
    "privacy_level" : <privacy_level> |
    "security_tags" : [ <string> ("," <string>)* ] |
    "param_types" : <type_schema> |
    "created_at" : <timestamp> |
    "modified_at" : <timestamp> |
    "version" : <version_string> |
    "dependencies" : [ <identifier> ("," <identifier>)* ] |
    "capabilities_required" : [ <string> ("," <string>)* ]

<provenance_info> ::= "{"
    "source" : <string>,
    "creator" : <string>,
    [ "," "creation_method" : <string> ]
    [ "," "parent_graph" : <identifier> ]
    [ "," "confidence" : <probability> ]
"}"

<audit_event> ::= "{"
    "user" : <string>,
    "action" : <string>,
    "timestamp" : <timestamp>
    [ "," "context" : <value> ]
"}"

<constraint_set> ::= "{"
    [ "max_nodes" : <positive_integer> ]
    [ "," "max_edges" : <positive_integer> ]
    [ "," "max_recursion_depth" : <positive_integer> ]
    [ "," "max_execution_time_ms" : <positive_integer> ]
    [ "," "max_memory_mb" : <positive_integer> ]
    [ "," "allowed_node_types" : [ <nodetype> ("," <nodetype>)* ] ]
    [ "," "required_capabilities" : [ <string> ("," <string>)* ] ]
"}"

<capability_set> ::= "{"
    ( <string> : <boolean> ("," <string> : <boolean>)* )?
"}"
// Examples: {"photonic": true, "quantum": false, "rlhf": true}

<custom_type_registry> ::= "{"
    ( <custom_nodetype> : <type_definition> ("," <custom_nodetype> : <type_definition>)* )?
"}"

<type_definition> ::= "{"
    "base_type" : <nodetype>,
    "param_schema" : <type_schema>,
    [ "," "description" : <string> ]
    [ "," "version" : <version_string> ]
"}"

<type_schema> ::= "{" /* JSON Schema subset */ "}"

<port_spec> ::= "{"
    "inputs" : [ <port_definition> ("," <port_definition>)* ],
    "outputs" : [ <port_definition> ("," <port_definition>)* ]
"}"

<port_definition> ::= "{"
    "name" : <port_name>,
    "type" : <data_type>
    [ "," "required" : <boolean> ]
    [ "," "default" : <value> ]
"}"

<data_type> ::= "tensor" | "scalar" | "string" | "graph" | "any" | <custom_type>

<resource_spec> ::= "{"
    [ "cpu_cores" : <positive_number> ]
    [ "," "memory_mb" : <positive_number> ]
    [ "," "gpu_count" : <non_negative_integer> ]
    [ "," "photonic_units" : <non_negative_integer> ]
    [ "," "energy_budget_nj" : <positive_number> ]
"}"

<condition> ::= "{"
    "type" : <condition_type>,
    "expression" : <string>
    [ "," "parameters" : <value> ]
"}"

<condition_type> ::= "threshold" | "comparison" | "custom"

<status> ::= "active" | "deprecated" | "experimental" | "disabled"

<error_response> ::= "{"
    "error_code" : <error_code>,
    "message" : <string>
    [ "," "details" : <value> ]
    [ "," "timestamp" : <timestamp> ]
    [ "," "node_id" : <identifier> ]
    [ "," "recoverable" : <boolean> ]
    [ "," "suggested_action" : <string> ]
"}"

<error_code> ::= 
    "AI_INVALID_REQUEST" | "AI_PHOTONIC_NOISE" | 
    "AI_COMPRESSION_INVALID" | "AI_RECURSION_DEPTH_EXCEEDED" |
    "AI_CONTRACT_VIOLATION" | "AI_RESOURCE_EXHAUSTED" |
    "AI_VALIDATION_FAILED" | "AI_CAPABILITY_MISSING" |
    "AI_EXECUTION_TIMEOUT" | "AI_SECURITY_VIOLATION" |
    <custom_error_code>

// Primitive types with constraints
<identifier> ::= <string>
    // CONSTRAINT: Must match ^[a-zA-Z][a-zA-Z0-9_-]*$ pattern
    // CONSTRAINT: Length 1-256 characters

<version_string> ::= <string>
    // CONSTRAINT: Must match semantic versioning ^[0-9]+\.[0-9]+\.[0-9]+$ pattern

<timestamp> ::= <string>
    // CONSTRAINT: Must be ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ)

<string> ::= '"' <unicode_char>* '"'
    // CONSTRAINT: Length 0-10000 characters
    // CONSTRAINT: Must be valid UTF-8

<unicode_char> ::= /* Any valid Unicode character, escaped as needed */

<number> ::= <integer> | <float>

<integer> ::= ["-"] <digit>+
    // CONSTRAINT: Range -9007199254740991 to 9007199254740991 (safe integer)

<positive_integer> ::= <digit>+
    // CONSTRAINT: Range 1 to 9007199254740991

<non_negative_integer> ::= <digit>+
    // CONSTRAINT: Range 0 to 9007199254740991

<float> ::= ["-"] <digit>+ "." <digit>+ [("e"|"E") ["+"|"-"] <digit>+]

<positive_number> ::= <digit>+ ["." <digit>+]
    // CONSTRAINT: > 0

<non_negative_number> ::= <digit>+ ["." <digit>+]
    // CONSTRAINT: >= 0

<probability> ::= <number>
    // CONSTRAINT: Range 0.0 to 1.0

<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

<boolean> ::= "true" | "false"

<value> ::= <string> | <number> | <boolean> | <list> | <dict> | "null"

<list> ::= "[" [ <value> ("," <value>)* ] "]"
    // CONSTRAINT: Maximum 10000 elements

<dict> ::= "{" [ <string> ":" <value> ("," <string> ":" <value>)* ] "}"
    // CONSTRAINT: Maximum 1000 key-value pairs

Validation Rules
Structural Constraints

Graph Size Limits (configurable via constraints):

Default max nodes: 1000
Default max edges: 5000
Default max recursion depth: 10
Maximum graph nesting: 5 levels


Node Validation:

Node IDs must be unique within a graph
Custom node types must be registered in custom_types
All referenced ports must exist in port specifications


Edge Validation:

Edge IDs must be unique within a graph
Source and target nodes must exist
Source and target ports must be compatible types
No self-loops unless explicitly allowed
Cycles allowed only for stream/feedback edges


Metadata Validation:

recursion_depth must be enforced: depth < max_recursion_depth
Timestamps must be valid ISO 8601
Audit trail must be append-only


Parameter Validation:

Contract accuracy must be in [0, 1]
Photonic noise_std should be validated: typically < 0.1
Compression modes must be from allowed list
RLHF temperature typically in (0, 2]



Runtime Validation Example
pythonasync def validate_and_execute_node(self, node: Dict, context: Dict, graph: Dict) -> Any:
    """Validate node before execution with comprehensive checks."""
    
    # Check recursion depth
    depth = node.get("metadata", {}).get("recursion_depth", 0)
    max_depth = graph.get("constraints", {}).get("max_recursion_depth", 10)
    
    if depth >= max_depth:
        return {
            "error_code": "AI_RECURSION_DEPTH_EXCEEDED",
            "message": f"Recursion depth {depth} >= {max_depth}",
            "details": {"node_id": node["id"], "depth": depth},
            "recoverable": False,
            "suggested_action": "Reduce graph nesting or increase max_recursion_depth"
        }
    
    # Validate node type
    node_type = node["type"]
    allowed_types = graph.get("constraints", {}).get("allowed_node_types")
    
    if allowed_types and node_type not in allowed_types:
        return {
            "error_code": "AI_VALIDATION_FAILED",
            "message": f"Node type {node_type} not in allowed types",
            "recoverable": False
        }
    
    # Validate capabilities
    required_caps = node.get("metadata", {}).get("capabilities_required", [])
    available_caps = graph.get("capabilities", {})
    
    for cap in required_caps:
        if not available_caps.get(cap, False):
            return {
                "error_code": "AI_CAPABILITY_MISSING",
                "message": f"Required capability '{cap}' not available",
                "suggested_action": f"Enable capability '{cap}' in graph configuration"
            }
    
    # Execute node
    return await self._execute_node_impl(node, context, graph)

Version Compatibility
Semantic Versioning
Graphix IR follows semantic versioning (MAJOR.MINOR.PATCH):

MAJOR: Incompatible grammar changes
MINOR: Backward-compatible new features
PATCH: Backward-compatible bug fixes

Compatibility Matrix
Graph VersionRuntime VersionCompatible?Notes2.0.x2.0.x✓Full compatibility2.0.x2.1.x✓Forward compatible2.1.x2.0.xPartialNew features unavailable2.x.x1.x.x✗Major version mismatch1.3.x2.x.xDeprecatedLegacy support only
Migration Guide
When upgrading from v1.3.1 to v2.0.0:

Update grammar_version field
Convert metadata from OR syntax to comma-separated fields
Add constraints section if using custom limits
Update error handling to use enhanced error_response
Register custom node types in custom_types


Enhanced Examples
Complete Graph with AI-Native Features
json{
  "grammar_version": "2.0.0",
  "id": "ai_optimization_pipeline",
  "type": "Graph",
  "nodes": [
    {
      "id": "input_1",
      "type": "InputNode",
      "params": {"value": "placeholder"},
      "ports": {
        "inputs": [],
        "outputs": [{"name": "data", "type": "tensor", "required": true}]
      }
    },
    {
      "id": "drift_detector_1",
      "type": "DriftDetectorNode",
      "params": {
        "drift_params": {
          "threshold": 0.15,
          "history_size": 10,
          "realignment_method": "procrustes"
        }
      },
      "metadata": {
        "explanation": "Monitors embedding drift",
        "capabilities_required": ["drift_detection"]
      },
      "resource_requirements": {
        "cpu_cores": 2,
        "memory_mb": 512
      }
    },
    {
      "id": "explainability_1",
      "type": "ExplainabilityNode",
      "params": {
        "explanation_params": {
          "method": "integrated_gradients",
          "coverage_threshold": 0.8,
          "require_validation": true
        }
      },
      "metadata": {
        "ethical_score": 0.95,
        "privacy_level": "private"
      }
    },
    {
      "id": "photonic_1",
      "type": "PhotonicMVMNode",
      "params": {
        "photonic_params": {
          "noise_std": 0.01,
          "multiplexing": "wavelength",
          "compression": "ITU-F.748-quantized",
          "bandwidth_ghz": 100,
          "latency_ps": 50,
          "device_type": "AIM_SOI_modulator"
        }
      },
      "metadata": {
        "diagnostic_info": "AIM photonic accelerator",
        "capabilities_required": ["photonic_hardware"]
      }
    },
    {
      "id": "feedback_1",
      "type": "FeedbackNode",
      "params": {
        "rlhf_params": {
          "temperature": 0.7,
          "max_tokens": 1000,
          "rlhf_train": true,
          "feedback_weight": 0.5
        }
      }
    },
    {
      "id": "output_1",
      "type": "OutputNode",
      "params": {"value": "result"}
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": {"node": "input_1", "port": "data"},
      "target": {"node": "drift_detector_1", "port": "input"},
      "type": "data"
    },
    {
      "id": "edge_2",
      "source": {"node": "drift_detector_1", "port": "output"},
      "target": {"node": "explainability_1", "port": "input"},
      "type": "data"
    },
    {
      "id": "edge_3",
      "source": {"node": "explainability_1", "port": "output"},
      "target": {"node": "photonic_1", "port": "input"},
      "type": "data"
    },
    {
      "id": "edge_4",
      "source": {"node": "photonic_1", "port": "output"},
      "target": {"node": "output_1", "port": "input"},
      "type": "data"
    },
    {
      "id": "edge_feedback",
      "source": {"node": "output_1", "port": "result"},
      "target": {"node": "feedback_1", "port": "input"},
      "type": "feedback"
    }
  ],
  "metadata": {
    "explanation": "AI-native optimization pipeline with drift detection and RLHF",
    "provenance": {
      "source": "autonomous_generator_v2",
      "creator": "ai_agent_001",
      "creation_method": "evolutionary_synthesis",
      "confidence": 0.92
    },
    "ethical_score": 0.88,
    "created_at": "2025-10-01T12:00:00.000Z",
    "version": "1.0.0"
  },
  "constraints": {
    "max_nodes": 100,
    "max_edges": 500,
    "max_recursion_depth": 5,
    "max_execution_time_ms": 10000,
    "max_memory_mb": 4096,
    "required_capabilities": ["drift_detection", "photonic_hardware"]
  },
  "capabilities": {
    "photonic_hardware": true,
    "quantum_computing": false,
    "rlhf": true,
    "drift_detection": true,
    "explainability": true
  }
}
Enhanced Error Response
json{
  "error_code": "AI_PHOTONIC_NOISE",
  "message": "Photonic noise_std exceeds safety threshold",
  "details": {
    "node_id": "photonic_1",
    "parameter": "noise_std",
    "value": 0.08,
    "threshold": 0.05,
    "device_type": "AIM_SOI_modulator"
  },
  "timestamp": "2025-10-01T12:00:01.523Z",
  "recoverable": true,
  "suggested_action": "Reduce noise_std to < 0.05 or enable noise compensation"
}
Contract Node with Extended Fields
json{
  "id": "contract_1",
  "type": "ContractNode",
  "params": {
    "contract": {
      "latency_ms": 10,
      "accuracy": 0.99,
      "ethical_label": "safe",
      "energy_budget_nj": 1000,
      "privacy_level": "private",
      "auditability": "full"
    }
  },
  "metadata": {
    "ethical_score": 0.95,
    "audit_trail": [
      {
        "user": "validator_ai",
        "action": "contract_verified",
        "timestamp": "2025-10-01T11:59:00.000Z",
        "context": {"verification_method": "formal_proof"}
      }
    ]
  }
}

AI-Native Design Principles
1. Autonomous Operation

Graphs are self-describing with metadata and provenance
Capabilities declare what's available to runtime
Constraints prevent unsafe operations
Error responses include suggested actions

2. Evolvability

Custom types extensible via registry
Semantic versioning ensures compatibility
Metadata tracks dependencies and versions
Evolution parameters built into grammar

3. Explainability

Every node can include explanation metadata
Audit trails track all modifications
Provenance tracks creation and lineage
Explainability nodes are first-class citizens

4. Safety by Design

Contracts specify requirements and constraints
Resource requirements declared upfront
Recursion depth limits enforced
Ethical labels and privacy levels mandatory for sensitive operations

5. Hardware Awareness

Photonic parameters for optical computing
Resource specs for heterogeneous hardware
Energy budgets for sustainability
Device-specific parameters supported


JSON Schema Reference
Auto-generate JSON Schema from grammar:
bashpython src/schema_auto_generator.py --input formal_grammar.md --output schemas/graph_v2_0_0.json
Validate graphs:
bashpython src/validate_graph.py --schema schemas/graph_v2_0_0.json --graph my_graph.json

Testing and Conformance
Conformance Test Suite
python# test_conformance_v2.py
def test_metadata_multiple_fields():
    """Metadata should allow multiple fields (v2.0 fix)."""
    graph = {
        "grammar_version": "2.0.0",
        "nodes": [{
            "id": "n1",
            "metadata": {
                "explanation": "test",
                "provenance": {"source": "test"},
                "ethical_score": 0.9,
                "created_at": "2025-10-01T00:00:00.000Z"
            }
        }]
    }
    assert validate_graph(graph) == True

def test_constraint_enforcement():
    """Runtime should enforce graph constraints."""
    graph = {
        "constraints": {"max_recursion_depth": 3},
        "nodes": [{
            "id": "meta1",
            "type": "MetaGraphNode",
            "metadata": {"recursion_depth": 5}
        }]
    }
    result = execute_graph(graph)
    assert result["error_code"] == "AI_RECURSION_DEPTH_EXCEEDED"

def test_capability_checking():
    """Runtime should check required capabilities."""
    graph = {
        "capabilities": {"photonic_hardware": false},
        "nodes": [{
            "type": "PhotonicMVMNode",
            "metadata": {"capabilities_required": ["photonic_hardware"]}
        }]
    }
    result = execute_graph(graph)
    assert result["error_code"] == "AI_CAPABILITY_MISSING"

Integration Guidelines
For AI Agents

Graph Generation: Use type schemas to validate parameters before generation
Capability Detection: Check available capabilities before using specialized nodes
Error Recovery: Parse error_response.suggested_action for autonomous recovery
Evolution: Use EvolutionNode to optimize graphs based on feedback
Monitoring: Track drift with DriftDetectorNode, explain with ExplainabilityNode

For Runtime Implementers

Validation: Validate all constraints before execution
Error Handling: Return structured error_response with recovery hints
Resource Management: Respect resource_requirements and energy_budget
Audit Logging: Maintain audit_trail in metadata
Versioning: Check grammar_version for compatibility


Future Extensions (v2.1+)
Planned enhancements:

Probabilistic graph execution (stochastic routing)
Temporal constraints (time-based scheduling)
Multi-graph coordination (graph composition)
Federated learning parameters
Quantum computing node types
Advanced optimization hints for AI agents