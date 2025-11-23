"""
Graphix Graph Executor
======================
This module executes Graphix IR graphs with a simple topological sort.

Key Features:
- Supports core nodes: InputNode, ComputeNode, OutputNode.
- Basic execution using standard Python libraries.

Dependencies:
- Standard Python libraries only.
"""

import json
import logging
import asyncio
import unittest
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)

class GraphExecutor:
    """
    Simple executor for Graphix IR graphs.
    """
    def __init__(self):
        self.logger = logging.getLogger("GraphExecutor")
        self.node_executors = {
            "InputNode": self._execute_input_node,
            "ComputeNode": self._execute_compute_node,
            "OutputNode": self._execute_output_node
        }
        self.logger.info("Executor initialized")

    async def _execute_input_node(self, node: Dict, context: Dict) -> Any:
        """Execute InputNode: Set input data in context."""
        input_data = node.get("value", "")
        context[node["id"]] = input_data
        self.logger.info(f"InputNode {node['id']} executed: {input_data}")
        return input_data

    async def _execute_compute_node(self, node: Dict, context: Dict) -> Any:
        """Execute ComputeNode: Transform input (e.g., uppercase)."""
        input_ref = node.get("in", "")
        input_data = context.get(input_ref, "")
        result = str(input_data).upper()  # Simple transformation
        context[node["id"]] = result
        self.logger.info(f"ComputeNode {node['id']} executed: {result}")
        return result

    async def _execute_output_node(self, node: Dict, context: Dict) -> Any:
        """Execute OutputNode: Return final result."""
        input_ref = node.get("in", "")
        output_data = context.get(input_ref, "")
        context["output"] = output_data
        self.logger.info(f"OutputNode {node['id']} executed: {output_data}")
        return output_data

    async def execute_graph(self, graph_json: str) -> Dict[str, Any]:
        """
        Execute a Graphix IR graph using topological sort.
        """
        try:
            graph = json.loads(graph_json)
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON graph")
            return {"status": "failed", "error": "Invalid JSON"}

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        context = {}

        # Simple topological sort using a list
        node_ids = [node["id"] for node in nodes]
        dependencies = {node["id"]: [] for node in nodes}
        for edge in edges:
            if edge["to"] in dependencies:
                dependencies[edge["to"]].append(edge["from"])

        sorted_nodes = []
        available = [node["id"] for node in nodes if not dependencies[node["id"]]]
        while available:
            node_id = available.pop(0)
            sorted_nodes.append(node_id)
            for dep_node_id, deps in list(dependencies.items()):
                if node_id in deps:
                    deps.remove(node_id)
                    if not deps:
                        available.append(dep_node_id)

        if len(sorted_nodes) != len(nodes):
            self.logger.error("Graph has cycles or invalid edges")
            return {"status": "failed", "error": "Graph has cycles"}

        # Execute nodes in topological order
        for node_id in sorted_nodes:
            node = next(n for n in nodes if n["id"] == node_id)
            node_type = node.get("type", "")
            executor = self.node_executors.get(node_type)
            if not executor:
                self.logger.error(f"Unknown node type: {node_type}")
                return {"status": "failed", "error": f"Unknown node type: {node_type}"}
            await executor(node, context)

        return {"status": "completed", "result": {"output": context.get("output", "")}}

class TestGraphExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = GraphExecutor()
        self.test_graph = json.dumps({
            "grammar_version": "1.0.0",
            "id": "test_graph",
            "type": "Graph",
            "nodes": [
                {"id": "in", "type": "InputNode", "value": "test"},
                {"id": "compute", "type": "ComputeNode", "in": "in"},
                {"id": "out", "type": "OutputNode", "in": "compute"}
            ],
            "edges": [
                {"id": "e1", "from": "in", "to": "compute", "type": "data"},
                {"id": "e2", "from": "compute", "to": "out", "type": "data"}
            ]
        })

    def test_basic_execution(self):
        result = asyncio.run(self.executor.execute_graph(self.test_graph))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["output"], "TEST")

    def test_invalid_json(self):
        result = asyncio.run(self.executor.execute_graph("invalid json"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("Invalid JSON", result["error"])

    def test_unknown_node_type(self):
        bad_graph = json.dumps({
            "grammar_version": "1.0.0",
            "id": "bad_graph",
            "type": "Graph",
            "nodes": [
                {"id": "in", "type": "UnknownNode"}
            ],
            "edges": []
        })
        result = asyncio.run(self.executor.execute_graph(bad_graph))
        self.assertEqual(result["status"], "failed")
        self.assertIn("Unknown node type", result["error"])

if __name__ == "__main__":
    unittest.main()
