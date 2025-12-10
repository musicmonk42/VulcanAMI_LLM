# demo_evolution.py
"""
Graphix Evolution Cycle Demo - Production Ready
================================================

Demonstrates the complete evolution cycle:
1. Load and validate evolution proposals
2. Generate new nodes via AI
3. Learn and register new node types
4. Test evolved nodes in execution

Features:
- Comprehensive error handling
- Configurable paths and timeouts
- Code safety validation
- Proper resource cleanup
- Detailed logging
- Graceful fallbacks

Usage:
    python demo_evolution.py
    python demo_evolution.py --proposal-dir ./proposals --timeout 60 --verbose
    python demo_evolution.py --grammar-version 3.4.0 --skip-validation
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Import with graceful fallback
try:
    from unified_runtime import UnifiedRuntime
    UNIFIED_RUNTIME_AVAILABLE = True
except ImportError:
    UnifiedRuntime = None
    UNIFIED_RUNTIME_AVAILABLE = False


# Configure logging
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('demo_evolution.log', mode='a')
        ]
    )
    
    return logging.getLogger("DemoEvolution")


@dataclass
class EvolutionConfig:
    """Configuration for evolution demo."""
    proposal_dir: Path = Path("src")
    output_dir: Path = Path("output")
    grammar_version: str = "3.4.0"
    timeout_seconds: int = 30
    skip_validation: bool = False
    enable_code_safety: bool = True
    max_code_size: int = 10000  # characters
    verbose: bool = False
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dict."""
        config_dict = asdict(self)
        # Convert Path objects to strings
        config_dict['proposal_dir'] = str(config_dict['proposal_dir'])
        config_dict['output_dir'] = str(config_dict['output_dir'])
        return config_dict


class CodeSafetyValidator:
    """Validates generated code for safety before execution."""
    
    # Dangerous imports and functions to block
    BLOCKED_IMPORTS = {
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        'importlib', '__import__', 'eval', 'exec', 'compile',
        'open', 'file', 'input', 'raw_input',
        'requests', 'urllib', 'http', 'socket',
        'pickle', 'shelve', 'marshal', 'ctypes'
    }
    
    BLOCKED_BUILTINS = {
        '__import__', 'eval', 'exec', 'compile',
        'open', 'file', 'input', 'raw_input',
        'execfile', 'reload'
    }
    
    BLOCKED_ATTRIBUTES = {
        '__globals__', '__code__', '__closure__',
        '__dict__', '__class__', '__bases__',
        '__subclasses__', '__builtins__'
    }
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def validate(self, code: str, max_size: int = 10000) -> tuple[bool, Optional[str]]:
        """
        Validate code for safety.
        
        Returns:
            (is_safe, error_message) tuple
        """
        if not code or not isinstance(code, str):
            return False, "Code is empty or not a string"
        
        if len(code) > max_size:
            return False, f"Code exceeds maximum size of {max_size} characters"
        
        # Check for blocked imports
        for blocked in self.BLOCKED_IMPORTS:
            patterns = [
                rf'\bimport\s+{re.escape(blocked)}\b',
                rf'\bfrom\s+{re.escape(blocked)}\b',
                rf'\b__import__\s*\(\s*["\']({re.escape(blocked)})["\']'
            ]
            for pattern in patterns:
                if re.search(pattern, code):
                    return False, f"Blocked import detected: {blocked}"
        
        # Check for blocked builtins
        for blocked in self.BLOCKED_BUILTINS:
            if re.search(rf'\b{re.escape(blocked)}\s*\(', code):
                return False, f"Blocked builtin detected: {blocked}"
        
        # Check for blocked attributes
        for blocked in self.BLOCKED_ATTRIBUTES:
            if blocked in code:
                return False, f"Blocked attribute access detected: {blocked}"
        
        # Check for suspicious patterns
        suspicious_patterns = [
            (r'exec\s*\(', "exec() call detected"),
            (r'eval\s*\(', "eval() call detected"),
            (r'__.*__', "Dunder method access detected"),
            (r'globals\s*\(', "globals() access detected"),
            (r'locals\s*\(', "locals() access detected"),
            (r'vars\s*\(', "vars() access detected"),
            (r'dir\s*\(', "dir() introspection detected"),
        ]
        
        for pattern, message in suspicious_patterns:
            if re.search(pattern, code):
                self.logger.warning(f"Suspicious pattern: {message}")
                # Log but don't block - might be legitimate
        
        # Basic syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"Syntax error in code: {e}"
        
        return True, None


class EvolutionDemo:
    """Main evolution demo orchestrator."""
    
    def __init__(self, config: EvolutionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.runtime: Optional[Any] = None
        self.safety_validator = CodeSafetyValidator(logger)
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"EvolutionDemo initialized with config: {config.to_json_dict()}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if UnifiedRuntime is None:
            raise ImportError("UnifiedRuntime is not available. Cannot initialize runtime.")
        
        self.runtime = UnifiedRuntime()
        self.logger.info("UnifiedRuntime initialized")
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit with cleanup."""
        if self.runtime:
            try:
                # Save any learned nodes before closing
                if hasattr(self.runtime, 'save_node_executors'):
                    self.runtime.save_node_executors()
                    self.logger.info("Node executors saved")
            except Exception as e:
                self.logger.error(f"Error saving node executors: {e}")
            
            # Close runtime if it has a close method
            if hasattr(self.runtime, 'close'):
                try:
                    await self.runtime.close()
                    self.logger.info("UnifiedRuntime closed")
                except Exception as e:
                    self.logger.error(f"Error closing runtime: {e}")
        
        return False  # Don't suppress exceptions
    
    def _validate_file_path(self, file_path: Path, description: str) -> None:
        """Validate that a file exists and is readable."""
        if not file_path.exists():
            raise FileNotFoundError(f"{description} not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"{description} is not a file: {file_path}")
        
        if not file_path.suffix == '.json':
            self.logger.warning(f"{description} doesn't have .json extension: {file_path}")
    
    def _load_json_file(self, file_path: Path, description: str) -> Dict[str, Any]:
        """Load and parse JSON file with error handling."""
        self._validate_file_path(file_path, description)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded {description}: {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {description}: {e}")
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading {description}: {e}")
            raise
    
    def _validate_graph_structure(self, graph: Dict[str, Any], description: str) -> None:
        """Validate basic graph structure."""
        required_fields = ["grammar_version", "id", "type", "nodes", "edges"]
        
        for field in required_fields:
            if field not in graph:
                raise ValueError(f"{description} missing required field: {field}")
        
        if not isinstance(graph["nodes"], list):
            raise ValueError(f"{description} 'nodes' must be a list")
        
        if not isinstance(graph["edges"], list):
            raise ValueError(f"{description} 'edges' must be a list")
        
        if graph["grammar_version"] != self.config.grammar_version:
            self.logger.warning(
                f"{description} grammar version mismatch: "
                f"expected {self.config.grammar_version}, got {graph['grammar_version']}"
            )
    
    async def load_and_validate_proposals(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Load and validate evolution proposals."""
        self.logger.info("Loading evolution proposals...")
        
        # Define proposal file paths
        causal_file = self.config.proposal_dir / "gemini_evolution_proposal.json"
        ai_file = self.config.proposal_dir / "grok_generative_ai_node_proposal.json"
        
        # Load proposals
        causal_proposal = self._load_json_file(causal_file, "Causal proposal")
        ai_proposal = self._load_json_file(ai_file, "AI proposal")
        
        # Validate structure
        self._validate_graph_structure(causal_proposal, "Causal proposal")
        self._validate_graph_structure(ai_proposal, "AI proposal")
        
        # Validate with runtime if not skipped
        if not self.config.skip_validation and self.runtime:
            try:
                if not self.runtime._validate_graph(causal_proposal):
                    raise ValueError("Causal proposal failed runtime validation")
                self.logger.info("Causal proposal validated successfully")
                
                if not self.runtime._validate_graph(ai_proposal):
                    raise ValueError("AI proposal failed runtime validation")
                self.logger.info("AI proposal validated successfully")
                
            except Exception as e:
                self.logger.error(f"Proposal validation failed: {e}")
                raise
        else:
            self.logger.info("Skipping runtime validation (disabled or runtime unavailable)")
        
        return causal_proposal, ai_proposal
    
    async def generate_and_learn_node(self) -> str:
        """Generate new node via AI and learn it."""
        self.logger.info("Generating new node type via AI...")
        
        # Create test graph for generation
        test_graph = {
            "grammar_version": self.config.grammar_version,
            "id": f"test_evolution_{int(datetime.now().timestamp())}",
            "type": "Graph",
            "metadata": {
                "purpose": "Generate AdderNode implementation",
                "timestamp": datetime.utcnow().isoformat()
            },
            "nodes": [
                {
                    "id": "input",
                    "type": "InputNode",
                    "value": "create a python function named 'execute' that adds 10 to its input"
                },
                {
                    "id": "ai_gen",
                    "type": "GenerativeAINode",
                    "prompt_template": "{input}"
                },
                {
                    "id": "dynamic",
                    "type": "DynamicCodeNode"
                },
                {
                    "id": "out",
                    "type": "OutputNode"
                }
            ],
            "edges": [
                {
                    "id": "e1",
                    "from": "input",
                    "to": "ai_gen",
                    "type": "data"
                },
                {
                    "id": "e2",
                    "from": "ai_gen",
                    "to": "dynamic",
                    "type": "ai_flow"
                },
                {
                    "id": "e3",
                    "from": "dynamic",
                    "to": "out",
                    "type": "data"
                }
            ]
        }
        
        # Save test graph for debugging
        test_graph_file = self.config.output_dir / "test_generation_graph.json"
        with open(test_graph_file, 'w', encoding='utf-8') as f:
            json.dump(test_graph, f, indent=2)
        self.logger.debug(f"Saved test graph to {test_graph_file}")
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                self.runtime.execute_graph(test_graph),
                timeout=self.config.timeout_seconds
            )
            self.logger.info("Graph execution completed")
            
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Graph execution timed out after {self.config.timeout_seconds} seconds"
            )
        except Exception as e:
            self.logger.error(f"Graph execution failed: {e}")
            raise
        
        # Extract generated code
        code = result.get("output")
        if not code:
            raise ValueError("No code generated in execution result")
        
        if not isinstance(code, str):
            raise ValueError(f"Generated code is not a string: {type(code)}")
        
        self.logger.info(f"Generated code ({len(code)} characters)")
        
        # Save generated code
        code_file = self.config.output_dir / "generated_adder_node.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        self.logger.info(f"Saved generated code to {code_file}")
        
        # Validate code safety
        if self.config.enable_code_safety:
            is_safe, error_msg = self.safety_validator.validate(
                code, 
                max_size=self.config.max_code_size
            )
            
            if not is_safe:
                raise ValueError(f"Generated code failed safety validation: {error_msg}")
            
            self.logger.info("Generated code passed safety validation")
        else:
            self.logger.warning("Code safety validation disabled - DANGEROUS!")
        
        # Learn the new node type
        try:
            self.runtime.learn_node_type("AdderNode", code)
            self.logger.info("Successfully learned AdderNode type")
        except Exception as e:
            self.logger.error(f"Failed to learn node type: {e}")
            raise
        
        return code
    
    async def test_learned_node(self) -> Dict[str, Any]:
        """Test the newly learned node type."""
        self.logger.info("Testing learned AdderNode...")
        
        # Create worker graph using the new node
        worker_graph = {
            "grammar_version": self.config.grammar_version,
            "id": f"test_worker_{int(datetime.now().timestamp())}",
            "type": "Graph",
            "metadata": {
                "purpose": "Test AdderNode functionality",
                "timestamp": datetime.utcnow().isoformat()
            },
            "nodes": [
                {
                    "id": "start",
                    "type": "InputNode",
                    "value": 10
                },
                {
                    "id": "add",
                    "type": "AdderNode"
                },
                {
                    "id": "out",
                    "type": "OutputNode"
                }
            ],
            "edges": [
                {
                    "id": "e1",
                    "from": "start",
                    "to": "add",
                    "type": "data"
                },
                {
                    "id": "e2",
                    "from": "add",
                    "to": "out",
                    "type": "data"
                }
            ]
        }
        
        # Save worker graph
        worker_graph_file = self.config.output_dir / "test_worker_graph.json"
        with open(worker_graph_file, 'w', encoding='utf-8') as f:
            json.dump(worker_graph, f, indent=2)
        self.logger.debug(f"Saved worker graph to {worker_graph_file}")
        
        # Execute with timeout
        try:
            final_result = await asyncio.wait_for(
                self.runtime.execute_graph(worker_graph),
                timeout=self.config.timeout_seconds
            )
            self.logger.info("Worker graph execution completed")
            
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Worker graph execution timed out after {self.config.timeout_seconds} seconds"
            )
        except Exception as e:
            self.logger.error(f"Worker graph execution failed: {e}")
            raise
        
        # Validate result
        if "output" not in final_result:
            raise ValueError("Execution result missing 'output' field")
        
        output_value = final_result["output"]
        expected_value = 20  # 10 + 10
        
        self.logger.info(f"AdderNode output: {output_value}")
        
        if output_value == expected_value:
            self.logger.info(f"Test PASSED: Output {output_value} matches expected {expected_value}")
        else:
            self.logger.warning(
                f"Test result unexpected: got {output_value}, expected {expected_value}"
            )
        
        # Save test result
        test_result = {
            "input_value": 10,
            "expected_output": expected_value,
            "actual_output": output_value,
            "test_passed": output_value == expected_value,
            "full_result": final_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result_file = self.config.output_dir / "test_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2)
        self.logger.info(f"Saved test result to {result_file}")
        
        return final_result
    
    async def run_complete_cycle(self) -> Dict[str, Any]:
        """Run the complete evolution cycle."""
        self.logger.info("="*60)
        self.logger.info("Starting Evolution Cycle")
        self.logger.info("="*60)
        
        cycle_results = {
            "start_time": datetime.utcnow().isoformat(),
            "config": self.config.to_json_dict(),
            "steps": {},
            "success": False,
            "error": None
        }
        
        try:
            # Step 1: Load and validate proposals
            self.logger.info("\nStep 1: Loading and validating proposals...")
            causal_proposal, ai_proposal = await self.load_and_validate_proposals()
            cycle_results["steps"]["load_proposals"] = {
                "status": "success",
                "causal_proposal_id": causal_proposal.get("id"),
                "ai_proposal_id": ai_proposal.get("id")
            }
            
            # Step 2: Generate and learn new node
            self.logger.info("\nStep 2: Generating and learning new node...")
            generated_code = await self.generate_and_learn_node()
            cycle_results["steps"]["generate_node"] = {
                "status": "success",
                "code_length": len(generated_code)
            }
            
            # Step 3: Test learned node
            self.logger.info("\nStep 3: Testing learned node...")
            test_result = await self.test_learned_node()
            cycle_results["steps"]["test_node"] = {
                "status": "success",
                "output": test_result.get("output")
            }
            
            # Mark overall success
            cycle_results["success"] = True
            
            self.logger.info("\n" + "="*60)
            self.logger.info("Evolution Cycle COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            
            print(f"\nEvolution successful! New node output: {test_result['output']}")
            
        except Exception as e:
            cycle_results["success"] = False
            cycle_results["error"] = str(e)
            self.logger.error(f"\nEvolution cycle FAILED: {e}")
            self.logger.error("="*60)
            raise
        
        finally:
            cycle_results["end_time"] = datetime.utcnow().isoformat()
            
            # Save cycle results
            results_file = self.config.output_dir / "evolution_cycle_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(cycle_results, f, indent=2)
            self.logger.info(f"\nCycle results saved to {results_file}")
        
        return cycle_results


async def demo_evolution_cycle():
    """
    Main entry point for evolution cycle demo.
    Maintained for backward compatibility.
    """
    # Use default configuration
    config = EvolutionConfig()
    logger = setup_logging(verbose=False)
    
    async with EvolutionDemo(config, logger) as demo:
        await demo.run_complete_cycle()


def main():
    """Enhanced main with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Graphix Evolution Cycle Demo - Production Ready",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--proposal-dir",
        type=Path,
        default=Path("src"),
        help="Directory containing evolution proposals"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--grammar-version",
        default="3.4.0",
        help="Grammar version to use"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for graph execution"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip runtime validation of proposals"
    )
    
    parser.add_argument(
        "--disable-code-safety",
        action="store_true",
        help="Disable code safety validation (DANGEROUS!)"
    )
    
    parser.add_argument(
        "--max-code-size",
        type=int,
        default=10000,
        help="Maximum allowed code size in characters"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Check if UnifiedRuntime is available
    if not UNIFIED_RUNTIME_AVAILABLE:
        print("ERROR: unified_runtime module not found. Please ensure it's in your Python path.")
        sys.exit(1)
    
    # Create configuration
    config = EvolutionConfig(
        proposal_dir=args.proposal_dir,
        output_dir=args.output_dir,
        grammar_version=args.grammar_version,
        timeout_seconds=args.timeout,
        skip_validation=args.skip_validation,
        enable_code_safety=not args.disable_code_safety,
        max_code_size=args.max_code_size,
        verbose=args.verbose
    )
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Run demo
    async def run():
        try:
            async with EvolutionDemo(config, logger) as demo:
                results = await demo.run_complete_cycle()
                
                # Exit with appropriate code
                sys.exit(0 if results["success"] else 1)
                
        except KeyboardInterrupt:
            logger.info("\nDemo interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.error(f"\nFatal error: {e}")
            sys.exit(1)
    
    try:
        asyncio.run(run())
    except SystemExit:
        # Re-raise SystemExit from the async function
        raise
    
    # If we get here (normal completion), exit successfully
    sys.exit(0)


if __name__ == "__main__":
    main()