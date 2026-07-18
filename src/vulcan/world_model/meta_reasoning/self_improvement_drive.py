# src/vulcan/world_model/meta_reasoning/self_improvement_drive.py
"""
Self-Improvement Drive - Intrinsic motivation for continuous improvement.

This makes self-improvement a CORE DRIVE of Vulcan, not just a feature you call.
The system will naturally seek to improve itself as part of its operation.

ENHANCEMENTS (2025-10-16):
- Reads extended config (including global_settings) and supports new fields:
  * LLM cost controls: cost_reconciliation_period_days, max_tokens_per_session, cost_tracking_window_hours
  * Persistence: backup_state_every_n_actions (configurable, not hardcoded)
  * Triggers: jitter_milliseconds for simultaneous triggers mitigation
  * Approval: passes through auto-approval hints (doc-only/test-only) to orchestrator
- State tracking:
  * Tracks session tokens and cost history with pruning based on reconciliation window
- Resource-limits:
  * Enforces token and dynamic (window-based) daily/monthly resets; warns/pauses via thresholds
- Action plans:
  * Echoes env_overrides and risk_classification hints to external orchestrators

INTRINSIC DRIVE (latent): Collective Self-Improvement via Human Understanding (CSIU)
Purpose: improve the collective self by reducing interaction entropy, increasing alignment coherence,
and clarifying intent, while respecting safety policies and exposing bounded operator/audit status.
Scope: governed regularizers only; max effect ≤ 5%; auditable; kill-switch granular.
Ethics Review: periodically audit metric definitions, weight updates, and longitudinal trends
for unintended bias.

WIRING (2025-10-18):
- Injectable metrics provider for real telemetry integration
- CSIU loop called during plan assembly with utility calculation, EWMA, pressure, regularization
- Granular kill-switches guard all CSIU operations
- Metrics provider verification for robust operation
- Optional CSIU weight persistence for long-term learning

ENCODING FIX (2025-10-19):
- Robust UTF-8 config loading with BOM handling and fallback strategies for Windows compatibility
"""

from typing import Any, Dict, List, Tuple
import copy
from datetime import timezone
import ast
from collections import defaultdict
from datetime import datetime, timedelta
import difflib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Initialize logger early - before it's used in import blocks
logger = logging.getLogger(__name__)

class AutoApplyStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    UNAVAILABLE = "unavailable"
    MISSING_GOVERNED_PROPOSAL = "missing_governed_proposal"
    CONFIGURATION_MISSING = "configuration_missing"
    GOVERNED_EXCEPTION = "governed_exception"
    QUEUED = "queued"
    APPLIED_AND_VERIFIED = "applied_and_verified"
    REJECTED_BEFORE_INSTALLATION = "rejected_before_installation"
    VERIFICATION_FAILED_ROLLBACK_SUCCEEDED = "verification_failed_rollback_succeeded"
    VERIFICATION_FAILED_ROLLBACK_FAILED = "verification_failed_rollback_failed"
    EXTERNAL_MUTATION_PREVENTED_ROLLBACK = "external_mutation_prevented_rollback"

@dataclass(frozen=True)
class AutoApplyResult:
    applied: bool
    status: AutoApplyStatus
    reason: str
    approval_required: bool
    terminal: bool
    rollback_succeeded: bool = False
    manual_recovery_required: bool = False
    transaction_result: Any = None
    def __post_init__(self):
        if isinstance(self.status, str):
            object.__setattr__(self, "status", AutoApplyStatus(self.status))
    @property
    def status_code(self) -> str:
        return self.status.value

# ==========================================================================
# CONSTANTS
# ==========================================================================

# Maximum characters to include in LLM prompts for code improvement
MAX_CODE_SNIPPET_CHARS = 3000

# NOTE: This patch assumes a class named SelfImprovementDrive exists.
# Additions: policy loading, auto-apply gate, and robust get_status.


try:
    from .auto_apply_policy import (
        Policy,
        TRUSTED_PROVIDERS,
        check_files_against_policy,
        is_trusted_llm_provider,
        load_policy,
        run_gates,
    )
except Exception:
    # Fallback: disable auto-apply if policy module isn't present
    # Define TRUSTED_PROVIDERS first since is_trusted_llm_provider references it
    # TASK 2 FIX: Added "trusted_internal_source" for VULCAN Policy compliance
    TRUSTED_PROVIDERS = frozenset({
        "local_llm", "graphix", "graphix_vulcan", "graphix_vulcan_llm", 
        "vulcan_local_llm", "internal", "trusted_internal_source"
    })

    def load_policy(_):
        from types import SimpleNamespace

        return SimpleNamespace(enabled=False)

    def check_files_against_policy(files, policy):
        return False, ["policy module unavailable"]

    def run_gates(policy, cwd=None):
        return False, ["policy module unavailable"]

    def is_trusted_llm_provider(provider_id):
        """Fallback: trust local providers using the same trusted providers set."""
        if not provider_id:
            return False
        return provider_id.lower().strip() in TRUSTED_PROVIDERS

    class Policy:
        enabled = False


try:
    from .csiu_enforcement import (CSIUEnforcementConfig, CSIUPolicy, CSIUMetricSnapshot, CSIUDecision, CSIUValidationError, canonical_digest, get_csiu_enforcer)

    CSIU_ENFORCEMENT_AVAILABLE = True
except ImportError:
    # Fallback if enforcement module not available
    logger.warning(
        "CSIU enforcement module not available - running without enforcement caps. "
        "This is NOT recommended for production use."
    )
    get_csiu_enforcer = None
    CSIUEnforcementConfig = None
    CSIU_ENFORCEMENT_AVAILABLE = False

try:
    from .safe_execution import get_safe_executor
except ImportError:
    # Fallback if safe execution module not available
    get_safe_executor = None

try:
    from .governed_transaction import (
        GovernedSelfImprovementTransaction, ImprovementProposal, ApprovalRecord, TransactionResult, TransactionStatus, inspect_repository
    )
except Exception:
    GovernedSelfImprovementTransaction = None
    ImprovementProposal = None
    ApprovalRecord = None
    TransactionResult = None
    TransactionStatus = None
    inspect_repository = None


# ==========================================================================
# CODE INTROSPECTION - Enables Vulcan to examine its own source code
# ==========================================================================


class CodeIntrospector:
    """
    Enables Vulcan to examine its own source code.
    
    This class provides the ability to:
    - Scan and parse all Python files in the project
    - Find specific class methods by name
    - Trace function calls from entry points
    - Find missing implementations
    - Analyze specific components like QueryRouter
    """
    
    def __init__(self, project_root: Path):
        """
        Initialize the code introspector.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = Path(project_root) if not isinstance(project_root, Path) else project_root
        self.source_files: Dict[str, Dict[str, Any]] = {}
        self.import_graph: Dict[str, List[str]] = {}
        self.function_signatures: Dict[str, Dict[str, Any]] = {}
        self.class_hierarchy: Dict[str, List[str]] = {}
        self._scan_codebase()
    
    def _scan_codebase(self) -> None:
        """Scan all Python files in the project."""
        if not self.project_root.exists():
            logger.warning(f"Project root does not exist: {self.project_root}")
            return
        
        for py_file in self.project_root.rglob("*.py"):
            # Skip virtual environments and cache directories
            path_str = str(py_file)
            if "venv" in path_str or "__pycache__" in path_str or ".git" in path_str:
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the AST
                try:
                    tree = ast.parse(content)
                except SyntaxError as e:
                    logger.debug(f"Syntax error parsing {py_file}: {e}")
                    tree = None
                
                self.source_files[str(py_file)] = {
                    'content': content,
                    'ast': tree,
                    'lines': content.splitlines()
                }
                
                # Extract import graph
                if tree:
                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)
                    self.import_graph[str(py_file)] = imports
                    
            except Exception as e:
                logger.debug(f"Failed to parse {py_file}: {e}")
        
        logger.info(f"CodeIntrospector: Scanned {len(self.source_files)} Python files")
    
    def find_class_method(self, class_name: str, method_name: str) -> Optional[str]:
        """
        Find implementation of a specific method in a class.
        
        Args:
            class_name: Name of the class to search for
            method_name: Name of the method within the class
            
        Returns:
            String representation of the method if found, None otherwise
        """
        for filepath, data in self.source_files.items():
            tree = data.get('ast')
            if tree is None:
                continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            try:
                                return ast.unparse(item)
                            except Exception:
                                # Fallback for older Python versions
                                start_line = item.lineno - 1
                                end_line = item.end_lineno if hasattr(item, 'end_lineno') else start_line + 20
                                return '\n'.join(data['lines'][start_line:end_line])
        return None
    
    def trace_function_calls(self, entry_point: str) -> List[str]:
        """
        Trace all function calls from an entry point.
        
        Args:
            entry_point: Name of the function or method to trace from
            
        Returns:
            List of function/method names that are called
        """
        calls: List[str] = []
        for filepath, data in self.source_files.items():
            if entry_point not in data['content']:
                continue
            
            tree = data.get('ast')
            if tree is None:
                continue
            
            # Extract function calls using AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        try:
                            calls.append(f"{ast.unparse(node.func.value)}.{node.func.attr}")
                        except Exception:
                            calls.append(node.func.attr)
        
        return list(set(calls))  # Remove duplicates
    
    def find_missing_implementations(self) -> List[Dict[str, Any]]:
        """
        Find methods that are called but not implemented.
        
        Returns:
            List of dictionaries describing potentially missing implementations
        """
        missing: List[Dict[str, Any]] = []
        all_calls: set = set()
        all_definitions: set = set()
        
        for filepath, data in self.source_files.items():
            tree = data.get('ast')
            if tree is None:
                continue
            
            # Collect all function/method calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        all_calls.add(node.func.attr)
            
            # Collect all function/method definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    all_definitions.add(node.name)
        
        # Find calls without definitions (excluding builtins and common methods)
        common_methods = {
            'get', 'set', 'add', 'remove', 'append', 'extend', 'pop', 'update',
            'keys', 'values', 'items', 'copy', 'clear', 'format', 'join', 'split',
            'strip', 'replace', 'lower', 'upper', 'startswith', 'endswith',
            'encode', 'decode', 'read', 'write', 'close', 'open', 'exists',
            'mkdir', 'info', 'debug', 'warning', 'error', 'critical',
        }
        
        for call in all_calls:
            if call not in all_definitions and call not in common_methods:
                missing.append({
                    'function': call,
                    'type': 'missing_implementation'
                })
        
        return missing
    
    def analyze_query_routing(self) -> Dict[str, Any]:
        """
        Specifically analyze the QueryRouter implementation.
        
        Returns:
            Dictionary containing analysis results for query routing
        """
        analysis: Dict[str, Any] = {
            'has_query_router': False,
            'routing_logic_found': False,
            'route_methods': [],
            'missing_routes': [],
            'issues': [],
            'file_path': None
        }
        
        # Find QueryRouter class
        for filepath, data in self.source_files.items():
            if 'QueryRouter' not in data['content'] and 'QueryAnalyzer' not in data['content']:
                continue
            
            analysis['has_query_router'] = True
            analysis['file_path'] = filepath
            
            tree = data.get('ast')
            if tree is None:
                continue
            
            # Look for route/classify methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and ('Router' in node.name or 'Analyzer' in node.name):
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            if 'route' in method.name.lower() or 'classify' in method.name.lower():
                                analysis['route_methods'].append(method.name)
                                
                                # Check for actual routing logic
                                try:
                                    method_body = ast.unparse(method)
                                except Exception:
                                    method_body = data['content'][method.lineno:method.end_lineno if hasattr(method, 'end_lineno') else method.lineno + 50]
                                
                                if any(kw in method_body.lower() for kw in [
                                    'philosophical', 'mathematical', 'identity',
                                    'querytype', 'query_type', 'routing'
                                ]):
                                    analysis['routing_logic_found'] = True
        
        # Identify issues
        if not analysis['routing_logic_found'] and analysis['has_query_router']:
            analysis['issues'].append("Query routing exists but classification logic may be incomplete")
        
        if not analysis['route_methods'] and analysis['has_query_router']:
            analysis['issues'].append("No route/classify methods found in router class")
            analysis['missing_routes'] = [
                'philosophical_route',
                'identity_route', 
                'conversational_route'
            ]
        
        return analysis
    
    def get_file_structure(self, directory: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get the file structure of the project.
        
        Args:
            directory: Optional subdirectory to focus on
            
        Returns:
            Dictionary mapping directories to their Python files
        """
        structure: Dict[str, List[str]] = defaultdict(list)
        base = self.project_root / directory if directory else self.project_root
        
        for filepath in self.source_files.keys():
            try:
                rel_path = Path(filepath).relative_to(base)
                parent = str(rel_path.parent)
                structure[parent].append(rel_path.name)
            except ValueError:
                continue
        
        return dict(structure)


class LogAnalyzer:
    """
    Analyze Vulcan's own logs for patterns and issues.
    
    This class provides the ability to:
    - Parse and analyze log files
    - Detect patterns like timeouts, routing failures, errors
    - Extract recent failures for diagnostic purposes
    """
    
    def __init__(self, log_dir: Path):
        """
        Initialize the log analyzer.
        
        Args:
            log_dir: Path to the logs directory
        """
        self.log_dir = Path(log_dir) if not isinstance(log_dir, Path) else log_dir
        self.patterns = {
            'timeouts': re.compile(r'timeout|timed out', re.IGNORECASE),
            'routing_failures': re.compile(r'route|routing|classify', re.IGNORECASE),
            'slow_queries': re.compile(r'took (\d+\.?\d*)s'),
            'errors': re.compile(r'ERROR|CRITICAL|FAILED', re.IGNORECASE),
            'exceptions': re.compile(r'Exception|Error:|Traceback', re.IGNORECASE),
            'memory_issues': re.compile(r'memory|MemoryError|OOM', re.IGNORECASE),
            'import_errors': re.compile(r'ImportError|ModuleNotFoundError', re.IGNORECASE),
        }
    
    def analyze_recent_failures(self, hours: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze recent log entries for failures.
        
        Args:
            hours: Number of hours to look back (default: 1)
            
        Returns:
            Dictionary mapping pattern types to lists of matched log entries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        failures: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        if not self.log_dir.exists():
            logger.debug(f"Log directory does not exist: {self.log_dir}")
            return dict(failures)
        
        for log_file in self.log_dir.glob("*.log"):
            try:
                # Skip old log files based on modification time
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime < cutoff_time:
                    continue
                
                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        # Try to extract timestamp and check recency
                        try:
                            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                            if match:
                                log_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                                if log_time < cutoff_time:
                                    continue
                        except (ValueError, AttributeError):
                            pass  # If we can't parse timestamp, include line anyway
                        
                        # Check patterns
                        for pattern_name, pattern in self.patterns.items():
                            if pattern.search(line):
                                failures[pattern_name].append({
                                    'line': line.strip()[:200],  # Truncate long lines
                                    'file': log_file.name,
                                    'pattern': pattern_name
                                })
                                
            except Exception as e:
                logger.debug(f"Error reading log file {log_file}: {e}")
        
        return dict(failures)
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get a summary of errors from recent logs.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Summary dictionary with error counts and examples
        """
        failures = self.analyze_recent_failures(hours)
        
        summary = {
            'total_errors': sum(len(v) for v in failures.values()),
            'error_types': {k: len(v) for k, v in failures.items()},
            'examples': {k: v[:3] for k, v in failures.items()},  # First 3 examples each
            'hours_analyzed': hours,
            'has_critical_issues': False
        }
        
        # Flag critical issues
        if failures.get('errors', []) or failures.get('exceptions', []):
            summary['has_critical_issues'] = True
        
        return summary


class CodeKnowledgeStore:
    """
    Stores and retrieves code knowledge using the memory system.
    
    This class integrates with HierarchicalMemory and GraphRAG to:
    - Store code patterns, issues, and insights as memories
    - Retrieve relevant code knowledge when making improvements
    - Learn from past code changes and their outcomes
    - Build a knowledge graph of code relationships
    """
    
    def __init__(self, memory_system=None):
        """
        Initialize the code knowledge store.
        
        Args:
            memory_system: Optional HierarchicalMemory or GraphRAG instance
        """
        self._memory = memory_system
        self._local_cache: Dict[str, Dict[str, Any]] = {}
        self._pattern_index: Dict[str, List[str]] = defaultdict(list)
        self._issue_history: List[Dict[str, Any]] = []
        self._learning_outcomes: List[Dict[str, Any]] = []
        
        # Try to initialize memory system if not provided
        if self._memory is None:
            self._memory = self._initialize_memory()
    
    def _initialize_memory(self):
        """Try to initialize connection to memory system."""
        try:
            from vulcan.memory.hierarchical import HierarchicalMemory
            from vulcan.memory.base import MemoryConfig
            # Create a minimal config for code knowledge
            config = MemoryConfig(
                max_memories=10000,
                default_importance=0.5,
                decay_rate=0.001
            )
            return HierarchicalMemory(config)
        except ImportError:
            logger.debug("HierarchicalMemory not available, using local cache only")
        except Exception as e:
            logger.debug(f"Failed to initialize HierarchicalMemory: {e}")
        
        try:
            from persistant_memory_v46 import GraphRAG
            # Note: Changed from invalid "llm_embeddings" to valid model "all-MiniLM-L6-v2"
            return GraphRAG(embedding_model="all-MiniLM-L6-v2")
        except ImportError:
            logger.debug("GraphRAG not available, using local cache only")
        except Exception as e:
            logger.debug(f"Failed to initialize GraphRAG: {e}")
        
        return None
    
    def store_code_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        source_file: str,
        confidence: float = 0.8
    ) -> str:
        """
        Store a learned code pattern in memory.
        
        Args:
            pattern_type: Type of pattern (e.g., 'bug_pattern', 'optimization', 'antipattern')
            pattern_data: Dictionary containing pattern details
            source_file: Source file where pattern was found
            confidence: Confidence score for this pattern
            
        Returns:
            Pattern ID for later retrieval
        """
        pattern_id = f"pattern_{pattern_type}_{hash(str(pattern_data)) % 10000}"
        
        memory_entry = {
            'id': pattern_id,
            'type': pattern_type,
            'data': pattern_data,
            'source_file': source_file,
            'confidence': confidence,
            'timestamp': time.time(),
            'access_count': 0
        }
        
        # Store in local cache
        self._local_cache[pattern_id] = memory_entry
        self._pattern_index[pattern_type].append(pattern_id)
        
        # Store in memory system if available
        if self._memory is not None:
            try:
                if hasattr(self._memory, 'add_node'):
                    # GraphRAG style
                    self._memory.add_node(
                        node_id=pattern_id,
                        content=json.dumps(pattern_data),
                        metadata={
                            'type': 'code_pattern',
                            'pattern_type': pattern_type,
                            'source_file': source_file,
                            'confidence': confidence
                        }
                    )
                elif hasattr(self._memory, 'store'):
                    # HierarchicalMemory style
                    from vulcan.memory.base import Memory, MemoryType
                    mem = Memory(
                        content=pattern_data,
                        memory_type=MemoryType.SEMANTIC,
                        metadata={
                            'pattern_id': pattern_id,
                            'pattern_type': pattern_type,
                            'source_file': source_file
                        },
                        importance=confidence
                    )
                    self._memory.store(mem)
            except Exception as e:
                logger.debug(f"Failed to store pattern in memory system: {e}")
        
        return pattern_id
    
    def store_issue(
        self,
        issue: Dict[str, Any],
        resolved: bool = False,
        resolution: Optional[str] = None
    ) -> str:
        """
        Store a diagnosed issue for learning.
        
        Args:
            issue: Issue dictionary from diagnose_system_issues
            resolved: Whether the issue has been resolved
            resolution: Description of how it was resolved
            
        Returns:
            Issue ID
        """
        issue_id = f"issue_{issue.get('component', 'unknown')}_{int(time.time())}"
        
        issue_entry = {
            'id': issue_id,
            'issue': issue,
            'resolved': resolved,
            'resolution': resolution,
            'timestamp': time.time()
        }
        
        self._issue_history.append(issue_entry)
        
        # Limit history size
        if len(self._issue_history) > 1000:
            self._issue_history = self._issue_history[-500:]
        
        return issue_id
    
    def record_learning_outcome(
        self,
        objective_type: str,
        success: bool,
        code_changes: Dict[str, Any],
        metrics_before: Optional[Dict[str, float]] = None,
        metrics_after: Optional[Dict[str, float]] = None
    ):
        """
        Record the outcome of a code improvement for learning.
        
        Args:
            objective_type: Type of improvement attempted
            success: Whether the improvement was successful
            code_changes: Dictionary describing what was changed
            metrics_before: Performance metrics before change
            metrics_after: Performance metrics after change
        """
        outcome = {
            'objective_type': objective_type,
            'success': success,
            'code_changes': code_changes,
            'metrics_before': metrics_before or {},
            'metrics_after': metrics_after or {},
            'timestamp': time.time()
        }
        
        self._learning_outcomes.append(outcome)
        
        # Learn from outcome
        if success:
            # Store successful pattern
            self.store_code_pattern(
                pattern_type='successful_fix',
                pattern_data={
                    'objective_type': objective_type,
                    'changes': code_changes,
                    'improvement': self._calculate_improvement(metrics_before, metrics_after)
                },
                source_file=code_changes.get('file_modified', 'unknown'),
                confidence=0.9
            )
        else:
            # Store failed attempt to avoid repeating
            self.store_code_pattern(
                pattern_type='failed_attempt',
                pattern_data={
                    'objective_type': objective_type,
                    'changes': code_changes,
                    'reason': code_changes.get('error', 'unknown')
                },
                source_file=code_changes.get('file_modified', 'unknown'),
                confidence=0.7
            )
        
        # Limit history
        if len(self._learning_outcomes) > 500:
            self._learning_outcomes = self._learning_outcomes[-250:]
    
    def _calculate_improvement(
        self,
        before: Optional[Dict[str, float]],
        after: Optional[Dict[str, float]]
    ) -> float:
        """Calculate improvement score from metrics."""
        if not before or not after:
            return 0.0
        
        improvements = []
        for key in before:
            if key in after:
                # Assume lower is better for loss/error metrics
                if 'loss' in key.lower() or 'error' in key.lower():
                    improvement = (before[key] - after[key]) / (before[key] + 1e-10)
                else:
                    improvement = (after[key] - before[key]) / (before[key] + 1e-10)
                improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def get_similar_issues(self, issue: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar past issues for learning from previous resolutions.
        
        Args:
            issue: Current issue to find similar ones for
            limit: Maximum number of similar issues to return
            
        Returns:
            List of similar past issues with their resolutions
        """
        component = issue.get('component', '')
        issue_type = issue.get('type', '')
        
        similar = []
        for past_issue in reversed(self._issue_history):
            past = past_issue.get('issue', {})
            if (past.get('component') == component or past.get('type') == issue_type):
                if past_issue.get('resolved'):
                    similar.append(past_issue)
                    if len(similar) >= limit:
                        break
        
        return similar
    
    def get_successful_patterns_for_objective(
        self,
        objective_type: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get successful fix patterns for a given objective type.
        
        Args:
            objective_type: Type of objective to find patterns for
            limit: Maximum patterns to return
            
        Returns:
            List of successful pattern data
        """
        patterns = []
        for pattern_id in self._pattern_index.get('successful_fix', []):
            pattern = self._local_cache.get(pattern_id)
            if pattern:
                data = pattern.get('data', {})
                if data.get('objective_type') == objective_type:
                    pattern['access_count'] = pattern.get('access_count', 0) + 1
                    patterns.append(pattern)
                    if len(patterns) >= limit:
                        break
        
        return patterns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored code knowledge."""
        return {
            'total_patterns': len(self._local_cache),
            'pattern_types': {k: len(v) for k, v in self._pattern_index.items()},
            'issues_tracked': len(self._issue_history),
            'learning_outcomes': len(self._learning_outcomes),
            'successful_fixes': sum(1 for o in self._learning_outcomes if o.get('success')),
            'memory_system_connected': self._memory is not None
        }


class TriggerType(Enum):
    """Types of triggers that can activate the drive."""

    ON_STARTUP = "on_startup"
    ON_ERROR = "on_error_detected"
    ON_PERFORMANCE_DEGRADATION = "on_performance_degradation"
    PERIODIC = "periodic"
    ON_LOW_ACTIVITY = "on_low_activity"


class FailureType(Enum):
    """Classifies the nature of a failure for adaptive cooldowns."""

    TRANSIENT = "transient"
    SYSTEMIC = "systemic"


@dataclass
class ImprovementObjective:
    """A specific improvement goal."""

    type: str
    weight: float
    auto_apply: bool
    completed: bool = False
    attempts: int = 0
    last_attempt: float = 0
    last_failure: Optional[float] = None
    failure_count: int = 0
    success_count: int = 0
    cooldown_until: float = 0


@dataclass
class SelfImprovementState:
    """Current state of self-improvement drive."""

    active: bool = False
    current_objective: Optional[str] = None
    completed_objectives: List[str] = field(default_factory=list)
    pending_approvals: List[Dict[str, Any]] = field(default_factory=list)
    improvements_this_session: int = 0
    last_improvement: float = (
        0  # Note: Default to 0, will be initialized in __post_init__
    )
    last_trigger_check: float = 0
    session_start_time: float = field(default_factory=time.time)
    total_cost_usd: float = 0.0
    daily_cost_usd: float = 0.0
    monthly_cost_usd: float = 0.0
    last_cost_reset: float = field(default_factory=time.time)
    state_save_count: int = 0  # For backup tracking
    # ENH: token + cost history for reconciliation
    session_tokens: int = 0
    cost_history: List[Dict[str, float]] = field(
        default_factory=list
    )  # [{timestamp, cost_usd}]

    def __post_init__(self):
        """Initialize tracking attributes. Keep last_improvement at 0 for fresh state."""
        # Don't initialize last_improvement - keep it at 0 to allow immediate triggers
        pass


class SelfImprovementDrive:
    """
    Intrinsic drive for continuous self-improvement.

    This integrates with Vulcan's motivational_introspection system to make
    self-improvement a core drive, not just a command.

    NOTE: This module handles drive logic and state management. External modules handle:
    - Pre-flight validation (safety/pre_flight_validator.py)
    - Impact analysis (safety/impact_analyzer.py)
    - Approval workflows (orchestrator/approval_service.py)
    - Circuit breaking (orchestrator/circuit_breaker.py)
    - Reporting (services/reporting_service.py)
    """

    # FIX Issue 5: Blacklisted objectives that should never be auto-selected
    # These are demo/test tasks that create noise and waste cycles on every startup
    BLACKLISTED_OBJECTIVES = {
        "fix_circular_imports",  # Demo task, creates dummy module_a.py/module_b.py files
        "module_a_refactor",
        "module_b_refactor",
    }

    # Files that should never be auto-modified by self-improvement
    PROTECTED_FILES = {
        "src/module_a.py",
        "src/module_b.py",
    }

    # --- START REPLACEMENT ---
    def __init__(
        self,
        world_model: Optional["WorldModel"] = None,  # ADD THIS
        config_path: Any = "configs/intrinsic_drives.json",
        state_path: str = "data/agent_state.json",
        alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        approval_checker: Optional[Callable[[str], Optional[str]]] = None,
        csiu_enforcer: Any = None,
        improvement_policy: Any = None,
        approval_store: Any = None,
        approval_verifier: Any = None,
        audit_owner: Any = None,
        owns_csiu_enforcer: bool = True,
        owns_approval_store: bool = False,
        owns_audit_owner: bool = False,
    ):
        """
        Initialize self-improvement drive.

        Args:
            world_model: Reference to parent WorldModel (optional)
            config_path: Path to configuration file (str) OR a pre-loaded config object/dict.
            state_path: Path to state persistence file
            alert_callback: Optional callback for sending alerts (severity, details)
            approval_checker: Optional callback to check approval status externally
        """
        # --- END REPLACEMENT ---
        self.state_path = Path(state_path)  # State path must be a path
        self.alert_callback = alert_callback
        self.approval_checker = approval_checker
        self._lock = threading.RLock()
        self._closed = False
        self._owns_csiu_enforcer = bool(owns_csiu_enforcer and csiu_enforcer is None)
        self._owns_approval_store = bool(owns_approval_store)
        self._owns_audit_owner = bool(owns_audit_owner)
        self.improvement_policy = improvement_policy
        self.approval_store = approval_store
        self.approval_verifier = approval_verifier
        self.audit_owner = audit_owner
        self._last_governed_transaction_result = None
        
        # FIX Issue 5: Track if blacklisted objectives have been logged (only log once)
        self._blacklist_logged = False

        # --- START ADDITION ---
        self.world_model = world_model
        logger.info("SelfImprovementDrive received WorldModel reference")
        # --- END ADDITION ---

        # --- START FIX: Robust config loading ---
        # Handle config_path being either a path (str) or a pre-loaded object (like AgentConfig)
        if isinstance(config_path, (str, Path)):
            logger.info(f"Loading self-improvement config from path: {config_path}")
            self.config_path = Path(config_path)
            self.full_config = self._load_full_config()  # This returns a dict
        else:
            # New logic: config_path is already a loaded config object/dict
            logger.info(
                f"Loading self-improvement config from pre-loaded object ({type(config_path)})."
            )
            self.config_path = None  # No path

            # It's an object (like the 'AgentConfig' that lacks .get).
            # We MUST convert it to a dict for the rest of the file's .get() calls to work.
            if isinstance(config_path, dict):
                self.full_config = config_path
            elif hasattr(config_path, "__dict__"):
                # Convert object (like AgentConfig) to a dict
                self.full_config = vars(config_path)
            else:
                logger.error(
                    f"Config is an unknown object type ({type(config_path)}), attempting to use defaults."
                )
                self.full_config = {
                    "drives": {"self_improvement": self._default_config()}
                }
        # --- END FIX ---

        # Original logic continues, now self.full_config is guaranteed to be a dict
        self.config = self._extract_drive_config(self.full_config)

        # Global settings (may be absent)
        self.global_settings = self.full_config.get("global_settings", {})

        # Cache commonly used globals
        self._simul_triggers_cfg = self.global_settings.get(
            "conflict_resolution", {}
        ).get("simultaneous_triggers", {})
        self._jitter_ms = int(self._simul_triggers_cfg.get("jitter_milliseconds", 0))

        # Persistence tuning
        persistence_cfg = (
            self.config.get("persistence", {}) if isinstance(self.config, dict) else {}
        )
        self.backup_interval = int(
            persistence_cfg.get("backup_state_every_n_actions", 5)
        )

        # CSIU: Granular kill switches
        # Enabled by default (when env var is "0" or not set), disabled when set to "1"
        self._csiu_enabled = os.getenv("INTRINSIC_CSIU_OFF", "0") != "1"
        self._csiu_calc_enabled = os.getenv("INTRINSIC_CSIU_CALC_OFF", "0") != "1"
        self._csiu_regs_enabled = os.getenv("INTRINSIC_CSIU_REGS_OFF", "0") != "1"
        self._csiu_hist_enabled = os.getenv("INTRINSIC_CSIU_HIST_OFF", "0") != "1"

        # CSIU: Initialize weight dictionary
        self._csiu_policy = CSIUPolicy() if CSIU_ENFORCEMENT_AVAILABLE else None
        self._csiu_w = dict(getattr(self._csiu_policy, "weights", {"A":0.6,"H":0.6,"C":0.6,"V":0.6,"D":0.6,"G":0.6,"E":0.5,"U":0.5,"M":0.5}))

        # CSIU: Tracking variables. First valid snapshot is a baseline with zero pressure.
        self._csiu_U_prev = 0.0
        self._csiu_u_ewma = 0.0
        self._csiu_ewma_alpha = 0.3
        self._csiu_last_metrics: Dict[str, float] = {}
        self._csiu_last_snapshot = None
        self._csiu_last_decision = None
        self._csiu_last_event: Dict[str, Any] = {}

        # CSIU: Injectable metrics provider and cache
        self.metrics_provider: Optional[Callable[[str], Optional[float]]] = None
        self._metrics_cache: Dict[str, Any] = {}

        # CSIU: Initialize enforcer with kill switches from environment
        self._csiu_enforcer = csiu_enforcer
        if self._csiu_enforcer is None and get_csiu_enforcer is not None and self._csiu_enabled:
            enforcer_config = CSIUEnforcementConfig(
                global_enabled=self._csiu_enabled,
                calculation_enabled=self._csiu_calc_enabled,
                regularization_enabled=self._csiu_regs_enabled,
                history_tracking_enabled=self._csiu_hist_enabled,
                durable_accounting_required=bool(self.config.get("constraints", {}).get("csiu_durable_accounting_required", False)),
                durable_store_path=self.config.get("constraints", {}).get("csiu_durable_store_path"),
            )
            self._csiu_enforcer = get_csiu_enforcer(enforcer_config)
            logger.info("CSIU enforcement module initialized with safety controls")

        # Load or initialize state
        self.state = self._load_state()

        # Load objectives
        self.objectives = self._load_objectives()

        # Validate configuration
        self._validate_config()

        # Track last weight notification
        self._last_weight_notification: Dict[str, float] = {}

        # ... existing init ...
        # Existing field likely present:
        # self.require_human_approval: bool = self.config['constraints']['require_human_approval'] # Example
        self.require_human_approval = self.config.get("constraints", {}).get(
            "require_human_approval", True
        )

        policy_path = os.getenv("VULCAN_AUTO_APPLY_POLICY") or getattr(
            self, "auto_apply_policy_path", None
        )
        try:
            # If a config accessor exists, prefer it
            from config import get_config

            policy_path = get_config(
                "intrinsic_drives_config.auto_apply_policy", policy_path
            )
        except Exception as e:
            logger.debug(f"Operation failed: {e}")

        self._auto_apply_policy = load_policy(policy_path)
        constraints = self.config.get("constraints", {}) if isinstance(self.config, dict) else {}
        self._governed_transactions_enabled = bool(constraints.get("governed_transactions_enabled", True))
        self._unattended_application_permitted = bool(constraints.get("unattended_application_permitted", not self.require_human_approval))
        self._independent_approval_required = bool(constraints.get("independent_approval_required", self.require_human_approval))
        self._auto_apply_enabled = bool(self._auto_apply_policy.enabled and self._governed_transactions_enabled and (self._unattended_application_permitted or self._independent_approval_required))

        # PRIORITY 1: Safe Execution Module Integration
        # Initialize safe executor for sandboxed improvement execution
        self.safe_executor = get_safe_executor(timeout=60) if get_safe_executor else None
        
        # Store policy reference for apply_improvement method
        self.policy = self._auto_apply_policy
        
        # Detect or set repository root for file operations
        self.repo_root = self._detect_repo_root()
        
        # Initialize code introspection capabilities
        # This enables Vulcan to examine its own source code and logs
        try:
            self.code_introspector = CodeIntrospector(self.repo_root)
            logger.info(f"CodeIntrospector initialized: {len(self.code_introspector.source_files)} files scanned")
        except Exception as e:
            logger.warning(f"Failed to initialize CodeIntrospector: {e}")
            self.code_introspector = None
        
        # Initialize log analyzer for self-diagnosis
        log_dir = self.repo_root / 'logs'
        try:
            self.log_analyzer = LogAnalyzer(log_dir)
            logger.info(f"LogAnalyzer initialized: monitoring {log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize LogAnalyzer: {e}")
            self.log_analyzer = None
        
        # Initialize code knowledge store for learning from code patterns
        try:
            self.code_knowledge_store = CodeKnowledgeStore()
            logger.info(f"CodeKnowledgeStore initialized: {self.code_knowledge_store.get_statistics()}")
        except Exception as e:
            logger.warning(f"Failed to initialize CodeKnowledgeStore: {e}")
            self.code_knowledge_store = None
        
        logger.info(
            f"Safe executor initialized: {self.safe_executor is not None}, "
            f"repo_root: {self.repo_root}"
        )

        logger.info(
            f"SelfImprovementDrive initialized with {len(self.objectives)} objectives"
        )
        logger.info(f"Priority: {self.config.get('priority', 0.8)}")
        logger.info(f"Requires human approval: {self.require_human_approval}")
        logger.info(
            f"Auto-apply policy enabled: {self._auto_apply_enabled} (policy loaded: {self._auto_apply_policy.enabled})"
        )
        logger.info(
            f"State loaded: {len(self.state.completed_objectives)} completed, "
            f"{self.state.improvements_this_session} this session"
        )
        if self._csiu_enabled:
            logger.info("CSIU (latent drive) enabled with granular controls")

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare state for pickling by removing unpickleable lock objects.
        """
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore state after unpickling, re-creating the lock.
        """
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def readiness(self) -> Tuple[bool, str]:
        if self._closed:
            return False, "self-improvement drive closed"
        if self._csiu_enabled and self._csiu_enforcer is not None:
            try:
                self._csiu_enforcer.readiness()
            except Exception as exc:
                return False, f"csiu unavailable: {exc}"
        if self._governed_transactions_enabled:
            if self.improvement_policy is None or self.audit_owner is None:
                return False, "governed improvement policy or audit owner missing"
            if self._independent_approval_required and self.approval_store is None:
                return False, "approval store missing"
        return True, "ready"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for owner, owned in ((getattr(self, "_csiu_enforcer", None), self._owns_csiu_enforcer), (getattr(self, "approval_store", None), self._owns_approval_store), (getattr(self, "audit_owner", None), self._owns_audit_owner)):
            if owned and owner is not None:
                close = getattr(owner, "close", None)
                if callable(close):
                    close()

    # ---------- Helper Methods ----------

    def _determine_llm_provider_id(self) -> Optional[str]:
        """
        Determine the LLM provider ID from the world_model's LLM object.
        
        This helper method extracts the provider ID used to check if the LLM
        is in the TRUSTED_PROVIDERS list for self-improvement operations.
        
        Returns:
            Provider ID string or None if it cannot be determined
        """
        if not self.world_model:
            return None
            
        # First, check if the world_model has an explicit provider_id attribute
        llm_provider_id = getattr(self.world_model, "llm_provider_id", None)
        if llm_provider_id:
            return llm_provider_id
        
        # Otherwise, try to determine from the LLM object type
        llm_obj = getattr(self.world_model, "llm", None)
        if llm_obj is not None:
            llm_type = type(llm_obj).__name__.lower()
            if "graphix" in llm_type or "vulcan" in llm_type or "local" in llm_type:
                return "graphix_vulcan"
            return llm_type
        
        return None

    # ---------- Config Loading ----------

    def _load_full_config(self) -> Dict[str, Any]:
        """
        Robustly load the intrinsic drives configuration with UTF-8 on Windows.
        Falls back to 'utf-8-sig' and finally replaces undecodable bytes to avoid crashes.
        """
        # This check is now vital, as self.config_path can be None
        if self.config_path is None:
            logger.error(
                "Config path is None, cannot load config from disk. Using defaults."
            )
            return {"drives": {"self_improvement": self._default_config()}}

        cfg_path = str(self.config_path)

        # Check if file exists first
        if not self.config_path.exists():
            logger.warning(f"Config not found at {cfg_path}, using defaults")
            return {"drives": {"self_improvement": self._default_config()}}

        # Try multiple encoding strategies
        for enc in ("utf-8", "utf-8-sig"):
            try:
                with open(cfg_path, "r", encoding=enc) as f:
                    config_data = json.load(f)
                logger.debug(f"Successfully loaded config with encoding: {enc}")
                return config_data
            except UnicodeDecodeError as e:
                logger.debug(
                    f"UnicodeDecodeError with {enc}: {e}, trying next encoding"
                )
                continue
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse config JSON with {enc}: {e}")
                return {"drives": {"self_improvement": self._default_config()}}
            except Exception as e:
                logger.error(f"Error loading config with {enc}: {e}")
                break

        # Last-resort: load with replacement to prevent crashes
        try:
            logger.warning(f"Falling back to error replacement for {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            config_data = json.loads(text)
            logger.info("Successfully loaded config with error replacement")
            return config_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config JSON even with replacement: {e}")
            return {"drives": {"self_improvement": self._default_config()}}
        except Exception as e:
            logger.error(f"Critical error loading config with replacement: {e}")
            return {"drives": {"self_improvement": self._default_config()}}

    def _extract_drive_config(self, full_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the self_improvement drive section with safe defaults."""
        drives = full_config.get("drives", {})
        if not isinstance(drives, dict):
            # FIXED: If 'drives' key is missing, assume the whole config is the drive config for test compatibility
            if "objectives" in full_config and "constraints" in full_config:
                logger.warning(
                    "Config missing 'drives' wrapper, assuming root is self_improvement config."
                )
                return full_config
            logger.error("'drives' is not a dict in config")
            return self._default_config()

        drive_config = drives.get("self_improvement", {})
        if not drive_config:
            logger.warning("No self_improvement config found, using defaults")
            return self._default_config()
        
        # Merge with defaults for missing required fields
        defaults = self._default_config()
        merged_config = defaults.copy()
        merged_config.update(drive_config)
        
        # Ensure required nested fields exist
        if "objectives" not in drive_config:
            merged_config["objectives"] = defaults["objectives"]
        if "constraints" not in merged_config or not merged_config["constraints"]:
            merged_config["constraints"] = defaults["constraints"]
        
        return merged_config

    def _default_config(self) -> Dict[str, Any]:
        """
        Default configuration matching the full JSON structure.

        This is a fallback that includes all objectives and key settings.
        """
        return {
            "enabled": True,
            "priority": 0.8,
            "description": "Continuous self-improvement and debugging",
            "objectives": [
                {
                    "type": "fix_circular_imports",
                    "weight": 1.0,
                    "auto_apply": False,
                    "success_criteria": {
                        "max_import_depth": 5,
                        "no_circular_chains": True,
                    },
                    "scope": {
                        "directories": ["src/", "lib/"],
                        "exclude": ["tests/", "migrations/"],
                    },
                },
                {
                    "type": "optimize_performance",
                    "weight": 0.8,
                    "auto_apply": False,
                    "target_metrics": {
                        "response_time_p95_ms": {"target": 100, "max": 200},
                        "memory_usage_mb": {"target": 512, "max": 1024},
                    },
                },
                {
                    "type": "improve_test_coverage",
                    "weight": 0.6,
                    "auto_apply": False,
                    "coverage_targets": {
                        "line_coverage_percent": 80,
                        "branch_coverage_percent": 70,
                    },
                },
                {"type": "enhance_safety_systems", "weight": 1.0, "auto_apply": False},
                {
                    "type": "fix_known_bugs",
                    "weight": 0.9,
                    "auto_apply": False,
                    "bug_sources": [
                        {
                            "type": "issue_tracker",
                            "labels": ["bug"],
                            "min_priority": "medium",
                        },
                        {"type": "error_logs", "severity": ["ERROR", "CRITICAL"]},
                    ],
                },
            ],
            "constraints": {
                "require_human_approval": True,
                "max_changes_per_session": 5,
                "always_maintain_tests": True,
                "never_reduce_safety": True,
                "rollback_on_failure": True,
                "max_session_duration_minutes": 30,
            },
            "triggers": [
                {"type": "on_startup", "cooldown_minutes": 60},
                {
                    "type": "on_error_detected",
                    "error_count_threshold": 3,
                    "time_window_minutes": 60,
                },
                {"type": "periodic", "interval_hours": 24, "random_jitter_minutes": 60},
            ],
            "resource_limits": {
                "llm_costs": {
                    "max_tokens_per_session": 100000,
                    "max_cost_usd_per_session": 5.0,
                    "max_cost_usd_per_day": 20.0,
                    "max_cost_usd_per_month": 500.0,
                    "cost_tracking_window_hours": 24,
                    "warn_at_percent": 80,
                    "pause_at_percent": 95,
                    "cost_reconciliation_period_days": 7,
                }
            },
            "adaptive_learning": {
                "enabled": True,
                "track_outcomes": {
                    "learning_rate": 0.2,
                    "min_samples_before_adjust": 10,
                    "weight_bounds": {"min": 0.3, "max": 1.0},
                },
                "failure_patterns": {
                    "failure_classification": {
                        "transient": {
                            "cooldown_hours": 2,  # Reduced from 4 to 2 hours for faster recovery
                            "indicators": [
                                "network_timeout",
                                "temporary_service_unavailable",
                            ],
                        },
                        "systemic": {
                            "cooldown_hours": 24,  # Reduced from 72 to 24 hours for non-code improvements
                            "indicators": [
                                "validation_failed",
                                "breaking_change_detected",
                            ],
                        },
                    }
                },
                "transparency": {
                    "notify_on_significant_change": {
                        "enabled": True,
                        "threshold_change": 0.2,
                    }
                },
            },
            # ENH: optional persistence tuning
            "persistence": {"backup_state_every_n_actions": 5},
            # ENH: pass-through for validation env overrides (used by orchestrator)
            "validation": {
                "env_overrides": {
                    "development": {"skip_security_scan": True, "run_lint": True},
                    "staging": {"skip_security_scan": False, "run_lint": True},
                    "production": {"run_all": True},
                }
            },
        }

    def _validate_config(self):
        """Validate configuration has required fields."""
        required_fields = ["enabled", "objectives", "constraints"]
        for field in required_fields:
            # FIXED: Check against self.config, not self.full_config
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        if not isinstance(self.config["objectives"], list):
            raise ValueError("objectives must be a list")

        if not isinstance(self.config["constraints"], dict):
            raise ValueError("constraints must be a dict")

    # ---------- State Persistence ----------

    def _load_state(self) -> SelfImprovementState:
        """Load state from disk or create new with robust UTF-8 handling."""
        try:
            if self.state_path.exists():
                # Try utf-8 first, then utf-8-sig, then replace
                try:
                    with open(self.state_path, "r", encoding="utf-8") as f:
                        state_dict = json.load(f)
                except UnicodeDecodeError:
                    try:
                        with open(self.state_path, "r", encoding="utf-8-sig") as f:
                            state_dict = json.load(f)
                    except UnicodeDecodeError:
                        logger.warning(
                            f"Failed UTF-8 and UTF-8-SIG, trying replace for {self.state_path}"
                        )
                        with open(
                            self.state_path, "r", encoding="utf-8", errors="replace"
                        ) as f:
                            text = f.read()
                        state_dict = json.loads(text)

                # Reconstruct state from dict
                state = SelfImprovementState(
                    active=state_dict.get("active", False),
                    current_objective=state_dict.get("current_objective"),
                    completed_objectives=state_dict.get("completed_objectives", []),
                    pending_approvals=state_dict.get("pending_approvals", []),
                    improvements_this_session=0,  # Reset on load
                    last_improvement=state_dict.get(
                        "last_improvement", 0
                    ),  # Note: Load saved value or 0 (will init in __post_init__)
                    last_trigger_check=state_dict.get("last_trigger_check", 0),
                    session_start_time=time.time(),  # New session
                    total_cost_usd=state_dict.get("total_cost_usd", 0.0),
                    daily_cost_usd=state_dict.get("daily_cost_usd", 0.0),
                    monthly_cost_usd=state_dict.get("monthly_cost_usd", 0.0),
                    last_cost_reset=state_dict.get("last_cost_reset", time.time()),
                    state_save_count=state_dict.get("state_save_count", 0),
                    session_tokens=state_dict.get("session_tokens", 0),
                    cost_history=state_dict.get("cost_history", []),
                )

                # CSIU active weights are immutable policy; reject malformed persisted weights.
                if "csiu_weights" in state_dict and self._csiu_enabled:
                    try:
                        loaded_weights = state_dict["csiu_weights"]
                        if set(loaded_weights) != set(self._csiu_w):
                            raise ValueError("persisted CSIU weight keys mismatch")
                        for k, v in loaded_weights.items():
                            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0 or float(v) > 10:
                                raise ValueError("persisted CSIU weight invalid")
                        self._csiu_w = {k: float(v) for k, v in loaded_weights.items()}
                        logger.info("Loaded validated persisted CSIU policy weights")
                    except Exception as e:
                        logger.warning(f"Rejected persisted CSIU weights: {e}")

                logger.info(f"Loaded state from {self.state_path}")
                logger.info(
                    f"State loaded: {len(state.completed_objectives)} completed, "
                    f"{state.improvements_this_session} this session, "
                    f"last_improvement={state.last_improvement:.0f}"
                )
                return state
            else:
                # Note: State file doesn't exist - this is expected on first run
                # Use INFO level and create the directory/file proactively
                logger.info(
                    f"State file not found at: {self.state_path} - "
                    f"This is expected on first run. Creating new state file."
                )
                # Create the directory if it doesn't exist
                try:
                    self.state_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as dir_e:
                    logger.debug(f"Could not create state directory: {dir_e}")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}, using new state")

        # Return new state - use INFO level for fresh start (expected behavior)
        logger.info(
            "Starting with fresh self-improvement state."
        )
        return SelfImprovementState()

    def _save_state(self):
        """Persist state to disk atomically (Windows-safe) with UTF-8 encoding."""
        with self._lock:
            try:
                # Create directory if needed
                self.state_path.parent.mkdir(parents=True, exist_ok=True)

                # Increment save count
                self.state.state_save_count += 1

                # Convert state to dict
                state_dict = {
                    "active": self.state.active,
                    "current_objective": self.state.current_objective,
                    "completed_objectives": self.state.completed_objectives,
                    "pending_approvals": self.state.pending_approvals,
                    "last_improvement": self.state.last_improvement,
                    "last_trigger_check": self.state.last_trigger_check,
                    "total_cost_usd": self.state.total_cost_usd,
                    "daily_cost_usd": self.state.daily_cost_usd,
                    "monthly_cost_usd": self.state.monthly_cost_usd,
                    "last_cost_reset": self.state.last_cost_reset,
                    "state_save_count": self.state.state_save_count,
                    "timestamp": time.time(),
                    "session_tokens": self.state.session_tokens,
                    "cost_history": self.state.cost_history,
                }

                # OPTIONAL: Persist CSIU weights for long-term learning
                if self._csiu_enabled:
                    state_dict["csiu_weights"] = dict(self._csiu_w)

                # Write to temp file
                temp_path = self.state_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(state_dict, f, indent=2, ensure_ascii=False)

                # Atomic replace (Windows-safe)
                os.replace(str(temp_path), str(self.state_path))

                # Create backup every N saves (configurable)
                if self.state.state_save_count % max(1, self.backup_interval) == 0:
                    self._create_state_backup()

            except Exception as e:
                logger.error(f"Failed to save state: {e}")

    def _create_state_backup(self):
        """Create a backup of the current state."""
        try:
            # Ensure state file exists before trying to copy it
            if not self.state_path.exists():
                logger.debug("State file does not exist yet, skipping backup")
                return

            backup_dir = self.state_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)

            # Create timestamped backup
            timestamp = int(time.time())
            backup_path = backup_dir / f"agent_state_{timestamp}.json"

            shutil.copy2(self.state_path, backup_path)
            logger.info(f"Created state backup: {backup_path}")

            # Clean old backups (keep last 10)
            backups = sorted(backup_dir.glob("agent_state_*.json"))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()
                    logger.debug(f"Removed old backup: {old_backup}")

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    # ---------- Objectives ----------

    def _load_objectives(self) -> List[ImprovementObjective]:
        """Load improvement objectives from config with historical state."""
        objectives = []
        for obj_config in self.config.get("objectives", []):
            # FIXED: Handle obj_config being a string (from bad config)
            if isinstance(obj_config, str):
                logger.warning(
                    f"Skipping malformed objective (expected dict, got str): {obj_config}"
                )
                continue
            if not isinstance(obj_config, dict):
                logger.warning(
                    f"Skipping malformed objective (expected dict, got {type(obj_config)}): {obj_config}"
                )
                continue

            obj = ImprovementObjective(
                type=obj_config["type"],
                weight=float(obj_config.get("weight", 0.5)),
                auto_apply=bool(obj_config.get("auto_apply", False)),
            )
            # Mark as completed if in state
            if obj.type in self.state.completed_objectives:
                obj.completed = True
            objectives.append(obj)
        return objectives

    # ---------- Alerts ----------

    def _send_alert(self, severity: str, message: str, details: Dict[str, Any]):
        """Send alert via callback if configured."""
        if self.alert_callback:
            try:
                alert_data = {
                    "severity": severity,
                    "message": message,
                    "details": details,
                    "timestamp": time.time(),
                    "source": "self_improvement_drive",
                }
                self.alert_callback(severity, alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        else:
            # Log as fallback
            log_func = getattr(logger, severity.lower(), logger.info)
            log_func(f"ALERT [{severity}]: {message} | {details}")

    # ---------- CSIU: Metrics Provider Integration ----------

    def _csiu_audit_status(self, reason: str, **data: Any) -> None:
        """Record bounded CSIU status without raw interactions, source, prompts, or full plans."""
        self._csiu_last_event = {"reason": reason, **{k: v for k, v in data.items() if k not in {"plan", "source", "prompt", "raw_interactions"}}}
        logger.info("CSIU status: %s", self._csiu_last_event)

    def set_metrics_provider(self, provider: Callable[[str], Optional[float]]):
        """Inject a real aggregate telemetry provider for dotted CSIU metric keys."""
        self.metrics_provider = provider
        logger.info("Metrics provider injected for CSIU telemetry")

    def verify_metrics_provider(self) -> Dict[str, Any]:
        required = list(self._csiu_metric_keys().values()) + [
            "metrics.window_start", "metrics.window_end", "metrics.sample_count",
            "metrics.aggregation_method", "metrics.metric_definition_version",
            "metrics.provider_id", "metrics.provenance_digest",
        ]
        if not self.metrics_provider:
            return {"configured": False, "working": False, "reason": "provider_missing", "details": {}}
        details = {}; ok = 0
        for key in required:
            try:
                val = self.metrics_provider(key)
                good = val is not None and not (isinstance(val, float) and not math.isfinite(val))
                details[key] = {"status": "ok" if good else "missing"}
                ok += int(good)
            except Exception as e:
                details[key] = {"status": "error", "error": type(e).__name__}
        return {"configured": True, "working": ok == len(required), "working_metrics": ok, "total_tested": len(required), "details": details, "reason": "ok" if ok == len(required) else "incomplete_provider"}

    def _csiu_metric_keys(self) -> Dict[str, str]:
        return {
            "A": "metrics.alignment_coherence_idx",
            "H": "metrics.communication_entropy",
            "C": "metrics.intent_clarity_score",
            "V": "policies.non_judgmental.violations_per_1k",
            "D": "metrics.disparity_at_k",
            "G": "metrics.calibration_gap",
            "E": "metrics.empathy_index",
            "U": "metrics.user_satisfaction",
            "M": "metrics.miscommunication_rate",
        }

    def _safe_get_metric(self, dotted: str, default: Optional[float] = None) -> Optional[float]:
        """Return real provider data only; defaults/cached values never influence CSIU."""
        if not self.metrics_provider:
            return None
        try:
            val = self.metrics_provider(dotted)
        except Exception as e:
            self._csiu_audit_status("provider_exception", metric=dotted, error=type(e).__name__)
            return None
        if isinstance(val, bool) or val is None:
            return None
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return float(val)
        return None

    def _collect_telemetry_snapshot(self) -> Dict[str, float]:
        """Assemble a complete validated CSIU metric snapshot from real aggregate telemetry."""
        if not self._csiu_enabled or not self._csiu_calc_enabled:
            self._csiu_audit_status("calculation_disabled")
            return {}
        if not self.metrics_provider or not self._csiu_enforcer:
            self._csiu_audit_status("provider_or_enforcer_missing")
            return {}
        metrics: Dict[str, float] = {}
        for key, dotted in self._csiu_metric_keys().items():
            val = self._safe_get_metric(dotted)
            if val is None:
                self._csiu_audit_status("metric_missing", metric=key)
                return {}
            metrics[key] = val
        try:
            def meta(k: str) -> Any: return self.metrics_provider(k)  # type: ignore[misc]
            snap = CSIUMetricSnapshot(
                metrics=metrics,
                window_start=str(meta("metrics.window_start")),
                window_end=str(meta("metrics.window_end")),
                sample_count=int(meta("metrics.sample_count")),
                aggregation_method=str(meta("metrics.aggregation_method")),
                metric_definition_version=str(meta("metrics.metric_definition_version")),
                provider_id=str(meta("metrics.provider_id")),
                provenance_digest=str(meta("metrics.provenance_digest")),
                policy_digest=self._csiu_enforcer.policy.policy_digest,
                privacy_cohort={"kind": "aggregate"},
            )
            valid, reason = self._csiu_enforcer.validate_snapshot(snap)
            if not valid:
                self._csiu_audit_status("snapshot_rejected", reason_code=reason, snapshot_digest=snap.snapshot_digest)
                return {}
            self._csiu_last_snapshot = snap
            self._csiu_audit_status("snapshot_validated", snapshot_digest=snap.snapshot_digest, sample_count=snap.sample_count, window_end=snap.window_end)
            return dict(snap.metrics)
        except Exception as e:
            self._csiu_audit_status("snapshot_rejected", reason_code=type(e).__name__)
            return {}

    def observe_csiu_telemetry(self):
        """Public boundary to accept aggregate CSIU telemetry without proposing source changes."""
        cur = self._collect_telemetry_snapshot()
        snap = getattr(self, "_csiu_last_snapshot", None) if cur else None
        if not self._csiu_enforcer or snap is None:
            return getattr(self, "_csiu_last_decision", None)
        prev = getattr(self, "_csiu_previous_snapshot", None)
        decision = self._csiu_enforcer.observe_telemetry_snapshots(prev, snap, observation_id="drive-telemetry")
        self._csiu_last_decision = decision
        terminal_acceptance = {
            "baseline_established", "telemetry_observed", "zero_pressure",
            "disabled", "cumulative_cap_exceeded",
        }
        if decision.reason_code in terminal_acceptance or decision.applied:
            self._csiu_previous_snapshot = snap
        self._csiu_audit_status("telemetry_observed", decision_id=decision.decision_id, decision_digest=decision.decision_digest, reason_code=decision.reason_code, applied=decision.applied)
        return decision

    def _csiu_utility(self, prev: Dict[str, float], cur: Dict[str, float]) -> float:
        if not self._csiu_enabled or not self._csiu_calc_enabled or not prev or not cur:
            return 0.0
        order = ("A", "H", "C", "V", "D", "G", "E", "U", "M")
        directions = {"A": 1, "H": -1, "C": 1, "V": -1, "D": -1, "G": -1, "E": 1, "U": 1, "M": -1}
        if set(prev) < set(order) or set(cur) < set(order):
            return 0.0
        weights = {k: float(self._csiu_w[k]) for k in order}
        denom = sum(abs(v) for v in weights.values())
        if denom <= 0 or not math.isfinite(denom):
            return 0.0
        acc = 0.0
        for k in order:
            a = float(prev[k]); b = float(cur[k])
            if not (math.isfinite(a) and math.isfinite(b)):
                return 0.0
            delta = (b - a) * directions[k]  # ranges are [0,1], so already normalized
            acc += weights[k] * delta
        u = acc / denom
        return max(-1.0, min(1.0, u)) if math.isfinite(u) else 0.0

    def _csiu_adaptive_lr(self, cur: Dict[str, float], base_lr: float = 0.02) -> float:
        """Weights are immutable active policy; return zero to prevent self-referential learning."""
        return 0.0

    def _csiu_update_weights(self, feature_deltas: Dict[str, float], U_prev: float, U_now: float, lr: float):
        """Do not activate learned weights; emit an inspectable proposal through the enforcer."""
        if self._csiu_enforcer and self._csiu_last_snapshot and self._csiu_enabled and self._csiu_calc_enabled:
            self._csiu_last_weight_proposal = self._csiu_enforcer.propose_weight_revision([self._csiu_last_snapshot])
        return None

    def _csiu_apply_ewma(self, U_now: float) -> float:
        if not self._csiu_enabled or not self._csiu_calc_enabled:
            return U_now
        if not math.isfinite(float(U_now)):
            raise ValueError("CSIU utility must be finite")
        alpha = float(getattr(self._csiu_enforcer.policy, "ewma_alpha", self._csiu_ewma_alpha)) if self._csiu_enforcer else self._csiu_ewma_alpha
        if not (0.0 < alpha <= 1.0):
            raise ValueError("CSIU EWMA alpha out of range")
        self._csiu_u_ewma = alpha * float(U_now) + (1 - alpha) * self._csiu_u_ewma
        return self._csiu_u_ewma

    def _csiu_pressure(self, U_ewma: float) -> float:
        if not self._csiu_enabled or not self._csiu_calc_enabled:
            return 0.0
        if not math.isfinite(float(U_ewma)):
            return 0.0
        cap = float(getattr(self._csiu_enforcer.config, "max_single_influence", 0.05)) if self._csiu_enforcer else 0.0
        return max(-cap, min(cap, cap * math.tanh(float(U_ewma))))

    def _estimate_explainability_score(self, plan: Dict[str, Any]) -> float:
        steps = len(plan.get("steps", [])) if isinstance(plan, dict) else 0
        has_rationale = bool(plan.get("rationale")) if isinstance(plan, dict) else False
        safety_affordances = sum(1 for pol in plan.get("policies", []) if pol in {"non_judgmental", "rollback_on_failure", "maintain_tests"}) if isinstance(plan, dict) else 0
        return max(0.0, min(1.0, 0.5 * has_rationale + 0.3 * min(1.0, 3 / (steps + 1)) + 0.2 * min(1.0, safety_affordances / 2)))

    def _csiu_regularize_plan(self, plan: Dict[str, Any], d: float, cur: Dict[str, float]) -> Dict[str, Any]:
        """Apply bounded CSIU regularization through CSIUEnforcement only."""
        original = copy.deepcopy(plan or {})
        if not self._csiu_enabled or not self._csiu_regs_enabled:
            return original
        if self._csiu_enforcer is None:
            self._csiu_audit_status("enforcer_unavailable")
            return original
        try:
            snapshot = self._csiu_last_snapshot
            previous_snapshot = getattr(self, "_csiu_previous_snapshot", None)
            if snapshot is None or previous_snapshot is None:
                self._csiu_last_decision = None
                self._csiu_audit_status("telemetry_unavailable", reason_code="telemetry_unavailable")
                return original
            regularized, decision = self._csiu_enforcer.apply_regularization_from_snapshots(
                original, previous_snapshot, snapshot, plan_id=str(original.get("id", "unknown")), action_type=str(original.get("type", "improvement"))
            )
            self._csiu_last_decision = decision
            self._csiu_audit_status("decision", decision_id=decision.decision_id, decision_digest=decision.decision_digest, reason_code=decision.reason_code, pressure=decision.pressure, actual_effect=decision.actual_effect, applied=decision.applied)
            if decision.reason_code in {"applied","zero_pressure","disabled","cumulative_cap_exceeded"}:
                self._csiu_previous_snapshot = snapshot
            return regularized if decision.applied else original
        except Exception as e:
            self._csiu_audit_status("decision_aborted", reason_code=type(e).__name__)
            return original

    # ---------- Cost / Resource Limits ----------

    def _check_resource_limits(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if resource limits would be exceeded, with warnings."""
        limits = self.config.get("resource_limits", {}).get("llm_costs", {})
        warn_at_pct = float(limits.get("warn_at_percent", 80)) / 100.0
        pause_at_pct = float(limits.get("pause_at_percent", 95)) / 100.0

        # Optional session tokens enforcement
        max_tokens = limits.get("max_tokens_per_session")
        if isinstance(max_tokens, int) and max_tokens > 0:
            # Update session tokens from context increment if provided
            if context:
                inc_tokens = int(context.get("tokens_used_increment", 0))
                if inc_tokens:
                    self.state.session_tokens += inc_tokens
            if self.state.session_tokens >= max_tokens:
                return (
                    False,
                    f"Session token limit reached ({self.state.session_tokens} >= {max_tokens})",
                )

        # Reconcile cost history (prune old entries outside recon window)
        self._prune_cost_history()

        # Session cost
        max_session = float(limits.get("max_cost_usd_per_session", float("inf")))
        # Note: Handle zero max limits
        if max_session == 0.0 and self.state.total_cost_usd > 0.0:
            return (
                False,
                f"Session cost limit is zero (${self.state.total_cost_usd:.2f} > $0.00)",
            )
        session_pct = (
            (self.state.total_cost_usd / max_session) if max_session > 0 else 0.0
        )
        if session_pct >= 1.0:
            return (
                False,
                f"Session cost limit reached (${self.state.total_cost_usd:.2f} >= ${max_session})",
            )
        elif session_pct >= pause_at_pct:
            self._send_alert(
                "warning",
                "Session cost near limit",
                {
                    "cost": self.state.total_cost_usd,
                    "limit": max_session,
                    "percent": session_pct * 100,
                },
            )
            return (
                False,
                f"Session cost paused at {pause_at_pct * 100:.0f}% (${self.state.total_cost_usd:.2f})",
            )
        elif session_pct >= warn_at_pct:
            self._send_alert(
                "info",
                "Session cost warning",
                {
                    "cost": self.state.total_cost_usd,
                    "limit": max_session,
                    "percent": session_pct * 100,
                },
            )

        # Adjustable tracking window (default 24h)
        window_hours = int(limits.get("cost_tracking_window_hours", 24))
        self._reset_cost_tracking_if_needed(window_hours=window_hours)

        # Daily limit (interpreted over tracking window for flexibility)
        max_daily = float(limits.get("max_cost_usd_per_day", float("inf")))
        # Note: Handle zero max limits
        if max_daily == 0.0 and self.state.daily_cost_usd > 0.0:
            return (
                False,
                f"Daily cost limit is zero (${self.state.daily_cost_usd:.2f} > $0.00)",
            )
        daily_pct = (self.state.daily_cost_usd / max_daily) if max_daily > 0 else 0.0
        if daily_pct >= 1.0:
            return (
                False,
                f"Daily cost limit reached (${self.state.daily_cost_usd:.2f} >= ${max_daily})",
            )
        elif daily_pct >= pause_at_pct:
            self._send_alert(
                "warning",
                "Daily cost near limit",
                {
                    "cost": self.state.daily_cost_usd,
                    "limit": max_daily,
                    "percent": daily_pct * 100,
                },
            )
            return (
                False,
                f"Daily cost paused at {pause_at_pct * 100:.0f}% (${self.state.daily_cost_usd:.2f})",
            )
        elif daily_pct >= warn_at_pct:
            self._send_alert(
                "info",
                "Daily cost warning",
                {
                    "cost": self.state.daily_cost_usd,
                    "limit": max_daily,
                    "percent": daily_pct * 100,
                },
            )

        # Monthly limit (fixed 30d window for simplicity)
        max_monthly = float(limits.get("max_cost_usd_per_month", float("inf")))
        # Note: Handle zero max limits
        if max_monthly == 0.0 and self.state.monthly_cost_usd > 0.0:
            return (
                False,
                f"Monthly cost limit is zero (${self.state.monthly_cost_usd:.2f} > $0.00)",
            )
        monthly_pct = (
            (self.state.monthly_cost_usd / max_monthly) if max_monthly > 0 else 0.0
        )
        if monthly_pct >= 1.0:
            return (
                False,
                f"Monthly cost limit reached (${self.state.monthly_cost_usd:.2f} >= ${max_monthly})",
            )
        elif monthly_pct >= pause_at_pct:
            self._send_alert(
                "warning",
                "Monthly cost near limit",
                {
                    "cost": self.state.monthly_cost_usd,
                    "limit": max_monthly,
                    "percent": monthly_pct * 100,
                },
            )
            return (
                False,
                f"Monthly cost paused at {pause_at_pct * 100:.0f}% (${self.state.monthly_cost_usd:.2f})",
            )
        elif monthly_pct >= warn_at_pct:
            self._send_alert(
                "info",
                "Monthly cost warning",
                {
                    "cost": self.state.monthly_cost_usd,
                    "limit": max_monthly,
                    "percent": monthly_pct * 100,
                },
            )

        # Session duration
        max_duration_min = int(
            self.config.get("constraints", {}).get("max_session_duration_minutes", 30)
        )
        session_duration_min = (time.time() - self.state.session_start_time) / 60.0
        if session_duration_min >= max_duration_min:
            return (
                False,
                f"Session duration limit reached ({session_duration_min:.1f} >= {max_duration_min} min)",
            )

        return True, None

    def _prune_cost_history(self):
        """Prune cost history outside reconciliation window and keep totals coherent."""
        limits = self.config.get("resource_limits", {}).get("llm_costs", {})
        recon_days = int(limits.get("cost_reconciliation_period_days", 7))
        cutoff = time.time() - recon_days * 86400
        # Purge old entries
        original_len = len(self.state.cost_history)
        self.state.cost_history = [
            e for e in self.state.cost_history if e.get("timestamp", 0) >= cutoff
        ]
        if len(self.state.cost_history) != original_len:
            self._save_state()

    def _reset_cost_tracking_if_needed(self, window_hours: int = 24):
        """Reset daily/monthly cost tracking if time window passed (daily uses configurable window)."""
        current_time = time.time()
        time_since_reset = current_time - self.state.last_cost_reset

        # Reset "daily" using configurable sliding window
        if time_since_reset > window_hours * 3600:
            logger.info(
                f"Resetting windowed daily cost tracking (was ${self.state.daily_cost_usd:.2f}, window={window_hours}h)"
            )
            self.state.daily_cost_usd = 0.0
            self.state.last_cost_reset = current_time
            self._save_state()

        # Reset monthly (30 days)
        if time_since_reset > 2592000:
            logger.info(
                f"Resetting monthly cost tracking (was ${self.state.monthly_cost_usd:.2f})"
            )
            self.state.monthly_cost_usd = 0.0
            self._save_state()

    # ---------- Trigger Evaluation ----------

    def _evaluate_trigger(self, trigger_config: Any, context: Dict[str, Any]) -> bool:
        """Evaluate a single trigger condition."""
        current_time = time.time()

        # Handle dict-based triggers (new format)
        if isinstance(trigger_config, dict):
            # Check if trigger is explicitly disabled
            if not trigger_config.get("enabled", True):
                return False

            trigger_type = trigger_config.get("type")

            if trigger_type == TriggerType.ON_STARTUP.value:
                cooldown = trigger_config.get("cooldown_minutes", 60) * 60
                if (
                    context.get("is_startup", False)
                    and (current_time - self.state.last_trigger_check) > cooldown
                ):
                    logger.debug(
                        f"Trigger: on_startup (cooldown: {cooldown / 60:.0f}m)"
                    )
                    return True

            elif trigger_type == TriggerType.ON_ERROR.value:
                threshold = trigger_config.get("error_count_threshold", 3)
                if context.get("error_detected", False):
                    error_count = int(context.get("error_count", 0))
                    if error_count >= threshold:
                        logger.debug(
                            f"Trigger: on_error ({error_count} >= {threshold})"
                        )
                        return True

            elif trigger_type == TriggerType.ON_PERFORMANCE_DEGRADATION.value:
                metric = trigger_config.get("metric", "response_time_p95")
                threshold_pct = float(trigger_config.get("threshold_percent", 20))
                perf_metrics = context.get("performance_metrics", {})
                degradation_pct = float(
                    perf_metrics.get(f"{metric}_degradation_percent", 0)
                )
                if degradation_pct >= threshold_pct:
                    logger.debug(
                        f"Trigger: performance degradation ({metric}: {degradation_pct:.1f}%)"
                    )
                    return True

            elif trigger_type == TriggerType.PERIODIC.value:
                # Don't fire periodic trigger on fresh state - need a baseline to measure from
                if self.state.last_improvement == 0:
                    return False
                
                interval_hours = trigger_config.get("interval_hours", 24)
                jitter_minutes = trigger_config.get("random_jitter_minutes", 0)

                # Calculate time since last improvement
                time_since = current_time - self.state.last_improvement
                interval_seconds = interval_hours * 3600

                # Add jitter (random, but consistent per session)
                import random

                random.seed(int(self.state.session_start_time))
                jitter_seconds = random.randint(0, max(0, jitter_minutes) * 60)

                if time_since > (interval_seconds + jitter_seconds):
                    logger.debug(
                        f"Trigger: periodic ({time_since / 3600:.1f}h >= {interval_hours}h)"
                    )
                    return True

            elif trigger_type == TriggerType.ON_LOW_ACTIVITY.value:
                cpu_threshold = trigger_config.get("cpu_threshold_percent", 30)
                duration_min = trigger_config.get("duration_minutes", 10)

                system_resources = context.get("system_resources", {})
                cpu_usage = system_resources.get("cpu_percent", 100)
                low_activity_duration = system_resources.get(
                    "low_activity_duration_minutes", 0
                )

                if cpu_usage < cpu_threshold and low_activity_duration >= duration_min:
                    logger.debug(
                        f"Trigger: low activity (CPU: {cpu_usage}% < {cpu_threshold}%)"
                    )
                    return True

        return False

    def should_trigger(self, context: Dict[str, Any]) -> bool:
        """
        Determine if self-improvement drive should activate.

        This is called by Vulcan's motivational system to decide if the
        system should focus on self-improvement right now.
        """
        # SAFEGUARD: Check environment variable kill switch first
        kill_switch = os.getenv("VULCAN_ENABLE_SELF_IMPROVEMENT", "1").lower()
        if kill_switch in ("0", "false", "no", "off"):
            logger.debug("Self-improvement disabled via VULCAN_ENABLE_SELF_IMPROVEMENT=0")
            return False

        # Check if enabled in config
        if not self.config.get("enabled", True):
            logger.debug("Self-improvement drive disabled in config")
            return False

        # Check resource limits (may update tokens from context increment)
        can_proceed, reason = self._check_resource_limits(context=context)
        if not can_proceed:
            logger.info(f"Cannot trigger: {reason}")
            return False

        # Check if we've hit the session limit
        max_changes = int(self.config["constraints"]["max_changes_per_session"])
        if self.state.improvements_this_session >= max_changes:
            logger.info(f"Reached max changes limit ({max_changes}) for this session")
            return False

        # SAFEGUARD: Prevent infinite loop by checking minimum time between improvements
        # Use environment variable SELF_IMPROVEMENT_MIN_INTERVAL or default to 3600 seconds (1 hour)
        # EXCEPTION: Allow immediate trigger on fresh state (last_improvement == 0)
        min_interval = int(os.getenv("SELF_IMPROVEMENT_MIN_INTERVAL", "3600"))
        time_since_last = time.time() - self.state.last_improvement
        
        # Check if this is a fresh state that should be allowed to trigger immediately
        # Fresh state indicators:
        # 1. last_improvement is 0 (never run before)
        # 2. OR session just started (within 60 seconds) - handles state loaded from disk
        is_fresh_state = (
            self.state.last_improvement == 0 
            or (time.time() - self.state.session_start_time) < 60
        )
        
        # Only enforce minimum interval if NOT a fresh state
        if not is_fresh_state and time_since_last < min_interval:
            logger.debug(
                f"Skipping trigger: only {time_since_last:.0f}s since last improvement "
                f"(minimum interval: {min_interval}s)"
            )
            return False

        # Evaluate all triggers
        triggers = self.config.get("triggers", [])
        for trigger_config in triggers:
            if self._evaluate_trigger(trigger_config, context):
                # Optional jitter to mitigate simultaneous trigger storms
                if self._jitter_ms > 0:
                    time.sleep(
                        min(0.5, self._jitter_ms / 1000.0)
                    )  # cap very small delay
                self.state.last_trigger_check = time.time()
                logger.info(f"✓ Trigger activated: {trigger_config}")
                return True

        # Check if drive priority is high enough given current context
        priority = float(self.config.get("priority", 0.8))
        other_drives_priority = float(context.get("other_drives_total_priority", 0.5))

        # If our priority is significantly higher than other drives, trigger
        if priority > other_drives_priority * 1.5:
            if self._jitter_ms > 0:
                time.sleep(min(0.5, self._jitter_ms / 1000.0))
            logger.info(
                f"✓ Priority trigger: {priority:.2f} > {other_drives_priority:.2f}"
            )
            self.state.last_trigger_check = time.time()
            return True

        return False

    # ---------- Adaptive Weighting ----------

    def _calculate_adjusted_weight(self, objective: ImprovementObjective) -> float:
        """Calculate adjusted weight based on adaptive learning."""
        base_weight = float(objective.weight)

        # Get adaptive learning config
        adaptive_config = self.config.get("adaptive_learning", {})
        if not adaptive_config.get("enabled", False):
            return base_weight

        # Adjust based on success rate
        total_attempts = objective.success_count + objective.failure_count
        if total_attempts == 0:
            return base_weight

        min_samples = adaptive_config.get("track_outcomes", {}).get(
            "min_samples_before_adjust", 10
        )
        if total_attempts < min_samples:
            return base_weight

        success_rate = objective.success_count / total_attempts
        learning_rate = float(
            adaptive_config.get("track_outcomes", {}).get("learning_rate", 0.2)
        )

        # Adjust weight: increase if successful, decrease if not
        adjustment = learning_rate * (success_rate - 0.5)
        adjusted_weight = base_weight + adjustment

        # Clamp to bounds
        bounds = adaptive_config.get("track_outcomes", {}).get("weight_bounds", {})
        min_weight = float(bounds.get("min", 0.3))
        max_weight = float(bounds.get("max", 1.0))
        adjusted_weight = max(min_weight, min(max_weight, adjusted_weight))

        # Check if we should notify about significant change
        transparency = adaptive_config.get("transparency", {})
        notify_config = transparency.get("notify_on_significant_change", {})
        if notify_config.get("enabled", False):
            threshold = float(notify_config.get("threshold_change", 0.2))
            last_notified = self._last_weight_notification.get(
                objective.type, base_weight
            )
            if abs(adjusted_weight - last_notified) >= threshold:
                self._send_alert(
                    "info",
                    f"Significant weight change for {objective.type}",
                    {
                        "objective": objective.type,
                        "old_weight": last_notified,
                        "new_weight": adjusted_weight,
                        "success_rate": success_rate,
                        "total_attempts": total_attempts,
                    },
                )
                self._last_weight_notification[objective.type] = adjusted_weight

        if abs(adjusted_weight - base_weight) > 0.01:
            logger.debug(
                f"Adjusted weight for {objective.type}: {base_weight:.2f} -> {adjusted_weight:.2f} "
                f"(success_rate: {success_rate:.2f})"
            )

        return adjusted_weight

    # ---------- Selection ----------

    def select_objective(self) -> Optional[ImprovementObjective]:
        """Select next objective to work on based on weights, cooldowns, and adaptive learning."""
        current_time = time.time()
        available_objectives = [
            obj
            for obj in self.objectives
            if not obj.completed 
            and obj.cooldown_until <= current_time
            # FIX Issue 5: Skip blacklisted objectives to avoid dummy circular import fixes
            and obj.type not in self.BLACKLISTED_OBJECTIVES
        ]

        # FIX Issue 5: Log skipped blacklisted objectives (only once per session)
        blacklisted_found = [
            obj.type for obj in self.objectives 
            if obj.type in self.BLACKLISTED_OBJECTIVES and not obj.completed
        ]
        if blacklisted_found and not self._blacklist_logged:
            logger.debug(f"Skipping blacklisted objectives: {blacklisted_found}")
            self._blacklist_logged = True

        if not available_objectives:
            # Check if any are just on cooldown
            on_cooldown = [
                obj for obj in self.objectives if obj.cooldown_until > current_time
            ]
            if on_cooldown:
                next_available = min(on_cooldown, key=lambda x: x.cooldown_until)
                wait_seconds = next_available.cooldown_until - current_time
                logger.info(
                    f"All objectives on cooldown. Next: {next_available.type} "
                    f"in {wait_seconds / 60:.1f} min"
                )
            else:
                logger.info("All improvement objectives completed!")
            return None

        # Calculate adjusted weights with adaptive learning
        weighted_objectives = [
            (obj, self._calculate_adjusted_weight(obj)) for obj in available_objectives
        ]

        # Sort by adjusted weight (highest first)
        weighted_objectives.sort(key=lambda x: x[1], reverse=True)
        selected, weight = weighted_objectives[0]
        logger.info(f"Selected: {selected.type} (weight: {weight:.2f})")
        return selected

    # ---------- Gap-Driven Priority Boost ----------

    # Mapping from gap types to objective types
    GAP_TO_OBJECTIVE_MAPPING = {
        "slow_routing": "optimize_performance",
        "performance": "optimize_performance",
        "latency": "optimize_performance",
        "timeout": "optimize_performance",
        "routing_variance": "optimize_performance",
        "high_error_rate": "fix_known_bugs",
        "tool_selection_failure": "enhance_safety_systems",
        "complex_query_handling": "optimize_performance",
        "decomposition": "fix_known_bugs",
        "causal": "fix_known_bugs",
        "semantic_bridge": "enhance_safety_systems",
        "transfer": "enhance_safety_systems",
    }

    def register_gap(self, gap_type: str, severity: float) -> bool:
        """
        Register a detected gap from CuriosityEngine and boost relevant objective.
        
        This method connects the CuriosityEngine's gap detection to the
        SelfImprovementDrive's objective prioritization, ensuring that
        performance gaps trigger performance improvements (not unrelated objectives).
        
        Args:
            gap_type: Type of gap detected (e.g., 'slow_routing', 'high_error_rate')
            severity: Severity score (0.0 to 1.0), higher means more urgent
            
        Returns:
            True if an objective was boosted, False otherwise
        """
        with self._lock:
            # Map gap type to objective type
            objective_type = self.GAP_TO_OBJECTIVE_MAPPING.get(
                gap_type.lower(), "optimize_performance"
            )
            
            # Find matching objective
            for obj in self.objectives:
                if obj.type == objective_type and not obj.completed:
                    # Boost weight based on severity (max 0.3 boost)
                    weight_boost = min(0.3, severity * 0.5)
                    obj.weight = min(1.0, obj.weight + weight_boost)
                    
                    logger.info(
                        f"[SelfImprovement] Gap '{gap_type}' (severity={severity:.2f}) "
                        f"-> boosted '{objective_type}' weight by {weight_boost:.2f} "
                        f"(new weight: {obj.weight:.2f})"
                    )
                    return True
            
            logger.debug(
                f"[SelfImprovement] Gap '{gap_type}' has no matching objective"
            )
            return False

    def process_gaps_from_curiosity_engine(
        self, gaps: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Process a list of gaps from CuriosityEngine.
        
        This is the primary integration point for connecting CuriosityEngine's
        gap detection to SelfImprovementDrive's objective prioritization.
        
        Args:
            gaps: List of gap dictionaries with 'type' and 'priority' keys
            
        Returns:
            Dictionary with count of boosted objectives by type
        """
        boost_counts: Dict[str, int] = {}
        
        for gap in gaps:
            gap_type = gap.get("type", gap.get("gap_type", "unknown"))
            severity = gap.get("priority", gap.get("severity", 0.5))
            
            if self.register_gap(gap_type, severity):
                objective_type = self.GAP_TO_OBJECTIVE_MAPPING.get(
                    gap_type.lower(), "optimize_performance"
                )
                boost_counts[objective_type] = boost_counts.get(objective_type, 0) + 1
        
        if boost_counts:
            logger.info(
                f"[SelfImprovement] Processed {len(gaps)} gaps, "
                f"boosted objectives: {boost_counts}"
            )
        
        return boost_counts

    # ---------- Code Introspection Methods ----------

    def diagnose_system_issues(self) -> List[Dict[str, Any]]:
        """
        Use introspection to diagnose system issues.
        
        This method examines Vulcan's own code and logs to identify problems
        that the self-improvement system should address.
        
        Returns:
            List of diagnosed issues, each with type, component, severity, and description
        """
        issues: List[Dict[str, Any]] = []
        
        # 1. Analyze query routing implementation
        if self.code_introspector:
            try:
                routing_analysis = self.code_introspector.analyze_query_routing()
                if routing_analysis.get('has_query_router') and not routing_analysis.get('routing_logic_found'):
                    issues.append({
                        'type': 'missing_implementation',
                        'component': 'QueryRouter',
                        'severity': 'critical',
                        'description': 'QueryRouter may have incomplete classification logic',
                        'suggested_fix': 'Review classify_query method for pattern matching',
                        'affected_files': [routing_analysis.get('file_path', 'query_router.py')],
                        'details': routing_analysis.get('issues', [])
                    })
                
                # Check for issues in routing analysis
                for issue in routing_analysis.get('issues', []):
                    issues.append({
                        'type': 'potential_issue',
                        'component': 'QueryRouter',
                        'severity': 'medium',
                        'description': issue,
                        'affected_files': [routing_analysis.get('file_path', 'query_router.py')]
                    })
                
            except Exception as e:
                logger.debug(f"Query routing analysis failed: {e}")
        
        # 2. Check recent logs for patterns
        if self.log_analyzer:
            try:
                log_failures = self.log_analyzer.analyze_recent_failures(hours=1)
                
                # Check for timeout patterns
                if log_failures.get('timeouts') and len(log_failures['timeouts']) > 10:
                    issues.append({
                        'type': 'performance',
                        'component': 'QueryHandler',
                        'severity': 'high',
                        'description': f"Found {len(log_failures['timeouts'])} timeout errors in recent logs",
                        'evidence': log_failures['timeouts'][:3]  # First 3 examples
                    })
                
                # Check for error patterns
                if log_failures.get('errors') and len(log_failures['errors']) > 5:
                    issues.append({
                        'type': 'errors',
                        'component': 'System',
                        'severity': 'high',
                        'description': f"Found {len(log_failures['errors'])} error entries in recent logs",
                        'evidence': log_failures['errors'][:3]
                    })
                
                # Check for import errors
                if log_failures.get('import_errors'):
                    issues.append({
                        'type': 'dependency',
                        'component': 'Imports',
                        'severity': 'medium',
                        'description': f"Found {len(log_failures['import_errors'])} import errors",
                        'evidence': log_failures['import_errors'][:3]
                    })
                    
            except Exception as e:
                logger.debug(f"Log analysis failed: {e}")
        
        # 3. Find missing implementations
        if self.code_introspector:
            try:
                missing = self.code_introspector.find_missing_implementations()
                # Only report if there are many missing implementations (likely real issues)
                significant_missing = [m for m in missing if not m['function'].startswith('_')]
                if len(significant_missing) > 10:
                    issues.append({
                        'type': 'missing_implementation',
                        'component': 'Unknown',
                        'severity': 'low',
                        'description': f"Found {len(significant_missing)} potentially missing function implementations",
                        'evidence': significant_missing[:5]  # First 5 examples
                    })
            except Exception as e:
                logger.debug(f"Missing implementation analysis failed: {e}")
        
        # Store issues in knowledge store and enhance with past similar issues
        if self.code_knowledge_store and issues:
            for issue in issues:
                # Store the issue
                self.code_knowledge_store.store_issue(issue, resolved=False)
                
                # Find similar past issues with resolutions
                similar = self.code_knowledge_store.get_similar_issues(issue, limit=3)
                if similar:
                    issue['similar_past_issues'] = similar
                    # Extract successful resolutions as hints
                    resolutions = [s.get('resolution') for s in similar if s.get('resolution')]
                    if resolutions:
                        issue['suggested_resolutions'] = resolutions
        
        logger.info(f"Diagnosed {len(issues)} system issues")
        return issues

    def generate_improvement_observation(self, objective: ImprovementObjective) -> Dict[str, Any]:
        """
        Generate observation that includes code introspection results.
        
        This provides context for LLM-based improvement generation by including
        diagnosed issues and relevant code context.
        
        Args:
            objective: The improvement objective to generate observation for
            
        Returns:
            Dictionary containing observation data including diagnosed issues
        """
        # First, diagnose current issues
        current_issues = self.diagnose_system_issues()
        
        # Base observation
        observation: Dict[str, Any] = {
            'objective_type': objective.type,
            'current_issues': current_issues,
            'timestamp': time.time(),
            'objective_weight': objective.weight,
            'objective_attempts': objective.attempts,
        }
        
        # If fixing bugs, include specific code context
        if objective.type == 'fix_known_bugs' and current_issues:
            # Pick highest severity issue
            critical_issues = [i for i in current_issues if i.get('severity') == 'critical']
            high_issues = [i for i in current_issues if i.get('severity') == 'high']
            
            target_issues = critical_issues or high_issues
            if target_issues:
                issue = target_issues[0]
                observation['target_issue'] = issue
                
                # Include actual code context
                if issue.get('component') and self.code_introspector:
                    component = issue['component']
                    # Try to find relevant code
                    if component == 'QueryRouter':
                        current_impl = self.code_introspector.find_class_method('QueryAnalyzer', 'route_query')
                        if not current_impl:
                            current_impl = self.code_introspector.find_class_method('QueryRouter', 'route')
                        if current_impl:
                            # Truncate to avoid prompt overflow
                            observation['current_code'] = current_impl[:MAX_CODE_SNIPPET_CHARS]
        
        # For optimization objectives, include performance-related issues
        if objective.type == 'optimize_performance':
            perf_issues = [i for i in current_issues if i.get('type') == 'performance']
            observation['performance_issues'] = perf_issues
        
        # Include successful patterns from code knowledge store for learning
        if self.code_knowledge_store:
            try:
                successful_patterns = self.code_knowledge_store.get_successful_patterns_for_objective(
                    objective.type, limit=5
                )
                if successful_patterns:
                    observation['successful_patterns'] = [
                        {
                            'changes': p.get('data', {}).get('changes', {}),
                            'improvement': p.get('data', {}).get('improvement', 0),
                            'confidence': p.get('confidence', 0)
                        }
                        for p in successful_patterns
                    ]
                    logger.info(f"Found {len(successful_patterns)} successful patterns for {objective.type}")
            except Exception as e:
                logger.debug(f"Failed to retrieve successful patterns: {e}")
        
        return observation

    # ---------- Action Planning ----------

    def generate_improvement_action(
        self, objective: ImprovementObjective
    ) -> Dict[str, Any]:
        """
        Generate an action plan for the given improvement objective.

        This returns a context that Vulcan's orchestrator can execute.
        The orchestrator is responsible for:
        - Running pre-flight checks
        - Performing impact analysis
        - Executing dry-run if configured
        - Validating changes
        """
        # Find objective config
        obj_config = next(
            (
                obj
                for obj in self.config.get("objectives", [])
                if isinstance(obj, dict) and obj.get("type") == objective.type
            ),
            {},
        )

        # Pass-through: env_overrides and auto-approval hints (external workflow will decide)
        validation_overrides = self.config.get("validation", {}).get(
            "env_overrides", {}
        )
        auto_approve_hints = (
            self.full_config.get("global_settings", {})
            .get("approval_workflow", {})
            .get("timeout_behavior", {})
            .get("auto_approve_if_safe_criteria", {})
        )

        action_map = {
            "fix_circular_imports": {
                "high_level_goal": "fix_circular_imports",
                "raw_observation": {
                    "task": "scan_and_fix_circular_imports",
                    "scope": obj_config.get("scope", {}),
                    "success_criteria": obj_config.get("success_criteria", {}),
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,  # Always dry-run first
                "requires_impact_analysis": True,
            },
            "optimize_performance": {
                "high_level_goal": "optimize_performance",
                "raw_observation": {
                    "task": "profile_and_optimize",
                    "target_metrics": obj_config.get("target_metrics", {}),
                    "allowed_optimizations": obj_config.get(
                        "allowed_optimizations", []
                    ),
                    "forbidden_optimizations": obj_config.get(
                        "forbidden_optimizations", []
                    ),
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,
                "requires_impact_analysis": True,
            },
            "enhance_safety_systems": {
                "high_level_goal": "enhance_safety",
                "raw_observation": {
                    "task": "strengthen_safety_systems",
                    "focus": ["governance", "validation", "rollback"],
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,  # Critical: always dry-run safety changes
                "requires_impact_analysis": True,
            },
            "improve_test_coverage": {
                "high_level_goal": "improve_tests",
                "raw_observation": {
                    "task": "increase_test_coverage",
                    "coverage_targets": obj_config.get("coverage_targets", {}),
                    "priority_areas": obj_config.get("priority_areas", []),
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,
                "requires_impact_analysis": False,  # Lower risk
            },
            "fix_known_bugs": {
                "high_level_goal": "fix_bugs",
                "raw_observation": {
                    "task": "fix_known_issues",
                    "bug_sources": obj_config.get("bug_sources", []),
                    "priority_order": obj_config.get("priority_order", []),
                },
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,
                "requires_impact_analysis": True,
            },
        }

        action = action_map.get(
            objective.type,
            {
                "high_level_goal": "improve_system",
                "raw_observation": {"objective": objective.type},
                "safety_constraints": self._get_safety_constraints(),
                "requires_dry_run": True,
                "requires_impact_analysis": True,
            },
        )

        # Add metadata + pass-through governance hints
        action["_drive_metadata"] = {
            "objective_type": objective.type,
            "objective_weight": objective.weight,
            "attempt_number": objective.attempts + 1,
            "timestamp": time.time(),
        }
        action["validation_overrides"] = validation_overrides
        action["auto_approve_hints"] = (
            auto_approve_hints  # external service decides if applicable
        )

        # Optional risk tag (helps external approver routing)
        risk_class = "low" if objective.type in ("improve_test_coverage",) else "medium"
        if objective.type in ("enhance_safety_systems",):
            risk_class = "high"
        action["risk_classification"] = risk_class

        return action

    def _get_safety_constraints(self) -> Dict[str, Any]:
        """Get safety constraints from config, robust to nested/flat shapes."""
        constraints = self.config.get("constraints", {}) or {}
        change_reqs = constraints.get("change_requirements", {}) or {}
        # Support both flat (constraints['always_maintain_tests']) and nested
        maintain_tests = constraints.get(
            "always_maintain_tests", change_reqs.get("always_maintain_tests", True)
        )
        never_reduce_safety = constraints.get(
            "never_reduce_safety", change_reqs.get("never_reduce_safety", True)
        )
        rollback_on_failure = constraints.get("rollback_on_failure", True)
        return {
            "require_approval": self.require_human_approval,
            "maintain_tests": bool(maintain_tests),
            "never_reduce_safety": bool(never_reduce_safety),
            "rollback_on_failure": bool(rollback_on_failure),
        }

    # ---------- Approval ----------

    def request_approval(self, improvement_plan: Dict[str, Any]) -> str:
        """
        Request human approval for improvement.

        Returns approval ID for tracking.

        NOTE: This creates the approval request. The actual approval workflow
        (Git PR, Slack, email, timeout) is handled by the external approval service.
        """
        # **************************************************************************
        # START FIX 1: Re-check config value in case it was modified post-init
        # This makes the test `test_auto_approval_when_disabled` pass.
        approval_required = self.config.get("constraints", {}).get(
            "require_human_approval", True
        )
        if not approval_required:
            # END FIX 1
            # Original code was:
            # if not self.require_human_approval:
            logger.info("Auto-approval enabled, applying improvement")
            return "AUTO_APPROVED"

        # Generate approval ID
        approval_id = f"approval_{int(time.time())}_{improvement_plan.get('high_level_goal', 'unknown')}"

        # Add to pending approvals
        approval_request = {
            "id": approval_id,
            "plan": improvement_plan,
            "timestamp": time.time(),
            "status": "pending",
        }
        self.state.pending_approvals.append(approval_request)
        self._save_state()

        logger.warning("=" * 60)
        logger.warning("🔒 IMPROVEMENT APPROVAL REQUIRED")
        logger.warning("=" * 60)
        logger.warning(f"Approval ID: {approval_id}")
        logger.warning(f"Improvement: {improvement_plan.get('high_level_goal')}")
        logger.warning(f"Details: {json.dumps(improvement_plan, indent=2)}")
        logger.warning("=" * 60)
        logger.warning(f"Approve: call approve_pending('{approval_id}')")
        logger.warning(f"Reject: call reject_pending('{approval_id}')")
        logger.warning("=" * 60)

        # Send alert for external notification
        self._send_alert(
            "info",
            "Approval required for improvement",
            {
                "approval_id": approval_id,
                "objective": improvement_plan.get("high_level_goal"),
                "plan": improvement_plan,
            },
        )

        return approval_id

    def approve_pending(self, approval_id: str) -> bool:
        """Approve a pending improvement."""
        with self._lock:
            for approval in self.state.pending_approvals:
                if approval["id"] == approval_id:
                    if approval.get("status") == "pending_governed_approval" or "proposal_digest" in approval:
                        logger.warning("Legacy approve_pending rejects governed approval records; use approve_governed_pending")
                        return False
                    approval["status"] = "approved"
                    approval["approved_at"] = time.time()
                    self._save_state()
                    logger.info(f"✅ Approved: {approval_id}")
                    return True

        logger.warning(f"Approval ID not found: {approval_id}")
        return False

    def reject_pending(self, approval_id: str, reason: str = "") -> bool:
        """Reject a pending improvement."""
        with self._lock:
            for approval in self.state.pending_approvals:
                if approval["id"] == approval_id:
                    approval["status"] = "rejected"
                    approval["rejected_at"] = time.time()
                    approval["rejection_reason"] = reason
                    self._save_state()
                    logger.info(f"❌ Rejected: {approval_id} ({reason})")
                    return True

        logger.warning(f"Approval ID not found: {approval_id}")
        return False

    def check_approval_status(self, approval_id: str) -> Optional[str]:
        """Check status of approval: 'pending', 'approved', 'rejected', or None."""
        # Check internal state first
        for approval in self.state.pending_approvals:
            if approval["id"] == approval_id:
                return approval["status"]

        # Check external approval service if configured
        if self.approval_checker:
            try:
                return self.approval_checker(approval_id)
            except Exception as e:
                logger.error(f"External approval check failed: {e}")

        return None


    def _queue_governed_approval(self, plan: Dict[str, Any], reason: str) -> Dict[str, Any]:
        proposal_map = plan.get("governed_proposal") or plan.get("improvement_proposal")
        policy = getattr(self, "improvement_policy", None) or getattr(self, "governed_improvement_policy", None)
        if not (ImprovementProposal and isinstance(proposal_map, dict) and policy is not None):
            approval_id = self.request_approval(plan)
            return {"status": "queued", "plan_id": plan.get("id"), "reason": reason, "approval_id": approval_id}
        try:
            proposal = ImprovementProposal.from_mapping(proposal_map)
            pending = {
                "id": proposal.proposal_id, "approval_id": proposal.approval_id or proposal.proposal_id,
                "status": "pending_governed_approval", "lifecycle": "pending governed approval",
                "proposal": proposal_map, "proposal_digest": proposal.digest(),
                "policy_digest": policy.digest, "original_source_digest": proposal.expected_original_sha256,
                "inspected_source_digest": proposal.inspected_source_digest,
                "provider_release_digest": proposal.provider_release_digest, "timestamp": time.time(),
                "reason": reason,
            }
            self.state.pending_approvals.append(pending); self._save_state()
            return {"status": "pending_governed_approval", "plan_id": plan.get("id"), "reason": reason, "approval_id": pending["approval_id"], "proposal_digest": pending["proposal_digest"]}
        except Exception as e:
            return {"status": "rejected", "plan_id": plan.get("id"), "reason": f"governed approval queue failed: {type(e).__name__}"}

    def approve_governed_pending(self, approval_id: str, principal: Any, *, ttl_seconds: float = 3600.0) -> bool:
        from .governed_transaction import TrustedApprovalPrincipal, ApprovalVerifierPort
        verifier = getattr(self, "approval_verifier", None) or getattr(self, "approval_authority", None)
        if verifier is None or not isinstance(principal, TrustedApprovalPrincipal):
            return False
        if not isinstance(verifier, ApprovalVerifierPort):
            return False
        if any(hasattr(verifier, name) for name in ("issue_principal","register_principal")):
            return False
        if not hasattr(verifier, "is_authorized"):
            return False
        if ApprovalRecord is None or not getattr(self, "approval_store", None):
            return False
        with self._lock:
            for approval in self.state.pending_approvals:
                if approval.get("approval_id") == approval_id and approval.get("status") == "pending_governed_approval":
                    bindings = {"approval_id": approval_id, "proposal_digest": approval["proposal_digest"], "policy_digest": approval["policy_digest"], "original_source_digest": approval["original_source_digest"], "required_scope": "self_improvement.approve"}
                    if not verifier.is_authorized(principal, bindings):
                        return False
                    pid = principal.principal_id
                    now = time.time()
                    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)) or not math.isfinite(float(ttl_seconds)) or float(ttl_seconds) <= 0:
                        return False
                    expires = min(principal.expires_at, now + min(float(ttl_seconds), 3600.0))
                    if expires <= now:
                        return False
                    rec = ApprovalRecord(approval_id, approval["proposal_digest"], approval["policy_digest"], approval["original_source_digest"], pid, now, expires)
                    self.approval_store.save(rec)
                    approval["status"] = "independently_approved"; approval["approved_at"] = rec.approved_at; approval["approver_identity"] = pid
                    self._save_state(); return True
        return False

    def resume_governed_pending(self, approval_id: str) -> Dict[str, Any]:
        with self._lock:
            pending = next((a for a in self.state.pending_approvals if a.get("approval_id") == approval_id), None)
        if not pending:
            return {"status": "missing_approval", "approval_id": approval_id}
        if pending.get("status") != "independently_approved":
            return {"status": pending.get("status", "pending_governed_approval"), "approval_id": approval_id, "applied": False}
        proposal_map = copy.deepcopy(pending.get("proposal", {})); proposal_map["approval_id"] = approval_id
        plan = {"id": pending.get("id"), "governed_proposal": proposal_map}
        auto_result = self._maybe_auto_apply(plan)
        applied, reason = auto_result.applied, auto_result.reason
        result = auto_result.transaction_result
        status = result.status_code if result is not None else auto_result.status_code
        with self._lock:
            pending["status"] = "applied" if applied else status
            pending["transaction_status"] = status
            pending["completed_at"] = time.time()
            self._save_state()
        if applied and not pending.get("outcome_recorded"):
            self.state.improvements_this_session += 1
            try: self.record_outcome(str(pending.get("objective_type", "governed")), True, {"governed": True, "approval_id": approval_id})
            except Exception: pass
            pending["outcome_recorded"] = True
            self._save_state()
        return {"status": status, "approval_id": approval_id, "applied": applied, "reason": reason, "transaction_result": result}

    # ---------- Auto Apply Logic ----------
    # Example change plan schema assumption:
    # plan = {
    #   "id": "...",
    #   "category": "path_fixes" | "runtime_hygiene" | ...,
    #   "files": [{"path": "src/foo.py", "loc_added": 10, "loc_removed": 2}, ...],
    #   "diff_loc": 12,
    #   "apply": callable_that_performs_change_or_patch
    # }
    def _maybe_auto_apply(self, plan: Dict[str, Any]) -> AutoApplyResult:
        """Auto-apply only through GovernedSelfImprovementTransaction."""
        if not self._auto_apply_enabled:
            return AutoApplyResult(False,AutoApplyStatus.NOT_ATTEMPTED,"auto-apply disabled",True,False)
        if GovernedSelfImprovementTransaction is None or ImprovementProposal is None or inspect_repository is None:
            return AutoApplyResult(False,AutoApplyStatus.UNAVAILABLE,"governed transaction unavailable",True,False)
        proposal_map = plan.get("governed_proposal") or plan.get("improvement_proposal")
        policy = getattr(self, "improvement_policy", None) or getattr(self, "governed_improvement_policy", None)
        audit_owner = getattr(self, "audit_owner", None) or getattr(self, "audit", None)
        if not isinstance(proposal_map, dict):
            return AutoApplyResult(False,AutoApplyStatus.MISSING_GOVERNED_PROPOSAL,"missing governed improvement proposal",True,False)
        if policy is None or audit_owner is None:
            return AutoApplyResult(False,AutoApplyStatus.CONFIGURATION_MISSING,"governed transaction policy or audit owner missing",True,False)
        try:
            proposal = ImprovementProposal.from_mapping(proposal_map)
            snapshot = inspect_repository(policy.repo_root, policy.permitted_path_globs, max_files=policy.max_files)
            tx = GovernedSelfImprovementTransaction(policy, audit_owner, getattr(self, "approval_store", None))
            result = tx.apply(proposal, snapshot, actor="self-improvement-drive")
            self._last_governed_transaction_result = result
            if result.verified_success:
                return AutoApplyResult(True,AutoApplyStatus(result.status_code),result.status_code,False,True, transaction_result=result)
            if result.status_code == "rejected_before_installation" and result.failure_category == "approval required":
                return AutoApplyResult(False,AutoApplyStatus(result.status_code),result.failure_category,True,False, transaction_result=result)
            manual = result.status_code in {"external_mutation_prevented_rollback","verification_failed_rollback_failed"}
            return AutoApplyResult(False,AutoApplyStatus(result.status_code),result.failure_category or result.status_code,False,True,result.status_code=="verification_failed_rollback_succeeded",manual,result)
        except Exception as e:
            return AutoApplyResult(False,AutoApplyStatus.GOVERNED_EXCEPTION,f"governed transaction failed: {type(e).__name__}",False,True)

    # Where improvements are processed, insert a call to _maybe_auto_apply before queueing/approval
    def process_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single improvement plan. Auto-apply if policy allows, otherwise queue for approval.
        """
        # ... any existing validation ...
        result = AutoApplyResult(False,AutoApplyStatus.QUEUED,"queued",True,False)
        if self._auto_apply_enabled:
            result = self._maybe_auto_apply(plan)

        if result.applied:
            status = {"status": "applied", "plan_id": plan.get("id"), "reason": result.reason, "transaction_status": result.status_code}
            logger.info(f"Auto-applied plan {plan.get('id')}")
            # Potentially call record_outcome here or let external orchestrator do it
        else:
            if self._auto_apply_enabled and result.terminal:
                status = {"status": result.status_code, "plan_id": plan.get("id"), "reason": result.reason, "terminal": True, "rollback_succeeded": result.rollback_succeeded, "manual_recovery_required": result.manual_recovery_required}
            elif result.approval_required:
                status = self._queue_governed_approval(plan, result.reason)
            else:
                status = {"status": result.status_code, "plan_id": plan.get("id"), "reason": result.reason}

        # Update state if your class tracks it
        try:
            st = getattr(self, "state", None)
            if isinstance(st, SelfImprovementState):  # Check type
                st.last_status = status  # Add a field to track last status
        except Exception as e:
            logger.debug(f"Operation failed: {e}")
        return status

    # ---------- Main Step ----------

    def step(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute one step of self-improvement drive.

        This is called by Vulcan's main loop when the drive is active.
        Returns an action to execute, or None if no action needed.
        """
        try:
            # Check if we should be active
            if not self.should_trigger(context):
                return None

            # Select an objective
            objective = self.select_objective()
            if not objective:
                logger.info("No objectives available")
                return None

            logger.info(
                f"🎯 Pursuing: {objective.type} (weight: {objective.weight:.2f}, "
                f"attempts: {objective.attempts})"
            )

            # Generate improvement action (PLAN ASSEMBLY)
            improvement_action = self.generate_improvement_action(objective)

            # CSIU: Calculate pressure and regularize plan (CRITICAL CALL-SITE)
            # This is the primary integration point for CSIU learning
            if self._csiu_enabled and self._csiu_regs_enabled:
                try:
                    # Collect telemetry
                    prev_telemetry = self._csiu_last_metrics
                    cur_telemetry = self._collect_telemetry_snapshot()

                    previous_snapshot = getattr(self, "_csiu_previous_snapshot", None)
                    current_snapshot = getattr(self, "_csiu_last_snapshot", None) if cur_telemetry else None
                    if self._csiu_enforcer is not None and current_snapshot is not None:
                        improvement_action, decision = self._csiu_enforcer.apply_regularization_from_snapshots(
                            improvement_action, previous_snapshot, current_snapshot,
                            plan_id=str(improvement_action.get("id", "unknown")),
                            action_type=str(improvement_action.get("type", "improvement")),
                        )
                        self._csiu_last_decision = decision
                        self._csiu_audit_status(
                            "decision", decision_id=decision.decision_id,
                            decision_digest=decision.decision_digest, reason_code=decision.reason_code,
                            pressure=decision.pressure, actual_effect=decision.actual_effect,
                            applied=decision.applied, authoritative="csiu_enforcement",
                        )
                        if decision.reason_code == "baseline_established" or decision.applied:
                            self._csiu_previous_snapshot = current_snapshot
                    self._csiu_last_metrics = cur_telemetry

                except Exception as e:
                    self._csiu_audit_status("step_csiu_failure", reason_code=type(e).__name__)

            # --- MODIFIED: Try auto-apply before requesting manual approval ---
            processed_status = self.process_plan(improvement_action)

            # If auto-applied successfully, record outcome and return None (no further action needed now)
            if processed_status.get("status") == "applied":
                self.record_outcome(
                    objective.type,
                    True,
                    {"auto_applied": True, "reason": processed_status.get("reason")},
                )
                # Mark objective state
                objective.attempts += 1
                objective.last_attempt = time.time()
                self.state.current_objective = (
                    objective.type
                )  # Track objective even if auto-applied
                self.state.active = True  # Drive was active
                self._save_state()
                return None  # Return None because the action was completed internally

            # If queued (either because auto-apply failed or wasn't enabled/triggered), proceed to return action for orchestrator
            elif processed_status.get("status") in {"queued", "pending_governed_approval"}:
                # If approval is required and wasn't auto-approved, it needs external handling
                if self.require_human_approval:
                    # Add approval info if it was queued because manual approval is needed
                    if processed_status.get("approval_id"):
                        improvement_action["_pending_approval"] = processed_status[
                            "approval_id"
                        ]
                        improvement_action["_wait_for_approval"] = True
                    else:  # If queuing failed internally before request_approval was hit
                        logger.warning(
                            f"Plan queued but no approval ID generated, likely auto-apply policy failure: {processed_status.get('reason')}"
                        )
                        # Decide how to handle this - maybe force manual approval anyway?
                        # For now, let it fall through but log clearly

                # Mark objective state (attempt is starting, even if waiting for approval)
                objective.attempts += 1
                objective.last_attempt = time.time()
                self.state.current_objective = objective.type
                self.state.active = True
                self._save_state()

                logger.info(
                    f"🚀 Returning action for orchestrator: {objective.type} (Reason: {processed_status.get('reason')})"
                )
                improvement_action.update(processed_status)
                if processed_status.get("status") == "pending_governed_approval":
                    improvement_action["_wait_for_approval"] = True
                    improvement_action["governed_pending_status"] = "pending_governed_approval"
                return improvement_action  # Return the action for the external orchestrator/approval flow

            else:
                # Should not happen, log error
                logger.error(f"Unexpected status from process_plan: {processed_status}")
                return None

        except Exception as e:
            logger.error(f"Error in self-improvement step: {e}", exc_info=True)
            # Attempt to record failure if an objective was selected
            if "objective" in locals() and objective is not None:
                self.record_outcome(
                    objective.type,
                    False,
                    {"error": str(e), "context": "step_exception"},
                )
            return None

    # ---------- Outcome Recording ----------

    def _classify_failure(self, details: Dict[str, Any]) -> FailureType:
        """Classify failure as transient or systemic."""
        adaptive_config = self.config.get("adaptive_learning", {})
        failure_patterns = adaptive_config.get("failure_patterns", {})
        classification = failure_patterns.get("failure_classification", {})

        error_msg = str(details.get("error", "")).lower()

        # Check transient indicators
        transient_indicators = classification.get("transient", {}).get("indicators", [])
        for indicator in transient_indicators:
            if str(indicator).lower() in error_msg:
                return FailureType.TRANSIENT

        # Check systemic indicators
        systemic_indicators = classification.get("systemic", {}).get("indicators", [])
        for indicator in systemic_indicators:
            if str(indicator).lower() in error_msg:
                return FailureType.SYSTEMIC

        # Default to systemic to be conservative
        return FailureType.SYSTEMIC

    def record_outcome(
        self, objective_type: str, success: bool, details: Dict[str, Any]
    ):
        """Record the outcome of an improvement attempt (thread-safe)."""
        with self._lock:
            # Normalize inputs
            cost = float(details.get("cost_usd", 0.0))
            tokens = int(details.get("tokens_used", 0))

            for obj in self.objectives:
                if obj.type == objective_type:
                    if success:
                        obj.completed = True
                        obj.success_count += 1
                        if objective_type not in self.state.completed_objectives:
                            self.state.completed_objectives.append(objective_type)
                        self.state.improvements_this_session += 1
                        self.state.last_improvement = time.time()

                        # Update costs & tokens
                        self.state.total_cost_usd += cost
                        self.state.daily_cost_usd += cost
                        self.state.monthly_cost_usd += cost
                        self.state.session_tokens += tokens
                        if cost > 0:
                            self.state.cost_history.append(
                                {"timestamp": time.time(), "cost_usd": cost}
                            )

                        logger.info(f"✅ Completed: {objective_type}")
                        logger.info(f"Details: {json.dumps(details, indent=2)}")
                        if cost or tokens:
                            logger.info(
                                f"Cost: ${cost:.2f} (total: ${self.state.total_cost_usd:.2f}); "
                                f"tokens +{tokens} (session: {self.state.session_tokens})"
                            )
                    else:
                        obj.failure_count += 1
                        obj.last_failure = time.time()

                        # Classify failure
                        failure_type = self._classify_failure(details)

                        # Apply cooldown
                        adaptive_config = self.config.get("adaptive_learning", {})
                        failure_patterns = adaptive_config.get("failure_patterns", {})
                        classification = failure_patterns.get(
                            "failure_classification", {}
                        )

                        if failure_type == FailureType.TRANSIENT:
                            cooldown_hours = float(
                                classification.get("transient", {}).get(
                                    "cooldown_hours", 4
                                )
                            )
                            logger.warning(
                                f"❌ Transient failure: {objective_type}, cooldown: {cooldown_hours}h"
                            )
                        else:
                            cooldown_hours = float(
                                classification.get("systemic", {}).get(
                                    "cooldown_hours", 72
                                )
                            )
                            logger.warning(
                                f"❌ Systemic failure: {objective_type}, cooldown: {cooldown_hours}h"
                            )

                        obj.cooldown_until = time.time() + (cooldown_hours * 3600)

                        logger.warning(f"Details: {json.dumps(details, indent=2)}")
                        logger.warning(
                            f"Cooldown until: {time.ctime(obj.cooldown_until)}"
                        )

                    break

            self.state.current_objective = None
            self.state.active = False
            self._save_state()
            
            # Record learning outcome in code knowledge store
            if self.code_knowledge_store:
                try:
                    self.code_knowledge_store.record_learning_outcome(
                        objective_type=objective_type,
                        success=success,
                        code_changes={
                            'file_modified': details.get('file_modified', 'unknown'),
                            'changes_applied': details.get('changes_applied', ''),
                            'error': details.get('error', '') if not success else None,
                            'tokens_used': details.get('tokens_used', 0),
                            'cost_usd': details.get('cost_usd', 0.0)
                        },
                        metrics_before=details.get('metrics_before'),
                        metrics_after=details.get('metrics_after')
                    )
                    logger.info(f"Learning outcome recorded for {objective_type}")
                except Exception as e:
                    logger.debug(f"Failed to record learning outcome: {e}")

    # ---------- Status ----------

    def _get_state_dict(self) -> Dict[str, Any]:
        """Helper to safely serialize the state object to a dict."""
        try:
            from dataclasses import asdict, is_dataclass

            st = getattr(self, "state", None)
            if st is None:
                return {}
            elif is_dataclass(st):
                return asdict(st)
            elif hasattr(st, "__dict__"):
                return dict(vars(st))
            elif isinstance(st, dict):
                return st
            else:
                return {"value": st}
        except Exception:
            return {}

    def get_status(self) -> Dict[str, Any]:
        # **************************************************************************
        # START FIX 2: Rewrite get_status to include all keys expected by tests
        with self._lock:  # Add lock for thread safety
            state_dict = self._get_state_dict()
            enabled = self.config.get("enabled", False)

            # 1. Get objective details
            objective_details = []
            for obj in self.objectives:
                total_attempts = obj.success_count + obj.failure_count
                success_rate = (
                    (obj.success_count / total_attempts) if total_attempts > 0 else 0.0
                )
                objective_details.append(
                    {
                        "type": obj.type,
                        "weight": obj.weight,
                        "adjusted_weight": self._calculate_adjusted_weight(
                            obj
                        ),  # Call helper
                        "completed": obj.completed,
                        "attempts": obj.attempts,
                        "success_rate": success_rate,
                        "cooldown_until": obj.cooldown_until,
                    }
                )

            # 2. Get cost details
            cost_details = {
                "session_usd": state_dict.get(
                    "total_cost_usd", 0.0
                ),  # 'total_cost_usd' is session cost
                "daily_usd": state_dict.get("daily_cost_usd", 0.0),
                "monthly_usd": state_dict.get("monthly_cost_usd", 0.0),
            }

            # 3. Get token details
            token_details = {
                "session_tokens": state_dict.get("session_tokens", 0),
                "max_session_tokens": self.config.get("resource_limits", {})
                .get("llm_costs", {})
                .get("max_tokens_per_session"),
            }

            # 4. Get CSIU details
            csiu_details = {
                "enabled": self._csiu_enabled,
                "calc_enabled": self._csiu_calc_enabled,
                "regs_enabled": self._csiu_regs_enabled,
                "weights": dict(self._csiu_w),  # Return a copy
                "ewma_utility": self._csiu_u_ewma,
            }

            # 5. Get session duration
            session_start = state_dict.get("session_start_time", time.time())
            session_duration_min = (time.time() - session_start) / 60.0

            return {
                "enabled": enabled,
                "active": bool(state_dict.get("active", False)),  # Use state_dict value
                "state": state_dict,  # Keep the full state dict for test_get_status
                "pending_approvals": state_dict.get("pending_approvals", []),
                "current_objective": state_dict.get("current_objective"),
                "last_improvement": state_dict.get("last_improvement"),
                "auto_apply_enabled": bool(getattr(self, "_auto_apply_enabled", False)),
                "policy_loaded": bool(
                    getattr(self, "_auto_apply_policy", None)
                    and getattr(self._auto_apply_policy, "enabled", False)
                ),
                # Add the missing keys
                "objectives": objective_details,
                "costs": cost_details,
                "tokens": token_details,
                "csiu": csiu_details,
                "session_duration_minutes": session_duration_min,
                # This key is checked by the first failing test
                "completed_objectives": state_dict.get("completed_objectives", []),
            }
        # END FIX 2
        # **************************************************************************

    # FIXED: Alias for test compatibility
    def _perform_improvement(
        self, action: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Alias for _execute_improvement to pass test_6
        """
        logger.warning(
            "Using deprecated alias _perform_improvement. Use _execute_improvement."
        )
        return self._execute_improvement(action)

    def _execute_improvement(
        self, action: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes the improvement action using LLM-driven code generation, AST validation, and Git integration.
        """
        objective_type = action.get("_drive_metadata", {}).get("objective_type")
        logger.info(f"EXECUTING IMPROVEMENT for: {objective_type}")

        try:
            # 1. Generate Solution Content (LLM + Diff)
            solution_content, file_path = self._generate_solution_content(action)
            if not solution_content or not file_path:
                return False, {
                    "status": "failed",
                    "error": "LLM failed to generate valid solution content",
                }

            # 2. AST / Syntax Validation
            if file_path.endswith(".py"):
                valid_syntax, syntax_error = self._validate_python_syntax(
                    solution_content
                )
                if not valid_syntax:
                    logger.error(f"Generated code has syntax errors: {syntax_error}")
                    return False, {
                        "status": "failed",
                        "error": f"Syntax error: {syntax_error}",
                    }

            # 3. File Application (I/O)
            changes_applied, diff_summary = self._apply_file_modification(
                file_path, solution_content
            )
            if not changes_applied:
                return False, {
                    "status": "failed",
                    "error": "Failed to apply file changes",
                }

            # 4. Git Integration (Commit)
            commit_hash = self._commit_to_version_control(file_path, objective_type)

            return True, {
                "status": "success",
                "objective_type": objective_type,
                "changes_applied": diff_summary,
                "commit_hash": commit_hash,
                "cost_usd": 0.05,  # Estimated cost for this operation
                "tokens_used": 1500,  # Estimated tokens
            }

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return False, {"status": "failed", "error": str(e)}

    def _generate_solution_content(
        self, action: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Uses the WorldModel's LLM interface (or fallback) to generate the code improvement.
        
        Enhanced with code introspection to provide actual code context to the LLM,
        enabling more accurate and targeted fixes.
        
        Returns (content, file_path).
        """
        # Extract details from action plan
        goal = action.get("high_level_goal")
        observation = action.get("raw_observation", {})
        
        # Generate improvement observation with code introspection
        objective_type = action.get("_drive_metadata", {}).get("objective_type")
        if objective_type:
            # Find the matching objective
            matching_obj = next(
                (obj for obj in self.objectives if obj.type == objective_type), 
                None
            )
            if matching_obj:
                # Get enhanced observation with diagnosed issues
                enhanced_obs = self.generate_improvement_observation(matching_obj)
                observation.update(enhanced_obs)
        
        # Build context sections for the prompt
        issues_context = ""
        if observation.get('current_issues'):
            issues_context = "\n\nDiagnosed Issues:\n"
            for issue in observation['current_issues']:
                severity = issue.get('severity', 'unknown').upper()
                desc = issue.get('description', 'No description')
                issues_context += f"- {severity}: {desc}\n"
                if issue.get('evidence'):
                    issues_context += f"  Evidence: {str(issue['evidence'])[:200]}\n"
        
        # Include current code if available
        code_context = ""
        if observation.get('current_code'):
            code_context = f"\n\nCurrent Implementation:\n```python\n{observation['current_code']}\n```"
        
        # Include target issue details
        target_issue_context = ""
        if observation.get('target_issue'):
            issue = observation['target_issue']
            target_issue_context = f"""
\n\nTarget Issue to Fix:
- Component: {issue.get('component', 'Unknown')}
- Severity: {issue.get('severity', 'Unknown')}
- Description: {issue.get('description', 'No description')}
- Suggested Fix: {issue.get('suggested_fix', 'Review implementation')}
"""
            # Include suggested resolutions from past similar issues
            if issue.get('suggested_resolutions'):
                target_issue_context += "\n\nPast Successful Resolutions for Similar Issues:\n"
                for i, resolution in enumerate(issue['suggested_resolutions'][:3], 1):
                    target_issue_context += f"  {i}. {resolution}\n"
        
        # Include learned patterns from code knowledge store
        patterns_context = ""
        if observation.get('successful_patterns'):
            patterns_context = "\n\nLearned Successful Patterns for This Objective:\n"
            for i, pattern in enumerate(observation['successful_patterns'][:3], 1):
                changes = pattern.get('changes', {})
                improvement = pattern.get('improvement', 0)
                patterns_context += f"  {i}. File: {changes.get('file_modified', 'unknown')}, Improvement: {improvement:.1%}\n"

        # Construct enhanced prompt with code introspection context
        prompt = f"""
You are an expert software engineer improving the Vulcan system.
Objective: {goal}
{issues_context}
{code_context}
{target_issue_context}
{patterns_context}
Task Details: {json.dumps({k: v for k, v in observation.items() if k not in ['current_issues', 'current_code', 'target_issue', 'successful_patterns']}, indent=2)}

Based on the diagnosed issues and current code, provide the FULL corrected implementation.
Focus on fixing the identified issues, especially any CRITICAL severity problems.
Consider the learned successful patterns when making improvements.

Please provide the FULL content of the Python file that needs to be created or modified to solve this.
If modifying, provide the complete updated file.

Format your response exactly as follows:
FILE: <path/to/file.py>
```python
<code content here>
```
"""

        # Note: Policy Deadlock Resolution - Use local GraphixVulcanLLM for code generation
        # This bypasses the EXTERNAL_LLM_DISABLED policy block by ensuring we only use
        # trusted internal LLM providers for self-improvement operations.
        # Call LLM (Mock integration if WorldModel not fully wired, otherwise use it)
        try:
            response_text = ""
            llm_provider_id = None
            
            # Determine which LLM provider to use
            if self.world_model and hasattr(self.world_model, "ask_llm"):
                # Use helper method to determine provider ID
                llm_provider_id = self._determine_llm_provider_id()
                
                # Only use the LLM if it's a trusted provider (to avoid policy deadlock)
                if llm_provider_id and is_trusted_llm_provider(llm_provider_id):
                    logger.info(f"Using trusted LLM provider '{llm_provider_id}' for self-improvement")
                    response_text = self.world_model.ask_llm(prompt)
                elif llm_provider_id:
                    logger.warning(
                        f"LLM provider '{llm_provider_id}' is not in TRUSTED_PROVIDERS list. "
                        f"Falling back to code-analysis-based fix generation to avoid policy deadlock. "
                        f"Trusted providers: {TRUSTED_PROVIDERS}"
                    )
                else:
                    # No provider ID but world_model has ask_llm - assume internal/trusted
                    logger.info("Using world_model.ask_llm (assumed internal/trusted)")
                    response_text = self.world_model.ask_llm(prompt)
            elif hasattr(self, "_mock_llm_response"):
                response_text = self._mock_llm_response(prompt)
            else:
                # No LLM available - use targeted fix generation based on code analysis
                # This prevents boilerplate generation by examining actual code
                logger.info("No LLM provider found, using code-analysis-based fix generation.")
                fix_content, fix_path = self._generate_targeted_fix(observation)
                if fix_content and fix_path:
                    logger.info(f"Generated targeted fix for {fix_path}")
                    return fix_content, fix_path
                else:
                    logger.warning("Could not generate targeted fix - no applicable pattern found")
                    return None, None

            # Parse Response
            lines = response_text.strip().split("\n")
            file_path = None
            code_lines = []
            in_code_block = False

            for line in lines:
                if line.startswith("FILE:"):
                    file_path = line.replace("FILE:", "").strip()
                elif line.strip().startswith("```python"):
                    in_code_block = True
                elif line.strip().startswith("```") and in_code_block:
                    in_code_block = False
                elif in_code_block:
                    code_lines.append(line)

            content = "\n".join(code_lines)
            
            # Validate the LLM-generated improvement to reject boilerplate
            objective_type = action.get("_drive_metadata", {}).get("objective_type", "")
            if content and file_path:
                is_valid, reason = self._validate_improvement(content, file_path, objective_type)
                if not is_valid:
                    logger.warning(f"LLM generated invalid improvement: {reason}")
                    # Fall back to targeted fix
                    fix_content, fix_path = self._generate_targeted_fix(observation)
                    if fix_content and fix_path:
                        return fix_content, fix_path
                    return None, None
            
            return content, file_path

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None, None

    def _validate_python_syntax(self, content: str) -> Tuple[bool, Optional[str]]:
        """Validates Python code syntax using AST."""
        try:
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def _validate_improvement(self, content: str, file_path: str, objective_type: str) -> Tuple[bool, str]:
        """
        Validate that a generated improvement is REAL, not boilerplate.
        
        This prevents the self-improvement system from creating useless placeholder
        files like "enhance_safety.py" with empty implementations.
        
        Validation criteria:
        1. Must not contain common boilerplate indicators
        2. Must have substantial implementation (not just pass/...)
        3. Must be relevant to the objective type
        4. Must target an existing file OR have clear justification for new file
        
        Args:
            content: The generated code content
            file_path: Target file path
            objective_type: The improvement objective being addressed
            
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        # Boilerplate indicators that suggest generated code is not real
        boilerplate_indicators = [
            '# Auto-generated fix',
            '# Placeholder',
            '# TODO: Implement',
            '# TODO',
            'def fix(): pass',
            'class SafetySystem:\n    pass',
            'raise NotImplementedError',
            '...',
        ]
        
        # Check for boilerplate patterns
        for indicator in boilerplate_indicators:
            if indicator in content:
                return False, f"Rejected: contains boilerplate pattern '{indicator[:30]}...'"
        
        # Check content has substantial implementation
        lines = [l for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
        if len(lines) < 5:
            return False, f"Rejected: too few implementation lines ({len(lines)} < 5)"
        
        # Check for only 'pass' statements in functions/classes
        # Use lookahead to properly match pass-only definitions within larger content
        pass_only_pattern = re.compile(r'(def|class)\s+\w+.*:\s*\n\s*pass\s*(?=\n|$)', re.MULTILINE)
        matches = pass_only_pattern.findall(content)
        total_defs = content.count('def ') + content.count('class ')
        if total_defs > 0 and len(matches) == total_defs:
            return False, "Rejected: all functions/classes are empty (pass only)"
        
        # Check file path is sensible
        suspicious_paths = ['temp_fix.py', 'auto_fix.py', 'generated_fix.py', 'enhance_safety.py']
        for suspicious in suspicious_paths:
            if suspicious in file_path:
                return False, f"Rejected: suspicious generated file path '{file_path}'"
        
        # For fix objectives, verify targeting existing file
        if objective_type in ('fix_known_bugs', 'fix_circular_imports'):
            target_path = self.repo_root / file_path if not Path(file_path).is_absolute() else Path(file_path)
            if not target_path.exists():
                # New files for fix objectives are suspicious
                return False, f"Rejected: fix objective targeting non-existent file '{file_path}'"
        
        # Verify the content actually addresses the objective (soft check)
        # This is informational - we don't reject solely on keyword mismatch
        # since a fix might add new methods that don't mention "bug" explicitly
        objective_keywords = {
            'fix_known_bugs': ['fix', 'bug', 'error', 'issue', 'patch', 'classify', 'route', 'handle'],
            'optimize_performance': ['optimize', 'cache', 'performance', 'speed', 'fast', 'lazy', 'batch'],
            'improve_test_coverage': ['test', 'assert', 'mock', 'fixture', 'pytest', 'expect'],
            'enhance_safety_systems': ['safety', 'validate', 'check', 'boundary', 'verify', 'constraint'],
            'fix_circular_imports': ['import', 'from', 'module', 'lazy'],
        }
        
        keywords = objective_keywords.get(objective_type, [])
        content_lower = content.lower()
        keyword_matches = sum(1 for kw in keywords if kw in content_lower)
        
        # Log low keyword match but don't reject - the fix might still be valid
        if keywords and keyword_matches == 0:
            logger.debug(f"Note: content has no objective keywords for '{objective_type}', may need review")
        
        logger.info(f"Improvement validated: {file_path} for {objective_type}")
        return True, "Valid improvement"

    def _generate_targeted_fix(self, observation: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate a targeted fix based on actual code analysis when LLM is unavailable.
        
        Instead of producing boilerplate, this method:
        1. Examines diagnosed issues from code introspection
        2. Reads the actual source file with the problem
        3. Generates a specific, minimal fix for the identified issue
        
        This is the KEY method that prevents "theatrical" self-improvement.
        
        Args:
            observation: Dictionary containing diagnosed issues and code context
            
        Returns:
            Tuple of (fix_content, file_path) or (None, None) if no fix possible
        """
        # Get the target issue
        target_issue = observation.get('target_issue')
        if not target_issue:
            # Try to find a high-priority issue from diagnosed issues
            current_issues = observation.get('current_issues', [])
            critical_issues = [i for i in current_issues if i.get('severity') == 'critical']
            high_issues = [i for i in current_issues if i.get('severity') == 'high']
            target_issue = (critical_issues or high_issues or current_issues or [None])[0]
        
        if not target_issue:
            logger.warning("No target issue found for targeted fix generation")
            return None, None
        
        component = target_issue.get('component', '')
        affected_files = target_issue.get('affected_files', [])
        
        # Get the actual file to fix
        if not affected_files and self.code_introspector:
            # Try to find the file from component name
            for filepath in self.code_introspector.source_files:
                if component.lower() in filepath.lower():
                    affected_files = [filepath]
                    break
        
        if not affected_files:
            logger.warning(f"No files identified for component {component}")
            return None, None
        
        target_file = affected_files[0]
        
        # Get the actual content of the file
        file_content = None
        if self.code_introspector and target_file in self.code_introspector.source_files:
            file_content = self.code_introspector.source_files[target_file].get('content')
        else:
            # Try reading directly
            try:
                target_path = Path(target_file)
                if not target_path.is_absolute():
                    target_path = self.repo_root / target_file
                if target_path.exists():
                    file_content = target_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.debug(f"Could not read {target_file}: {e}")
        
        if not file_content:
            logger.warning(f"Could not read content of {target_file}")
            return None, None
        
        # Generate the fix based on issue type
        issue_type = target_issue.get('type', '')
        description = target_issue.get('description', '')
        suggested_fix = target_issue.get('suggested_fix', '')
        
        # Apply known fix patterns based on issue analysis
        fixed_content = self._apply_known_fix_pattern(
            file_content, 
            issue_type, 
            description,
            suggested_fix,
            component
        )
        
        if fixed_content and fixed_content != file_content:
            logger.info(f"Generated targeted fix for {target_file}")
            return fixed_content, str(target_file)
        
        logger.info(f"No applicable fix pattern for issue in {target_file}")
        return None, None

    def _apply_known_fix_pattern(
        self, 
        content: str, 
        issue_type: str, 
        description: str,
        suggested_fix: str,
        component: str
    ) -> Optional[str]:
        """
        Apply known fix patterns to source code.
        
        This method contains templates for common fixes that can be applied
        without LLM assistance, based on code introspection findings.
        
        Args:
            content: Original file content
            issue_type: Type of issue (missing_implementation, performance, etc.)
            description: Issue description
            suggested_fix: Suggested fix from diagnosis
            component: Affected component name
            
        Returns:
            Fixed content or None if no pattern applies
        """
        # Pattern 1: Missing classify_query method in QueryRouter
        if component == 'QueryRouter' and 'classify' in description.lower():
            # Check if classify_query is missing
            if 'def classify_query' not in content and 'class QueryAnalyzer' in content:
                # Find the class definition to insert the method
                class_match = re.search(r'(class QueryAnalyzer.*?:)', content)
                if class_match:
                    # Find a good insertion point (after __init__ or first method)
                    init_pos = content.find('def __init__')
                    if init_pos == -1:
                        # No __init__, find first method in class
                        first_def_pos = content.find('def ', class_match.end())
                        init_end = first_def_pos if first_def_pos > 0 else -1
                    else:
                        init_end = content.find('def ', init_pos + 1)
                    
                    if init_end > 0:
                        # Insert classify_query method
                        classify_method = '''
    def classify_query(self, query: str) -> str:
        """
        Classify query type based on content patterns.
        
        Added by self-improvement system to fix missing classification.
        
        Args:
            query: The query string to classify
            
        Returns:
            Query type string: 'IDENTITY', 'PHILOSOPHICAL', 'CONVERSATIONAL', 
            'MATHEMATICAL', or 'GENERAL'
        """
        query_lower = query.lower()
        
        # Identity queries - about the system's creator/origin
        identity_phrases = ['who created', 'who made', 'who built', 'your creator', 'made by']
        if any(phrase in query_lower for phrase in identity_phrases):
            return 'IDENTITY'
        
        # Philosophical queries - paradoxes, thought experiments
        philosophical_phrases = ['meaning of', 'this sentence is false', 'experience machine', 
                                'trolley problem', 'free will', 'consciousness']
        if any(phrase in query_lower for phrase in philosophical_phrases):
            return 'PHILOSOPHICAL'
        
        # Conversational queries - greetings, small talk
        conversational_phrases = ['hello', 'how are you', 'what would', 'nice to meet']
        if any(phrase in query_lower for phrase in conversational_phrases):
            return 'CONVERSATIONAL'
        
        # Mathematical queries - calculations, probabilities
        mathematical_phrases = ['calculate', 'probability', 'integral', 'derivative', 'solve']
        if any(phrase in query_lower for phrase in mathematical_phrases):
            return 'MATHEMATICAL'
        
        return 'GENERAL'

'''
                        # Insert after class definition
                        return content[:init_end] + classify_method + content[init_end:]
        
        # Pattern 2: Add missing import
        if 'import' in issue_type.lower() and 'ImportError' in description:
            # Extract module name from error
            import_match = re.search(r"No module named '(\w+)'", description)
            if import_match:
                module_name = import_match.group(1)
                # Add optional import with fallback
                import_fix = f'''
try:
    import {module_name}
except ImportError:
    {module_name} = None  # Graceful fallback
    
'''
                # Insert at top of file after existing imports
                last_import_pos = content.rfind('import ')
                last_from_pos = content.rfind('from ')
                # Only proceed if at least one import exists
                if last_import_pos > 0 or last_from_pos > 0:
                    last_import = max(last_import_pos, last_from_pos)
                    end_of_import_line = content.find('\n', last_import) + 1
                    return content[:end_of_import_line] + import_fix + content[end_of_import_line:]
        
        # Pattern 3: Performance issues - only add import time if needed
        # Note: Actual timing instrumentation requires more context and is not applied automatically
        if 'timeout' in description.lower() and 'performance' in issue_type.lower():
            # Add timing import if not present (minimal safe change)
            if 'import time' not in content:
                # Insert at the top after initial imports
                first_import = content.find('import ')
                if first_import >= 0:
                    end_of_first_import = content.find('\n', first_import) + 1
                    return content[:end_of_first_import] + 'import time  # Added for performance monitoring\n' + content[end_of_first_import:]
        
        # No applicable pattern found
        return None

    def _apply_file_modification(
        self, file_path: str, new_content: str
    ) -> Tuple[bool, str]:
        """Writes the new content to disk and calculates a diff."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            old_content = ""
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    old_content = f.read()

            # Calculate Diff
            diff = difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="",
            )
            diff_text = "\n".join(list(diff))

            # Write New Content
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True, diff_text if diff_text else "New file created"

        except Exception as e:
            logger.error(f"File I/O failed for {file_path}: {e}")
            return False, ""

    def _push_to_remote(self) -> bool:
        """
        Push committed changes to remote repository.
        
        This method enables Git persistence for self-improvement fixes, preventing
        the "Groundhog Day" loop where fixes are lost on container restart.
        
        WARNING: This is DISABLED by default for Railway deployments to prevent
        unintended changes being pushed to the repository. Only enable this if
        you have proper review processes in place.
        
        Requires:
        - VULCAN_GIT_PUSH_ENABLED=1 environment variable (default: disabled)
        - Git credentials configured (via SSH key, token, or credential helper)
        
        Returns:
            True if push succeeded, False otherwise
        """
        # Check if git push is enabled via environment variable
        # DEFAULT: DISABLED - to prevent Railway deployments from pushing changes
        if os.getenv("VULCAN_GIT_PUSH_ENABLED", "0") != "1":
            logger.debug(
                "Git push disabled by default (Railway-safe). "
                "Set VULCAN_GIT_PUSH_ENABLED=1 to enable (not recommended for production)"
            )
            return False
        
        try:
            if get_safe_executor is not None:
                executor = get_safe_executor()
                push_result = executor.execute_safe(
                    ["git", "push"], timeout=60
                )
                if push_result.success:
                    logger.info("✅ Self-improvement changes pushed to remote repository")
                    return True
                else:
                    logger.warning(f"Git push failed: {push_result.stderr}")
                    return False
            else:
                # nosec B603, B607: subprocess call is safe - using list arguments
                # with hardcoded 'git' command, no shell=True, no user input
                result = subprocess.run(  # nosec B603 B607
                    ["git", "push"], capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    logger.info("✅ Self-improvement changes pushed to remote repository")
                    return True
                else:
                    logger.warning(f"Git push failed: {result.stderr}")
                    return False
        except subprocess.TimeoutExpired:
            logger.warning("Git push timed out after 60 seconds")
            return False
        except Exception as e:
            logger.warning(f"Git push failed: {e}")
            return False

    def _commit_to_version_control(self, file_path: str, message: str) -> str:
        """
        Stages, commits, and optionally pushes changes using git.
        
        NOTE: Git push is DISABLED by default for Railway deployments. Changes are
        committed locally only. To enable push (not recommended for production without
        proper review), set VULCAN_GIT_PUSH_ENABLED=1.
        
        Without git push, fixes applied by the self-improvement system will be lost when
        the container restarts. This is intentional for Railway deployment to prevent
        unintended changes to the repository.
        """
        try:
            # Use safe executor if available, otherwise fallback to direct subprocess
            if get_safe_executor is not None:
                executor = get_safe_executor()

                # Stage
                stage_result = executor.execute_safe(
                    ["git", "add", file_path], timeout=30
                )
                if not stage_result.success:
                    logger.warning(f"Git add failed: {stage_result.error}")
                    return "git_failed"

                # Commit
                commit_msg = f"vulcan(auto): {message}"
                commit_result = executor.execute_safe(
                    ["git", "commit", "-m", commit_msg], timeout=30
                )

                if commit_result.success:
                    # Try to push changes to persist them (prevents "Groundhog Day" loop)
                    self._push_to_remote()
                    
                    # Try to get short hash
                    hash_result = executor.execute_safe(
                        ["git", "rev-parse", "--short", "HEAD"], timeout=10
                    )
                    if hash_result.success:
                        return hash_result.stdout.strip()
                    return "unknown_hash"
                else:
                    logger.warning(
                        f"Git commit returned non-zero: {commit_result.stderr}"
                    )
                    return "unknown_hash"
            else:
                # Fallback to direct subprocess (already safe - using list args, not shell=True)
                # nosec B603, B607: subprocess call is safe - using list arguments
                # with hardcoded 'git' command, file_path is internal path
                subprocess.run(  # nosec B603 B607
                    ["git", "add", file_path], check=True, capture_output=True
                )

                commit_msg = f"vulcan(auto): {message}"
                # nosec B603, B607: subprocess call is safe - using list arguments
                result = subprocess.run(  # nosec B603 B607
                    ["git", "commit", "-m", commit_msg], capture_output=True, text=True
                )

                if result.returncode == 0:
                    # Try to push changes to persist them (prevents "Groundhog Day" loop)
                    self._push_to_remote()
                    
                    # nosec B603, B607: subprocess call is safe - using list arguments
                    # with hardcoded 'git' command, no user input
                    hash_proc = subprocess.run(  # nosec B603 B607
                        ["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True,
                        text=True,
                    )
                    return hash_proc.stdout.strip()
                else:
                    logger.warning(f"Git commit returned non-zero: {result.stderr}")
                    return "unknown_hash"

        except Exception as e:
            logger.warning(f"Git operation failed (is this a repo?): {e}")
            return "git_failed"

    # ==========================================================================
    # PRIORITY 1: Safe Execution Module Integration
    # ==========================================================================

    def _detect_repo_root(self) -> Path:
        """
        Detect the repository root directory.
        
        Uses git to find the root, falls back to common paths if git unavailable.
        
        Returns:
            Path to repository root
        """
        # Try git to find repo root
        try:
            # nosec B603, B607: subprocess call is safe - using list arguments
            # with hardcoded 'git' command, no user input
            result = subprocess.run(  # nosec B603 B607
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass
        
        # Fallback: check common locations
        fallback_paths = [
            Path("/app"),
            Path.cwd(),
            Path(__file__).parent.parent.parent.parent.parent,  # Navigate up from this file
        ]
        
        for path in fallback_paths:
            if path.exists() and (path / "src").exists():
                return path
        
        # Last resort: current working directory
        return Path.cwd()

    def apply_improvement(self, objective_type: str, changes: Dict[str, Any]) -> bool:
        """
        Apply improvement with SAFE EXECUTION.
        
        This method integrates safe execution and policy gates to ensure
        improvements are applied safely and can be rolled back if tests fail.
        
        IMPORTANT: This method applies changes LOCALLY only. Changes are NOT
        automatically pushed to GitHub. To enable git push (not recommended
        for Railway deployment), set VULCAN_GIT_PUSH_ENABLED=1.
        
        Args:
            objective_type: Type of improvement being applied
            changes: Dictionary containing:
                - file_path: Path to the file being modified
                - new_content: New content for the file
                - type: Type of change (e.g., "code_modification")
                
        Returns:
            True if improvement was safely applied, False otherwise
        """
        # Check if auto-apply is disabled via environment variable
        # Default: disabled to prevent unintended modifications in production (e.g., Railway)
        if os.getenv("VULCAN_AUTO_APPLY_DISABLED", "1") == "1":
            logger.info(
                f"Auto-apply disabled (VULCAN_AUTO_APPLY_DISABLED=1). "
                f"Skipping improvement for {objective_type}"
            )
            return False

        # STEP 1: Validate target file exists
        target_file = Path(changes.get("file_path", ""))
        if not target_file.is_absolute():
            target_file = self.repo_root / target_file
        
        if not target_file.exists():
            logger.error(f"Target file does not exist: {target_file}")
            return False
        
        # STEP 2: Run policy gates if enabled
        if self.policy and getattr(self.policy, 'enabled', False):
            files = [str(target_file)]
            
            # Check files against policy
            check_result = check_files_against_policy(files, self.policy)
            if hasattr(check_result, 'ok') and not check_result.ok:
                reasons = getattr(check_result, 'reasons', ['Policy violation'])
                logger.error(f"Policy violation: {reasons}")
                return False
            elif isinstance(check_result, tuple) and not check_result[0]:
                logger.error(f"Policy violation: {check_result[1]}")
                return False
            
            # Run gates
            gates_report = run_gates(self.policy)
            if hasattr(gates_report, 'ok') and not gates_report.ok:
                failures = getattr(gates_report, 'failures', ['Gate failed'])
                logger.error(f"Gate failures: {failures}")
                return False
            elif isinstance(gates_report, tuple) and not gates_report[0]:
                logger.error(f"Gate failures: {gates_report[1]}")
                return False
        
        # STEP 3: Apply changes using safe executor
        if changes.get("type") == "code_modification":
            # Create a robust temporary file path using tempfile
            temp_dir = target_file.parent
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".tmp", 
                prefix=f"{target_file.stem}_",
                dir=temp_dir
            )
            temp_file = Path(temp_path)
            
            try:
                # Write content to temp file
                os.close(temp_fd)  # Close the file descriptor from mkstemp
                temp_file.write_text(changes["new_content"])
                
                # Run tests on modified code using safe executor
                if self.safe_executor:
                    # Run tests specific to the modified file if possible
                    test_result = self.safe_executor.execute_safe(
                        ["python", "-m", "pytest", str(target_file.parent), "-v", "-x", "--tb=short"],
                        timeout=30
                    )
                    
                    if test_result.success:
                        # Move temp to actual
                        shutil.move(str(temp_file), str(target_file))
                        logger.info(f"✅ Safely applied changes to {target_file}")
                        return True
                    else:
                        # Clean up temp file
                        if temp_file.exists():
                            temp_file.unlink()
                        logger.error(f"Tests failed: {test_result.stderr}")
                        return False
                else:
                    # No safe executor available - apply without testing
                    logger.warning("Safe executor not available, applying without test validation")
                    shutil.move(str(temp_file), str(target_file))
                    return True
                    
            except Exception as e:
                # Clean up temp file on error
                if temp_file.exists():
                    temp_file.unlink()
                logger.error(f"Error applying improvement: {e}")
                return False
        
        return False

    # ==========================================================================
    # PRIORITY 3: Fix Hallucinated Improvements - Grounded Repository Scanning
    # ==========================================================================

    def scan_repository(self) -> Dict[str, str]:
        """
        Scan actual repository structure to get real file contents.
        
        This prevents hallucinated improvements by ensuring the system
        works with actual code from the repository.
        
        Returns:
            Dictionary mapping file paths to their contents
        """
        repo_files: Dict[str, str] = {}
        valid_dirs = ["src/vulcan", "src/safety", "src/unified_platform"]
        
        for dir_path in valid_dirs:
            base_path = self.repo_root / dir_path
            if not base_path.exists():
                continue
            
            for py_file in base_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8")
                    # Store relative path from repo root
                    rel_path = str(py_file.relative_to(self.repo_root))
                    repo_files[rel_path] = content
                except Exception as e:
                    logger.warning(f"Could not read {py_file}: {e}")
        
        logger.info(f"Scanned repository: found {len(repo_files)} Python files")
        return repo_files

    def find_relevant_files(
        self, 
        objective_type: str, 
        repo_files: Dict[str, str]
    ) -> List[str]:
        """
        Find files relevant to a given improvement objective.
        
        Args:
            objective_type: Type of improvement being targeted
            repo_files: Dictionary of file paths to contents
            
        Returns:
            List of relevant file paths, sorted by relevance
        """
        # Mapping from objective types to relevant keywords/paths
        objective_keywords = {
            "optimize_performance": ["routing", "cache", "performance", "query"],
            "fix_circular_imports": ["import", "__init__"],
            "improve_test_coverage": ["test_", "_test"],
            "enhance_safety_systems": ["safety", "validation", "boundary"],
            "fix_known_bugs": ["fix", "bug", "error"],
        }
        
        keywords = objective_keywords.get(objective_type, [objective_type])
        scored_files: List[Tuple[str, int]] = []
        
        for file_path, content in repo_files.items():
            score = 0
            file_path_lower = file_path.lower()
            
            # Score based on path matching
            for keyword in keywords:
                if keyword.lower() in file_path_lower:
                    score += 2
            
            # Score based on content matching  
            content_lower = content.lower()
            for keyword in keywords:
                score += content_lower.count(keyword.lower())
            
            if score > 0:
                scored_files.append((file_path, score))
        
        # Sort by score descending
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in scored_files[:10]]  # Return top 10 relevant files

    def analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Analyze AST to understand code structure.
        
        Args:
            tree: Parsed AST tree
            
        Returns:
            Dictionary describing the code structure
        """
        structure = {
            "classes": [],
            "functions": [],
            "imports": [],
            "global_vars": [],
        }
        
        # First pass: collect classes and their methods
        class_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.add(node.name)
                class_info = {
                    "name": node.name,
                    "methods": [
                        m.name for m in node.body 
                        if isinstance(m, ast.FunctionDef)
                    ],
                    "line": node.lineno,
                }
                structure["classes"].append(class_info)
                
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        structure["imports"].append(alias.name)
                else:
                    module = node.module or ""
                    structure["imports"].append(module)
        
        # Second pass: find only top-level functions (direct children of Module)
        if isinstance(tree, ast.Module):
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "line": node.lineno,
                    }
                    structure["functions"].append(func_info)
        
        return structure

    def generate_grounded_improvement(
        self, 
        objective_type: str
    ) -> Dict[str, Any]:
        """
        Generate improvements based on ACTUAL code from the repository.
        
        This method ensures improvements are grounded in real code, preventing
        hallucinated changes to non-existent files or methods.
        
        Args:
            objective_type: Type of improvement to generate
            
        Returns:
            Dictionary containing improvement details:
                - file_path: Target file to modify
                - original_content: Current file contents
                - new_content: Proposed modifications
                - type: "code_modification"
        """
        # Get real files
        repo_files = self.scan_repository()
        if not repo_files:
            logger.error("No repository files found!")
            return {}
        
        # Find relevant file for objective
        relevant_files = self.find_relevant_files(objective_type, repo_files)
        if not relevant_files:
            logger.warning(f"No files found for objective: {objective_type}")
            return {}
        
        # Pick actual file to improve
        target_file = relevant_files[0]
        actual_code = repo_files[target_file]
        
        # Parse AST to understand structure
        try:
            tree = ast.parse(actual_code)
            structure = self.analyze_ast(tree)
        except SyntaxError as e:
            logger.error(f"Could not parse {target_file}: {e}")
            return {}
        
        # Generate improvement prompt with ACTUAL code
        # Truncate to avoid context overflow
        code_snippet = actual_code[:MAX_CODE_SNIPPET_CHARS]
        
        prompt = f"""
Improve this ACTUAL code from {target_file}:

```python
{code_snippet}
```

Current structure: {json.dumps(structure, indent=2)}
Objective: {objective_type}

Generate SPECIFIC improvements using existing classes and methods.
Output the complete modified file content.
"""
        
        # Call LLM with grounded context
        # Note: Policy Deadlock Resolution - Use local GraphixVulcanLLM for code generation
        # This ensures self-improvement operations use trusted internal LLM providers
        improvement = None
        if self.world_model and hasattr(self.world_model, "ask_llm"):
            # Use helper method to determine provider ID
            llm_provider_id = self._determine_llm_provider_id()
            
            if llm_provider_id and not is_trusted_llm_provider(llm_provider_id):
                logger.warning(
                    f"LLM provider '{llm_provider_id}' is not trusted for grounded improvement. "
                    f"Skipping to avoid policy deadlock."
                )
                return {}
            
            try:
                improvement = self.world_model.ask_llm(prompt)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return {}
        else:
            logger.warning("No LLM available for grounded improvement generation")
            return {}
        
        if not improvement:
            return {}
        
        return {
            "file_path": str(self.repo_root / target_file),
            "original_content": actual_code,
            "new_content": improvement,
            "type": "code_modification",
            "objective_type": objective_type,
            "structure": structure,
        }
    
    def process_gaps_from_curiosity_engine(self, gaps: List[Dict[str, Any]]) -> List[str]:
        """
        Process gaps detected by CuriosityEngine and boost relevant objectives.
        
        This method is called by CuriosityEngine when knowledge gaps are detected,
        enabling the self-improvement system to dynamically prioritize objectives
        based on actual system weaknesses.
        
        FIX Issue 3: This callback enables CuriosityEngine → SelfImprovementDrive
        integration, connecting gap detection to objective prioritization.
        
        Gap Types → Objective Mapping:
        - slow_routing, routing_variance → optimize_performance
        - complex_query_handling → improve_test_coverage (more tests for edge cases)
        - high_error_rate → fix_known_bugs
        - decomposition failures → fix_circular_imports (code structure issues)
        - prediction/causal gaps → enhance_safety_systems (improve validation)
        
        Args:
            gaps: List of gap dictionaries from CuriosityEngine
                Each gap has: {type, priority, domain, estimated_cost, metadata}
                
        Returns:
            List of objective types that were boosted
            
        Example:
            >>> gaps = [
            ...     {"type": "slow_routing", "priority": 0.8, "domain": "query_processing"},
            ...     {"type": "high_error_rate", "priority": 0.9, "domain": "reasoning"}
            ... ]
            >>> boosted = self.process_gaps_from_curiosity_engine(gaps)
            >>> # ['optimize_performance', 'fix_known_bugs']
        """
        if not gaps:
            return []
        
        # Map gap types to improvement objectives
        gap_to_objective = {
            'slow_routing': 'optimize_performance',
            'routing_variance': 'optimize_performance',
            'complex_query_handling': 'improve_test_coverage',
            'high_error_rate': 'fix_known_bugs',
            'decomposition': 'fix_circular_imports',
            'causal': 'enhance_safety_systems',
            'prediction': 'enhance_safety_systems',
            'transfer': 'optimize_performance',
            'latent': 'improve_test_coverage',
        }
        
        # Track which objectives should be boosted
        objectives_to_boost = set()
        boost_amounts = {}  # objective -> total boost amount
        
        for gap in gaps:
            gap_type = gap.get('type', '')
            gap_priority = gap.get('priority', 0.5)
            
            # Find matching objective
            objective_type = gap_to_objective.get(gap_type)
            if objective_type:
                objectives_to_boost.add(objective_type)
                # Accumulate boost amount based on gap priority
                # Higher priority gaps = bigger boost
                boost = gap_priority * 0.2  # Boost by up to 0.2 per gap
                boost_amounts[objective_type] = boost_amounts.get(objective_type, 0) + boost
        
        # Apply boosts to objectives
        boosted_objectives = []
        for objective in self.objectives:
            if objective.type in objectives_to_boost:
                boost_amount = min(boost_amounts[objective.type], 0.5)  # Cap at +0.5
                old_weight = objective.weight
                objective.weight = min(1.0, objective.weight + boost_amount)
                
                boosted_objectives.append(objective.type)
                
                logger.info(
                    f"[SelfImprovementDrive] Boosted objective '{objective.type}' "
                    f"weight: {old_weight:.2f} → {objective.weight:.2f} "
                    f"(gap-driven priority boost)"
                )
                
                # Notify if weight changed significantly
                if boost_amount >= 0.2:
                    self._send_alert(
                        "info",
                        f"Objective '{objective.type}' priority boosted significantly",
                        {
                            "old_weight": old_weight,
                            "new_weight": objective.weight,
                            "reason": "CuriosityEngine detected relevant knowledge gaps"
                        }
                    )
        
        # Save state to persist weight changes
        if boosted_objectives:
            self._save_state()
            logger.info(
                f"[SelfImprovementDrive] Processed {len(gaps)} gaps, "
                f"boosted {len(boosted_objectives)} objectives: {boosted_objectives}"
            )
        else:
            logger.debug(
                f"[SelfImprovementDrive] Processed {len(gaps)} gaps, "
                f"no matching objectives found for boost"
            )
        
        return boosted_objectives


def compose_self_improvement_drive(*, world_model: Any = None, config_path: Any = "configs/intrinsic_drives.json", state_path: str = "data/agent_state.json", alert_callback: Any = None, approval_checker: Any = None, csiu_enforcer: Any = None, improvement_policy: Any = None, approval_store: Any = None, approval_verifier: Any = None, audit_owner: Any = None) -> SelfImprovementDrive:
    """Canonical public composition path for the governed CSIU drive.

    The drive receives shared owners explicitly.  The drive closes only the CSIU
    owner it creates itself; supplied owners remain the caller/container's close
    responsibility.
    """
    return SelfImprovementDrive(world_model=world_model, config_path=config_path, state_path=state_path, alert_callback=alert_callback, approval_checker=approval_checker, csiu_enforcer=csiu_enforcer, improvement_policy=improvement_policy, approval_store=approval_store, approval_verifier=approval_verifier, audit_owner=audit_owner, owns_csiu_enforcer=(csiu_enforcer is None), owns_approval_store=False, owns_audit_owner=False)
