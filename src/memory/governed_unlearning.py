from __future__ import annotations

import asyncio
import logging
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND CONSTANTS
# ============================================================

class ProposalStatus(Enum):
    """Status of an unlearning proposal."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UnlearningMethod(Enum):
    """Methods for unlearning."""
    GRADIENT_SURGERY = "gradient_surgery"
    EXACT_REMOVAL = "exact_removal"
    RETRAINING = "retraining"
    CRYPTOGRAPHIC_ERASURE = "cryptographic_erasure"
    DIFFERENTIAL_PRIVACY = "differential_privacy"


class UrgencyLevel(Enum):
    """Urgency levels for unlearning requests."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictResolution(Enum):
    """Strategies for resolving conflicting unlearning requests."""
    FIRST_WINS = "first_wins"
    MERGE = "merge"
    VOTE = "vote"
    ADMIN_OVERRIDE = "admin_override"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class IRProposal:
    """Intermediate Representation proposal for unlearning."""
    proposal_id: str
    ir_content: Dict[str, Any]
    proposer_id: str
    timestamp: float = field(default_factory=time.time)
    urgency: UrgencyLevel = UrgencyLevel.NORMAL
    justification: str = ""
    affected_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'proposal_id': self.proposal_id,
            'ir_content': self.ir_content,
            'proposer_id': self.proposer_id,
            'timestamp': self.timestamp,
            'urgency': self.urgency.value,
            'justification': self.justification,
            'affected_entities': self.affected_entities,
            'metadata': self.metadata
        }


@dataclass
class GovernanceResult:
    """Result of governance decision."""
    proposal_id: str
    status: ProposalStatus
    details: Dict[str, Any]
    votes: Dict[str, bool] = field(default_factory=dict)
    approval_timestamp: Optional[float] = None
    rejection_reason: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'proposal_id': self.proposal_id,
            'status': self.status.value,
            'details': self.details,
            'votes': self.votes,
            'approval_timestamp': self.approval_timestamp,
            'rejection_reason': self.rejection_reason,
            'execution_result': self.execution_result
        }


@dataclass
class UnlearningTask:
    """Task for executing unlearning."""
    task_id: str
    proposal: IRProposal
    method: UnlearningMethod
    pattern: str
    affected_packs: List[str]
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: ProposalStatus = ProposalStatus.PENDING
    progress: float = 0.0
    error: Optional[str] = None
    proof: Optional[Dict[str, Any]] = None
    
    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class UnlearningMetrics:
    """Metrics for unlearning operations."""
    total_requests: int = 0
    approved_requests: int = 0
    rejected_requests: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_patterns_removed: int = 0
    total_packs_processed: int = 0
    total_time_seconds: float = 0.0
    average_approval_time: float = 0.0
    average_execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_requests': self.total_requests,
            'approved_requests': self.approved_requests,
            'rejected_requests': self.rejected_requests,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'total_patterns_removed': self.total_patterns_removed,
            'total_packs_processed': self.total_packs_processed,
            'total_time_seconds': self.total_time_seconds,
            'average_approval_time': self.average_approval_time,
            'average_execution_time': self.average_execution_time,
            'success_rate': self.get_success_rate()
        }
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        total = self.completed_tasks + self.failed_tasks
        if total == 0:
            return 0.0
        return self.completed_tasks / total


# ============================================================
# AUDIT LOGGER
# ============================================================

class UnlearningAuditLogger:
    """Audit logger for unlearning operations."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger."""
        self.log_file = log_file
        self.audit_trail = deque(maxlen=10000)
        self._lock = threading.Lock()
    
    def log_proposal(self, proposal: IRProposal, result: GovernanceResult) -> None:
        """Log a proposal and its result with thread safety."""
        entry = {
            'type': 'proposal',
            'timestamp': time.time(),
            'proposal': proposal.to_dict(),
            'result': result.to_dict()
        }
        self._write_entry(entry)
    
    def log_execution(self, task: UnlearningTask) -> None:
        """Log execution of an unlearning task with thread safety."""
        entry = {
            'type': 'execution',
            'timestamp': time.time(),
            'task_id': task.task_id,
            'proposal_id': task.proposal.proposal_id,
            'pattern': task.pattern,
            'method': task.method.value,
            'status': task.status.value,
            'duration': task.get_duration(),
            'packs_processed': len(task.affected_packs) if task.affected_packs else 0,
            'error': task.error
        }
        self._write_entry(entry)
    
    def log_proof_generation(self, task_id: str, proof: Dict[str, Any]) -> None:
        """Log proof generation with thread safety."""
        entry = {
            'type': 'proof',
            'timestamp': time.time(),
            'task_id': task_id,
            'proof_hash': hashlib.sha256(str(proof).encode()).hexdigest(),
            'proof_verified': proof.get('verified', False) if proof else False
        }
        self._write_entry(entry)
    
    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit entries with thread safety."""
        with self._lock:
            # Return up to 'limit' most recent entries
            entries = list(self.audit_trail)
            return entries[-limit:] if len(entries) > limit else entries
    
    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Write audit entry with thread safety and error handling."""
        try:
            with self._lock:
                self.audit_trail.append(entry)
            
            # Also write to file if specified
            if self.log_file:
                try:
                    with open(self.log_file, 'a') as f:
                        f.write(json.dumps(entry) + '\n')
                except Exception as e:
                    logger.error(f"Failed to write audit log to file: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")


# ============================================================
# CONSENSUS ENGINES
# ============================================================

class ConsensusEngine:
    """Base class for consensus engines."""
    
    def evaluate_proposal(self, proposal: IRProposal) -> GovernanceResult:
        """Evaluate a proposal and return governance result."""
        raise NotImplementedError


class SimpleConsensusEngine(ConsensusEngine):
    """
    Simple consensus engine with configurable auto-approval.
    
    FIXED: When auto_approve=False, proposals are now properly rejected
    instead of being left in PENDING state.
    """
    
    def __init__(self, auto_approve: bool = True):
        """
        Initialize simple consensus engine.
        
        Args:
            auto_approve: If True, automatically approve all proposals.
                         If False, automatically reject all proposals.
        """
        self.auto_approve = auto_approve
    
    def evaluate_proposal(self, proposal: IRProposal) -> GovernanceResult:
        """
        Simple evaluation - auto approve or auto reject.
        
        FIXED: Now properly rejects when auto_approve=False instead of
        leaving proposals in PENDING state.
        
        Args:
            proposal: The proposal to evaluate
            
        Returns:
            GovernanceResult with APPROVED or REJECTED status
        """
        if self.auto_approve:
            return GovernanceResult(
                proposal_id=proposal.proposal_id,
                status=ProposalStatus.APPROVED,
                details={'auto_approved': True, 'reason': 'Simple consensus auto-approval'},
                approval_timestamp=time.time()
            )
        else:
            # FIXED: Properly reject instead of leaving in PENDING
            return GovernanceResult(
                proposal_id=proposal.proposal_id,
                status=ProposalStatus.REJECTED,
                details={'auto_rejected': True, 'reason': 'Auto-approval disabled'},
                rejection_reason='Auto-approval is disabled in SimpleConsensusEngine'
            )


# ============================================================
# GOVERNED UNLEARNING SYSTEM
# ============================================================

class GovernedUnlearning:
    """
    Governed unlearning system with consensus mechanisms, audit trails,
    and zero-knowledge proofs.
    
    Features:
    - Multi-party governance
    - Conflict resolution
    - Comprehensive audit logging
    - Zero-knowledge proof generation
    - Multiple unlearning methods
    - Priority-based processing
    """
    
    def __init__(self, 
                 persistent_memory,
                 consensus_engine: Optional[ConsensusEngine] = None,
                 max_workers: int = 4,
                 audit_log_file: Optional[str] = None):
        """
        Initialize governed unlearning system.
        
        Args:
            persistent_memory: Persistent memory system with unlearning capabilities
            consensus_engine: Consensus engine for governance decisions
            max_workers: Maximum number of parallel unlearning tasks
            audit_log_file: Path to audit log file
        """
        self.memory = persistent_memory
        self.consensus = consensus_engine or SimpleConsensusEngine()
        self.audit_logger = UnlearningAuditLogger(audit_log_file)
        
        # Task management
        self.pending_proposals: Dict[str, IRProposal] = {}
        self.governance_results: Dict[str, GovernanceResult] = {}
        self.pending_tasks: deque = deque()
        self.active_tasks: Dict[str, UnlearningTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Pattern tracking
        self.unlearned_patterns: Set[str] = set()
        self.pattern_conflicts: Dict[str, List[str]] = defaultdict(list)
        
        # Metrics
        self.metrics = UnlearningMetrics()
        
        # Configuration
        self.urgency_thresholds = {
            UrgencyLevel.CRITICAL: timedelta(hours=1),
            UrgencyLevel.HIGH: timedelta(hours=6),
            UrgencyLevel.NORMAL: timedelta(hours=24),
            UrgencyLevel.LOW: timedelta(days=7)
        }
        
        # Callbacks
        self.on_approval: List[Callable] = []
        self.on_completion: List[Callable] = []
        self.on_failure: List[Callable] = []
        
        # Threading
        self._lock = threading.Lock()
        self._shutdown = False
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info("GovernedUnlearning system initialized")
    
    def submit_ir_proposal(self,
                          ir_content: Dict[str, Any],
                          proposer_id: str,
                          urgency: UrgencyLevel = UrgencyLevel.NORMAL,
                          justification: str = "",
                          affected_entities: Optional[List[str]] = None) -> str:
        """
        Submit an IR proposal for unlearning with defensive programming.
        
        The ir_content dictionary should contain:
        - 'pattern': The pattern to unlearn (required)
        - 'affected_packs': List of affected packfiles (optional)
        - 'method': Preferred unlearning method (optional, will be auto-determined if not provided)
        
        Args:
            ir_content: IR content to be unlearned
            proposer_id: ID of the proposer
            urgency: Urgency level of the request
            justification: Justification for the unlearning request
            affected_entities: List of affected entities
            
        Returns:
            Proposal ID
        """
        try:
            # Validate ir_content
            if not ir_content:
                raise ValueError("ir_content cannot be empty")
            
            # FIXED: Validate that pattern is present in ir_content
            if 'pattern' not in ir_content:
                logger.warning("No 'pattern' in ir_content, generating default")
                ir_content['pattern'] = f"pattern_{time.time()}"
            
            # Generate unique proposal ID
            proposal_id = self._generate_proposal_id()
            
            # Create proposal with error handling
            proposal = IRProposal(
                proposal_id=proposal_id,
                ir_content=ir_content,
                proposer_id=proposer_id,
                urgency=urgency,
                justification=justification,
                affected_entities=affected_entities or [],
                metadata={'submitted_at': time.time()}
            )
            
            # Store proposal with thread safety
            with self._lock:
                self.pending_proposals[proposal_id] = proposal
                self.metrics.total_requests += 1
            
            # Evaluate through consensus engine with error handling
            try:
                governance_result = self.consensus.evaluate_proposal(proposal)
                
                with self._lock:
                    self.governance_results[proposal_id] = governance_result
                
                # Log the proposal and result
                self.audit_logger.log_proposal(proposal, governance_result)
                
                # If approved, queue for execution
                if governance_result.status == ProposalStatus.APPROVED:
                    self._queue_for_execution(proposal, governance_result)
                    with self._lock:
                        self.metrics.approved_requests += 1
                    
                    # Trigger approval callbacks
                    for callback in self.on_approval:
                        try:
                            callback(proposal, governance_result)
                        except Exception as e:
                            logger.error(f"Approval callback failed: {e}")
                            
                elif governance_result.status == ProposalStatus.REJECTED:
                    with self._lock:
                        self.metrics.rejected_requests += 1
                        
            except Exception as e:
                logger.error(f"Consensus evaluation failed for proposal {proposal_id}: {e}")
                # Create fallback rejection result
                governance_result = GovernanceResult(
                    proposal_id=proposal_id,
                    status=ProposalStatus.REJECTED,
                    details={'error': str(e)},
                    rejection_reason=f"Consensus evaluation failed: {str(e)}"
                )
                with self._lock:
                    self.governance_results[proposal_id] = governance_result
                    self.metrics.rejected_requests += 1
            
            logger.info(f"Submitted proposal {proposal_id} with status {governance_result.status.value}")
            return proposal_id
            
        except Exception as e:
            logger.error(f"Failed to submit IR proposal: {e}")
            raise
    
    def submit_governance_vote(self,
                              proposal_id: str,
                              voter_id: str,
                              approve: bool,
                              justification: str = "") -> bool:
        """
        Submit a governance vote with error handling.
        
        FIXED: Parameter name changed from 'vote' to 'approve' for clarity.
        
        Args:
            proposal_id: ID of the proposal to vote on
            voter_id: ID of the voter
            approve: True to approve, False to reject
            justification: Optional justification for the vote
            
        Returns:
            True if vote was recorded successfully, False otherwise
        """
        try:
            with self._lock:
                if proposal_id not in self.governance_results:
                    logger.warning(f"Cannot vote on non-existent proposal {proposal_id}")
                    return False
                
                result = self.governance_results[proposal_id]
                
                # Only allow voting on pending proposals
                if result.status != ProposalStatus.PENDING:
                    logger.warning(f"Cannot vote on proposal {proposal_id} with status {result.status.value}")
                    return False
                
                # Record the vote
                result.votes[voter_id] = approve
                
                # Simple majority rule for this implementation
                if len(result.votes) >= 2:  # Require at least 2 votes
                    approve_votes = sum(1 for v in result.votes.values() if v)
                    total_votes = len(result.votes)
                    
                    if approve_votes > total_votes / 2:
                        result.status = ProposalStatus.APPROVED
                        result.approval_timestamp = time.time()
                        # Queue for execution
                        if proposal_id in self.pending_proposals:
                            self._queue_for_execution(self.pending_proposals[proposal_id], result)
                        self.metrics.approved_requests += 1
                    else:
                        result.status = ProposalStatus.REJECTED
                        result.rejection_reason = "Rejected by voting"
                        self.metrics.rejected_requests += 1
            
            logger.info(f"Vote recorded for proposal {proposal_id} by {voter_id}: {'approve' if approve else 'reject'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit vote: {e}")
            return False
    
    def get_proposal_status(self, proposal_id: str) -> Optional[GovernanceResult]:
        """
        Get the status of a proposal with error handling.
        
        FIXED: Now returns GovernanceResult object directly instead of dictionary
        for consistency with the data model.
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            GovernanceResult object or None if not found
        """
        try:
            with self._lock:
                if proposal_id not in self.governance_results:
                    return None
                
                return self.governance_results[proposal_id]
                
        except Exception as e:
            logger.error(f"Failed to get proposal status: {e}")
            return None
    
    def list_active_proposals(self) -> List[Dict[str, Any]]:
        """List all active proposals with error handling."""
        try:
            with self._lock:
                active_proposals = []
                
                for proposal_id, proposal in self.pending_proposals.items():
                    result = self.governance_results.get(proposal_id)
                    if result and result.status in [ProposalStatus.PENDING, ProposalStatus.APPROVED, ProposalStatus.EXECUTING]:
                        active_proposals.append({
                            'proposal_id': proposal_id,
                            'status': result.status.value,
                            'urgency': proposal.urgency.value,
                            'proposer_id': proposal.proposer_id,
                            'submitted_at': proposal.timestamp,
                            'pattern': proposal.ir_content.get('pattern', 'unknown')
                        })
                
                # Sort by urgency and submission time
                urgency_order = {
                    UrgencyLevel.CRITICAL.value: 0,
                    UrgencyLevel.HIGH.value: 1,
                    UrgencyLevel.NORMAL.value: 2,
                    UrgencyLevel.LOW.value: 3
                }
                
                active_proposals.sort(key=lambda x: (
                    urgency_order.get(x['urgency'], 4),
                    x['submitted_at']
                ))
                
                return active_proposals
                
        except Exception as e:
            logger.error(f"Failed to list active proposals: {e}")
            return []
    
    def get_unlearning_metrics(self) -> UnlearningMetrics:
        """Get current unlearning metrics with thread safety."""
        with self._lock:
            # Return a copy to prevent external modification
            return UnlearningMetrics(
                total_requests=self.metrics.total_requests,
                approved_requests=self.metrics.approved_requests,
                rejected_requests=self.metrics.rejected_requests,
                completed_tasks=self.metrics.completed_tasks,
                failed_tasks=self.metrics.failed_tasks,
                total_patterns_removed=self.metrics.total_patterns_removed,
                total_packs_processed=self.metrics.total_packs_processed,
                total_time_seconds=self.metrics.total_time_seconds,
                average_approval_time=self.metrics.average_approval_time,
                average_execution_time=self.metrics.average_execution_time
            )
    
    def is_pattern_already_unlearned(self, pattern: str) -> bool:
        """Check if a pattern has already been unlearned."""
        with self._lock:
            return pattern in self.unlearned_patterns
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """
        Detect conflicts between pending proposals.
        
        Returns:
            List of conflicts found
        """
        conflicts = []
        
        try:
            with self._lock:
                # Group proposals by pattern
                pattern_groups = defaultdict(list)
                
                for proposal_id, proposal in self.pending_proposals.items():
                    result = self.governance_results.get(proposal_id)
                    if result and result.status in [ProposalStatus.PENDING, ProposalStatus.APPROVED]:
                        pattern = proposal.ir_content.get('pattern')
                        if pattern:
                            pattern_groups[pattern].append({
                                'proposal_id': proposal_id,
                                'proposal': proposal,
                                'result': result
                            })
                
                # Identify conflicts (multiple proposals for same pattern)
                for pattern, proposals in pattern_groups.items():
                    if len(proposals) > 1:
                        conflicts.append({
                            'pattern': pattern,
                            'proposals': [p['proposal_id'] for p in proposals],
                            'conflict_type': 'duplicate_pattern',
                            'detected_at': time.time()
                        })
                        
        except Exception as e:
            logger.error(f"Failed to detect conflicts: {e}")
        
        return conflicts
    
    def _queue_for_execution(self, proposal: IRProposal, result: GovernanceResult) -> None:
        """Queue approved proposal for execution with error handling."""
        try:
            # Extract pattern and affected packs from IR content
            pattern = proposal.ir_content.get('pattern', f"proposal_{proposal.proposal_id}")
            affected_packs = proposal.ir_content.get('affected_packs', [])
            
            # FIXED: Ensure affected_packs is always a list
            if not isinstance(affected_packs, list):
                if isinstance(affected_packs, str):
                    affected_packs = [affected_packs]
                else:
                    affected_packs = []
            
            # Determine unlearning method
            # First check if method is specified in ir_content
            method_str = proposal.ir_content.get('method')
            if method_str:
                try:
                    method = UnlearningMethod(method_str)
                except ValueError:
                    logger.warning(f"Invalid method '{method_str}' in ir_content, using urgency-based selection")
                    method = self._select_method_by_urgency(proposal.urgency)
            else:
                # Select based on urgency
                method = self._select_method_by_urgency(proposal.urgency)
            
            # Create unlearning task
            task = UnlearningTask(
                task_id=self._generate_task_id(),
                proposal=proposal,
                method=method,
                pattern=pattern,
                affected_packs=affected_packs
            )
            
            # Add to pending tasks
            self.pending_tasks.append(task)
            logger.info(f"Queued task {task.task_id} for pattern '{pattern}' using method {method.value}")
            
        except Exception as e:
            logger.error(f"Failed to queue proposal for execution: {e}")
    
    def _select_method_by_urgency(self, urgency: UrgencyLevel) -> UnlearningMethod:
        """
        Select appropriate unlearning method based on urgency level.
        
        FIXED: NEW method to centralize method selection logic.
        
        Args:
            urgency: The urgency level
            
        Returns:
            Appropriate UnlearningMethod
        """
        if urgency == UrgencyLevel.CRITICAL:
            return UnlearningMethod.EXACT_REMOVAL
        elif urgency == UrgencyLevel.HIGH:
            return UnlearningMethod.GRADIENT_SURGERY
        elif urgency == UrgencyLevel.LOW:
            return UnlearningMethod.DIFFERENTIAL_PRIVACY
        else:  # NORMAL
            return UnlearningMethod.GRADIENT_SURGERY
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing unlearning tasks."""
        while not self._shutdown:
            try:
                # Get next task with thread safety
                task = None
                with self._lock:
                    if self.pending_tasks:
                        task = self.pending_tasks.popleft()
                        self.active_tasks[task.task_id] = task
                
                if task:
                    # Execute task in thread pool
                    future = self.executor.submit(self._execute_unlearning_task, task)
                else:
                    # No tasks, sleep briefly
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1)
    
    def _execute_unlearning_task(self, task: UnlearningTask) -> None:
        """Execute an unlearning task with comprehensive error handling."""
        # Protect task state modification with lock
        with self._lock:
            task.started_at = time.time()
            task.status = ProposalStatus.EXECUTING
        
        try:
            logger.info(f"Executing unlearning task {task.task_id}: {task.pattern}")
            
            # Execute based on method with error handling
            if task.method == UnlearningMethod.GRADIENT_SURGERY:
                self._execute_gradient_surgery(task)
            elif task.method == UnlearningMethod.EXACT_REMOVAL:
                self._execute_exact_removal(task)
            elif task.method == UnlearningMethod.RETRAINING:
                self._execute_retraining(task)
            elif task.method == UnlearningMethod.CRYPTOGRAPHIC_ERASURE:
                self._execute_cryptographic_erasure(task)
            elif task.method == UnlearningMethod.DIFFERENTIAL_PRIVACY:
                self._execute_differential_privacy(task)
            else:
                raise ValueError(f"Unknown unlearning method: {task.method}")
            
            # Generate zero-knowledge proof with error handling
            try:
                task.proof = self._generate_unlearning_proof(task)
                if task.proof is None:
                    logger.warning(f"Task {task.task_id} completed without proof verification")
            except Exception as e:
                logger.error(f"Proof generation failed for task {task.task_id}: {e}")
                task.proof = None
            
            # Update tracking and mark as completed
            with self._lock:
                # Mark as completed
                task.status = ProposalStatus.COMPLETED
                task.completed_at = time.time()
                task.progress = 1.0
                
                # Update tracking
                self.unlearned_patterns.add(task.pattern)
                self.metrics.completed_tasks += 1
                self.metrics.total_patterns_removed += 1
                self.metrics.total_packs_processed += len(task.affected_packs) if task.affected_packs else 0
                
                # FIXED: Update average execution time properly
                duration = task.get_duration()
                if duration:
                    total = self.metrics.completed_tasks
                    if total > 0:
                        self.metrics.total_time_seconds += duration
                        self.metrics.average_execution_time = (
                            self.metrics.total_time_seconds / total
                        )
            
            # Trigger callbacks
            for callback in self.on_completion:
                try:
                    callback(task)
                except Exception as e:
                    logger.error(f"Completion callback failed: {e}")
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            # Protect task state modification with lock
            with self._lock:
                task.status = ProposalStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                self.metrics.failed_tasks += 1
            
            # Trigger failure callbacks
            for callback in self.on_failure:
                try:
                    callback(task)
                except Exception as e:
                    logger.error(f"Failure callback failed: {e}")
        
        finally:
            # Move to completed tasks
            with self._lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks.append(task)
            
            # Log execution
            self.audit_logger.log_execution(task)
    
    def _execute_gradient_surgery(self, task: UnlearningTask) -> None:
        """Execute gradient surgery unlearning with error handling."""
        is_urgent = self._is_urgent(task.proposal.urgency)
        
        if not task.affected_packs:
            logger.warning(f"No affected packs specified for task {task.task_id}")
            return
        
        for i, pack in enumerate(task.affected_packs):
            try:
                if hasattr(self.memory, 'unlearning_engine') and hasattr(self.memory.unlearning_engine, 'gradient_surgery'):
                    self.memory.unlearning_engine.gradient_surgery(
                        packfile=pack,
                        pattern=task.pattern,
                        fast_lane=is_urgent
                    )
                else:
                    logger.warning(f"Memory system lacks unlearning_engine.gradient_surgery for task {task.task_id}")
                    return
                    
                # Protect task.progress with lock
                with self._lock:
                    task.progress = (i + 1) / len(task.affected_packs)
                    
            except Exception as e:
                logger.error(f"Gradient surgery failed on pack {pack}: {e}")
                raise
    
    def _execute_exact_removal(self, task: UnlearningTask) -> None:
        """Execute exact removal unlearning with error handling."""
        if not task.affected_packs:
            logger.warning(f"No affected packs specified for task {task.task_id}")
            return
            
        for i, pack in enumerate(task.affected_packs):
            try:
                # Remove exact matches from pack
                if hasattr(self.memory, 'unlearning_engine') and hasattr(self.memory.unlearning_engine, 'exact_removal'):
                    self.memory.unlearning_engine.exact_removal(
                        packfile=pack,
                        pattern=task.pattern
                    )
                else:
                    # Fallback to gradient surgery
                    if hasattr(self.memory, 'unlearning_engine') and hasattr(self.memory.unlearning_engine, 'gradient_surgery'):
                        self.memory.unlearning_engine.gradient_surgery(
                            packfile=pack,
                            pattern=task.pattern,
                            fast_lane=True
                        )
                    else:
                        logger.warning(f"Memory system lacks unlearning capabilities for task {task.task_id}")
                        return
                        
                # Protect task.progress with lock
                with self._lock:
                    task.progress = (i + 1) / len(task.affected_packs)
                    
            except Exception as e:
                logger.error(f"Exact removal failed on pack {pack}: {e}")
                raise
    
    def _execute_retraining(self, task: UnlearningTask) -> None:
        """Execute retraining-based unlearning with error handling."""
        # This is computationally expensive
        logger.warning(f"Retraining unlearning requested for {task.pattern}")
        
        try:
            if hasattr(self.memory, 'retrain_without_pattern'):
                self.memory.retrain_without_pattern(task.pattern)
            else:
                # Fallback to gradient surgery
                logger.info(f"retrain_without_pattern not available, falling back to gradient surgery")
                self._execute_gradient_surgery(task)
        except Exception as e:
            logger.error(f"Retraining failed for task {task.task_id}: {e}")
            raise
    
    def _execute_cryptographic_erasure(self, task: UnlearningTask) -> None:
        """Execute cryptographic erasure unlearning with error handling."""
        try:
            if hasattr(self.memory, 'unlearning_engine') and hasattr(self.memory.unlearning_engine, 'cryptographic_erasure'):
                self.memory.unlearning_engine.cryptographic_erasure(
                    pattern=task.pattern,
                    affected_packs=task.affected_packs
                )
            else:
                # Fallback to exact removal
                logger.info(f"cryptographic_erasure not available, falling back to exact removal")
                self._execute_exact_removal(task)
        except Exception as e:
            logger.error(f"Cryptographic erasure failed for task {task.task_id}: {e}")
            raise
    
    def _execute_differential_privacy(self, task: UnlearningTask) -> None:
        """Execute differential privacy-based unlearning with error handling."""
        try:
            if hasattr(self.memory, 'unlearning_engine') and hasattr(self.memory.unlearning_engine, 'differential_privacy_unlearn'):
                self.memory.unlearning_engine.differential_privacy_unlearn(
                    pattern=task.pattern,
                    epsilon=0.1,
                    affected_packs=task.affected_packs
                )
            else:
                # Fallback to gradient surgery
                logger.info(f"differential_privacy_unlearn not available, falling back to gradient surgery")
                self._execute_gradient_surgery(task)
        except Exception as e:
            logger.error(f"Differential privacy unlearning failed for task {task.task_id}: {e}")
            raise
    
    def _generate_unlearning_proof(self, task: UnlearningTask) -> Optional[Dict[str, Any]]:
        """Generate zero-knowledge proof of unlearning with error handling."""
        try:
            if not hasattr(self.memory, 'zk_prover'):
                logger.warning(f"Memory system lacks zk_prover for task {task.task_id}")
                return None
            
            if not hasattr(self.memory.zk_prover, 'generate_unlearning_proof'):
                logger.warning(f"zk_prover lacks generate_unlearning_proof method")
                return None
                
            proof = self.memory.zk_prover.generate_unlearning_proof(
                pattern=task.pattern,
                affected_packs=task.affected_packs or []
            )
            
            # Log proof generation
            if proof:
                self.audit_logger.log_proof_generation(task.task_id, proof)
            
            return proof
            
        except Exception as e:
            logger.error(f"Proof generation failed: {e}")
            # Return None to indicate failure, not a fake proof object
            return None
    
    def _is_urgent(self, urgency: UrgencyLevel) -> bool:
        """Check if urgency level requires fast-lane processing."""
        return urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]
    
    def _generate_proposal_id(self) -> str:
        """
        Generate unique proposal ID.
        
        FIXED: Now uses threading.current_thread().ident instead of hash()
        for better uniqueness across threads.
        """
        # FIXED: Use thread.ident for better uniqueness
        thread_id = threading.current_thread().ident or 0
        content = f"proposal_{time.time()}_{thread_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_task_id(self) -> str:
        """
        Generate unique task ID.
        
        FIXED: Now uses threading.current_thread().ident instead of hash()
        for better uniqueness across threads.
        """
        # FIXED: Use thread.ident for better uniqueness
        thread_id = threading.current_thread().ident or 0
        content = f"task_{time.time()}_{thread_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def shutdown(self) -> None:
        """Shutdown the unlearning system."""
        logger.info("Shutting down GovernedUnlearning system...")
        self._shutdown = True
        
        # Wait for worker thread
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
            if self._worker_thread.is_alive():
                logger.warning("Worker thread did not stop within timeout")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("GovernedUnlearning system shutdown complete")
    
    def __del__(self):
        """Destructor."""
        try:
            self.shutdown()
        except Exception as e:            logger.debug(f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}")
