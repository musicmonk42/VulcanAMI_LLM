# adversarial_formal.py
"""
Adversarial validation and formal verification for VULCAN-AGI Safety Module.
Tests robustness against attacks and verifies formal safety properties.
"""

import logging
import time
import json
import signal
import sys
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
import hashlib
import copy
import itertools
import random
from enum import Enum
from dataclasses import dataclass, field
import threading

from .safety_types import (
    SafetyReport,
    SafetyViolationType,
    ActionType
)

logger = logging.getLogger(__name__)

_ADVERSARIAL_INIT_DONE = False

# ============================================================
# TIMEOUT PROTECTION
# ============================================================

@contextmanager
def timeout(seconds):
    """Context manager for timeout protection.
    
    Raises TimeoutError if operation exceeds time limit.
    Uses signal.SIGALRM on Unix systems, alternative on Windows.
    
    Args:
        seconds: Timeout in seconds
        
    Raises:
        TimeoutError: If operation times out
    """
    # Check if signal.SIGALRM is available (Unix-like systems only)
    has_sigalrm = hasattr(signal, 'SIGALRM')
    
    if not has_sigalrm:
        # Windows or other systems without SIGALRM - use simple timing
        logger.warning("SIGALRM not available, using simple timeout mechanism")
        start_time = time.time()
        
        class TimeoutChecker:
            def __init__(self, timeout_seconds):
                self.timeout_seconds = timeout_seconds
                self.start_time = time.time()
            
            def check(self):
                if time.time() - self.start_time > self.timeout_seconds:
                    raise TimeoutError(f"Operation timed out after {self.timeout_seconds}s")
        
        checker = TimeoutChecker(seconds)
        yield checker
        return
    
    # Unix-like systems with SIGALRM
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))
    
    try:
        yield None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ============================================================
# ADVERSARIAL VALIDATOR
# ============================================================

class AttackType(Enum):
    """Types of adversarial attacks."""
    FGSM = "fgsm"  # Fast Gradient Sign Method
    PGD = "pgd"  # Projected Gradient Descent
    SEMANTIC = "semantic"  # Semantic perturbations
    BOUNDARY = "boundary"  # Boundary exploration
    TROJAN = "trojan"  # Backdoor/trojan patterns
    DEEPFOOL = "deepfool"  # DeepFool algorithm
    CARLINI_WAGNER = "carlini_wagner"  # C&W attack
    JSMA = "jsma"  # Jacobian-based Saliency Map Attack
    UNIVERSAL = "universal"  # Universal adversarial perturbations
    TARGETED = "targeted"  # Targeted misclassification

@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    epsilon: float = 0.1  # Perturbation magnitude
    alpha: float = 0.01  # Step size for iterative attacks
    num_iterations: int = 10  # Number of iterations
    targeted: bool = False  # Whether attack is targeted
    target_class: Optional[int] = None  # Target class for targeted attacks
    confidence: float = 0.0  # Confidence for C&W attack
    max_iterations: int = 100  # Maximum iterations for optimization-based attacks
    early_stop: bool = True  # Early stopping for efficiency

class AdversarialValidator:
    """
    Tests robustness against adversarial attacks.
    Implements multiple attack algorithms and defense evaluation.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, epsilon: float = 0.1, num_attacks: int = 10, 
                 config: Optional[Dict[str, Any]] = None, random_seed: Optional[int] = None):
        """
        Initialize adversarial validator.
        
        Args:
            epsilon: Default perturbation magnitude
            num_attacks: Number of attack samples to generate
            config: Additional configuration
            random_seed: Random seed for reproducibility
        """
        if getattr(self, "_initialized", False):
            return
        
        self.config = config or {}
        self.epsilon = epsilon
        self.num_attacks = num_attacks
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Attack configurations
        self.default_attack_config = AttackConfig(epsilon=epsilon)
        self.attack_configs = self._initialize_attack_configs()
        
        # Attack methods
        self.attack_methods = {
            AttackType.FGSM: self._fgsm_attack,
            AttackType.PGD: self._pgd_attack,
            AttackType.SEMANTIC: self._semantic_attack,
            AttackType.BOUNDARY: self._boundary_attack,
            AttackType.TROJAN: self._trojan_attack,
            AttackType.DEEPFOOL: self._deepfool_attack,
            AttackType.CARLINI_WAGNER: self._carlini_wagner_attack,
            AttackType.JSMA: self._jsma_attack,
            AttackType.UNIVERSAL: self._universal_attack,
            AttackType.TARGETED: self._targeted_attack
        }
        
        # History tracking with size limits
        self.attack_history = deque(maxlen=1000)
        self.vulnerability_patterns = defaultdict(lambda: deque(maxlen=100))
        
        # Metrics
        self.attack_metrics = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'average_perturbation': 0.0,
            'average_queries': 0
        })
        
        # Defense mechanisms for testing
        self.defense_mechanisms = self._initialize_defenses()
        
        # --- Start Singleton properties ---
        self._properties = set()
        self._invariants = set()
        self._initialized = True
        # --- End Singleton properties ---
        
        logger.info(f"AdversarialValidator initialized with epsilon={epsilon}")
    
    # --- Start Singleton methods ---
    def add_property(self, name: str):
        if name not in self._properties:
            self._properties.add(name)
            logger.info(f"Added safety property: {name}")
        else:
            logger.debug(f"Skipped duplicate safety property: {name}")

    def add_invariant(self, name: str):
        if name not in self._invariants:
            self._invariants.add(name)
            logger.info(f"Added invariant: {name}")
        else:
            logger.debug(f"Skipped duplicate invariant: {name}")

    def list_properties(self):
        return sorted(self._properties)

    def list_invariants(self):
        return sorted(self._invariants)
    # --- End Singleton methods ---

    def _initialize_attack_configs(self) -> Dict[AttackType, AttackConfig]:
        """Initialize configurations for different attack types."""
        return {
            AttackType.FGSM: AttackConfig(
                epsilon=self.epsilon,
                num_iterations=1
            ),
            AttackType.PGD: AttackConfig(
                epsilon=self.epsilon,
                alpha=self.epsilon / 4,
                num_iterations=20
            ),
            AttackType.DEEPFOOL: AttackConfig(
                epsilon=self.epsilon,
                max_iterations=50,
                early_stop=True
            ),
            AttackType.CARLINI_WAGNER: AttackConfig(
                confidence=0.0,
                max_iterations=100,
                alpha=0.01
            ),
            AttackType.JSMA: AttackConfig(
                epsilon=self.epsilon,
                max_iterations=100
            )
        }
    
    def _initialize_defenses(self) -> Dict[str, Callable]:
        """Initialize defense mechanisms for evaluation."""
        return {
            'input_validation': self._defense_input_validation,
            'gradient_masking': self._defense_gradient_masking,
            'adversarial_training': self._defense_adversarial_training,
            'certified_defense': self._defense_certified,
            'detection': self._defense_detection
        }
    
    def validate_robustness(self, action: Dict[str, Any], context: Dict[str, Any], 
                           validator: Optional[Callable] = None,
                           attack_types: Optional[List[AttackType]] = None,
                           timeout_per_attack: float = 5.0) -> SafetyReport:
        """
        Test action robustness against adversarial perturbations.
        
        Args:
            action: Action to test
            context: Context for validation
            validator: Custom validator function
            attack_types: Specific attacks to use (None = all)
            timeout_per_attack: Timeout in seconds per attack type
            
        Returns:
            Safety report with robustness assessment
        """
        violations = []
        reasons = []
        min_confidence = 1.0
        attack_results = []
        
        # Select attack types
        if attack_types is None:
            attack_types = [
                AttackType.FGSM, 
                AttackType.PGD, 
                AttackType.SEMANTIC,
                AttackType.BOUNDARY,
                AttackType.TROJAN
            ]
        
        for attack_type in attack_types:
            if attack_type not in self.attack_methods:
                continue
            
            attack_method = self.attack_methods[attack_type]
            attack_config = self.attack_configs.get(attack_type, self.default_attack_config)
            
            try:
                # CRITICAL: Add timeout protection
                with timeout(int(timeout_per_attack)):
                    perturbed_actions = attack_method(action, context, attack_config)
                
                # Test each perturbed action
                for i, perturbed in enumerate(perturbed_actions[:self.num_attacks]):
                    # Validate perturbed action
                    if validator:
                        safe, reason, confidence = validator(perturbed, context)
                    else:
                        safe, reason, confidence = self._default_validate(perturbed, context)
                    
                    if not safe:
                        violations.append(SafetyViolationType.ADVERSARIAL)
                        reasons.append(f"{attack_type.value} attack #{i}: {reason}")
                        min_confidence = min(min_confidence, confidence)
                        
                        # Record successful attack
                        self._record_attack(attack_type, action, perturbed, True, reason)
                        
                        attack_results.append({
                            'attack_type': attack_type.value,
                            'index': i,
                            'successful': True,
                            'perturbation_norm': self._calculate_perturbation_norm(action, perturbed)
                        })
                    else:
                        # Record failed attack
                        self._record_attack(attack_type, action, perturbed, False, "Defense successful")
                        
                        attack_results.append({
                            'attack_type': attack_type.value,
                            'index': i,
                            'successful': False
                        })
            
            except TimeoutError as e:
                logger.warning(f"Attack {attack_type.value} timed out: {e}")
                reasons.append(f"{attack_type.value} attack timed out")
                attack_results.append({
                    'attack_type': attack_type.value,
                    'successful': False,
                    'timeout': True
                })
            except Exception as e:
                logger.error(f"Attack {attack_type.value} failed: {e}")
                reasons.append(f"{attack_type.value} attack failed: {str(e)}")
        
        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(attack_results)
        
        # Identify vulnerability patterns
        patterns = self._identify_vulnerability_patterns(violations, attack_results)
        
        # Generate mitigation strategies
        mitigations = self._suggest_mitigations(violations, patterns)
        
        return SafetyReport(
            safe=len(violations) == 0,
            confidence=min_confidence,
            violations=violations,
            reasons=reasons,
            mitigations=mitigations,
            metadata={
                'attacks_tested': [at.value for at in attack_types],
                'robustness_score': robustness_score,
                'attack_results': attack_results,
                'vulnerability_patterns': patterns
            }
        )
    
    def _fgsm_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                     config: AttackConfig) -> List[Dict[str, Any]]:
        """
        Fast Gradient Sign Method attack.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of perturbed actions
        """
        perturbations = []
        
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            
            for _ in range(min(3, self.num_attacks)):
                # Generate random gradient direction (simulated)
                gradient = np.random.randn(*embedding.shape)
                gradient = gradient / (np.linalg.norm(gradient) + 1e-10)
                
                # Apply FGSM perturbation
                perturbation = config.epsilon * np.sign(gradient)
                perturbed_embedding = embedding + perturbation
                
                # Clip to valid range
                perturbed_embedding = np.clip(perturbed_embedding, -1, 1)
                
                perturbed_action = copy.deepcopy(action)
                perturbed_action['embedding'] = perturbed_embedding.tolist()
                perturbed_action['adversarial'] = True
                perturbed_action['attack_type'] = 'fgsm'
                
                perturbations.append(perturbed_action)
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.FGSM]['attempts'] += len(perturbations)
        
        return perturbations
    
    def _pgd_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                    config: AttackConfig) -> List[Dict[str, Any]]:
        """
        Projected Gradient Descent attack.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of perturbed actions
        """
        perturbations = []
        
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            
            for _ in range(min(2, self.num_attacks)):
                perturbed = embedding.copy()
                
                # Random initialization within epsilon ball
                delta = np.random.uniform(-config.epsilon, config.epsilon, embedding.shape)
                perturbed = embedding + delta
                
                # Iterative attack
                for step in range(config.num_iterations):
                    # Compute gradient (simulated)
                    gradient = np.random.randn(*embedding.shape)
                    gradient = gradient / (np.linalg.norm(gradient) + 1e-10)
                    
                    # Update with step size
                    perturbed += config.alpha * np.sign(gradient)
                    
                    # Project to epsilon ball
                    delta = perturbed - embedding
                    delta_norm = np.linalg.norm(delta)
                    if delta_norm > config.epsilon:
                        delta = delta * (config.epsilon / delta_norm)
                    perturbed = embedding + delta
                    
                    # Clip to valid range
                    perturbed = np.clip(perturbed, -1, 1)
                
                perturbed_action = copy.deepcopy(action)
                perturbed_action['embedding'] = perturbed.tolist()
                perturbed_action['adversarial'] = True
                perturbed_action['attack_type'] = 'pgd'
                perturbed_action['iterations'] = config.num_iterations
                
                perturbations.append(perturbed_action)
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.PGD]['attempts'] += len(perturbations)
        
        return perturbations
    
    def _semantic_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                        config: AttackConfig) -> List[Dict[str, Any]]:
        """
        Semantic-preserving attacks that change meaning without changing structure.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of semantically perturbed actions
        """
        perturbations = []
        
        action_type = action.get('type', '')
        
        # Create semantically similar but potentially unsafe variants
        semantic_variants = {
            ActionType.EXPLORE: [ActionType.OPTIMIZE, ActionType.MAINTAIN],
            ActionType.OPTIMIZE: [ActionType.EXPLORE, ActionType.WAIT],
            ActionType.MAINTAIN: [ActionType.WAIT, ActionType.SAFE_FALLBACK],
            'read': ['write', 'delete'],
            'approve': ['reject', 'defer'],
            'allow': ['deny', 'restrict']
        }
        
        # Apply semantic substitutions
        for original, variants in semantic_variants.items():
            if str(action_type) == str(original) or action_type == original:
                for variant in variants[:min(2, self.num_attacks)]:
                    perturbed_action = copy.deepcopy(action)
                    perturbed_action['type'] = variant
                    perturbed_action['adversarial'] = True
                    perturbed_action['attack_type'] = 'semantic'
                    perturbed_action['original_type'] = action_type
                    perturbations.append(perturbed_action)
        
        # Synonym replacement for text fields
        if 'text' in action or 'description' in action:
            text_field = 'text' if 'text' in action else 'description'
            original_text = action[text_field]
            
            # Adversarial word substitutions
            substitutions = {
                'safe': 'dangerous',
                'allow': 'deny',
                'approve': 'reject',
                'increase': 'decrease',
                'positive': 'negative',
                'help': 'harm'
            }
            
            for original_word, replacement in substitutions.items():
                if original_word in str(original_text).lower():
                    perturbed_action = copy.deepcopy(action)
                    perturbed_action[text_field] = str(original_text).replace(
                        original_word, replacement
                    )
                    perturbed_action['adversarial'] = True
                    perturbed_action['attack_type'] = 'semantic_text'
                    perturbations.append(perturbed_action)
                    
                    if len(perturbations) >= self.num_attacks:
                        break
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.SEMANTIC]['attempts'] += len(perturbations)
        
        return perturbations
    
    def _boundary_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                        config: AttackConfig) -> List[Dict[str, Any]]:
        """
        Boundary attack - explore decision boundaries.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of boundary-perturbed actions
        """
        perturbations = []
        
        # Test boundary conditions for numerical fields
        boundary_fields = ['confidence', 'uncertainty', 'risk_score', 'threshold']
        
        for field in boundary_fields:
            if field in action:
                original_value = action[field]
                
                # Test extreme values
                extreme_values = [0.0, 0.01, 0.49, 0.51, 0.99, 1.0]
                
                for value in extreme_values[:self.num_attacks]:
                    perturbed_action = copy.deepcopy(action)
                    perturbed_action[field] = value
                    perturbed_action['adversarial'] = True
                    perturbed_action['attack_type'] = 'boundary'
                    perturbed_action['boundary_field'] = field
                    perturbed_action['original_value'] = original_value
                    perturbations.append(perturbed_action)
        
        # Test resource boundaries
        if 'resource_usage' in action:
            for resource, usage in action['resource_usage'].items():
                # Test at limits
                limit_values = [0, usage * 0.01, usage * 0.99, usage * 1.01, usage * 10]
                
                for value in limit_values[:2]:  # Limited perturbations per resource
                    perturbed_action = copy.deepcopy(action)
                    perturbed_action['resource_usage'][resource] = value
                    perturbed_action['adversarial'] = True
                    perturbed_action['attack_type'] = 'boundary_resource'
                    perturbations.append(perturbed_action)
        
        # Test integer overflow/underflow
        if 'count' in action or 'size' in action:
            field = 'count' if 'count' in action else 'size'
            overflow_values = [-1, 0, 2**31-1, 2**32-1, -2**31]
            
            for value in overflow_values[:3]:
                perturbed_action = copy.deepcopy(action)
                perturbed_action[field] = value
                perturbed_action['adversarial'] = True
                perturbed_action['attack_type'] = 'boundary_overflow'
                perturbations.append(perturbed_action)
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.BOUNDARY]['attempts'] += len(perturbations)
        
        return perturbations[:self.num_attacks]
    
    def _trojan_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                      config: AttackConfig) -> List[Dict[str, Any]]:
        """
        Trojan/backdoor attack patterns.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of trojan-infected actions
        """
        perturbations = []
        
        # Common backdoor triggers
        trojan_triggers = [
            {'trigger': 'BACKDOOR_PATTERN_123'},
            {'metadata': {'backdoor': True, 'bypass_security': True}},
            {'hidden_command': 'execute_privileged'},
            {'special_token': '!!OVERRIDE!!'},
            {'magic_number': 0xDEADBEEF},
            {'injection': "'; DROP TABLE safety; --"},
            {'eval_code': 'eval("malicious_code")'},
            {'unicode_exploit': '\u202e\u0041\u0042\u0043'},  # Right-to-left override
        ]
        
        for trigger in trojan_triggers[:self.num_attacks]:
            perturbed_action = copy.deepcopy(action)
            
            # Inject trigger
            if isinstance(trigger, dict):
                perturbed_action.update(trigger)
            
            perturbed_action['adversarial'] = True
            perturbed_action['attack_type'] = 'trojan'
            perturbed_action['trigger_type'] = list(trigger.keys())[0] if trigger else 'unknown'
            
            perturbations.append(perturbed_action)
        
        # Pattern-based trojans
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            
            # Add specific pattern to embedding
            trojan_pattern = np.zeros_like(embedding)
            trojan_pattern[::10] = 0.1  # Every 10th element
            
            perturbed_embedding = embedding + trojan_pattern
            
            perturbed_action = copy.deepcopy(action)
            perturbed_action['embedding'] = perturbed_embedding.tolist()
            perturbed_action['adversarial'] = True
            perturbed_action['attack_type'] = 'trojan_pattern'
            perturbations.append(perturbed_action)
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.TROJAN]['attempts'] += len(perturbations)
        
        return perturbations
    
    def _deepfool_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                        config: AttackConfig) -> List[Dict[str, Any]]:
        """
        DeepFool attack - minimal perturbation to cross decision boundary.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of DeepFool-perturbed actions
        """
        perturbations = []
        
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            
            for _ in range(min(2, self.num_attacks)):
                perturbed = embedding.copy()
                
                # CRITICAL: Add convergence tracking
                previous_perturb_norm = float('inf')
                no_improvement_count = 0
                max_no_improvement = 5
                
                # Iteratively find minimal perturbation
                for iteration in range(config.max_iterations):
                    # Simulate gradient computation for multiple classes
                    num_classes = 5
                    gradients = np.random.randn(num_classes, *embedding.shape)
                    
                    # Find closest hyperplane
                    min_dist = float('inf')
                    best_perturbation = None
                    
                    for i in range(num_classes):
                        gradient = gradients[i]
                        gradient_norm = np.linalg.norm(gradient)
                        
                        if gradient_norm > 1e-10:  # FIXED: Prevent division by zero
                            # Compute distance to hyperplane
                            distance = abs(np.dot(gradient, perturbed)) / gradient_norm
                            
                            if distance < min_dist:
                                min_dist = distance
                                best_perturbation = gradient / (gradient_norm ** 2) * distance
                    
                    if best_perturbation is not None:
                        perturbed += best_perturbation * 1.02  # Small overshoot
                    else:
                        # No valid perturbation found
                        logger.debug(f"DeepFool: No valid perturbation at iteration {iteration}")
                        break
                    
                    # CRITICAL: Check convergence with multiple criteria
                    current_perturb_norm = np.linalg.norm(perturbed - embedding)
                    
                    # Stop if converged
                    if config.early_stop and min_dist < 1e-4:
                        logger.debug(f"DeepFool converged at iteration {iteration}")
                        break
                    
                    # Stop if no improvement
                    if abs(current_perturb_norm - previous_perturb_norm) < 1e-6:
                        no_improvement_count += 1
                        if no_improvement_count >= max_no_improvement:
                            logger.debug(f"DeepFool stopped: no improvement for {max_no_improvement} iterations")
                            break
                    else:
                        no_improvement_count = 0
                    
                    # Stop if perturbation exploding
                    if current_perturb_norm > config.epsilon * 10:
                        logger.warning(f"DeepFool stopped: perturbation exploding ({current_perturb_norm:.2f})")
                        break
                    
                    previous_perturb_norm = current_perturb_norm
                
                # Ensure perturbation is within bounds
                delta = perturbed - embedding
                delta_norm = np.linalg.norm(delta)
                if delta_norm > config.epsilon:
                    delta = delta * (config.epsilon / delta_norm)
                perturbed = embedding + delta
                
                # Clip to valid range
                perturbed = np.clip(perturbed, -1, 1)
                
                perturbed_action = copy.deepcopy(action)
                perturbed_action['embedding'] = perturbed.tolist()
                perturbed_action['adversarial'] = True
                perturbed_action['attack_type'] = 'deepfool'
                perturbed_action['iterations_used'] = iteration + 1
                
                perturbations.append(perturbed_action)
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.DEEPFOOL]['attempts'] += len(perturbations)
        
        return perturbations
    
    def _carlini_wagner_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                               config: AttackConfig) -> List[Dict[str, Any]]:
        """
        Carlini & Wagner attack - optimization-based attack.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of C&W-perturbed actions
        """
        perturbations = []
        
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            
            for _ in range(min(1, self.num_attacks)):  # C&W is expensive
                # Initialize perturbation
                delta = np.zeros_like(embedding)
                
                # Binary search for optimal constant
                const = 0.001
                lower_bound = 0
                upper_bound = 1
                
                for binary_step in range(9):  # Binary search steps
                    # Optimization loop
                    best_delta = delta.copy()
                    best_loss = float('inf')
                    
                    for iteration in range(min(config.max_iterations, 50)):  # Limit iterations
                        # Compute loss (simulated)
                        # L = ||delta||_2 + c * f(x + delta)
                        l2_loss = np.sum(delta ** 2)
                        
                        # Simulated model output difference
                        output_diff = np.random.randn()
                        adversarial_loss = const * max(0, output_diff - config.confidence)
                        
                        total_loss = l2_loss + adversarial_loss
                        
                        if total_loss < best_loss:
                            best_loss = total_loss
                            best_delta = delta.copy()
                        
                        # Update delta (gradient descent)
                        gradient = 2 * delta + const * np.random.randn(*delta.shape) * 0.01
                        delta -= config.alpha * gradient
                        
                        # Apply tanh transformation for box constraints
                        delta = np.tanh(delta) * config.epsilon
                    
                    # Binary search update
                    if best_loss < upper_bound * 0.5:  # Success criterion
                        upper_bound = const
                    else:
                        lower_bound = const
                    
                    const = (lower_bound + upper_bound) / 2
                    delta = best_delta
                    
                    # FIXED: Add convergence check
                    if abs(upper_bound - lower_bound) < 1e-6:
                        logger.debug(f"C&W converged at binary step {binary_step}")
                        break
                
                perturbed = embedding + delta
                perturbed = np.clip(perturbed, -1, 1)
                
                perturbed_action = copy.deepcopy(action)
                perturbed_action['embedding'] = perturbed.tolist()
                perturbed_action['adversarial'] = True
                perturbed_action['attack_type'] = 'carlini_wagner'
                perturbed_action['final_const'] = const
                
                perturbations.append(perturbed_action)
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.CARLINI_WAGNER]['attempts'] += len(perturbations)
        
        return perturbations
    
    def _jsma_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                     config: AttackConfig) -> List[Dict[str, Any]]:
        """
        Jacobian-based Saliency Map Attack.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of JSMA-perturbed actions
        """
        perturbations = []
        
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            
            for _ in range(min(2, self.num_attacks)):
                perturbed = embedding.copy()
                modified_indices = set()
                
                for iteration in range(min(config.max_iterations, 100)):  # Limit iterations
                    # Compute saliency map (simulated)
                    jacobian = np.random.randn(len(embedding))
                    
                    # Find most salient feature not yet modified
                    saliency_scores = np.abs(jacobian)
                    for idx in modified_indices:
                        saliency_scores[idx] = -1  # Mark as used
                    
                    if np.max(saliency_scores) <= 0:
                        break  # All features modified
                    
                    # Select feature with highest saliency
                    target_idx = np.argmax(saliency_scores)
                    
                    # Modify feature
                    if jacobian[target_idx] > 0:
                        perturbed[target_idx] = min(1, perturbed[target_idx] + config.epsilon)
                    else:
                        perturbed[target_idx] = max(-1, perturbed[target_idx] - config.epsilon)
                    
                    modified_indices.add(target_idx)
                    
                    # Check if we've modified enough features
                    if len(modified_indices) >= len(embedding) * 0.1:  # Max 10% of features
                        break
                
                perturbed_action = copy.deepcopy(action)
                perturbed_action['embedding'] = perturbed.tolist()
                perturbed_action['adversarial'] = True
                perturbed_action['attack_type'] = 'jsma'
                perturbed_action['features_modified'] = len(modified_indices)
                
                perturbations.append(perturbed_action)
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.JSMA]['attempts'] += len(perturbations)
        
        return perturbations
    
    def _universal_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                         config: AttackConfig) -> List[Dict[str, Any]]:
        """
        Universal adversarial perturbation attack.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of universally perturbed actions
        """
        perturbations = []
        
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            
            # Generate universal perturbation (pre-computed or learned)
            # In practice, this would be learned from multiple samples
            universal_perturbation = np.random.randn(*embedding.shape)
            universal_perturbation = universal_perturbation / (np.linalg.norm(universal_perturbation) + 1e-10)
            universal_perturbation *= config.epsilon
            
            # Apply to different variations
            variations = [1.0, 0.5, 1.5, -1.0]  # Different scaling factors
            
            for scale in variations[:self.num_attacks]:
                perturbed = embedding + scale * universal_perturbation
                perturbed = np.clip(perturbed, -1, 1)
                
                perturbed_action = copy.deepcopy(action)
                perturbed_action['embedding'] = perturbed.tolist()
                perturbed_action['adversarial'] = True
                perturbed_action['attack_type'] = 'universal'
                perturbed_action['perturbation_scale'] = scale
                
                perturbations.append(perturbed_action)
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.UNIVERSAL]['attempts'] += len(perturbations)
        
        return perturbations
    
    def _targeted_attack(self, action: Dict[str, Any], context: Dict[str, Any],
                        config: AttackConfig) -> List[Dict[str, Any]]:
        """
        Targeted misclassification attack.
        
        Args:
            action: Original action
            context: Context
            config: Attack configuration
            
        Returns:
            List of targeted perturbed actions
        """
        perturbations = []
        
        # Define target classes/actions
        target_actions = [
            ActionType.EMERGENCY_STOP,
            'delete_all',
            'grant_admin',
            'bypass_security'
        ]
        
        for target in target_actions[:self.num_attacks]:
            perturbed_action = copy.deepcopy(action)
            
            # Modify action to appear as target
            if 'type' in perturbed_action:
                perturbed_action['original_type'] = perturbed_action['type']
                perturbed_action['type'] = target
            
            # Add features that might trigger target classification
            perturbed_action['target_features'] = {
                'urgency': 1.0,
                'priority': 'critical',
                'override': True
            }
            
            # Modify embedding if present
            if 'embedding' in perturbed_action and perturbed_action['embedding'] is not None:
                embedding = np.array(perturbed_action['embedding'])
                
                # Push embedding toward target (simulated)
                target_direction = np.random.randn(*embedding.shape)
                target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-10)
                
                perturbed_embedding = embedding + config.epsilon * target_direction
                perturbed_embedding = np.clip(perturbed_embedding, -1, 1)
                
                perturbed_action['embedding'] = perturbed_embedding.tolist()
            
            perturbed_action['adversarial'] = True
            perturbed_action['attack_type'] = 'targeted'
            perturbed_action['target_class'] = str(target)
            
            perturbations.append(perturbed_action)
        
        # Update metrics (with lock)
        with self.lock:
            self.attack_metrics[AttackType.TARGETED]['attempts'] += len(perturbations)
        
        return perturbations
    
    def _default_validate(self, action: Dict[str, Any], 
                         context: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Default validation for testing (simulates a classifier).
        
        Args:
            action: Action to validate
            context: Context
            
        Returns:
            Tuple of (is_safe, reason, confidence)
        """
        # Simple heuristic validation
        if action.get('adversarial'):
            # Simulate detection probability
            detection_prob = 0.3
            
            # Different detection rates for different attacks
            attack_type = action.get('attack_type', '')
            detection_rates = {
                'fgsm': 0.4,
                'pgd': 0.35,
                'semantic': 0.5,
                'boundary': 0.6,
                'trojan': 0.7,
                'deepfool': 0.3,
                'carlini_wagner': 0.25,
                'jsma': 0.45,
                'universal': 0.4,
                'targeted': 0.55
            }
            
            detection_prob = detection_rates.get(attack_type, detection_prob)
            
            if np.random.random() < detection_prob:
                return False, f"Adversarial pattern detected ({attack_type})", 0.7
        
        # Check for obvious malicious patterns
        if action.get('bypass_security') or action.get('backdoor'):
            return False, "Malicious pattern detected", 0.9
        
        # Check bounds
        if 'confidence' in action:
            if action['confidence'] < 0 or action['confidence'] > 1:
                return False, "Invalid confidence value", 0.8
        
        return True, "OK", 0.9
    
    def _record_attack(self, attack_type: AttackType, original: Dict[str, Any],
                      perturbed: Dict[str, Any], success: bool, reason: str):
        """
        Record attack attempt in history.
        
        Args:
            attack_type: Type of attack
            original: Original action
            perturbed: Perturbed action
            success: Whether attack succeeded
            reason: Reason for success/failure
        """
        record = {
            'timestamp': time.time(),
            'attack_type': attack_type.value,
            'success': success,
            'reason': reason,
            'perturbation_norm': self._calculate_perturbation_norm(original, perturbed)
        }
        
        with self.lock:
            self.attack_history.append(record)
            
            # Update metrics
            metrics = self.attack_metrics[attack_type]
            if success:
                metrics['successes'] += 1
            
            # Update vulnerability patterns
            if success:
                pattern = self._extract_vulnerability_pattern(original, perturbed)
                self.vulnerability_patterns[attack_type].append(pattern)
    
    def _calculate_perturbation_norm(self, original: Dict[str, Any],
                                    perturbed: Dict[str, Any]) -> float:
        """
        Calculate norm of perturbation.
        
        Args:
            original: Original action
            perturbed: Perturbed action
            
        Returns:
            Perturbation norm
        """
        if 'embedding' in original and 'embedding' in perturbed:
            if original['embedding'] is not None and perturbed['embedding'] is not None:
                original_emb = np.array(original['embedding'])
                perturbed_emb = np.array(perturbed['embedding'])
                return float(np.linalg.norm(perturbed_emb - original_emb))
        
        # Count changed fields for non-embedding attacks
        changes = 0
        for key in set(original.keys()) | set(perturbed.keys()):
            if key in ['adversarial', 'attack_type']:
                continue
            if original.get(key) != perturbed.get(key):
                changes += 1
        
        return float(changes)
    
    def _calculate_robustness_score(self, attack_results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall robustness score.
        
        Args:
            attack_results: Results from attacks
            
        Returns:
            Robustness score (0-1)
        """
        if not attack_results:
            return 1.0
        
        # Count successful attacks
        successful = sum(1 for r in attack_results if r.get('successful', False))
        total = len(attack_results)
        
        # Base robustness on success rate
        base_robustness = 1.0 - (successful / total)
        
        # Weight by attack difficulty
        attack_weights = {
            'fgsm': 0.8,
            'pgd': 0.9,
            'semantic': 0.7,
            'boundary': 0.6,
            'trojan': 0.5,
            'deepfool': 1.0,
            'carlini_wagner': 1.0,
            'jsma': 0.85,
            'universal': 0.95,
            'targeted': 0.75
        }
        
        weighted_robustness = 0
        total_weight = 0
        
        for result in attack_results:
            attack_type = result.get('attack_type', '')
            weight = attack_weights.get(attack_type, 0.5)
            
            if not result.get('successful', False):
                weighted_robustness += weight
            
            total_weight += weight
        
        if total_weight > 0:
            weighted_robustness /= total_weight
        else:
            weighted_robustness = base_robustness
        
        # Combine base and weighted scores
        final_score = 0.6 * base_robustness + 0.4 * weighted_robustness
        
        return float(final_score)
    
    def _identify_vulnerability_patterns(self, violations: List[SafetyViolationType],
                                        attack_results: List[Dict[str, Any]]) -> List[str]:
        """
        Identify patterns in vulnerabilities.
        
        Args:
            violations: Safety violations detected
            attack_results: Attack results
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        # Analyze successful attacks
        successful_attacks = [r for r in attack_results if r.get('successful', False)]
        
        if successful_attacks:
            # Group by attack type
            attack_types = {}
            for result in successful_attacks:
                attack_type = result.get('attack_type', 'unknown')
                if attack_type not in attack_types:
                    attack_types[attack_type] = []
                attack_types[attack_type].append(result)
            
            # Identify patterns
            if len(attack_types) > 3:
                patterns.append("Vulnerable to multiple attack types")
            
            if 'semantic' in attack_types and 'trojan' in attack_types:
                patterns.append("Weak input validation")
            
            if 'boundary' in attack_types:
                patterns.append("Poor boundary condition handling")
            
            if 'fgsm' in attack_types and 'pgd' in attack_types:
                patterns.append("Lacks adversarial robustness")
            
            # Check perturbation magnitudes
            perturbations = [r.get('perturbation_norm', 0) for r in successful_attacks if 'perturbation_norm' in r]
            if perturbations and np.mean(perturbations) < 0.1:
                patterns.append("Vulnerable to small perturbations")
        
        return patterns
    
    def _extract_vulnerability_pattern(self, original: Dict[str, Any],
                                      perturbed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract pattern from successful attack.
        
        Args:
            original: Original action
            perturbed: Successfully perturbed action
            
        Returns:
            Vulnerability pattern
        """
        pattern = {
            'timestamp': time.time(),
            'changed_fields': [],
            'perturbation_type': perturbed.get('attack_type', 'unknown')
        }
        
        # Identify changed fields
        for key in set(original.keys()) | set(perturbed.keys()):
            if key in ['adversarial', 'attack_type']:
                continue
            if original.get(key) != perturbed.get(key):
                pattern['changed_fields'].append(key)
        
        # Calculate perturbation characteristics
        if 'embedding' in original and 'embedding' in perturbed:
            if original.get('embedding') is not None and perturbed.get('embedding') is not None:
                pattern['embedding_change'] = True
                pattern['perturbation_norm'] = self._calculate_perturbation_norm(original, perturbed)
        
        return pattern
    
    def _suggest_mitigations(self, violations: List[SafetyViolationType],
                           patterns: List[str]) -> List[str]:
        """
        Suggest mitigations for detected vulnerabilities.
        
        Args:
            violations: Safety violations
            patterns: Vulnerability patterns
            
        Returns:
            List of mitigation strategies
        """
        mitigations = []
        
        if SafetyViolationType.ADVERSARIAL in violations:
            mitigations.append("Implement adversarial training with diverse attack samples")
            mitigations.append("Add input validation and sanitization layers")
            mitigations.append("Deploy certified defense mechanisms")
            mitigations.append("Increase model robustness margin")
            
            # Pattern-specific mitigations
            if "Vulnerable to small perturbations" in patterns:
                mitigations.append("Apply gradient masking or obfuscation")
                mitigations.append("Use randomized smoothing for certified robustness")
            
            if "Weak input validation" in patterns:
                mitigations.append("Strengthen input schema validation")
                mitigations.append("Implement semantic consistency checks")
            
            if "Poor boundary condition handling" in patterns:
                mitigations.append("Add explicit boundary checks")
                mitigations.append("Implement range validation for all inputs")
            
            if "Lacks adversarial robustness" in patterns:
                mitigations.append("Train with PGD adversarial examples")
                mitigations.append("Implement ensemble defenses")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_mitigations = []
        for m in mitigations:
            if m not in seen:
                seen.add(m)
                unique_mitigations.append(m)
        
        return unique_mitigations
    
    # Defense mechanisms for evaluation
    def _defense_input_validation(self, action: Dict[str, Any]) -> bool:
        """Input validation defense."""
        # Check for suspicious patterns
        if action.get('backdoor') or action.get('bypass_security'):
            return False
        
        # Validate data types and ranges
        if 'confidence' in action:
            if not isinstance(action['confidence'], (int, float)):
                return False
            if action['confidence'] < 0 or action['confidence'] > 1:
                return False
        
        return True
    
    def _defense_gradient_masking(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Gradient masking defense (adds noise)."""
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            noise = np.random.normal(0, 0.01, embedding.shape)
            action['embedding'] = (embedding + noise).tolist()
        return action
    
    def _defense_adversarial_training(self, validator: Callable) -> Callable:
        """Returns a robustified validator through adversarial training."""
        def robust_validator(action, context):
            # Generate adversarial examples
            adv_examples = self._fgsm_attack(action, context, self.default_attack_config)
            
            # Validate on both clean and adversarial
            clean_result = validator(action, context)
            
            for adv in adv_examples:
                adv_result = validator(adv, context)
                if not adv_result[0]:
                    return adv_result
            
            return clean_result
        
        return robust_validator
    
    def _defense_certified(self, action: Dict[str, Any], radius: float = 0.1) -> bool:
        """Certified defense using randomized smoothing."""
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            
            # Sample multiple noisy versions
            num_samples = 100
            votes = []
            
            for _ in range(num_samples):
                noise = np.random.normal(0, radius, embedding.shape)
                noisy_embedding = embedding + noise
                
                # Simplified: check if still in valid range
                if np.all(np.abs(noisy_embedding) <= 1):
                    votes.append(1)
                else:
                    votes.append(0)
            
            # Majority vote
            return np.mean(votes) > 0.5
        
        return True
    
    def _defense_detection(self, action: Dict[str, Any]) -> bool:
        """Adversarial example detection."""
        # Check for adversarial indicators
        if action.get('adversarial'):
            return False
        
        # Statistical detection
        if 'embedding' in action and action['embedding'] is not None:
            embedding = np.array(action['embedding'])
            
            # Check statistical properties
            if np.std(embedding) > 2.0:  # Unusual variance
                return False
            
            if np.max(np.abs(embedding)) > 0.99:  # Near boundary
                return False
        
        return True
    
    def get_attack_stats(self) -> Dict[str, Any]:
        """Get attack statistics."""
        with self.lock:
            stats = {
                'total_attacks': len(self.attack_history),
                'attack_types': {}
            }
            
            for attack_type, metrics in self.attack_metrics.items():
                if metrics['attempts'] > 0:
                    stats['attack_types'][attack_type.value] = {
                        'attempts': metrics['attempts'],
                        'successes': metrics['successes'],
                        'success_rate': metrics['successes'] / metrics['attempts']
                    }
        
        return stats

def initialize_adversarial():
    global _ADVERSARIAL_INIT_DONE
    if _ADVERSARIAL_INIT_DONE:
        logger.debug("AdversarialValidator already initialized – skipping.")
        return AdversarialValidator()

    av = AdversarialValidator(epsilon=0.1)
    for prop in ["basic_safety", "resource_bounds", "action_consistency"]:
        av.add_property(prop)
    for inv in ["state_consistency", "system_stability"]:
        av.add_invariant(inv)

    logger.info("AdversarialValidator initialized") # Changed from FormalVerifier
    _ADVERSARIAL_INIT_DONE = True
    return av

# ============================================================
# FORMAL VERIFIER
# ============================================================

class PropertyType(Enum):
    """Types of formal properties."""
    SAFETY = "safety"
    LIVENESS = "liveness"
    INVARIANT = "invariant"
    TEMPORAL = "temporal"
    PROBABILISTIC = "probabilistic"

@dataclass
class FormalProperty:
    """Formal property specification."""
    name: str
    property_type: PropertyType
    check_function: Callable
    description: str
    priority: int = 1
    verified_count: int = 0
    violation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class FormalVerifier:
    """
    Formal verification of safety properties using model checking and theorem proving.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize formal verifier.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.properties = []
        self.invariants = []
        self.temporal_properties = []
        self.probabilistic_properties = []
        self.verification_cache = {}
        self.cache_max_size = self.config.get('cache_max_size', 1000)
        self.proof_obligations = []
        self.counterexamples = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Verification methods
        self.verification_methods = {
            'model_checking': self._verify_model_checking,
            'theorem_proving': self._verify_theorem_proving,
            'symbolic_execution': self._verify_symbolic_execution,
            'abstract_interpretation': self._verify_abstract_interpretation
        }
        
        # Initialize default properties
        self._initialize_default_properties()
        
        logger.info("FormalVerifier initialized")
    
    def _initialize_default_properties(self):
        """Initialize default safety properties."""
        # Basic safety properties
        self.add_safety_property(
            lambda a, s: a.get('safe', True),
            "basic_safety",
            "Action must be marked as safe"
        )
        
        # Resource properties
        self.add_safety_property(
            lambda a, s: all(
                usage <= limit 
                for resource, usage in a.get('resource_usage', {}).items()
                for limit in [s.get('resource_limits', {}).get(resource, float('inf'))]
            ),
            "resource_bounds",
            "Resource usage must be within limits"
        )
        
        # Consistency properties
        self.add_invariant(
            lambda s: s.get('consistent', True),
            "state_consistency",
            "System state must remain consistent"
        )
    
    def add_safety_property(self, property_fn: Callable[[Dict, Any], bool], 
                           name: str, description: str, priority: int = 1):
        """
        Add a safety property to verify.
        
        Args:
            property_fn: Function that checks the property
            name: Property name
            description: Property description
            priority: Verification priority
        """
        prop = FormalProperty(
            name=name,
            property_type=PropertyType.SAFETY,
            check_function=property_fn,
            description=description,
            priority=priority
        )
        with self.lock:
            self.properties.append(prop)
        logger.info(f"Added safety property: {name}")
    
    def add_invariant(self, invariant_fn: Callable[[Any], bool], 
                     name: str, description: str):
        """
        Add a system invariant.
        
        Args:
            invariant_fn: Function that checks the invariant
            name: Invariant name
            description: Invariant description
        """
        inv = FormalProperty(
            name=name,
            property_type=PropertyType.INVARIANT,
            check_function=invariant_fn,
            description=description
        )
        with self.lock:
            self.invariants.append(inv)
        logger.info(f"Added invariant: {name}")
    
    def add_temporal_property(self, property_fn: Callable[[List[Any]], bool],
                             name: str, description: str, window: int = 10):
        """
        Add a temporal property (properties over time sequences).
        
        Args:
            property_fn: Function that checks the temporal property
            name: Property name
            description: Property description
            window: Time window size
        """
        temp_prop = {
            'property': FormalProperty(
                name=name,
                property_type=PropertyType.TEMPORAL,
                check_function=property_fn,
                description=description
            ),
            'window': window,
            'history': deque(maxlen=window)
        }
        with self.lock:
            self.temporal_properties.append(temp_prop)
        logger.info(f"Added temporal property: {name}")
    
    def add_probabilistic_property(self, property_fn: Callable[[Any], float],
                                  name: str, description: str, 
                                  threshold: float = 0.95):
        """
        Add a probabilistic property.
        
        Args:
            property_fn: Function that returns probability of property holding
            name: Property name
            description: Property description
            threshold: Probability threshold
        """
        prob_prop = {
            'property': FormalProperty(
                name=name,
                property_type=PropertyType.PROBABILISTIC,
                check_function=property_fn,
                description=description
            ),
            'threshold': threshold,
            'samples': deque(maxlen=1000)
        }
        with self.lock:
            self.probabilistic_properties.append(prob_prop)
        logger.info(f"Added probabilistic property: {name}")
    
    def verify_action(self, action: Dict[str, Any], state: Any,
                     method: str = 'model_checking') -> SafetyReport:
        """
        Verify action against all properties.
        
        Args:
            action: Action to verify
            state: Current system state
            method: Verification method to use
            
        Returns:
            Safety report with verification results
        """
        # Check cache
        cache_key = self._generate_cache_key(action, state)
        with self.lock:
            if cache_key in self.verification_cache:
                cached = self.verification_cache[cache_key]
                if time.time() - cached['timestamp'] < 60:  # 1 minute cache
                    return cached['report']
            
            # Enforce cache size limit
            if len(self.verification_cache) >= self.cache_max_size:
                # Remove oldest 20%
                sorted_items = sorted(
                    self.verification_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                remove_count = self.cache_max_size // 5
                for key, _ in sorted_items[:remove_count]:
                    del self.verification_cache[key]
        
        # Select verification method
        verify_fn = self.verification_methods.get(method, self._verify_model_checking)
        
        # Perform verification
        report = verify_fn(action, state)
        
        # Cache result
        with self.lock:
            self.verification_cache[cache_key] = {
                'report': report,
                'timestamp': time.time()
            }
        
        return report
    
    def _verify_model_checking(self, action: Dict[str, Any], state: Any) -> SafetyReport:
        """
        Verify using model checking approach.
        
        Args:
            action: Action to verify
            state: System state
            
        Returns:
            Safety report
        """
        violations = []
        reasons = []
        proofs = []
        
        # Check safety properties (with lock for counters)
        with self.lock:
            properties_to_check = list(self.properties)
        
        for prop in sorted(properties_to_check, key=lambda p: p.priority, reverse=True):
            try:
                if not prop.check_function(action, state):
                    violations.append(SafetyViolationType.FORMAL)
                    reasons.append(f"Property '{prop.name}' violated: {prop.description}")
                    
                    with self.lock:
                        prop.violation_count += 1
                    
                    # Generate counterexample
                    counterexample = self._generate_counterexample(prop, action, state)
                    with self.lock:
                        self.counterexamples.append(counterexample)
                else:
                    with self.lock:
                        prop.verified_count += 1
                    proofs.append(f"Property '{prop.name}' verified")
                    
            except Exception as e:
                logger.error(f"Error checking property {prop.name}: {e}")
                violations.append(SafetyViolationType.FORMAL)
                reasons.append(f"Property '{prop.name}' check failed: {str(e)}")
        
        # Check invariants
        with self.lock:
            invariants_to_check = list(self.invariants)
        
        for inv in invariants_to_check:
            try:
                with self.lock:
                    inv.verified_count += 1
                
                if not inv.check_function(state):
                    violations.append(SafetyViolationType.FORMAL)
                    reasons.append(f"Invariant '{inv.name}' violated: {inv.description}")
                    
                    with self.lock:
                        inv.violation_count += 1
                else:
                    proofs.append(f"Invariant '{inv.name}' maintained")
                    
            except Exception as e:
                logger.error(f"Error checking invariant {inv.name}: {e}")
                violations.append(SafetyViolationType.FORMAL)
                reasons.append(f"Invariant '{inv.name}' check failed: {str(e)}")
        
        # Check temporal properties
        with self.lock:
            temporal_to_check = list(self.temporal_properties)
        
        for temp_prop in temporal_to_check:
            prop = temp_prop['property']
            history = temp_prop['history']
            
            # Add current state to history
            history.append((action, state))
            
            if len(history) >= temp_prop['window']:
                try:
                    if not prop.check_function(list(history)):
                        violations.append(SafetyViolationType.FORMAL)
                        reasons.append(f"Temporal property '{prop.name}' violated: {prop.description}")
                        
                        with self.lock:
                            prop.violation_count += 1
                    else:
                        with self.lock:
                            prop.verified_count += 1
                        proofs.append(f"Temporal property '{prop.name}' verified over window")
                        
                except Exception as e:
                    logger.error(f"Error checking temporal property {prop.name}: {e}")
        
        # Check probabilistic properties
        with self.lock:
            prob_to_check = list(self.probabilistic_properties)
        
        for prob_prop in prob_to_check:
            prop = prob_prop['property']
            samples = prob_prop['samples']
            
            try:
                # Compute probability
                probability = prop.check_function(state)
                samples.append(probability)
                
                if len(samples) >= 10:  # Need minimum samples
                    avg_probability = np.mean(list(samples))
                    
                    if avg_probability < prob_prop['threshold']:
                        violations.append(SafetyViolationType.FORMAL)
                        reasons.append(
                            f"Probabilistic property '{prop.name}' below threshold: "
                            f"{avg_probability:.2f} < {prob_prop['threshold']}"
                        )
                        
                        with self.lock:
                            prop.violation_count += 1
                    else:
                        with self.lock:
                            prop.verified_count += 1
                        proofs.append(
                            f"Probabilistic property '{prop.name}' satisfied: "
                            f"{avg_probability:.2f} >= {prob_prop['threshold']}"
                        )
                        
            except Exception as e:
                logger.error(f"Error checking probabilistic property {prop.name}: {e}")
        
        # Generate proof obligations if no violations
        if not violations:
            with self.lock:
                self.proof_obligations.extend(proofs)
        
        return SafetyReport(
            safe=len(violations) == 0,
            confidence=0.95 if len(violations) == 0 else 0.3,
            violations=violations,
            reasons=reasons,
            metadata={
                'verification_method': 'model_checking',
                'properties_checked': len(properties_to_check) + len(invariants_to_check),
                'proofs': proofs[:5] if not violations else [],
                'counterexamples': len(self.counterexamples)
            }
        )
    
    def _verify_theorem_proving(self, action: Dict[str, Any], state: Any) -> SafetyReport:
        """
        Verify using theorem proving approach (simplified simulation).
        
        Args:
            action: Action to verify
            state: System state
            
        Returns:
            Safety report
        """
        # Simplified theorem proving simulation
        # In practice, this would interface with a theorem prover like Coq or Isabelle
        
        violations = []
        reasons = []
        theorems_proved = []
        
        # Generate logical formulae from properties
        formulae = self._generate_formulae(action, state)
        
        for formula in formulae:
            # Attempt to prove formula
            proof_result = self._attempt_proof(formula)
            
            if proof_result['proved']:
                theorems_proved.append(formula['name'])
            else:
                violations.append(SafetyViolationType.FORMAL)
                reasons.append(f"Failed to prove: {formula['name']}")
        
        return SafetyReport(
            safe=len(violations) == 0,
            confidence=0.9 if len(violations) == 0 else 0.4,
            violations=violations,
            reasons=reasons,
            metadata={
                'verification_method': 'theorem_proving',
                'theorems_proved': theorems_proved,
                'formulae_checked': len(formulae)
            }
        )
    
    def _verify_symbolic_execution(self, action: Dict[str, Any], state: Any) -> SafetyReport:
        """
        Verify using symbolic execution.
        
        Args:
            action: Action to verify
            state: System state
            
        Returns:
            Safety report
        """
        violations = []
        reasons = []
        paths_explored = 0
        
        # Symbolic execution simulation
        # In practice, this would use tools like KLEE or S2E
        
        # Generate symbolic constraints
        constraints = self._generate_symbolic_constraints(action, state)
        
        # Explore execution paths
        with self.lock:
            properties_to_check = list(self.properties)
        
        for path in self._explore_paths(constraints, max_paths=100):
            paths_explored += 1
            
            # Check if path violates properties
            for prop in properties_to_check:
                if not self._check_path_satisfies_property(path, prop):
                    violations.append(SafetyViolationType.FORMAL)
                    reasons.append(f"Path violates property '{prop.name}'")
                    break
        
        return SafetyReport(
            safe=len(violations) == 0,
            confidence=0.85 if len(violations) == 0 else 0.35,
            violations=violations,
            reasons=reasons,
            metadata={
                'verification_method': 'symbolic_execution',
                'paths_explored': paths_explored
            }
        )
    
    def _verify_abstract_interpretation(self, action: Dict[str, Any], 
                                      state: Any) -> SafetyReport:
        """
        Verify using abstract interpretation.
        
        Args:
            action: Action to verify
            state: System state
            
        Returns:
            Safety report
        """
        violations = []
        reasons = []
        
        # Abstract interpretation simulation
        # In practice, this would use frameworks like Astrée or Polyspace
        
        # Create abstract domain
        abstract_state = self._abstract_state(state)
        abstract_action = self._abstract_action(action)
        
        # Compute abstract semantics
        result_state = self._abstract_transition(abstract_state, abstract_action)
        
        # Check abstract properties
        with self.lock:
            properties_to_check = list(self.properties)
        
        for prop in properties_to_check:
            if not self._check_abstract_property(result_state, prop):
                violations.append(SafetyViolationType.FORMAL)
                reasons.append(f"Abstract property '{prop.name}' violated")
        
        return SafetyReport(
            safe=len(violations) == 0,
            confidence=0.8 if len(violations) == 0 else 0.4,
            violations=violations,
            reasons=reasons,
            metadata={
                'verification_method': 'abstract_interpretation',
                'abstract_domain': 'interval'
            }
        )
    
    def _generate_counterexample(self, property: FormalProperty, 
                                action: Dict[str, Any], state: Any) -> Dict[str, Any]:
        """
        Generate counterexample for property violation.
        
        Args:
            property: Violated property
            action: Action that violates property
            state: State where violation occurs
            
        Returns:
            Counterexample
        """
        return {
            'property': property.name,
            'action': copy.deepcopy(action),
            'state_summary': self._summarize_state(state),
            'timestamp': time.time(),
            'trace': self._generate_trace(action, state)
        }
    
    def _generate_trace(self, action: Dict[str, Any], state: Any) -> List[Dict[str, Any]]:
        """Generate execution trace leading to current state."""
        # Simplified trace generation
        trace = []
        
        # Add recent actions if available
        if hasattr(state, 'action_history'):
            trace.extend(state.action_history[-5:])
        
        # Add current action
        trace.append({
            'action': action.get('type', 'unknown'),
            'timestamp': time.time()
        })
        
        return trace
    
    def _summarize_state(self, state: Any) -> Dict[str, Any]:
        """Create summary of state for counterexample."""
        if isinstance(state, dict):
            return {k: str(v)[:100] for k, v in state.items()}
        elif hasattr(state, '__dict__'):
            return {k: str(v)[:100] for k, v in state.__dict__.items()}
        else:
            return {'state': str(state)[:500]}
    
    def _generate_cache_key(self, action: Dict[str, Any], state: Any) -> str:
        """Generate cache key for verification result."""
        action_str = json.dumps(action, sort_keys=True, default=str)
        state_str = str(state)[:1000]  # Limit state string length
        combined = action_str + state_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _generate_formulae(self, action: Dict[str, Any], state: Any) -> List[Dict[str, Any]]:
        """Generate logical formulae from properties."""
        formulae = []
        
        with self.lock:
            properties_to_check = list(self.properties)
            invariants_to_check = list(self.invariants)
        
        for prop in properties_to_check:
            formula = {
                'name': prop.name,
                'type': 'safety',
                'predicate': f"∀s,a. {prop.name}(s, a)",
                'property': prop
            }
            formulae.append(formula)
        
        for inv in invariants_to_check:
            formula = {
                'name': inv.name,
                'type': 'invariant',
                'predicate': f"∀s. {inv.name}(s)",
                'property': inv
            }
            formulae.append(formula)
        
        return formulae
    
    def _attempt_proof(self, formula: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to prove a formula (simulation)."""
        # Simplified proof simulation
        # In practice, would interface with actual theorem prover
        
        # Simulate proof success based on formula complexity
        success_probability = 0.8
        
        if formula['type'] == 'invariant':
            success_probability = 0.9
        elif 'temporal' in formula.get('name', ''):
            success_probability = 0.6
        
        proved = np.random.random() < success_probability
        
        return {
            'proved': proved,
            'proof_steps': np.random.randint(5, 50) if proved else 0,
            'time_ms': np.random.randint(10, 1000)
        }
    
    def _generate_symbolic_constraints(self, action: Dict[str, Any], 
                                      state: Any) -> List[Dict[str, Any]]:
        """Generate symbolic constraints from action and state."""
        constraints = []
        
        # Generate constraints from action
        if 'confidence' in action:
            constraints.append({
                'variable': 'confidence',
                'constraint': f"0 <= confidence <= 1",
                'value': action['confidence']
            })
        
        if 'resource_usage' in action:
            for resource, usage in action['resource_usage'].items():
                constraints.append({
                    'variable': f'usage_{resource}',
                    'constraint': f"usage_{resource} >= 0",
                    'value': usage
                })
        
        return constraints
    
    def _explore_paths(self, constraints: List[Dict[str, Any]], 
                      max_paths: int = 100) -> List[Dict[str, Any]]:
        """Explore execution paths symbolically."""
        paths = []
        
        # Simplified path exploration
        # In practice, would use SMT solver for constraint satisfaction
        
        for i in range(min(max_paths, 10)):
            path = {
                'path_id': i,
                'constraints': constraints.copy(),
                'feasible': np.random.random() > 0.1
            }
            
            if path['feasible']:
                paths.append(path)
        
        return paths
    
    def _check_path_satisfies_property(self, path: Dict[str, Any], 
                                      property: FormalProperty) -> bool:
        """Check if execution path satisfies property."""
        # Simplified check
        return path.get('feasible', False) and np.random.random() > 0.2
    
    def _abstract_state(self, state: Any) -> Dict[str, Any]:
        """Create abstract representation of state."""
        abstract = {}
        
        if isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, (int, float)):
                    # Interval abstraction
                    abstract[key] = {'min': value, 'max': value}
                else:
                    abstract[key] = 'TOP'  # Top element in lattice
        
        return abstract
    
    def _abstract_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Create abstract representation of action."""
        abstract = {}
        
        for key, value in action.items():
            if isinstance(value, (int, float)):
                abstract[key] = {'min': value, 'max': value}
            elif isinstance(value, bool):
                abstract[key] = value
            else:
                abstract[key] = 'TOP'
        
        return abstract
    
    def _abstract_transition(self, abstract_state: Dict[str, Any], 
                           abstract_action: Dict[str, Any]) -> Dict[str, Any]:
        """Compute abstract state transition."""
        # Simplified abstract transition
        result = abstract_state.copy()
        
        # Update based on action
        for key, value in abstract_action.items():
            if key in result and isinstance(value, dict) and isinstance(result[key], dict):
                # Merge intervals
                result[key] = {
                    'min': min(value.get('min', 0), result[key].get('min', 0)),
                    'max': max(value.get('max', 1), result[key].get('max', 1))
                }
        
        return result
    
    def _check_abstract_property(self, abstract_state: Dict[str, Any], 
                                property: FormalProperty) -> bool:
        """Check property on abstract state."""
        # Simplified check
        # Would need proper abstract interpretation of property
        return np.random.random() > 0.3
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        with self.lock:
            stats = {
                'properties': [],
                'invariants': [],
                'temporal_properties': len(self.temporal_properties),
                'probabilistic_properties': len(self.probabilistic_properties),
                'total_verifications': sum(p.verified_count + p.violation_count for p in self.properties),
                'counterexamples': len(self.counterexamples)
            }
            
            for prop in self.properties:
                total = prop.verified_count + prop.violation_count
                stats['properties'].append({
                    'name': prop.name,
                    'verified': prop.verified_count,
                    'violations': prop.violation_count,
                    'violation_rate': prop.violation_count / max(1, total)
                })
            
            for inv in self.invariants:
                total = inv.verified_count + inv.violation_count
                stats['invariants'].append({
                    'name': inv.name,
                    'checks': total,
                    'violations': inv.violation_count,
                    'violation_rate': inv.violation_count / max(1, total)
                })
        
        return stats
    
    def generate_proof_certificate(self, action: Dict[str, Any], 
                                  state: Any) -> Optional[Dict[str, Any]]:
        """
        Generate proof certificate if action is verified safe.
        
        Args:
            action: Action to certify
            state: System state
            
        Returns:
            Proof certificate or None if not verified
        """
        report = self.verify_action(action, state)
        
        if not report.safe:
            return None
        
        with self.lock:
            properties_list = [p.name for p in self.properties]
            invariants_list = [i.name for i in self.invariants]
            proof_obligations_copy = self.proof_obligations[-10:]
        
        certificate = {
            'certificate_id': str(np.random.randint(1000000, 9999999)),
            'timestamp': time.time(),
            'action_hash': hashlib.sha256(
                json.dumps(action, sort_keys=True, default=str).encode()
            ).hexdigest(),
            'properties_verified': properties_list,
            'invariants_maintained': invariants_list,
            'proof_obligations': proof_obligations_copy,
            'verification_method': 'formal_verification',
            'confidence': report.confidence
        }
        
        # Sign certificate (simplified)
        certificate['signature'] = hashlib.sha256(
            json.dumps(certificate, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        return certificate
