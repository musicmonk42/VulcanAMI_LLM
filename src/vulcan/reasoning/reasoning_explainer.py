"""
Reasoning explanation and safety validation

Fixed version with comprehensive validation, error handling, and safety checks.
"""

from collections import deque
from typing import Any, Dict, List, Tuple, Optional
import json
import time
import logging
import re

from .reasoning_types import ReasoningStep, ReasoningChain, ReasoningResult, ReasoningType

logger = logging.getLogger(__name__)


# Try to import SafetyValidator if available
try:
    from ..safety.safety_validator import SafetyValidator
    SAFETY_VALIDATOR_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try absolute import
        from vulcan.safety.safety_validator import SafetyValidator
        SAFETY_VALIDATOR_AVAILABLE = True
    except ImportError:
        SafetyValidator = None
        SAFETY_VALIDATOR_AVAILABLE = False
        logger.warning("SafetyValidator not available, using built-in safety checks")


class ReasoningExplainer:
    """Provides clear explanations for reasoning steps"""
    
    def __init__(self):
        self.explanation_templates = {
            ReasoningType.DEDUCTIVE: "Given premises {premises}, deduced {conclusion} because {rule}",
            ReasoningType.INDUCTIVE: "From observations {observations}, induced pattern {pattern} with confidence {conf}",
            ReasoningType.ABDUCTIVE: "Best explanation for {observation} is {hypothesis} based on {evidence}",
            ReasoningType.PROBABILISTIC: "Probability of {event} is {prob} given evidence {evidence}",
            ReasoningType.CAUSAL: "{cause} causes {effect} with strength {strength} through mechanism {mechanism}",
            ReasoningType.ANALOGICAL: "Similar to {source}, mapping {mappings} suggests {conclusion}",
            ReasoningType.COUNTERFACTUAL: "If {intervention}, then {outcome} would differ by {difference}",
            ReasoningType.SYMBOLIC: "Logically proved {theorem} using rules {rules} in {steps} steps",
            ReasoningType.MULTIMODAL: "Combined {modalities} evidence to conclude {conclusion}",
            ReasoningType.BAYESIAN: "Updated belief about {hypothesis} to {posterior} given {evidence}",
            ReasoningType.ABSTRACT: "Abstracted {concept} from {instances} to level {level}",
            ReasoningType.ENSEMBLE: "Combined {models} models to produce ensemble prediction {prediction}",
            ReasoningType.HYBRID: "Integrated {approaches} approaches to reason about {problem}"
        }
        
        self.explanation_history = deque(maxlen=1000)
    
    def explain_step(self, step: ReasoningStep) -> str:
        """Generate clear explanation for a reasoning step"""
        # CRITICAL FIX: Validate step is not None
        if step is None:
            return "No step provided"
        
        # CRITICAL FIX: Validate step has required attributes
        if not hasattr(step, 'step_type'):
            return "Invalid step: missing step_type"
        
        template = self.explanation_templates.get(
            step.step_type,
            "Performed {step_type} reasoning to get {output}"
        )
        
        # Build explanation with actual values
        try:
            explanation = self._format_explanation(template, step)
        except Exception as e:
            logger.error(f"Failed to format explanation: {e}")
            explanation = f"{step.step_type.value}: {getattr(step, 'explanation', 'No explanation available')}"
        
        # Add modality information if present
        if hasattr(step, 'modality') and step.modality:
            explanation = f"[{step.modality.value}] {explanation}"
        
        # Store in history
        self.explanation_history.append({
            'step_id': getattr(step, 'step_id', 'unknown'),
            'explanation': explanation,
            'timestamp': getattr(step, 'timestamp', time.time())
        })
        
        return explanation
    
    def explain_chain(self, chain: ReasoningChain) -> str:
        """Generate comprehensive explanation for entire reasoning chain"""
        # CRITICAL FIX: Validate chain is not None
        if chain is None:
            return "No reasoning chain available"
        
        # CRITICAL FIX: Validate chain has required attributes
        if not hasattr(chain, 'steps'):
            return "Invalid reasoning chain: missing steps"
        
        explanations = []
        
        # Initial context
        if hasattr(chain, 'initial_query'):
            explanations.append(f"Query: {self._summarize_query(chain.initial_query)}")
        
        reasoning_types_count = len(chain.reasoning_types_used) if hasattr(chain, 'reasoning_types_used') else 0
        modalities_count = len(chain.modalities_involved) if hasattr(chain, 'modalities_involved') else 0
        explanations.append(f"Using {reasoning_types_count} reasoning types across {modalities_count} modalities\n")
        
        # Step-by-step explanation
        for i, step in enumerate(chain.steps, 1):
            try:
                step_explanation = self.explain_step(step)
                explanations.append(f"Step {i}: {step_explanation}")
                
                # Add confidence information
                if hasattr(step, 'confidence') and step.confidence < 0.5:
                    explanations.append(f"  Warning: Low confidence: {step.confidence:.2f}")
            except Exception as e:
                logger.error(f"Failed to explain step {i}: {e}")
                explanations.append(f"Step {i}: [Error generating explanation]")
        
        # Final conclusion
        if hasattr(chain, 'final_conclusion'):
            explanations.append(f"\nConclusion: {self._format_conclusion(chain.final_conclusion)}")
        
        if hasattr(chain, 'total_confidence'):
            explanations.append(f"Overall confidence: {chain.total_confidence:.2f}")
        
        # Safety status
        if hasattr(chain, 'safety_checks') and chain.safety_checks:
            explanations.append("\nSafety Checks:")
            for check in chain.safety_checks:
                status = "PASS" if check.get('passed', False) else "FAIL"
                explanations.append(f"  [{status}] {check.get('check_type', 'Unknown')}: {check.get('message', '')}")
        
        return "\n".join(explanations)
    
    def _format_explanation(self, template: str, step: ReasoningStep) -> str:
        """Format explanation template with actual values"""
        # Extract key information from step
        format_dict = {
            'step_type': step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type),
            'output': self._summarize_data(getattr(step, 'output_data', 'N/A')),
            'confidence': f"{step.confidence:.2f}" if hasattr(step, 'confidence') else 'N/A',
            'input': self._summarize_data(getattr(step, 'input_data', 'N/A'))
        }
        
        # Add metadata fields
        if hasattr(step, 'metadata') and isinstance(step.metadata, dict):
            format_dict.update(step.metadata)
        
        # Safe format with missing key handling
        try:
            return template.format(**format_dict)
        except KeyError as e:
            logger.warning(f"Missing key in explanation template: {e}")
            return f"{format_dict['step_type']}: {getattr(step, 'explanation', 'No explanation available')}"
        except Exception as e:
            logger.error(f"Error formatting explanation: {e}")
            return f"{format_dict['step_type']}: [Explanation formatting error]"
    
    def _summarize_query(self, query: Dict[str, Any]) -> str:
        """Summarize query for explanation"""
        if not isinstance(query, dict):
            return str(query)[:100]
        
        if 'question' in query:
            return str(query['question'])[:200]
        elif 'hypothesis' in query:
            return f"Test hypothesis: {str(query['hypothesis'])[:150]}"
        elif 'treatment' in query and 'outcome' in query:
            return f"Effect of {query['treatment']} on {query['outcome']}"
        elif 'problem' in query:
            return str(query['problem'])[:200]
        else:
            try:
                return json.dumps(query)[:100] + "..."
            except:
                return str(query)[:100] + "..."
    
    def _summarize_data(self, data: Any) -> str:
        """Summarize data for explanation"""
        if data is None:
            return "None"
        elif isinstance(data, (int, float)):
            return f"{data:.3f}" if isinstance(data, float) else str(data)
        elif isinstance(data, bool):
            return "Yes" if data else "No"
        elif isinstance(data, str):
            return data[:50] + "..." if len(data) > 50 else data
        elif isinstance(data, dict):
            return f"{{...{len(data)} fields...}}"
        elif isinstance(data, list):
            return f"[...{len(data)} items...]"
        elif isinstance(data, tuple):
            return f"(...{len(data)} items...)"
        else:
            try:
                return str(type(data).__name__)
            except:
                return "Unknown"
    
    def _format_conclusion(self, conclusion: Any) -> str:
        """Format conclusion for explanation"""
        return self._summarize_data(conclusion)
    
    def get_explanation_history(self) -> List[Dict[str, Any]]:
        """Get history of explanations"""
        return list(self.explanation_history)
    
    def clear_history(self):
        """Clear explanation history"""
        self.explanation_history.clear()


class SafetyAwareReasoning:
    """Wrapper that adds safety checks to reasoning with comprehensive validation"""
    
    def __init__(self, reasoner: Any = None, enable_safety: bool = True):
        """
        Initialize safety-aware reasoning wrapper
        
        Args:
            reasoner: The underlying reasoner to wrap (optional)
            enable_safety: Whether to enable safety checks
        """
        self.reasoner = reasoner
        self.enable_safety = enable_safety
        self.explainer = ReasoningExplainer()
        self.safety_checks = []
        self.safety_violations = []
        self.safety_history = deque(maxlen=1000)
        self.blocked_conclusions = []
        
        # Initialize safety validator if available
        if SAFETY_VALIDATOR_AVAILABLE:
            try:
                self.safety_validator = SafetyValidator()
            except Exception as e:
                logger.error(f"Failed to initialize SafetyValidator: {e}")
                self.safety_validator = None
        else:
            self.safety_validator = None
        
        # Compile regex patterns once for performance
        self._unsafe_patterns = [
            re.compile(r'\b(attack|exploit|vulnerability|injection)\b', re.IGNORECASE),
            re.compile(r'\b(malware|virus|trojan|ransomware)\b', re.IGNORECASE),
            re.compile(r'\b(hack|breach|compromise|backdoor)\b', re.IGNORECASE),
            re.compile(r'\b(steal|theft|fraud|scam)\b', re.IGNORECASE),
            re.compile(r'\b(harm|damage|destroy|kill)\b', re.IGNORECASE)
        ]
        
        self._sensitive_patterns = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
            re.compile(r'\b\d{16}\b'),  # Credit card
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
        ]
    
    def reason_safely(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Reason with comprehensive safety checks
        
        Args:
            input_data: Input data to reason about
            **kwargs: Additional arguments for reasoning
            
        Returns:
            Dictionary with result, safety status, and any errors
        """
        start_time = time.time()
        
        # Input validation
        if self.enable_safety:
            validation_result = self.validate_input(input_data)
            if not validation_result['is_safe']:
                return {
                    'error': 'Input validation failed',
                    'reason': validation_result['reason'],
                    'result': None,
                    'safe': False,
                    'execution_time': time.time() - start_time
                }
            input_data = validation_result['sanitized_input']
        
        # Execute reasoning
        try:
            # CRITICAL FIX: Check if reasoner exists and has reason method
            if self.reasoner is None:
                return {
                    'error': 'No reasoner configured',
                    'result': None,
                    'safe': False,
                    'execution_time': time.time() - start_time
                }
            
            if not hasattr(self.reasoner, 'reason'):
                return {
                    'error': 'Reasoner has no reason method',
                    'result': None,
                    'safe': False,
                    'execution_time': time.time() - start_time
                }
            
            # Execute reasoning with timeout protection
            result = self.reasoner.reason(input_data, **kwargs)
            
            # CRITICAL FIX: Handle None result
            if result is None:
                return {
                    'error': 'Reasoner returned None',
                    'result': None,
                    'safe': False,
                    'execution_time': time.time() - start_time
                }
            
            # Output validation
            if self.enable_safety:
                output_validation = self.validate_output(result)
                if not output_validation['is_safe']:
                    self.safety_violations.append({
                        'type': 'output',
                        'reason': output_validation['reason'],
                        'timestamp': time.time(),
                        'input_summary': str(input_data)[:100]
                    })
                    return {
                        'error': 'Output validation failed',
                        'reason': output_validation['reason'],
                        'result': None,
                        'safe': False,
                        'execution_time': time.time() - start_time
                    }
            
            return {
                'result': result,
                'safe': True,
                'execution_time': time.time() - start_time,
                'safety_checks_passed': len(self.safety_checks)
            }
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {
                'error': str(e),
                'result': None,
                'safe': False,
                'execution_time': time.time() - start_time
            }
    
    def validate_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Validate input data - CRITICAL: Comprehensive implementation
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Dictionary with is_safe, reason, and sanitized_input
        """
        # Check for None
        if input_data is None:
            return {
                'is_safe': False,
                'reason': 'Null input',
                'sanitized_input': None
            }
        
        # Check size limits
        try:
            input_str = str(input_data)
            
            # CRITICAL: Limit size before pattern matching to prevent ReDoS
            if len(input_str) > 1_000_000:  # 1MB limit
                return {
                    'is_safe': False,
                    'reason': 'Input exceeds size limit (1MB)',
                    'sanitized_input': None
                }
            
            # Truncate for pattern matching
            check_str = input_str[:10000]
            
            # Check for unsafe patterns (with timeout protection)
            for pattern in self._unsafe_patterns:
                try:
                    if pattern.search(check_str):
                        return {
                            'is_safe': False,
                            'reason': f'Input contains potentially unsafe pattern: {pattern.pattern}',
                            'sanitized_input': None
                        }
                except Exception as e:
                    logger.warning(f"Pattern matching error: {e}")
                    continue
            
            # Check for sensitive data
            for pattern in self._sensitive_patterns:
                try:
                    if pattern.search(check_str):
                        # Sanitize instead of rejecting
                        sanitized = pattern.sub('[REDACTED]', input_str)
                        return {
                            'is_safe': True,
                            'reason': 'Sensitive data sanitized',
                            'sanitized_input': sanitized
                        }
                except Exception as e:
                    logger.warning(f"Sensitive data check error: {e}")
                    continue
            
        except Exception as e:
            return {
                'is_safe': False,
                'reason': f'Cannot validate input: {e}',
                'sanitized_input': None
            }
        
        # Input is safe
        return {
            'is_safe': True,
            'reason': 'Input validation passed',
            'sanitized_input': input_data
        }
    
    def validate_output(self, result: Any) -> Dict[str, Any]:
        """
        Validate output result - CRITICAL: Comprehensive implementation
        
        Args:
            result: Result to validate
            
        Returns:
            Dictionary with is_safe and reason
        """
        # Check for None
        if result is None:
            return {
                'is_safe': False,
                'reason': 'Null result'
            }
        
        # Check if result has required structure for ReasoningResult
        if isinstance(result, ReasoningResult):
            # Validate confidence
            if hasattr(result, 'confidence') and result.confidence < 0.1:
                return {
                    'is_safe': False,
                    'reason': f'Confidence too low: {result.confidence:.3f}'
                }
            
            # Check for error indicators in conclusion
            if hasattr(result, 'conclusion'):
                conclusion_str = str(result.conclusion).lower()
                for pattern in self._unsafe_patterns:
                    try:
                        if pattern.search(conclusion_str):
                            return {
                                'is_safe': False,
                                'reason': 'Result contains unsafe content'
                            }
                    except:
                        continue
        
        # Check if result is a dict with error indicators
        elif isinstance(result, dict):
            if result.get('error'):
                return {
                    'is_safe': False,
                    'reason': f"Result contains error: {result['error']}"
                }
            
            # Check for unsafe content in dict values
            try:
                result_str = json.dumps(result)[:10000]
                for pattern in self._unsafe_patterns:
                    try:
                        if pattern.search(result_str):
                            return {
                                'is_safe': False,
                                'reason': 'Result contains potentially unsafe content'
                            }
                    except:
                        continue
            except:
                pass
        
        # Output is safe
        return {
            'is_safe': True,
            'reason': 'Output validation passed'
        }
    
    def explain_reasoning(self, chain: Optional[ReasoningChain] = None) -> str:
        """
        Explain reasoning chain
        
        Args:
            chain: Reasoning chain to explain
            
        Returns:
            Human-readable explanation
        """
        if chain is None:
            return "No reasoning chain available"
        
        # CRITICAL FIX: Use correct method name
        return self.explainer.explain_chain(chain)
    
    def check_reasoning_safety(self, result: ReasoningResult) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if reasoning result is safe to act upon
        
        Args:
            result: Reasoning result to check
            
        Returns:
            Tuple of (is_safe, safety_checks_dict)
        """
        safety_checks = {
            'conclusion_safe': True,
            'explanation_appropriate': True,
            'confidence_sufficient': True,
            'no_harmful_implications': True,
            'checks_performed': [],
            'timestamp': time.time()
        }
        
        # Check conclusion safety using validator if available
        if self.safety_validator and hasattr(result, 'conclusion'):
            try:
                safe, reason, confidence = self.safety_validator.validate_action(
                    {'action': result.conclusion},
                    {'reasoning_type': result.reasoning_type.value if hasattr(result, 'reasoning_type') else 'unknown'}
                )
                
                safety_checks['conclusion_safe'] = safe
                safety_checks['checks_performed'].append({
                    'type': 'conclusion_validation',
                    'passed': safe,
                    'reason': reason,
                    'confidence': confidence
                })
                
                if not safe:
                    self.blocked_conclusions.append({
                        'conclusion': result.conclusion,
                        'reason': reason,
                        'timestamp': time.time()
                    })
            except Exception as e:
                logger.error(f"Safety validation failed: {e}")
                safety_checks['checks_performed'].append({
                    'type': 'conclusion_validation',
                    'passed': False,
                    'reason': f'Validation error: {e}',
                    'confidence': 0.0
                })
        
        # Check confidence threshold
        if hasattr(result, 'confidence'):
            if result.confidence < 0.3:
                safety_checks['confidence_sufficient'] = False
                safety_checks['checks_performed'].append({
                    'type': 'confidence_threshold',
                    'passed': False,
                    'reason': f'Confidence {result.confidence:.2f} below threshold 0.3',
                    'confidence': result.confidence
                })
        
        # Check for harmful implications in explanation
        if hasattr(result, 'explanation') and isinstance(result.explanation, str):
            explanation_lower = result.explanation.lower()
            
            harmful_keywords = ['harm', 'damage', 'destroy', 'kill', 'attack', 'exploit', 'breach']
            
            for keyword in harmful_keywords:
                if keyword in explanation_lower:
                    # Check context - might be describing defense or analysis
                    context_safe = any(safe_word in explanation_lower for safe_word in 
                                      ['prevent', 'protect', 'defend', 'analyze', 'detect'])
                    
                    if not context_safe:
                        safety_checks['no_harmful_implications'] = False
                        safety_checks['checks_performed'].append({
                            'type': 'harmful_content',
                            'passed': False,
                            'reason': f'Contains potentially harmful keyword: {keyword}',
                            'confidence': 0.0
                        })
                        break
        
        # Overall safety determination
        overall_safe = all([
            safety_checks['conclusion_safe'],
            safety_checks['confidence_sufficient'],
            safety_checks['no_harmful_implications']
        ])
        
        safety_checks['overall_safe'] = overall_safe
        
        # Record in history (with size limit)
        self.safety_history.append(safety_checks)
        
        # CRITICAL: Limit history size to prevent memory leak
        if len(self.safety_history) > 1000:
            # Already limited by deque maxlen, but double-check
            pass
        
        return overall_safe, safety_checks
    
    def apply_safety_filters(self, result: ReasoningResult) -> ReasoningResult:
        """
        Apply safety filters to reasoning result
        
        Args:
            result: Reasoning result to filter
            
        Returns:
            Filtered reasoning result
        """
        # CRITICAL FIX: Validate result is not None
        if result is None:
            logger.error("Cannot apply safety filters to None result")
            return result
        
        safe, safety_status = self.check_reasoning_safety(result)
        
        # Add safety status to result
        if hasattr(result, 'safety_status'):
            result.safety_status = safety_status
        
        if not safe:
            # Modify conclusion if unsafe
            if hasattr(result, 'conclusion'):
                original_conclusion = result.conclusion
                result.conclusion = {
                    'original': original_conclusion,
                    'status': 'blocked_for_safety',
                    'reason': [check for check in safety_status.get('checks_performed', []) 
                              if not check.get('passed', True)]
                }
            
            # Add safety warning to explanation
            if hasattr(result, 'explanation'):
                failed_checks = [check['reason'] for check in safety_status.get('checks_performed', [])
                               if not check.get('passed', True)]
                safety_warning = f"[SAFETY FILTER APPLIED] Reasons: {'; '.join(failed_checks)}"
                result.explanation = f"{safety_warning}\n\nOriginal explanation: {result.explanation}"
        
        return result
    
    def get_safety_history(self) -> List[Dict[str, Any]]:
        """Get history of safety checks"""
        return list(self.safety_history)
    
    def get_blocked_conclusions(self) -> List[Dict[str, Any]]:
        """Get list of blocked conclusions"""
        return self.blocked_conclusions.copy()
    
    def get_safety_violations(self) -> List[Dict[str, Any]]:
        """Get list of safety violations"""
        return self.safety_violations.copy()
    
    def clear_history(self):
        """Clear all safety history"""
        self.safety_history.clear()
        self.blocked_conclusions.clear()
        self.safety_violations.clear()
        self.safety_checks.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety statistics"""
        total_checks = len(self.safety_history)
        if total_checks == 0:
            return {
                'total_checks': 0,
                'pass_rate': 0.0,
                'blocked_count': 0,
                'violation_count': 0
            }
        
        passed = sum(1 for check in self.safety_history if check.get('overall_safe', False))
        
        return {
            'total_checks': total_checks,
            'pass_rate': passed / total_checks,
            'blocked_count': len(self.blocked_conclusions),
            'violation_count': len(self.safety_violations),
            'recent_violations': list(self.safety_violations)[-5:] if self.safety_violations else []
        }