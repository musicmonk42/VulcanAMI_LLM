"""
Reasoning Result Formatters

This module provides functions for formatting reasoning engine results into
human-readable responses. Handles various reasoning types and result formats.

The formatters handle:
- ReasoningResult dataclass objects
- Moral uncertainty analysis (trolley problem, etc.)
- Deontic analysis (normative logic)
- Formal proofs and symbolic reasoning
- Pareto dominance analysis
- Debug wrapper unwrapping
- ReasoningType Enum to string conversion

Functions:
    get_reasoning_attr          - Safe attribute extraction from ReasoningResult
    reasoning_result_to_dict    - Convert ReasoningResult to dictionary
    reasoning_type_to_string    - Convert ReasoningType Enum/str to string safely
    format_reasoning_type_display - Format reasoning type for human display
    format_direct_reasoning_response - Format complete reasoning response
    format_conclusion_for_user  - Format conclusion value for human output
    format_moral_uncertainty_result - Format MEC analysis
    format_deontic_analysis_result - Format deontic logic
    format_formal_proof_result - Format proof results
    format_dominance_analysis_result - Format Pareto analysis
"""

import ast
import logging
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)

# Maximum size for ast.literal_eval to prevent DoS attacks
MAX_LITERAL_EVAL_SIZE = 10000  # 10KB limit


def get_reasoning_attr(result: Any, attr: str, default: Any = None) -> Any:
    """
    Safely extract an attribute from a ReasoningResult object or dictionary.
    
    This helper handles the case where reasoning results may be either:
    1. A dataclass object with attributes (like ReasoningResult)
    2. A dictionary with keys
    3. Any other object
    
    Args:
        result: The reasoning result (ReasoningResult, dict, or other)
        attr: The attribute/key name to extract
        default: Default value if attribute not found
        
    Returns:
        The attribute value or the default
    
    Note:
        This resolves "'ReasoningResult' object has no attribute 'get'" errors
        that occur when code tries to call .get() on ReasoningResult dataclasses.
    """
    if result is None:
        return default
    
    # Handle dictionary
    if isinstance(result, dict):
        return result.get(attr, default)
    
    # Handle objects with attributes (dataclasses, named tuples, etc.)
    if hasattr(result, attr):
        return getattr(result, attr, default)
    
    return default


def reasoning_result_to_dict(result: Any) -> Dict[str, Any]:
    """
    Convert a ReasoningResult object or similar to a dictionary safely.
    
    Extracts common reasoning attributes and handles enum values.
    
    Args:
        result: The reasoning result to convert
        
    Returns:
        A dictionary representation of the result
    
    Note:
        If the result doesn't have standard attributes, returns {"value": str(result)}
    """
    if result is None:
        return {}
    
    # Already a dict
    if isinstance(result, dict):
        return result
    
    # Handle objects with common reasoning attributes
    result_dict = {}
    for attr in ["conclusion", "confidence", "reasoning_type", "explanation", 
                 "uncertainty", "evidence", "safety_status", "metadata"]:
        if hasattr(result, attr):
            value = getattr(result, attr, None)
            # Handle enum values
            if hasattr(value, "value"):
                value = value.value
            result_dict[attr] = value
    
    return result_dict if result_dict else {"value": str(result)}


def reasoning_type_to_string(reasoning_type: Any) -> str:
    """
    Convert a reasoning_type to its string representation safely.
    
    Industry Standard: Type-safe conversion handling both Enum and string types.
    This prevents AttributeError when calling .replace() on Enum objects.
    
    Handles:
    1. ReasoningType Enum instances - extracts .value (the string representation)
    2. String values - returns as-is
    3. None - returns empty string
    4. Other objects - converts via str()
    
    Args:
        reasoning_type: The reasoning type (ReasoningType Enum, str, or other)
        
    Returns:
        String representation suitable for display formatting.
        
    Example:
        >>> from vulcan.reasoning.reasoning_types import ReasoningType
        >>> reasoning_type_to_string(ReasoningType.PROBABILISTIC)
        'probabilistic'
        >>> reasoning_type_to_string("causal_reasoning")
        'causal_reasoning'
        >>> reasoning_type_to_string(None)
        ''
        
    Note:
        This resolves "'ReasoningType' object has no attribute 'replace'" errors
        that occur when Enum objects are passed to string formatting code.
    """
    if reasoning_type is None:
        return ""
    
    # Handle Enum instances (ReasoningType, etc.)
    # Industry Standard: Check for .value first (standard Enum attribute)
    if hasattr(reasoning_type, 'value'):
        return str(reasoning_type.value)
    
    # Handle objects with .name attribute (Enum alternative)
    if hasattr(reasoning_type, 'name') and not isinstance(reasoning_type, str):
        return str(reasoning_type.name)
    
    # String or other - convert to string
    return str(reasoning_type)


def format_reasoning_type_display(reasoning_type: Any, default: str = "Hybrid") -> str:
    """
    Format a reasoning_type for human-readable display.
    
    Industry Standard: Single responsibility function for consistent display formatting.
    Converts underscores to spaces and applies title case.
    
    Args:
        reasoning_type: The reasoning type (ReasoningType Enum, str, or other)
        default: Default display value if reasoning_type is None/empty
        
    Returns:
        Human-readable formatted string (e.g., "Probabilistic", "Causal Reasoning")
        
    Example:
        >>> format_reasoning_type_display(ReasoningType.PROBABILISTIC)
        'Probabilistic'
        >>> format_reasoning_type_display("causal_reasoning")
        'Causal Reasoning'
        >>> format_reasoning_type_display(None)
        'Hybrid'
    """
    type_str = reasoning_type_to_string(reasoning_type)
    
    if not type_str:
        return default
    
    # Format: replace underscores with spaces and apply title case
    return type_str.replace("_", " ").title()


def format_direct_reasoning_response(
    conclusion: Any,
    confidence: float,
    reasoning_type: Union[str, Any],  # Accepts str or ReasoningType Enum
    explanation: str,
    reasoning_results: Dict[str, Any] = None,
) -> str:
    """
    Format reasoning engine result as final user response.
    
    This function formats high-confidence reasoning results directly for user
    output WITHOUT passing through OpenAI, ensuring that specialized reasoning
    engine outputs are preserved with their full context and accuracy.
    
    Handles:
    1. ReasoningResult dataclass objects - extracts meaningful content
    2. Debug wrapper dicts ({"original": ..., "filtered": True}) - unwraps them
    3. Philosophical/MEC conclusions - formats with human-readable analysis
    4. Reasoning chains - includes step-by-step explanations
    
    Args:
        conclusion: The main result/answer from the reasoning engine
        confidence: Confidence score (0.0 to 1.0)
        reasoning_type: Type of reasoning used (e.g., "probabilistic", "causal")
        explanation: Additional explanation from the engine
        reasoning_results: Full reasoning results dict for additional context
        
    Returns:
        Formatted string response suitable for direct user output
    
    Example:
        >>> response = format_direct_reasoning_response(
        ...     conclusion="P(D|+) = 0.166667",
        ...     confidence=0.95,
        ...     reasoning_type="probabilistic",
        ...     explanation="Applied Bayes' theorem with given parameters"
        ... )
        >>> print(response)
        P(D|+) = 0.166667
        
        Applied Bayes' theorem with given parameters
        
        Reasoning type: Probabilistic | Confidence: 95%
    
    Note:
        Previously, raw Python objects (like ReasoningResult) were dumped to users
        as their repr() string causing 6000+ token outputs of technical internals.
        This function properly extracts and formats user-facing content.
    """
    if reasoning_results is None:
        reasoning_results = {}
    
    response_parts = []
    
    # Handle ReasoningResult dataclass objects
    if hasattr(conclusion, 'conclusion') and hasattr(conclusion, 'confidence'):
        # This is a ReasoningResult object - extract meaningful fields
        inner_conclusion = getattr(conclusion, 'conclusion', None)
        inner_explanation = getattr(conclusion, 'explanation', '')
        inner_confidence = getattr(conclusion, 'confidence', confidence)
        inner_reasoning_type = getattr(conclusion, 'reasoning_type', None)
        
        # Update confidence if the inner one is higher
        if isinstance(inner_confidence, (int, float)) and inner_confidence > confidence:
            confidence = inner_confidence
        
        # Get reasoning type value if it's an enum
        if inner_reasoning_type and hasattr(inner_reasoning_type, 'value'):
            reasoning_type = inner_reasoning_type.value
        
        # Use inner explanation if outer one is empty
        if inner_explanation and (not explanation or explanation == str(conclusion)):
            explanation = inner_explanation
        
        # Replace conclusion with the unwrapped inner conclusion
        conclusion = inner_conclusion
    
    # Handle debug wrapper dicts that shouldn't be shown to users
    # Detect {"original": ..., "filtered": True, "reason": ...} pattern
    if isinstance(conclusion, dict) and "filtered" in conclusion and "original" in conclusion:
        # This is a debug wrapper - extract the original conclusion
        inner_conclusion = conclusion.get("original")
        filter_reason = conclusion.get("reason", "")
        
        # If reason mentions low confidence, add a user-friendly note
        if filter_reason and "below threshold" in filter_reason.lower():
            explanation = explanation or "Note: This analysis has moderate confidence due to query complexity."
        
        # Use the original conclusion
        conclusion = inner_conclusion
    
    # Main conclusion (the answer)
    if conclusion:
        formatted_conclusion = format_conclusion_for_user(conclusion, reasoning_type)
        if formatted_conclusion:
            response_parts.append(formatted_conclusion)
    
    # Add explanation if available and meaningful
    if explanation and explanation.strip():
        # Don't duplicate if explanation is same as conclusion
        explanation_str = str(explanation).strip()
        conclusion_str = str(conclusion).strip() if conclusion else ""
        if explanation_str != conclusion_str and explanation_str not in response_parts:
            response_parts.append(f"\n{explanation}")
    
    # Add reasoning chain/steps if available
    unified = reasoning_results.get("unified", {})
    if isinstance(unified, dict):
        reasoning_steps = unified.get("reasoning_steps", [])
        if reasoning_steps and len(reasoning_steps) > 0:
            response_parts.append("\n**Reasoning steps:**")
            for i, step in enumerate(reasoning_steps[:10], 1):  # Limit to 10 steps
                if isinstance(step, dict):
                    step_explanation = step.get("explanation", "")
                    if step_explanation:
                        response_parts.append(f"{i}. {step_explanation}")
                elif isinstance(step, str):
                    response_parts.append(f"{i}. {step}")
    
    # Add transparency footer with reasoning type and confidence
    confidence_pct = int(confidence * 100)
    # Industry Standard: Use helper function for type-safe enum/string conversion
    reasoning_type_display = format_reasoning_type_display(reasoning_type, default="Hybrid")
    response_parts.append(
        f"\nReasoning type: {reasoning_type_display} | Confidence: {confidence_pct}%"
    )
    
    return "\n".join(response_parts)


def format_conclusion_for_user(conclusion: Any, reasoning_type: str = "") -> str:
    """
    Format a conclusion value for human-readable output.
    
    Handles various conclusion types including:
    - ReasoningResult objects (recursively unwrap)
    - Moral uncertainty analysis dicts (format nicely for trolley problem)
    - Debug wrapper dicts (unwrap)
    - Embedded JSON/dict strings (parse and format)
    - Plain values (int, float, str)
    - Collections (list, tuple, dict)
    
    Args:
        conclusion: The conclusion value to format
        reasoning_type: Type of reasoning used (for context-specific formatting)
        
    Returns:
        Human-readable string representation of the conclusion
    
    Note:
        Some reasoning engines (like World Model in philosophical mode) return
        conclusions like "Some preamble text...\\n{'type': 'moral_uncertainty_analysis', ...}"
        This function detects and formats these embedded dicts properly.
    """
    if conclusion is None:
        return ""
    
    # Handle ReasoningResult objects recursively
    if hasattr(conclusion, 'conclusion') and hasattr(conclusion, 'confidence'):
        inner = getattr(conclusion, 'conclusion', conclusion)
        return format_conclusion_for_user(inner, reasoning_type)
    
    # Handle debug wrapper dicts
    if isinstance(conclusion, dict) and "filtered" in conclusion and "original" in conclusion:
        inner = conclusion.get("original", conclusion)
        return format_conclusion_for_user(inner, reasoning_type)
    
    # Handle simple types
    if isinstance(conclusion, (int, float)):
        return str(conclusion)
    
    if isinstance(conclusion, str):
        # Handle strings with embedded JSON/dict
        # Some reasoning engines return conclusions with embedded dict structures
        if "{" in conclusion and "}" in conclusion:
            # Find the start of the embedded dict
            dict_start = -1
            for i, char in enumerate(conclusion):
                if char == "{":
                    # Check if this looks like a dict start
                    remaining = conclusion[i:].lstrip()
                    if remaining.startswith("{'") or remaining.startswith('{"'):
                        dict_start = i
                        break
            
            if dict_start >= 0:
                # Extract preamble text before the dict
                preamble = conclusion[:dict_start].strip()
                dict_str = conclusion[dict_start:]
                
                # Security: Limit size to prevent DoS
                if len(dict_str) > MAX_LITERAL_EVAL_SIZE:
                    logger.debug(f"Dict string too large ({len(dict_str)} chars), skipping parse")
                    return conclusion
                
                try:
                    # Parse as Python dict literal using ast.literal_eval (safe)
                    # ast.literal_eval only evaluates literal structures - cannot execute code
                    embedded_dict = ast.literal_eval(dict_str)
                    if isinstance(embedded_dict, dict):
                        # Check if this is a known analysis type we can format nicely
                        if embedded_dict.get("type") == "moral_uncertainty_analysis":
                            formatted_analysis = format_moral_uncertainty_result(embedded_dict)
                            if preamble:
                                return f"{preamble}\n\n{formatted_analysis}"
                            return formatted_analysis
                        
                        # For other embedded dicts, format them nicely
                        formatted_dict = format_conclusion_for_user(embedded_dict, reasoning_type)
                        if preamble:
                            return f"{preamble}\n\n{formatted_dict}"
                        return formatted_dict
                except (ValueError, SyntaxError):
                    # Not a valid Python dict literal - return as-is
                    pass
        
        return conclusion
    
    if isinstance(conclusion, (list, tuple)):
        return "\n".join(str(item) for item in conclusion)
    
    # Handle dict conclusions with special formatting
    if isinstance(conclusion, dict):
        # Handle Moral Uncertainty Analysis (MEC) output
        if conclusion.get("type") == "moral_uncertainty_analysis":
            return format_moral_uncertainty_result(conclusion)
        
        # Handle deontic analysis output
        if conclusion.get("type") == "deontic_analysis":
            return format_deontic_analysis_result(conclusion)
        
        # Handle formal proof output
        if conclusion.get("type") == "formal_proof":
            return format_formal_proof_result(conclusion)
        
        # Handle dominance analysis output
        if conclusion.get("type") == "dominance_analysis":
            return format_dominance_analysis_result(conclusion)
        
        # Handle fallback analysis output
        if conclusion.get("type") == "fallback":
            error = conclusion.get("error", "Unknown error")
            partial = conclusion.get("partial_analysis", [])
            response = f"Analysis could not complete fully: {error}"
            if partial:
                response += f"\n\nPartial analysis concepts: {', '.join(partial)}"
            return response
        
        # Standard dict handling
        if "answer" in conclusion:
            return str(conclusion["answer"])
        if "result" in conclusion:
            return str(conclusion["result"])
        if "recommended_action" in conclusion:
            return f"Recommended action: {conclusion['recommended_action']}"
        
        # Filter out internal/debug fields and format nicely
        user_facing_fields = {
            k: v for k, v in conclusion.items() 
            if v is not None and not k.startswith("_") and k not in {
                "type", "metadata", "debug", "internal", "timestamp", "query_id"
            }
        }
        
        if user_facing_fields:
            lines = []
            for key, value in user_facing_fields.items():
                # Format key nicely
                formatted_key = key.replace("_", " ").title()
                lines.append(f"{formatted_key}: {value}")
            return "\n".join(lines)
    
    # Fallback: convert to string
    return str(conclusion)


def format_moral_uncertainty_result(conclusion: Dict[str, Any]) -> str:
    """
    Format a moral uncertainty analysis result for human-readable output.
    
    This handles the output from World Model philosophical reasoning for queries like
    the trolley problem, producing a clear analysis instead of raw Python dicts.
    
    Args:
        conclusion: Dict with keys: recommended_action, expected_choiceworthiness,
                   confidence, theory_evaluations, variance_voting
    
    Returns:
        Formatted multi-line string with ethical analysis
    """
    lines = []
    
    # Main recommendation
    recommended = conclusion.get("recommended_action", "Unknown")
    ec = conclusion.get("expected_choiceworthiness", 0.0)
    conf = conclusion.get("confidence", 0.0)
    
    lines.append(f"Answer: {recommended}")
    lines.append("")
    lines.append("This is a classic moral dilemma where both choices involve harm.")
    lines.append("")
    
    # Theory evaluations
    theory_evals = conclusion.get("theory_evaluations", {})
    if theory_evals:
        lines.append("Ethical framework analysis:")
        for theory, score in theory_evals.items():
            # Format theory name and score
            formatted_theory = theory.replace("_", " ")
            score_pct = score * 100 if isinstance(score, (int, float)) else 50
            lines.append(f"  • {formatted_theory}: {score_pct:.0f}%")
    
    # Variance voting if present
    variance_voting = conclusion.get("variance_voting", {})
    if variance_voting:
        winner = variance_voting.get("winner", "")
        votes = variance_voting.get("votes", {})
        if winner:
            lines.append("")
            lines.append(f"Consensus winner: {winner}")
            if votes:
                vote_str = ", ".join(f"{k}: {v}" for k, v in votes.items())
                lines.append(f"Votes: {vote_str}")
    
    # Confidence interpretation
    lines.append("")
    if conf < 0.5:
        lines.append(f"ℹ️ Low confidence ({conf:.0%}): This reflects genuine moral uncertainty, not lack of reasoning.")
    elif conf < 0.7:
        lines.append(f"ℹ️ Moderate confidence ({conf:.0%}): This dilemma has no clear answer.")
    else:
        lines.append(f"ℹ️ Confidence: {conf:.0%}")
    
    return "\n".join(lines)


def format_deontic_analysis_result(conclusion: Dict[str, Any]) -> str:
    """
    Format deontic analysis result for human-readable output.
    
    Args:
        conclusion: Dict with keys: formulas, inferences, consistent
    
    Returns:
        Formatted multi-line string with deontic logic analysis
    """
    lines = []
    
    formulas = conclusion.get("formulas", [])
    inferences = conclusion.get("inferences", [])
    consistent = conclusion.get("consistent", True)
    
    if formulas:
        lines.append("Deontic formulas analyzed:")
        for f in formulas:
            lines.append(f"  • {f}")
    
    if inferences:
        lines.append("")
        lines.append("Inferences:")
        for inf in inferences:
            lines.append(f"  • {inf}")
    
    if not consistent:
        lines.append("")
        lines.append("⚠️ Warning: Inconsistency detected in the deontic system.")
    
    return "\n".join(lines) if lines else "Deontic analysis completed."


def format_formal_proof_result(conclusion: Dict[str, Any]) -> str:
    """
    Format formal proof result for human-readable output.
    
    Args:
        conclusion: Dict with keys: success, proof_results
    
    Returns:
        Formatted multi-line string with proof status
    """
    lines = []
    
    success = conclusion.get("success", False)
    proof_results = conclusion.get("proof_results", [])
    
    if success:
        lines.append("✓ Proof successful")
    else:
        lines.append("✗ Proof not found")
    
    if proof_results:
        lines.append("")
        for pr in proof_results:
            formula = pr.get("formula", "Unknown")
            proven = pr.get("proven", False)
            status = "PROVEN ✓" if proven else "NOT PROVEN"
            lines.append(f"  • {formula}: {status}")
    
    return "\n".join(lines) if lines else "Formal proof attempted."


def format_dominance_analysis_result(conclusion: Dict[str, Any]) -> str:
    """
    Format Pareto dominance analysis result for human-readable output.
    
    Args:
        conclusion: Dict with keys: pareto_frontier, dominated_actions
    
    Returns:
        Formatted multi-line string with dominance analysis
    """
    lines = []
    
    frontier = conclusion.get("pareto_frontier", [])
    dominated = conclusion.get("dominated_actions", [])
    
    if frontier:
        lines.append("Pareto-optimal actions (non-dominated):")
        for action in frontier:
            lines.append(f"  • {action}")
    
    if dominated:
        lines.append("")
        lines.append("Dominated actions:")
        for action in dominated:
            lines.append(f"  • {action}")
    
    return "\n".join(lines) if lines else "Dominance analysis completed."
