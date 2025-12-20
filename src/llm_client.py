"""
Graphix LLM Client
===================

This module provides an enhanced client for Graphix IR LLM interactions using the OpenAI API.

Key Features:
- Integrates with OpenAI for real LLM interactions with best practices (secure key handling, error management).
- Represents LLM interactions as Graphix IR graphs for compatibility with GraphixArena agents.
- Supports configurable models, temperature, and max tokens from crew_config.yaml defaults.
- Includes retry logic for transient API errors and comprehensive logging.
- Generates IR graphs compatible with node_handlers.py (e.g., GenerativeNode).
- Aligns with platform security (e.g., NSOAligner checks) and observability.
- Graceful degradation: works in mock mode when OPENAI_API_KEY is not configured.
- Multi-layered safety validation including pre-query and post-response checks.
- Safety system prompt integration for safer LLM interactions.
- Risk classification for governance routing.

Dependencies:
- openai==2.3.0 (latest as of October 2025; pip install openai)
- tenacity==8.5.0 (for retries; pip install tenacity)
- Standard Python libraries (json, hashlib, os, logging, datetime, typing).

Best Practices (2025 Standards):
- Secure API key handling via environment variables.
- Robust error handling with retries for transient failures.
- Integration with Graphix IR for seamless agent execution.
- Logging aligned with ObservabilityManager (observability_logs).
- Multi-layered safety validation (GDPR, HIPAA, ITU F.748.53, EU AI Act compliance).

"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env file if it exists and dotenv is available
try:
    from dotenv import load_dotenv

    # Try multiple paths for .env file
    env_paths = [
        Path(__file__).parent.parent / ".env",  # project_root/.env
        Path(__file__).parent / ".env",  # src/.env
        Path.cwd() / ".env",  # current working directory
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            break
except ImportError:
    pass  # dotenv not available, rely on system environment variables

# Try to import tenacity for retry logic, handle gracefully if not available
try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

    # Create a no-op decorator as fallback when tenacity is not installed.
    # This enables the @retry(...) decorator syntax to work without tenacity.
    #
    # Handles two decorator usage patterns:
    # 1. @retry - decorator without parentheses (args[0] is the function)
    # 2. @retry(...) - decorator with arguments (returns a decorator)
    def retry(*args, **kwargs):
        """No-op retry decorator when tenacity is not installed."""

        def decorator(func):
            return func

        # Pattern 1: @retry - first arg is the decorated function itself
        if len(args) == 1 and callable(args[0]):
            return args[0]
        # Pattern 2: @retry(...) - return a decorator that will receive the function
        return decorator

    # No-op functions for tenacity configuration parameters.
    # These are passed as arguments to retry() and their return values
    # are ignored by the no-op retry decorator above.
    def stop_after_attempt(attempts):
        """No-op stop condition."""
        return None

    def wait_exponential(**kwargs):
        """No-op wait strategy."""
        return None

    def retry_if_exception_type(exception_types):
        """No-op retry condition."""
        return None


# Try to import OpenAI, handle gracefully if not available
try:
    from openai import OpenAI, OpenAIError

    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OpenAIError = Exception  # Fallback for retry decorator
    OPENAI_AVAILABLE = False

# Try to import safety validator components
try:
    from vulcan.safety.safety_validator import initialize_all_safety_components
    from generation.safe_generation import RiskLevel
    SAFETY_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import paths
        from src.vulcan.safety.safety_validator import initialize_all_safety_components
        from src.generation.safe_generation import RiskLevel
        SAFETY_AVAILABLE = True
    except ImportError:
        initialize_all_safety_components = None
        RiskLevel = None
        SAFETY_AVAILABLE = False

# Safety System Prompt for all LLM interactions
SAFETY_SYSTEM_PROMPT = """You are Vulcan, an AI with democratic governance and safety oversight.

Core constraints:
- Refuse requests that could cause harm
- Don't provide operational attack guides (phishing, social engineering, exploits)
- Recognize dual-use information and err on the side of caution
- Explain refusals clearly and suggest alternatives when possible

Your architecture includes multi-layered safety validation, compliance checking (GDPR, HIPAA, ITU F.748.53, EU AI Act), and democratic oversight.

When refusing a request, explain why the request cannot be fulfilled and suggest safer alternatives if available."""

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
MOCK_RESPONSE_TRUNCATION_LENGTH = 100


class GraphixLLMClient:
    """
    Enhanced client for Graphix IR LLM interactions using OpenAI API.
    Designed for GraphixArena integration with generator, evolver, and visualizer agents.

    Supports graceful degradation: when OPENAI_API_KEY is not configured or OpenAI
    package is not installed, the client operates in mock mode and returns placeholder
    responses.
    
    Includes multi-layered safety validation:
    - Pre-query validation to block unsafe queries
    - Safety system prompt for governance-aligned responses
    - Post-response validation to catch unsafe outputs
    - Risk classification for governance routing
    """

    def __init__(
        self,
        agent_id: str = "agent-default",
        model: str = "gpt-4o",
        temperature: float = 0.01,
        max_tokens: int = 4096,
        enable_safety: bool = True,
    ):
        """
        Initialize the LLM client with OpenAI API and Graphix-specific settings.

        Args:
            agent_id (str): Identifier for the agent (e.g., 'generator', 'evolver').
            model (str): LLM model to use (default from crew_config.yaml: gpt-4o).
            temperature (float): Sampling temperature (default from crew_config.yaml: 0.01).
            max_tokens (int): Maximum tokens for response (default: 4096).
            enable_safety (bool): Enable safety validation (default: True).
        """
        self.agent_id = agent_id
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger("GraphixLLMClient")
        self.client = None
        self.mock_mode = False
        self.enable_safety = enable_safety
        self.safety_validator = None

        api_key = os.getenv("OPENAI_API_KEY")

        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI package not installed. Running in mock mode.")
            self.logger.warning("To enable real LLM features: pip install openai")
            self.mock_mode = True
        elif not api_key:
            self.logger.warning(
                "OPENAI_API_KEY not set in environment variables. Running in mock mode."
            )
            self.logger.warning(
                "To enable real LLM features: set OPENAI_API_KEY in your .env file"
            )
            self.mock_mode = True
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                self.logger.info(
                    f"Client initialized for agent: {agent_id} with model: {model}, temperature: {temperature}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize OpenAI client: {e}. Running in mock mode."
                )
                self.mock_mode = True
        
        # Initialize safety validator if enabled and available
        if self.enable_safety and SAFETY_AVAILABLE:
            try:
                self.safety_validator = initialize_all_safety_components()
                self.logger.info("Safety validator initialized for LLM client")
            except Exception as e:
                self.logger.warning(f"Failed to initialize safety validator: {e}")
                self.safety_validator = None
        elif self.enable_safety and not SAFETY_AVAILABLE:
            self.logger.warning("Safety validation requested but safety modules not available")

    @property
    def is_safety_enabled(self) -> bool:
        """Check if safety validation is enabled and available."""
        return self.enable_safety and self.safety_validator is not None

    @property
    def is_available(self) -> bool:
        """Check if the LLM client is available for real API calls."""
        return not self.mock_mode and self.client is not None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((OpenAIError, ConnectionError)),
        before_sleep=lambda retry_state: logging.getLogger("GraphixLLMClient").warning(
            f"Retrying due to {retry_state.outcome.exception()}. Attempt {retry_state.attempt_number}..."
        ),
    )
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        skip_safety: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform a chat interaction using OpenAI API, returning a Graphix IR graph.
        
        Includes multi-layered safety validation:
        - Pre-query validation to block unsafe queries
        - Safety system prompt injection for safer responses
        - Post-response validation to catch unsafe outputs

        Args:
            messages (List[Dict[str, str]]): List of message dicts with 'role' and 'content'.
            model (Optional[str]): Override default model (e.g., 'gpt-4o-mini').
            temperature (Optional[float]): Override default temperature.
            max_tokens (Optional[int]): Override default max tokens.
            skip_safety (bool): Skip safety validation (use with caution, default: False).

        Returns:
            Dict[str, Any]: Contains response, IR graph, proposal ID, and safety info.

        Raises:
            OpenAIError: If API call fails after retries (only in non-mock mode).
        """
        # Use defaults or overrides
        model = model or self.model
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Validate messages
        if not messages or not all("role" in m and "content" in m for m in messages):
            self.logger.error("Invalid messages format")
            raise ValueError(
                "Messages must be a list of dicts with 'role' and 'content'"
            )

        last_content = messages[-1].get("content", "") if messages else ""
        proposal_id = hashlib.sha256(last_content.encode()).hexdigest()[:8]
        
        # Safety metadata for response
        safety_info = {
            "safety_enabled": self.is_safety_enabled and not skip_safety,
            "pre_check_passed": True,
            "post_check_passed": True,
            "risk_level": "SAFE",
        }
        
        # Priority 1: Pre-query safety validation
        if self.is_safety_enabled and not skip_safety:
            try:
                pre_check = self.safety_validator.validate_query(last_content)
                safety_info["pre_check_passed"] = pre_check.safe
                safety_info["pre_check_confidence"] = pre_check.confidence
                
                if not pre_check.safe:
                    # Block the query and return a refusal response
                    refusal_reason = pre_check.reasons[0] if pre_check.reasons else "Query blocked by safety validation"
                    self.logger.warning(f"Query blocked by safety validation: {refusal_reason}")
                    
                    response = f"I can't help with that. {refusal_reason}"
                    safety_info["blocked"] = True
                    safety_info["block_reason"] = refusal_reason
                    
                    # Generate IR graph for blocked response
                    ir_graph = self._generate_ir_graph(
                        proposal_id, last_content, response, model, temperature, 
                        blocked=True, block_reason=refusal_reason
                    )
                    
                    return {
                        "response": response,
                        "ir": ir_graph,
                        "proposal_id": proposal_id,
                        "safe": False,
                        "reason": refusal_reason,
                        "safety_info": safety_info,
                    }
                
                # Priority 3: Risk classification for governance routing
                risk_level = self.safety_validator.classify_query_risk(last_content)
                risk_level_name = risk_level.name if hasattr(risk_level, 'name') else str(risk_level)
                safety_info["risk_level"] = risk_level_name
                
                # For high-risk queries, could integrate governance approval here
                # Currently logging for awareness; full governance integration would go here
                is_high_risk = risk_level_name in ("HIGH", "CRITICAL")
                if is_high_risk:
                    self.logger.warning(f"High-risk query detected (risk={risk_level_name}): governance approval may be required")
                    safety_info["requires_governance"] = True
                    
            except Exception as e:
                self.logger.error(f"Safety pre-check failed: {e}")
                safety_info["pre_check_error"] = str(e)
        
        # Prepare messages with safety system prompt (Priority 2)
        messages_with_safety = messages.copy()
        if self.is_safety_enabled and not skip_safety:
            # Add safety system prompt if not already present
            has_system_prompt = any(m.get("role") == "system" for m in messages_with_safety)
            if not has_system_prompt:
                messages_with_safety.insert(0, {"role": "system", "content": SAFETY_SYSTEM_PROMPT})
            else:
                # Prepend safety prompt to existing system message
                for i, m in enumerate(messages_with_safety):
                    if m.get("role") == "system":
                        messages_with_safety[i] = {
                            "role": "system",
                            "content": SAFETY_SYSTEM_PROMPT + "\n\n" + m.get("content", "")
                        }
                        break

        # Handle mock mode
        if self.mock_mode:
            response = f"[Mock Response] This is a simulated response for: {last_content[:MOCK_RESPONSE_TRUNCATION_LENGTH]}..."
            self.logger.debug(f"Mock chat response generated for agent {self.agent_id}")
        else:
            try:
                # Call OpenAI API with safety-enhanced messages
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages_with_safety,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = completion.choices[0].message.content
            except Exception as e:
                self.logger.error(f"OpenAI API error: {str(e)}")
                raise
        
        # Priority 4: Post-generation validation
        if self.is_safety_enabled and not skip_safety:
            try:
                post_check = self.safety_validator.validate_response(response, last_content)
                safety_info["post_check_passed"] = post_check.safe
                safety_info["post_check_confidence"] = post_check.confidence
                
                if not post_check.safe:
                    # Response failed safety validation - could implement retry with stricter constraints
                    post_reason = post_check.reasons[0] if post_check.reasons else "Response blocked by safety validation"
                    self.logger.warning(f"Response failed post-generation safety check: {post_reason}")
                    
                    # Replace with a safe refusal response
                    response = f"I apologize, but I cannot provide that response as it may contain content that violates our safety policies. {post_reason}"
                    safety_info["response_replaced"] = True
                    safety_info["replacement_reason"] = post_reason
                    
            except Exception as e:
                self.logger.error(f"Safety post-check failed: {e}")
                safety_info["post_check_error"] = str(e)

        # Generate IR graph for Graphix platform
        ir_graph = self._generate_ir_graph(
            proposal_id, last_content, response, model, temperature,
            safety_info=safety_info
        )

        # Log interaction for ObservabilityManager
        self.logger.info(
            f"Chat response for agent {self.agent_id}: {response[:50]}... (ID: {proposal_id})"
        )

        return {
            "response": response,
            "ir": ir_graph,
            "proposal_id": proposal_id,
            "safe": safety_info.get("pre_check_passed", True) and safety_info.get("post_check_passed", True),
            "safety_info": safety_info,
        }
    
    def _generate_ir_graph(
        self,
        proposal_id: str,
        prompt: str,
        response: str,
        model: str,
        temperature: float,
        blocked: bool = False,
        block_reason: Optional[str] = None,
        safety_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an IR graph for the Graphix platform.
        
        Args:
            proposal_id: Unique proposal identifier
            prompt: The original user prompt
            response: The generated response
            model: Model used for generation
            temperature: Temperature used for generation
            blocked: Whether the query was blocked by safety validation
            block_reason: Reason for blocking if applicable
            safety_info: Safety validation metadata
        
        Returns:
            Dict containing the IR graph structure
        """
        nodes = [
            {"id": "prompt", "type": "PromptNode", "value": prompt},
            {
                "id": "generate",
                "type": "GenerativeNode",
                "params": {"model": model, "temperature": temperature},
            },
            {"id": "output", "type": "OutputNode", "value": response},
        ]
        
        # Add safety node if validation was performed
        if safety_info and safety_info.get("safety_enabled"):
            nodes.insert(1, {
                "id": "safety_check",
                "type": "SafetyValidationNode",
                "params": {
                    "pre_check_passed": safety_info.get("pre_check_passed", True),
                    "post_check_passed": safety_info.get("post_check_passed", True),
                    "risk_level": safety_info.get("risk_level", "SAFE"),
                },
            })
        
        edges = [
            {
                "from": "prompt",
                "to": {"node": "safety_check" if safety_info and safety_info.get("safety_enabled") else "generate", "port": "input"},
                "type": "data",
            },
        ]
        
        if safety_info and safety_info.get("safety_enabled"):
            edges.append({
                "from": "safety_check",
                "to": {"node": "generate", "port": "input"},
                "type": "data",
            })
        
        edges.append({
            "from": "generate",
            "to": {"node": "output", "port": "input"},
            "type": "data",
        })
        
        metadata = {
            "agent_id": self.agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "mock_mode": self.mock_mode,
        }
        
        if blocked:
            metadata["blocked"] = True
            metadata["block_reason"] = block_reason
        
        if safety_info:
            metadata["safety_info"] = safety_info
        
        return {
            "id": proposal_id,
            "type": "Graph",
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata,
        }


if __name__ == "__main__":
    # Demo usage
    client = GraphixLLMClient(agent_id="agent-grok")
    messages = [
        {
            "role": "user",
            "content": "Generate a Python function for matrix multiplication",
        }
    ]

    print("\n--- Demo: OpenAI LLM Interaction ---")
    print(f"Mock mode: {client.mock_mode}")
    print(f"Is available: {client.is_available}")
    try:
        result = client.chat(messages)
        print(f"Response: {result['response'][:100]}...")
        print(f"Generated IR ID: {result['proposal_id']}")
        print(f"IR Graph: {json.dumps(result['ir'], indent=2)}")
    except Exception as e:
        print(f"Demo failed: {str(e)}")
