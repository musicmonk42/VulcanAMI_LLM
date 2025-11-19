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

Dependencies:
- openai==2.3.0 (latest as of October 2025; pip install openai)
- tenacity==8.5.0 (for retries; pip install tenacity)
- Standard Python libraries (json, hashlib, os, logging, datetime, typing).

Best Practices (2025 Standards):
- Secure API key handling via environment variables.
- Robust error handling with retries for transient failures.
- Integration with Graphix IR for seamless agent execution.
- Logging aligned with ObservabilityManager (observability_logs).

"""

import json
import hashlib
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, OpenAIError  # Requires pip install openai==2.3.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class GraphixLLMClient:
    """
    Enhanced client for Graphix IR LLM interactions using OpenAI API.
    Designed for GraphixArena integration with generator, evolver, and visualizer agents.
    """
    def __init__(self, agent_id: str = "agent-default", model: str = "gpt-4o", temperature: float = 0.01, max_tokens: int = 4096):
        """
        Initialize the LLM client with OpenAI API and Graphix-specific settings.
        
        Args:
            agent_id (str): Identifier for the agent (e.g., 'generator', 'evolver').
            model (str): LLM model to use (default from crew_config.yaml: gpt-4o).
            temperature (float): Sampling temperature (default from crew_config.yaml: 0.01).
            max_tokens (int): Maximum tokens for response (default: 4096).
        """
        self.agent_id = agent_id
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger("GraphixLLMClient")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OPENAI_API_KEY not set in environment variables")
            raise ValueError("OPENAI_API_KEY not set in .env")
        
        self.client = OpenAI(api_key=api_key)
        self.logger.info(f"Client initialized for agent: {agent_id} with model: {model}, temperature: {temperature}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((OpenAIError, ConnectionError)),
        before_sleep=lambda retry_state: logging.getLogger("GraphixLLMClient").warning(
            f"Retrying due to {retry_state.outcome.exception()}. Attempt {retry_state.attempt_number}..."
        )
    )
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform a chat interaction using OpenAI API, returning a Graphix IR graph.

        Args:
            messages (List[Dict[str, str]]): List of message dicts with 'role' and 'content'.
            model (Optional[str]): Override default model (e.g., 'gpt-4o-mini').
            temperature (Optional[float]): Override default temperature.
            max_tokens (Optional[int]): Override default max tokens.

        Returns:
            Dict[str, Any]: Contains response, IR graph, and proposal ID.

        Raises:
            OpenAIError: If API call fails after retries.
        """
        try:
            # Use defaults or overrides
            model = model or self.model
            temperature = temperature or self.temperature
            max_tokens = max_tokens or self.max_tokens

            # Validate messages
            if not messages or not all('role' in m and 'content' in m for m in messages):
                self.logger.error("Invalid messages format")
                raise ValueError("Messages must be a list of dicts with 'role' and 'content'")

            # Call OpenAI API
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = completion.choices[0].message.content

            # Generate IR graph for Graphix platform
            last_content = messages[-1].get("content", "") if messages else ""
            proposal_id = hashlib.sha256(last_content.encode()).hexdigest()[:8]
            ir_graph = {
                "id": proposal_id,
                "type": "Graph",
                "nodes": [
                    {"id": "prompt", "type": "PromptNode", "value": last_content},
                    {"id": "generate", "type": "GenerativeNode", "params": {"model": model, "temperature": temperature}},
                    {"id": "output", "type": "OutputNode", "value": response}
                ],
                "edges": [
                    {"from": "prompt", "to": {"node": "generate", "port": "input"}, "type": "data"},
                    {"from": "generate", "to": {"node": "output", "port": "input"}, "type": "data"}
                ],
                "metadata": {
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "model": model
                }
            }

            # Log interaction for ObservabilityManager
            self.logger.info(f"Chat response for agent {self.agent_id}: {response[:50]}... (ID: {proposal_id})")

            return {
                "response": response,
                "ir": ir_graph,
                "proposal_id": proposal_id
            }

        except OpenAIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in chat: {str(e)}")
            raise

if __name__ == "__main__":
    # Demo usage
    client = GraphixLLMClient(agent_id="agent-grok")
    messages = [{"role": "user", "content": "Generate a Python function for matrix multiplication"}]

    print("\n--- Demo: OpenAI LLM Interaction ---")
    try:
        result = client.chat(messages)
        print(f"Response: {result['response'][:100]}...")
        print(f"Generated IR ID: {result['proposal_id']}")
        print(f"IR Graph: {json.dumps(result['ir'], indent=2)}")
    except Exception as e:
        print(f"Demo failed: {str(e)}")