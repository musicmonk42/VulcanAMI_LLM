"""
AI Runtime Integration Module for Graphix IR
Provides unified interface for AI providers and services
"""

import asyncio
import json
import time
import hashlib
import os
from typing import Dict, Any, Optional, List, Union, Callable, Awaitable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import random
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# ERROR CODES
# ============================================================================

class AI_ERRORS(Enum):
    """Standardized AI runtime error codes"""
    AI_INVALID_REQUEST = "AI_INVALID_REQUEST"
    AI_UNAUTHORIZED = "AI_UNAUTHORIZED"
    AI_TIMEOUT = "AI_TIMEOUT"
    AI_INTERNAL_ERROR = "AI_INTERNAL_ERROR"
    AI_UNSUPPORTED = "AI_UNSUPPORTED"
    AI_RESOURCE_LIMIT = "AI_RESOURCE_LIMIT"
    AI_SAFETY_VIOLATION = "AI_SAFETY_VIOLATION"
    AI_RATE_LIMIT = "AI_RATE_LIMIT"
    AI_QUOTA_EXCEEDED = "AI_QUOTA_EXCEEDED"
    AI_MODEL_NOT_FOUND = "AI_MODEL_NOT_FOUND"
    AI_PROVIDER_ERROR = "AI_PROVIDER_ERROR"
    AI_NETWORK_ERROR = "AI_NETWORK_ERROR"
    AI_RESPONSE_PARSE_ERROR = "AI_RESPONSE_PARSE_ERROR"
    AI_VALIDATION_ERROR = "AI_VALIDATION_ERROR" # Added from test failure


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class AIContract:
    """Contract specifying SLA and constraints for AI operations"""
    max_latency_ms: Optional[float] = 1000.0 # Match test default
    min_accuracy: Optional[float] = 0.9 # Match test default
    max_cost: Optional[float] = 0.01
    required_safety_level: str = "standard"
    allow_cached: bool = True
    require_deterministic: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate contract parameters"""
        if self.max_latency_ms is not None and self.max_latency_ms <= 0:
            return False, "max_latency_ms must be positive"
        if self.min_accuracy is not None and not (0 <= self.min_accuracy <= 1):
            return False, "min_accuracy must be between 0 and 1"
        # Increased upper bound to 2.0 based on OpenAI docs
        if self.temperature is not None and not (0 <= self.temperature <= 2.0):
            return False, "temperature must be between 0 and 2.0"
        # Add more validation as needed (e.g., safety_level enum)
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AITask:
    """Task specification for AI operations"""
    operation: str  # EMBED, GENERATE, CLASSIFY, etc.
    provider: str  # OpenAI, Anthropic, Grok, etc.
    model: str  # Model identifier
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 0  # Higher = more important
    deadline: Optional[datetime] = None
    trace_id: Optional[str] = None

    def __post_init__(self):
        if not self.trace_id:
            # Generate trace ID for tracking
            trace_data = f"{self.operation}_{self.provider}_{self.model}_{time.time()}_{random.random()}"
            self.trace_id = hashlib.md5(trace_data.encode()).hexdigest()[:16]

    def is_expired(self) -> bool:
        """Check if task has passed deadline"""
        if self.deadline:
            # Ensure timezone-aware comparison if deadline might have timezone
            # For simplicity, assuming naive datetime for now()
            # Make comparison robust even if deadline is naive
            now = datetime.now(self.deadline.tzinfo) if self.deadline.tzinfo else datetime.now()
            return now > self.deadline
        return False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.deadline:
            data['deadline'] = self.deadline.isoformat()
        return data


@dataclass
class AIResult:
    """Result from AI operation"""
    status: str  # SUCCESS, FAILED, TIMEOUT, etc.
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    latency_ms: float = 0.0 # Default to float
    cost: float = 0.0 # Default to float
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    provider_response: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None

    def is_success(self) -> bool:
        return self.status == "SUCCESS"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# RATE LIMITER (Adjusted based on test failures)
# ============================================================================
class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self):
        self._lock = threading.Lock()
        # provider -> {"timestamps": deque(), "calls": limit, "window": seconds}
        self.limits: Dict[str, Dict[str, Any]] = {
            # Add default limits, can be overridden by config
            "OpenAI": {"calls": 60, "window": 60},
            "Anthropic": {"calls": 30, "window": 60},
            # Add others as needed
        }
        # Fixed: Use bounded deques to prevent unbounded memory growth
        # Each provider gets a deque with maxlen = 2x their rate limit (enough for rolling window)
        self.calls: Dict[str, deque] = {}

    def _get_or_create_deque(self, provider_name: str) -> deque:
        """Get or create a bounded deque for the provider"""
        if provider_name not in self.calls:
            limit_info = self.limits.get(provider_name, {"calls": 100})
            max_len = limit_info["calls"] * 2  # 2x the limit is enough for rolling window
            self.calls[provider_name] = deque(maxlen=max_len)
        return self.calls[provider_name]

    def check_limit(self, provider_name: str) -> bool:
        """Check if call is within rate limits"""
        now = time.time()
        # Use provider_name directly, assuming it's already normalized or matches keys
        limit_info = self.limits.get(provider_name)
        if not limit_info:
            logger.debug(f"No rate limits defined for provider '{provider_name}'. Allowing call.")
            return True # No limits defined for this provider

        max_calls = limit_info["calls"]
        window_seconds = limit_info["window"]

        with self._lock:
            timestamps = self._get_or_create_deque(provider_name)
            # Remove timestamps older than the window
            window_start = now - window_seconds
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()

            # Check count against limit
            if len(timestamps) >= max_calls:
                logger.warning(f"Rate limit exceeded for provider {provider_name}")
                return False

            # Add current timestamp and allow call
            timestamps.append(now)
            return True

    def reset(self, provider: Optional[str] = None):
        """Reset rate limit counters"""
        with self._lock:
            if provider:
                if provider in self.calls:
                    self.calls[provider].clear()
            else:
                self.calls.clear()


# ============================================================================
# PROVIDER INTERFACES
# ============================================================================

class AIProvider(ABC):
    """Abstract base class for AI providers"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limiter = RateLimiter() # Instantiate RateLimiter

    @abstractmethod
    async def execute(self, task: AITask, contract: AIContract) -> AIResult:
        """Execute AI task with contract constraints (must be async)"""
        pass

    @abstractmethod
    def supports_operation(self, operation: str) -> bool:
        """Check if provider supports operation"""
        pass

    @abstractmethod
    def get_models(self) -> List[str]:
        """Get list of available models"""
        pass

    async def validate_credentials(self) -> bool:
        """Validate API credentials"""
        return self.api_key is not None


class OpenAIProvider(AIProvider):
    """OpenAI API provider implementation"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key, "https://api.openai.com/v1")
        self.supported_operations = ["EMBED", "GENERATE", "COMPLETE", "CLASSIFY"]
        self.models = {
            "EMBED": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            "GENERATE": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "COMPLETE": ["gpt-4", "gpt-3.5-turbo"], # Usually same models as GENERATE
            "CLASSIFY": ["gpt-4", "gpt-3.5-turbo"] # Often done via GENERATE/COMPLETE
        }
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
             if AIOHTTP_AVAILABLE:
                 self._session = aiohttp.ClientSession()
             else:
                  raise ImportError("aiohttp is required for async OpenAI calls but not installed.")
        return self._session

    async def execute(self, task: AITask, contract: AIContract) -> AIResult:
        """Execute OpenAI API call (async implementation)"""
        start_time_ns = time.time_ns()

        if not self.api_key:
             return AIResult(status="FAILED", error="OpenAI API key not configured", error_code=AI_ERRORS.AI_UNAUTHORIZED.value, trace_id=task.trace_id)

        # Check rate limits using the provider's specific name
        if not self.rate_limiter.check_limit("OpenAI"):
            return AIResult(
                status="FAILED",
                error="Rate limit exceeded",
                error_code=AI_ERRORS.AI_RATE_LIMIT.value,
                trace_id=task.trace_id
            )

        result: Optional[AIResult] = None
        try:
            if task.operation == "EMBED":
                result = await self._embed(task, contract)
            elif task.operation in ["GENERATE", "COMPLETE"]:
                result = await self._generate_or_complete(task, contract)
            elif task.operation == "CLASSIFY":
                 result = await self._classify(task, contract) # Typically uses generate/complete
            else:
                result = AIResult(
                    status="FAILED",
                    error=f"Unsupported operation: {task.operation}",
                    # Corrected error code based on test
                    error_code=AI_ERRORS.AI_UNSUPPORTED.value,
                    trace_id=task.trace_id
                )

            latency_ms = (time.time_ns() - start_time_ns) / 1_000_000.0
            if result: result.latency_ms = latency_ms # Set latency on the result object

            if contract.max_latency_ms and latency_ms > contract.max_latency_ms:
                if result:
                     if result.metadata is None: result.metadata = {}
                     result.metadata['warning'] = f"Latency exceeded contract: {latency_ms:.2f}ms > {contract.max_latency_ms}ms"
                else:
                     result = AIResult(status="FAILED", error="Result object missing after operation", error_code=AI_ERRORS.AI_INTERNAL_ERROR.value, trace_id=task.trace_id, latency_ms=latency_ms)

            return result

        except Exception as e:
            logger.error(f"OpenAI provider error during {task.operation}: {e}", exc_info=True)
            latency_ms = (time.time_ns() - start_time_ns) / 1_000_000.0
            return AIResult(
                status="FAILED",
                error=str(e),
                error_code=AI_ERRORS.AI_PROVIDER_ERROR.value,
                latency_ms=latency_ms,
                trace_id=task.trace_id
            )

    async def _embed(self, task: AITask, contract: AIContract) -> AIResult:
        """Generate embeddings using OpenAI API"""
        text = task.payload.get('text', '')
        requested_dim = task.payload.get('dim') # Get the requested dimension

        if not text:
            return AIResult(status="FAILED", error="No text provided for embedding", error_code=AI_ERRORS.AI_INVALID_REQUEST.value, trace_id=task.trace_id)

        if not AIOHTTP_AVAILABLE:
            return AIResult(status="FAILED", error="aiohttp is required for OpenAI provider", error_code=AI_ERRORS.AI_INTERNAL_ERROR.value, trace_id=task.trace_id)

        api_payload = {
            "input": text,
            "model": task.model
        }

        # --- >>> ADDED DIMENSION LOGIC <<< ---
        # If a specific dimension is requested in the task, add it to the API payload.
        # The parameter name for the OpenAI API is 'dimensions'.
        if requested_dim is not None:
            try:
                api_payload["dimensions"] = int(requested_dim)
                logger.info(f"Requesting OpenAI embedding with dimensions: {api_payload['dimensions']}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid 'dim' value ({requested_dim}) provided in task payload, ignoring.")
        # --- >>> END ADDED DIMENSION LOGIC <<< ---

        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        api_timeout = aiohttp.ClientTimeout(total=contract.max_latency_ms / 1000 if contract.max_latency_ms else 30) # Default 30s timeout

        try:
            async with session.post(f"{self.base_url}/embeddings",
                                    headers=headers,
                                    json=api_payload,
                                    timeout=api_timeout) as response:
                response_data = await response.json()
                if response.status == 200:
                    embedding_data = response_data.get('data', [{}])[0]
                    embedding = embedding_data.get('embedding')
                    usage = response_data.get('usage', {})
                    cost = 0.0001 * usage.get('total_tokens', 0) # Example cost calculation

                    if embedding is None:
                         return AIResult(
                            status="FAILED",
                            error="Embedding data missing in successful API response",
                            error_code=AI_ERRORS.AI_RESPONSE_PARSE_ERROR.value,
                            provider_response=response_data,
                            trace_id=task.trace_id
                        )

                    return AIResult(
                        status="SUCCESS",
                        data={"vector": embedding},
                        cost=cost,
                        provider_response=response_data,
                        metadata={"model_used": task.model, "dimensions_returned": len(embedding)},
                        trace_id=task.trace_id
                    )
                else:
                    error_info = response_data.get('error', {})
                    error_message = error_info.get('message', 'Unknown OpenAI API error')
                    error_code = error_info.get('code', AI_ERRORS.AI_PROVIDER_ERROR.value)
                    status_map = { 401: AI_ERRORS.AI_UNAUTHORIZED, 429: AI_ERRORS.AI_RATE_LIMIT, 500: AI_ERRORS.AI_INTERNAL_ERROR }
                    api_error_code = status_map.get(response.status, AI_ERRORS.AI_PROVIDER_ERROR).value
                    logger.error(f"OpenAI API error ({response.status}): {error_message}")
                    return AIResult(
                        status="FAILED",
                        error=error_message,
                        error_code=api_error_code, # Use mapped code
                        provider_response=response_data,
                        trace_id=task.trace_id
                    )

        except aiohttp.ClientConnectorError as e:
            logger.error(f"OpenAI network error: {e}")
            return AIResult(status="FAILED", error=f"Network error: {e}", error_code=AI_ERRORS.AI_NETWORK_ERROR.value, trace_id=task.trace_id)
        except asyncio.TimeoutError:
            logger.error("OpenAI request timed out.")
            return AIResult(status="FAILED", error="Request timed out", error_code=AI_ERRORS.AI_TIMEOUT.value, trace_id=task.trace_id)
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI embedding: {e}", exc_info=True)
            return AIResult(status="FAILED", error=str(e), error_code=AI_ERRORS.AI_INTERNAL_ERROR.value, trace_id=task.trace_id)


    async def _generate_or_complete(self, task: AITask, contract: AIContract) -> AIResult:
        """Generate text completion (Mock implementation - needs actual API call)"""
        prompt = task.payload.get('prompt') or task.payload.get('messages')
        if not prompt:
            return AIResult(status="FAILED", error="No prompt/messages provided", error_code=AI_ERRORS.AI_INVALID_REQUEST.value, trace_id=task.trace_id)

        # --- THIS NEEDS TO BE REPLACED WITH ACTUAL API CALL LOGIC ---
        await asyncio.sleep(random.uniform(0.1, 0.5)) # Simulate network latency

        response_text = f"Mock completion for '{str(prompt)[:30]}...' using {task.model}"
        prompt_tokens = len(str(prompt).split())
        completion_tokens = len(response_text.split())
        total_tokens = prompt_tokens + completion_tokens

        mock_provider_response = {
            "id": f"chatcmpl-{random.randint(1000,9999)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": task.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}
        }

        return AIResult(
            status="SUCCESS",
            data={"text": response_text, "tokens_used": total_tokens},
            cost=0.002 * (total_tokens / 1000), # Mock cost
            provider_response=mock_provider_response,
            metadata={"model_used": task.model, "finish_reason": "stop"},
            trace_id=task.trace_id
        )
        # --- END MOCK ---

    async def _classify(self, task: AITask, contract: AIContract) -> AIResult:
        """Classify text using generation/completion (Mock)"""
        text_to_classify = task.payload.get('text', '')
        classes = task.payload.get('classes', ['positive', 'negative', 'neutral'])
        if not text_to_classify:
             return AIResult(status="FAILED", error="No text provided for classification", error_code=AI_ERRORS.AI_INVALID_REQUEST.value, trace_id=task.trace_id)

        prompt = f"Classify the following text into one of these categories: {', '.join(classes)}.\n\nText: \"{text_to_classify}\"\n\nClass:"

        generation_task = AITask(
            operation="GENERATE", provider=task.provider, model=task.model,
            payload={"prompt": prompt}, trace_id=task.trace_id
        )
        classification_contract = AIContract(**contract.to_dict())
        classification_contract.temperature = 0.1
        classification_contract.max_tokens = 5

        gen_result = await self._generate_or_complete(generation_task, classification_contract)

        if gen_result.is_success():
            predicted_class = gen_result.data.get("text", "").strip().lower()
            final_class = "unknown"
            for cls in classes:
                 if cls.lower() in predicted_class:
                      final_class = cls
                      break

            return AIResult(
                status="SUCCESS",
                data={"class": final_class, "confidence": random.uniform(0.6, 0.95)}, # Mock confidence
                cost=gen_result.cost,
                provider_response=gen_result.provider_response,
                metadata={"model_used": task.model, "classification_prompt": prompt},
                trace_id=task.trace_id,
                latency_ms=gen_result.latency_ms
            )
        else:
            gen_result.error = f"Classification failed (underlying generation error: {gen_result.error})"
            return gen_result


    def supports_operation(self, operation: str) -> bool:
        return operation in self.supported_operations

    def get_models(self) -> List[str]:
        all_models = set()
        for models in self.models.values():
            all_models.update(models)
        return sorted(list(all_models))

    async def close_session(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


class AnthropicProvider(AIProvider):
    """Anthropic Claude API provider (Placeholder/Mock)"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key, "https://api.anthropic.com/v1")
        self.supported_operations = ["GENERATE", "COMPLETE", "ANALYZE"]
        self.models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1"]

    async def execute(self, task: AITask, contract: AIContract) -> AIResult:
        """Execute Anthropic API call (Mock implementation)"""
        start_time_ns = time.time_ns()

        if task.operation not in self.supported_operations:
            return AIResult(status="FAILED", error=f"Operation {task.operation} not supported", error_code=AI_ERRORS.AI_UNSUPPORTED.value, trace_id=task.trace_id)

        # Check rate limits using the provider's specific name
        if not self.rate_limiter.check_limit("Anthropic"):
            return AIResult(
                status="FAILED",
                error="Rate limit exceeded",
                error_code=AI_ERRORS.AI_RATE_LIMIT.value,
                trace_id=task.trace_id
            )

        await asyncio.sleep(random.uniform(0.1, 0.6)) # Simulate network latency
        response = f"Mock Claude response for {task.operation}: {task.payload.get('prompt', '')[:30]}..."
        tokens = len(response.split())
        mock_provider_response = { "id": f"msg_{random.randint(1000,9999)}", "type": "message", "role": "assistant", "content": [{"type": "text", "text": response}], "model": task.model, "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": tokens} }

        latency_ms = (time.time_ns() - start_time_ns) / 1_000_000.0
        return AIResult(
            status="SUCCESS",
            data={"text": response, "model": task.model, "tokens_used": tokens},
            latency_ms=latency_ms,
            cost=0.003 * (tokens / 1000), # Mock cost
            provider_response=mock_provider_response,
            trace_id=task.trace_id
        )

    def supports_operation(self, operation: str) -> bool:
        return operation in self.supported_operations

    def get_models(self) -> List[str]:
        return self.models


class GrokProvider(AIProvider):
    """xAI Grok API provider (Placeholder/Mock)"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key, "https://api.x.ai/v1") # Note: Actual URL might differ
        self.supported_operations = ["GENERATE", "EMBED", "REASON", "ANALYZE"]
        # Updated models based on test failure
        self.models = ["grok-1", "grok-1.5", "grok-3", "grok-4"]

    async def execute(self, task: AITask, contract: AIContract) -> AIResult:
        """Execute Grok API call (Mock implementation)"""
        start_time_ns = time.time_ns()

        # Check rate limits (assuming a 'Grok' entry in RateLimiter.limits)
        if not self.rate_limiter.check_limit("Grok"): # Use consistent name
            return AIResult(
                status="FAILED",
                error="Rate limit exceeded",
                error_code=AI_ERRORS.AI_RATE_LIMIT.value,
                trace_id=task.trace_id
            )

        data_payload = {}
        response_content: Union[str, Dict, List] = "" # Store the core response content

        if task.operation == "REASON":
            steps = self._grok_reasoning_steps(task.payload) # Get steps as list
            # Updated data payload based on test failure
            data_payload = {"reasoning_steps": steps}
            response_content = steps # Store for mock provider response
        elif task.operation == "EMBED":
             dim = 768
             if NUMPY_AVAILABLE: embedding = np.random.randn(dim).tolist()
             else: embedding = [random.gauss(0,1) for _ in range(dim)]
             data_payload = {"vector": embedding}
             response_content = embedding
        elif task.operation in ["GENERATE", "ANALYZE"]: # Combine these for mock
            text_response = f"Mock Grok analysis/generation: {task.payload.get('prompt', 'No input')[:50]}..."
            data_payload = {"text": text_response}
            response_content = text_response
        else: # Should not happen if supports_operation is checked, but handle defensively
            return AIResult(status="FAILED", error=f"Unsupported Grok operation: {task.operation}", error_code=AI_ERRORS.AI_UNSUPPORTED.value, trace_id=task.trace_id)

        await asyncio.sleep(random.uniform(0.08, 0.4)) # Simulate network latency

        # Estimate tokens (simplistic)
        tokens = 10
        if isinstance(response_content, str): tokens = len(response_content.split())
        elif isinstance(response_content, list) and len(response_content) > 0 and isinstance(response_content[0], str): tokens = sum(len(s.split()) for s in response_content)
        elif isinstance(response_content, list): tokens = len(response_content) # e.g. embedding dimension

        mock_provider_response = { "id": f"grok_{random.randint(1000,9999)}", "response": response_content, "model": task.model, "usage": {"tokens": tokens} }

        latency_ms = (time.time_ns() - start_time_ns) / 1_000_000.0

        return AIResult(
            status="SUCCESS",
            data=data_payload, # Use the correctly structured payload
            latency_ms=latency_ms,
            cost=0.004 * (tokens / 1000), # Mock cost
            metadata={"provider": "xAI", "version": "1.0"},
            provider_response=mock_provider_response,
            trace_id=task.trace_id
        )

    def _grok_reasoning_steps(self, payload: Dict[str, Any]) -> List[str]:
        """Simulate Grok's reasoning capability, returning steps"""
        prompt = payload.get('prompt', '')
        steps = [ f"Analyzing: {prompt[:30]}", "Applying principles", "Synthesizing response" ]
        return steps

    def supports_operation(self, operation: str) -> bool:
        return operation in self.supported_operations

    def get_models(self) -> List[str]:
        return self.models


class MockProvider(AIProvider):
    """Mock provider for testing and development"""

    def __init__(self):
        super().__init__(api_key="mock-key", base_url="mock://localhost")
        self.call_count = 0
        # Renamed based on test failure
        self.latency_range_ms = (10.0, 100.0) # Ensure float

    async def execute(self, task: AITask, contract: AIContract) -> AIResult:
        """Execute mock operation"""
        start_time_ns = time.time_ns()
        self.call_count += 1

        latency_ms = random.uniform(*self.latency_range_ms)
        await asyncio.sleep(latency_ms / 1000)

        data = {}
        mock_provider_response = {"mock": True, "call_count": self.call_count, "operation": task.operation}

        if task.operation == "EMBED":
            # <<< --- START Mock Dimension Fix --- >>>
            # Respect 'dim' from payload, default to 768 if missing/invalid
            dim = task.payload.get('dim', 768)
            try:
                dim = int(dim)
                if not (1 <= dim <= 8192): # Clamp to a reasonable range
                    logger.warning(f"MockProvider received unreasonable dim {dim}, clamping to 768.")
                    dim = 768
            except (ValueError, TypeError):
                 logger.warning(f"MockProvider received invalid dim '{dim}', using 768.")
                 dim = 768
            # <<< --- END Mock Dimension Fix --- >>>

            if NUMPY_AVAILABLE: embedding = np.random.randn(dim).tolist()
            else: embedding = [random.gauss(0,1) for _ in range(dim)]
            data = {"vector": embedding}
            mock_provider_response["embedding_dim"] = dim
        elif task.operation == "GENERATE":
            text = f"Mock response #{self.call_count} to: {task.payload.get('prompt', '')[:20]}"
            data = {"text": text, "tokens": len(text.split())}
            mock_provider_response["generated_text"] = text
        elif task.operation == "CLASSIFY":
            cls = random.choice(task.payload.get('classes', ["A", "B"]))
            conf = random.uniform(0.5, 0.99)
            data = {"class": cls, "confidence": conf}
            mock_provider_response["predicted_class"] = cls
        else:
            data = {"result": f"mock_result_for_{task.operation}"}

        latency_ms_actual = (time.time_ns() - start_time_ns) / 1_000_000.0
        return AIResult(
            status="SUCCESS",
            data=data,
            latency_ms=latency_ms_actual,
            cost=0.0,
            cached=False,
            # Changed metadata key based on test failure
            metadata={"mock_provider": True, "simulated_latency_ms": latency_ms},
            provider_response=mock_provider_response,
            trace_id=task.trace_id
        )

    def supports_operation(self, operation: str) -> bool:
        return True # Mock supports everything

    def get_models(self) -> List[str]:
        return ["mock-model-v1", "mock-model-v2"]


# ============================================================================
# LOCAL GPT PROVIDER SHIM (wraps your fine-tuned LocalGPTProvider)
# ============================================================================

class LocalGPTAIProvider(AIProvider):
    """
    Shim provider that wraps the LocalGPTProvider adapter so it can be registered
    inside AIRuntime like other providers. This class adapts LocalGPTProvider's
    synchronous generate() into the async execute() interface and maps runtime
    operations to local generation.

    Supported operations:
      - "GENERATE", "COMPLETE": returns {"text": "...", "tokens_used": int}
    Unsupported operations:
      - "EMBED", "CLASSIFY", etc. (returns AI_UNSUPPORTED)
    """

    def __init__(self, artifacts_dir: str):
        super().__init__(api_key=None, base_url="local://gpt")
        self.artifacts_dir = artifacts_dir
        # Defer import to avoid hard dependency if artifacts are not used
        try:
            from src.local_llm.provider.local_gpt_provider import build_provider_from_artifacts
        except Exception as e:
            raise ImportError(f"LocalGPTProvider import failed. Ensure files are added to src/local_llm. Error: {e}")
        # Build underlying local provider (loads model + vocab + config)
        self._builder = build_provider_from_artifacts
        self._provider = self._builder(artifacts_dir)
        # Supported operations for routing
        self._supported = {"GENERATE", "COMPLETE"}

    def supports_operation(self, operation: str) -> bool:
        return operation in self._supported

    def get_models(self) -> List[str]:
        # Single logical model label
        return ["local-gpt"]

    async def execute(self, task: AITask, contract: AIContract) -> AIResult:
        if task.operation not in self._supported:
            return AIResult(
                status="FAILED",
                error=f"Operation {task.operation} not supported by LocalGPT",
                error_code=AI_ERRORS.AI_UNSUPPORTED.value,
                trace_id=task.trace_id
            )

        # Extract prompt from payload; support simple message lists by concatenation
        prompt: Optional[str] = None
        if "prompt" in task.payload and isinstance(task.payload["prompt"], str):
            prompt = task.payload["prompt"]
        elif "messages" in task.payload and isinstance(task.payload["messages"], list):
            try:
                # Join message contents with roles (simple baseline)
                parts = []
                for m in task.payload["messages"]:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    parts.append(f"{role}: {content}")
                prompt = "\n".join(parts)
            except Exception:
                prompt = str(task.payload.get("messages"))

        if not prompt:
            return AIResult(
                status="FAILED",
                error="No prompt/messages provided",
                error_code=AI_ERRORS.AI_INVALID_REQUEST.value,
                trace_id=task.trace_id
            )

        max_new_tokens = int(task.payload.get("max_new_tokens") or task.payload.get("max_tokens") or (contract.max_tokens or 128))
        temperature = float(task.payload.get("temperature") or contract.temperature or 0.7)
        top_p = float(task.payload.get("top_p") or (contract.top_p if contract.top_p is not None else 0.95))
        top_k = int(task.payload.get("top_k") or 64)
        repetition_penalty = float(task.payload.get("repetition_penalty") or 1.05)

        # Run synchronous generation in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        def _do_generate() -> Tuple[str, Dict[str, Any]]:
            return self._provider.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        try:
            text, meta = await loop.run_in_executor(None, _do_generate)
            # Rough token estimate (space-delimited)
            tokens_used = len((text or "").split())
            return AIResult(
                status="SUCCESS",
                data={"text": text, "tokens_used": tokens_used},
                cost=0.0,
                metadata={
                    "model_used": "local-gpt",
                    "params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty
                    }
                },
                provider_response={"meta": meta},
                trace_id=task.trace_id
            )
        except Exception as e:
            logger.error(f"LocalGPT generation error: {e}", exc_info=True)
            return AIResult(
                status="FAILED",
                error=f"LocalGPT error: {str(e)}",
                error_code=AI_ERRORS.AI_INTERNAL_ERROR.value,
                trace_id=task.trace_id
            )


# ============================================================================
# CACHING
# ============================================================================

class ResultCache:
    """Cache for AI operation results"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _compute_key(self, task: AITask, contract: AIContract) -> str:
        """Compute cache key for task"""
        key_data = {
            "op": task.operation,
            "provider": task.provider,
            "model": task.model,
            "payload": task.payload,
            "temp": contract.temperature,
        }
        try:
            key_str = json.dumps(key_data, sort_keys=True, default=str)
        except (TypeError, ValueError) as e:
            logger.warning(f"Non-serializable data in cache key computation: {e}")
            key_str = json.dumps({k: v for k, v in key_data.items() if k != 'payload'}, sort_keys=True)

        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, task: AITask, contract: AIContract) -> Optional[AIResult]:
        """Get cached result if available and valid"""
        if not contract.allow_cached:
            return None

        key = self._compute_key(task, contract)

        with self._lock:
            cached_data = self.cache.get(key)
            if cached_data:
                timestamp = cached_data["ts"]
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    return None

                try:
                    # Reconstruct AIResult, making sure status is present
                    result_dict = cached_data["result_dict"]
                    if "status" not in result_dict:
                        raise ValueError("Cached result dict missing 'status'")
                    result = AIResult(**result_dict)
                    result.cached = True
                    return result
                except Exception as e:
                     logger.error(f"Failed to reconstruct cached AIResult for key {key}: {e}")
                     del self.cache[key]
                     return None
        return None

    def put(self, task: AITask, contract: AIContract, result: AIResult):
        """Store result in cache"""
        if not contract.allow_cached or not result.is_success(): # Corrected check
            return

        key = self._compute_key(task, contract)

        with self._lock:
            if len(self.cache) >= self.max_size:
                try:
                    oldest_key = min(self.cache, key=lambda k: self.cache[k]['ts'])
                    del self.cache[oldest_key]
                except ValueError:
                    pass

            # Ensure result has status before caching its dict representation
            result_dict = result.to_dict()
            if "status" not in result_dict:
                 logger.error(f"Attempting to cache result without status for key {key}")
                 return # Don't cache invalid results
            self.cache[key] = {"ts": time.time(), "result_dict": result_dict}

    def clear(self):
        """Clear all cached results"""
        with self._lock:
            self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = len(self.cache)
            # Add max_size and ttl_seconds based on test expectation
            stats_dict: Dict[str, Any] = {
                "size": total,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds
            }
            if total > 0:
                now = time.time()
                ages = [now - data['ts'] for data in self.cache.values()]
                stats_dict["avg_age_seconds"] = sum(ages) / total if ages else 0
                stats_dict["oldest_seconds"] = max(ages) if ages else 0
            else:
                 stats_dict["avg_age_seconds"] = 0.0
                 stats_dict["oldest_seconds"] = 0.0
            return stats_dict


# ============================================================================
# MAIN RUNTIME (Refactored per request)
# ============================================================================

class AIRuntime:
    """
    Unified AI runtime for managing providers, routing, and execution
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.providers: Dict[str, AIProvider] = {}
        self._lock = threading.Lock()
        self.cache = ResultCache(
            max_size=self.config.get("cache_size", 1000),
            # Corrected TTL default based on test file
            ttl_seconds=self.config.get("cache_ttl_seconds", 60)
        )
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))
        self.metrics = AIMetrics()

        self._initialize_providers()

        self.routing_strategy = self.config.get("routing_strategy", "round_robin")
        self.provider_weights = self.config.get("provider_weights", {})
        self._last_provider_idx = 0

        logger.info(f"AIRuntime initialized with providers: {list(self.providers.keys())}")

    def _initialize_providers(self):
        """Initialize configured providers"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                 provider = OpenAIProvider(api_key=api_key)
                 # Register under lowercase name for consistency
                 self.providers["openai"] = provider
                 logger.info("Registered OpenAI provider from OPENAI_API_KEY")
            else:
                 logger.warning("OPENAI_API_KEY not found, OpenAI provider not registered.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")

        # Always register Mock provider under lowercase 'mock'
        self.providers["mock"] = MockProvider()

        # Optionally register LocalGPT provider if artifacts are configured
        try:
            artifacts_dir = self.config.get("local_gpt_artifacts_dir") or os.getenv("LOCAL_GPT_ARTIFACTS_DIR") or ""
            if artifacts_dir and os.path.isdir(artifacts_dir):
                try:
                    local_provider = LocalGPTAIProvider(artifacts_dir=artifacts_dir)
                    self.providers["local-gpt"] = local_provider
                    logger.info(f"Registered LocalGPT provider from artifacts: {artifacts_dir}")
                except Exception as e:
                    logger.error(f"Failed to initialize LocalGPT provider: {e}")
            else:
                if artifacts_dir:
                    logger.error(f"Configured local_gpt_artifacts_dir does not exist: {artifacts_dir}")
                else:
                    logger.info("No LocalGPT artifacts configured; skipping local-gpt registration.")
        except Exception as e:
            logger.error(f"LocalGPT provider registration error: {e}")

        # Set default provider only if 'default' isn't already set (e.g. by OpenAI)
        if "default" not in self.providers:
             # Priority: explicit default in config, else OpenAI, else local-gpt, else mock
             desired_default = (self.config.get("default_provider") or "").lower()
             if desired_default and desired_default in self.providers:
                 self.providers["default"] = self.providers[desired_default]
                 logger.info(f"Registered {desired_default} provider as default (from config).")
             elif "openai" in self.providers:
                 self.providers["default"] = self.providers["openai"]
                 logger.info("Registered OpenAI provider as default.")
             elif "local-gpt" in self.providers:
                 self.providers["default"] = self.providers["local-gpt"]
                 logger.info("Registered LocalGPT provider as default.")
             else:
                 # Fallback to mock if no other provider is default yet
                 self.providers["default"] = self.providers["mock"]
                 logger.info("Registered Mock provider as default.")


    def _get_provider(self, name: str) -> Optional[AIProvider]:
        """Gets a provider by name (case-insensitive), falling back to default."""
        # Use lowercase for lookup consistency
        name_lower = name.lower() if name else "default"

        provider = self.providers.get(name_lower) or self.providers.get("default")

        if not provider:
             logger.error(f"Could not find provider '{name}' or default provider.")

        return provider

    def execute_task(self, task: AITask, contract: AIContract) -> AIResult:
        """
        Sync API so embed_node can call it in a thread via asyncio.to_thread.
        Internally awaits provider.execute() if needed using a managed event loop.
        """
        start_t_ns = time.time_ns()

        # Check for expiration FIRST
        if task.is_expired():
             logger.warning(f"Task {task.trace_id} is expired (deadline: {task.deadline}).")
             latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
             expired_result = AIResult(
                 status="FAILED",
                 error="Task deadline expired before execution",
                 error_code=AI_ERRORS.AI_TIMEOUT.value, # Use timeout for expired
                 trace_id=task.trace_id,
                 latency_ms=latency_ms
             )
             self.metrics.record_execution(task.provider, task.operation, expired_result)
             return expired_result


        ok, err = contract.validate()
        if not ok:
            logger.error(f"Invalid AIContract: {err}. Task: {task.to_dict()}")
            latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0 # Calculate latency even for validation failure
            validation_result = AIResult(
                status="FAILED",
                error=f"Invalid contract: {err}",
                error_code=AI_ERRORS.AI_VALIDATION_ERROR.value,
                trace_id=task.trace_id,
                latency_ms=latency_ms
            )
            # Record metrics for validation failures too
            self.metrics.record_execution(task.provider, task.operation, validation_result)
            return validation_result

        provider = self._get_provider(task.provider)
        if not provider:
            logger.error(f"Provider '{task.provider}' not found or configured.")
            latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
            no_provider_result = AIResult(
                status="FAILED",
                error=f"No provider '{task.provider}' available",
                error_code=AI_ERRORS.AI_PROVIDER_ERROR.value,
                trace_id=task.trace_id,
                latency_ms=latency_ms
            )
            self.metrics.record_execution(task.provider or "unknown", task.operation, no_provider_result)
            return no_provider_result

        # Check cache AFTER validation and provider check
        if contract.allow_cached:
            cached_result = self.cache.get(task, contract)
            if cached_result:
                # No need to set cached=True again, cache.get does it
                self.metrics.record_cache_hit()
                logger.debug(f"Cache hit for task {task.trace_id}")
                # Recalculate latency for cache hit (should be very low)
                cached_result.latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
                # Record metrics for cache hits (important for cost/latency analysis)
                self.metrics.record_execution(task.provider, task.operation, cached_result)
                return cached_result
        self.metrics.record_cache_miss()

        result: Optional[AIResult] = None
        try:
            if not asyncio.iscoroutinefunction(provider.execute):
                logger.warning(f"Provider {task.provider} execute method is synchronous.")
                # Directly call the sync method
                result = provider.execute(task, contract) # Type checker might complain here
            else:
                 try:
                     # Check if an event loop is already running in this thread
                     _ = asyncio.get_running_loop()
                     # If yes, raise an error because asyncio.run() cannot be used.
                     # The caller should use execute_task_async instead.
                     logger.error(f"execute_task (sync) called from within an async context for task {task.trace_id}. Use execute_task_async.", exc_info=True)
                     raise RuntimeError("Synchronous execute_task cannot be called directly from an async function. Use await runtime.execute_task_async() instead or call from sync context.")

                 except RuntimeError: # No running event loop, so asyncio.run is appropriate
                     logger.debug(f"Creating new event loop for provider.execute task {task.trace_id}")
                     result = asyncio.run(provider.execute(task, contract))


            if not isinstance(result, AIResult):
                 raise TypeError(f"Provider {task.provider} returned unexpected type {type(result)}")

            # Latency is calculated from the start of *this sync function*
            result.latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
            result.trace_id = task.trace_id

            if result.status == "SUCCESS" and contract.allow_cached:
                self.cache.put(task, contract, result)

            self.metrics.record_execution(task.provider, task.operation, result)
            return result

        except Exception as e:
            logger.error(f"Error executing task {task.trace_id} with provider {task.provider}: {e}", exc_info=True)
            latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
            error_result = AIResult(
                status="FAILED",
                error=f"Internal error during task execution: {str(e)}",
                error_code=AI_ERRORS.AI_INTERNAL_ERROR.value,
                trace_id=task.trace_id,
                latency_ms=latency_ms
            )
            self.metrics.record_execution(task.provider or "unknown", task.operation, error_result)
            return error_result

    # Add an async version for callers already in an event loop
    async def execute_task_async(self, task: AITask, contract: AIContract) -> AIResult:
         """Asynchronous version of execute_task."""
         start_t_ns = time.time_ns()

         if task.is_expired():
             logger.warning(f"Task {task.trace_id} is expired (deadline: {task.deadline}).")
             latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
             expired_result = AIResult(status="FAILED", error="Task deadline expired", error_code=AI_ERRORS.AI_TIMEOUT.value, trace_id=task.trace_id, latency_ms=latency_ms)
             self.metrics.record_execution(task.provider, task.operation, expired_result)
             return expired_result

         ok, err = contract.validate()
         if not ok:
             logger.error(f"Invalid AIContract: {err}. Task: {task.to_dict()}")
             latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
             validation_result = AIResult(status="FAILED", error=f"Invalid contract: {err}", error_code=AI_ERRORS.AI_VALIDATION_ERROR.value, trace_id=task.trace_id, latency_ms=latency_ms)
             self.metrics.record_execution(task.provider, task.operation, validation_result)
             return validation_result

         provider = self._get_provider(task.provider)
         if not provider:
             logger.error(f"Provider '{task.provider}' not found.")
             latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
             no_provider_result = AIResult(status="FAILED", error=f"No provider '{task.provider}' available", error_code=AI_ERRORS.AI_PROVIDER_ERROR.value, trace_id=task.trace_id, latency_ms=latency_ms)
             self.metrics.record_execution(task.provider or "unknown", task.operation, no_provider_result)
             return no_provider_result

         if contract.allow_cached:
             cached_result = self.cache.get(task, contract)
             if cached_result:
                 self.metrics.record_cache_hit()
                 logger.debug(f"Cache hit for task {task.trace_id}")
                 cached_result.latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
                 self.metrics.record_execution(task.provider, task.operation, cached_result)
                 return cached_result
         self.metrics.record_cache_miss()

         result: Optional[AIResult] = None
         try:
             # Directly await the provider's execute method
             result = await provider.execute(task, contract)

             if not isinstance(result, AIResult):
                  raise TypeError(f"Provider {task.provider} returned invalid type {type(result)}")

             result.latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
             result.trace_id = task.trace_id

             if result.is_success() and contract.allow_cached:
                  self.cache.put(task, contract, result)

             self.metrics.record_execution(task.provider, task.operation, result)
             return result

         except Exception as e:
            logger.error(f"Async error executing task {task.trace_id} with provider {task.provider}: {e}", exc_info=True)
            latency_ms = (time.time_ns() - start_t_ns) / 1_000_000.0
            error_result = AIResult(status="FAILED", error=f"Async execution error: {str(e)}", error_code=AI_ERRORS.AI_INTERNAL_ERROR.value, trace_id=task.trace_id, latency_ms=latency_ms)
            self.metrics.record_execution(task.provider or "unknown", task.operation, error_result)
            return error_result

    async def shutdown(self):
        """Gracefully shuts down the AI Runtime and its providers."""
        logger.info("Shutting down AIRuntime...")
        self.executor.shutdown(wait=True)
        for provider_name, provider in self.providers.items():
             try:
                 if hasattr(provider, 'close_session') and asyncio.iscoroutinefunction(provider.close_session):
                     logger.debug(f"Closing session for provider {provider_name}...")
                     await provider.close_session()
             except Exception as e:
                 logger.error(f"Error closing session for provider {provider_name}: {e}")
        logger.info("AIRuntime shutdown complete.")

    def cleanup(self):
         """Synchronous cleanup wrapper."""
         logger.info("Cleaning up AIRuntime (sync)...")
         try:
             # Try running async shutdown in a loop
             asyncio.run(self.shutdown())
         except RuntimeError as e:
              logger.warning(f"Could not run async shutdown during sync cleanup: {e}. Proceeding with sync cleanup.")
              # Fallback sync cleanup if loop is closed or error occurs
              self.executor.shutdown(wait=False, cancel_futures=True)
              # Sync provider cleanup (if any providers implement sync cleanup)
              for provider in self.providers.values():
                   if hasattr(provider, 'sync_cleanup'): provider.sync_cleanup()
         except Exception as e:
              logger.error(f"Error during AIRuntime sync cleanup: {e}")
              # Still try to shut down executor even if async shutdown had other errors
              if hasattr(self, 'executor'):
                   self.executor.shutdown(wait=False, cancel_futures=True)

         # Explicitly clear the cache during cleanup
         self.cache.clear()

         logger.info("AIRuntime sync cleanup finished.")

    def register_provider(self, name: str, provider: AIProvider):
        """Register a custom provider (use lowercase name)."""
        # Store provider under lowercase name for consistent lookup
        self.providers[name.lower()] = provider
        logger.info(f"Registered provider: {name.lower()}")

    def _route_to_provider(self, task: AITask) -> Optional[AIProvider]:
        """Route task to appropriate provider based on strategy."""
        # Provider selection logic - simplified for now, use _get_provider logic
        # Could add round-robin or weighted logic here if task.provider is empty
        # Example round-robin (if task.provider is empty/default and strategy is round_robin)
        if (not task.provider or task.provider.lower() == "default") and self.routing_strategy == "round_robin":
             available_providers = [p for name, p in self.providers.items() if name != "default" and p.supports_operation(task.operation)]
             if available_providers:
                 with self._lock: # Protect index access
                     provider_idx = self._last_provider_idx % len(available_providers)
                     selected_provider = available_providers[provider_idx]
                     self._last_provider_idx = (self._last_provider_idx + 1)
                 logger.debug(f"Round-robin routing task {task.trace_id} to provider index {provider_idx}")
                 return selected_provider
             else:
                 logger.warning(f"Round-robin failed: No non-default providers support operation '{task.operation}'. Falling back.")

        # Default: Use specified provider or the registered default
        return self._get_provider(task.provider)

    # execute_task_sync IS the primary sync method now
    def execute_task_sync(self, task: AITask, contract: Optional[AIContract] = None) -> AIResult:
        """Synchronous wrapper for execute_task (this IS the main method now)"""
        if contract is None: contract = AIContract()
        # Simply call the main execute_task method
        return self.execute_task(task, contract)

    async def batch_execute(self, tasks: List[AITask], contract: Optional[AIContract] = None) -> List[AIResult]:
        """Execute multiple tasks in parallel (using the async execute_task_async)"""
        if contract is None: contract = AIContract()

        # Create coroutines for each task using the async version
        coroutines = [self.execute_task_async(task, contract) for task in tasks]

        # Run them concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results, handling potential exceptions caught by gather
        final_results: List[AIResult] = []
        for i, res in enumerate(results):
             if isinstance(res, Exception):
                 logger.error(f"Exception during batch execution for task {tasks[i].trace_id}: {res}", exc_info=res)
                 final_results.append(AIResult(
                     status="FAILED",
                     error=f"Batch execution exception: {str(res)}",
                     error_code=AI_ERRORS.AI_INTERNAL_ERROR.value,
                     trace_id=tasks[i].trace_id
                 ))
             elif isinstance(res, AIResult):
                  final_results.append(res)
             else: # Should not happen
                  logger.error(f"Unexpected result type in batch execution for task {tasks[i].trace_id}: {type(res)}")
                  final_results.append(AIResult(
                     status="FAILED",
                     error=f"Unexpected result type: {type(res)}",
                     error_code=AI_ERRORS.AI_INTERNAL_ERROR.value,
                     trace_id=tasks[i].trace_id
                 ))

        return final_results

    def get_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics"""
        return {
            "providers": list(self.providers.keys()),
            "cache_stats": self.cache.stats(),
            "execution_metrics": self.metrics.get_summary()
        }


class AIMetrics:
    """Metrics tracking for AI operations"""

    def __init__(self):
        self.executions = defaultdict(int)
        self.successes = defaultdict(int)
        self.failures = defaultdict(int)
        self.total_latency = defaultdict(float)
        self.total_cost = defaultdict(float)
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.Lock()

    def record_execution(self, provider: str, operation: str, result: AIResult):
        """Record execution metrics"""
        provider_name = provider.lower() if provider else 'unknown' # Use lowercase
        key = f"{provider_name}:{operation}"

        with self._lock:
            self.executions[key] += 1

            if result.is_success():
                self.successes[key] += 1
            else:
                self.failures[key] += 1

            latency = result.latency_ms if isinstance(result.latency_ms, (int, float)) else 0.0
            cost = result.cost if isinstance(result.cost, (int, float)) else 0.0

            self.total_latency[key] += latency
            self.total_cost[key] += cost

    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self):
        with self._lock:
            self.cache_misses += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        with self._lock:
            total_lookups = self.cache_hits + self.cache_misses
            summary = {
                "total_executions": sum(self.executions.values()),
                "total_successes": sum(self.successes.values()),
                "total_failures": sum(self.failures.values()),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": (self.cache_hits / total_lookups) if total_lookups > 0 else 0.0, # Avoid division by zero
                "by_provider_operation": {} # Corrected key name
            }

            for key in self.executions:
                try:
                     provider, operation = key.split(":", 1)
                except ValueError:
                     provider, operation = key, "unknown"

                if provider not in summary["by_provider_operation"]:
                    summary["by_provider_operation"][provider] = {}

                count = self.executions[key]
                success_count = self.successes[key]
                latency_sum = self.total_latency[key]
                cost_sum = self.total_cost[key]

                summary["by_provider_operation"][provider][operation] = {
                    "count": count,
                    "success_rate": (success_count / count) if count > 0 else 0.0, # Avoid division by zero
                    "avg_latency_ms": (latency_sum / count) if count > 0 else 0.0, # Avoid division by zero
                    "total_cost": cost_sum
                }

            # Add top-level avg latency and total cost (optional)
            total_latency_all = sum(self.total_latency.values())
            total_cost_all = sum(self.total_cost.values())
            total_execs = summary["total_executions"]
            summary["avg_latency_ms_overall"] = (total_latency_all / total_execs) if total_execs > 0 else 0.0
            summary["total_cost_overall"] = total_cost_all

            return summary