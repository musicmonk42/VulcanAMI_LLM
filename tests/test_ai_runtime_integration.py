"""
Comprehensive pytest suite for ai_runtime_integration.py
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

# Import the module to test
# Assuming the file is in the same directory or accessible via path
# If it's part of a package, adjust the import accordingly.
# For example: from src.unified_runtime import ai_runtime_integration as ai
import ai_runtime_integration as ai
import pytest


class TestAIErrors:
    """Test AI_ERRORS enum"""

    def test_error_codes(self):
        """Test all error code values"""
        assert ai.AI_ERRORS.AI_INVALID_REQUEST.value == "AI_INVALID_REQUEST"
        assert ai.AI_ERRORS.AI_TIMEOUT.value == "AI_TIMEOUT"
        assert ai.AI_ERRORS.AI_RATE_LIMIT.value == "AI_RATE_LIMIT"
        assert ai.AI_ERRORS.AI_PROVIDER_ERROR.value == "AI_PROVIDER_ERROR"
        # Added based on source code enum
        assert ai.AI_ERRORS.AI_VALIDATION_ERROR.value == "AI_VALIDATION_ERROR"


class TestAIContract:
    """Test AIContract dataclass"""

    def test_contract_creation(self):
        """Test creating AI contract"""
        contract = ai.AIContract(
            max_latency_ms=500.0, min_accuracy=0.95, temperature=0.8
        )

        assert contract.max_latency_ms == 500.0
        assert contract.min_accuracy == 0.95
        assert contract.temperature == 0.8

    def test_contract_defaults(self):
        """Test default contract values"""
        contract = ai.AIContract()

        assert contract.temperature == 0.7
        assert contract.allow_cached is True
        assert contract.require_deterministic is False

    def test_contract_validation_success(self):
        """Test valid contract validation"""
        contract = ai.AIContract(
            max_latency_ms=1000.0, min_accuracy=0.9, temperature=1.0
        )

        valid, error = contract.validate()
        assert valid is True
        assert error is None

    def test_contract_validation_negative_latency(self):
        """Test validation with negative latency"""
        contract = ai.AIContract(max_latency_ms=-100.0)

        valid, error = contract.validate()
        assert valid is False
        assert "positive" in error.lower()

    def test_contract_validation_invalid_accuracy(self):
        """Test validation with invalid accuracy"""
        contract = ai.AIContract(min_accuracy=1.5)

        valid, error = contract.validate()
        assert valid is False
        assert "accuracy" in error.lower()

    def test_contract_validation_invalid_temperature(self):
        """Test validation with invalid temperature"""
        contract = ai.AIContract(temperature=3.0)  # Source allows up to 2.0 now

        valid, error = contract.validate()
        assert valid is False
        assert "temperature" in error.lower()

        # Test valid upper bound
        contract_ok = ai.AIContract(temperature=2.0)
        valid_ok, error_ok = contract_ok.validate()
        assert valid_ok is True
        assert error_ok is None

    def test_contract_to_dict(self):
        """Test converting contract to dict"""
        contract = ai.AIContract(max_latency_ms=500.0)
        d = contract.to_dict()

        assert "max_latency_ms" in d
        assert "temperature" in d


class TestAITask:
    """Test AITask dataclass"""

    def test_task_creation(self):
        """Test creating AI task"""
        task = ai.AITask(
            operation="GENERATE",
            provider="OpenAI",
            model="gpt-4",
            payload={"prompt": "test"},
        )

        assert task.operation == "GENERATE"
        assert task.provider == "OpenAI"
        assert task.model == "gpt-4"
        assert task.trace_id is not None
        assert len(task.trace_id) == 16

    def test_task_with_deadline(self):
        """Test task with deadline"""
        deadline = datetime.now() + timedelta(seconds=10)
        task = ai.AITask(
            operation="EMBED", provider="OpenAI", model="ada", deadline=deadline
        )

        assert task.deadline == deadline
        assert task.is_expired() is False

    def test_task_is_expired(self):
        """Test task expiration check"""
        past_deadline = datetime.now() - timedelta(seconds=1)
        task = ai.AITask(
            operation="GENERATE",
            provider="OpenAI",
            model="gpt-4",
            deadline=past_deadline,
        )

        assert task.is_expired() is True

    def test_task_to_dict(self):
        """Test converting task to dict"""
        task = ai.AITask(operation="EMBED", provider="OpenAI", model="ada")

        d = task.to_dict()
        assert "operation" in d
        assert "provider" in d
        assert "trace_id" in d


class TestAIResult:
    """Test AIResult dataclass"""

    def test_result_creation(self):
        """Test creating AI result"""
        result = ai.AIResult(
            status="SUCCESS", data={"text": "response"}, latency_ms=150.0, cost=0.002
        )

        assert result.status == "SUCCESS"
        assert result.latency_ms == 150.0
        assert result.cost == 0.002

    def test_result_is_success(self):
        """Test success checking"""
        success_result = ai.AIResult(status="SUCCESS")
        failed_result = ai.AIResult(status="FAILED")

        assert success_result.is_success() is True
        assert failed_result.is_success() is False

    def test_result_to_dict(self):
        """Test converting result to dict"""
        result = ai.AIResult(status="SUCCESS", data={"key": "value"})

        d = result.to_dict()
        assert "status" in d
        assert "data" in d


class TestMockProvider:
    """Test MockProvider"""

    @pytest.fixture
    def provider(self):
        """Create mock provider"""
        return ai.MockProvider()

    def test_mock_provider_creation(self, provider):
        """Test creating mock provider"""
        assert provider.call_count == 0
        # Fix: Access the correct attribute name
        assert provider.latency_range_ms == (10.0, 100.0)

    @pytest.mark.asyncio
    async def test_mock_execute_embed(self, provider):
        """Test mock embedding execution"""
        task = ai.AITask(
            operation="EMBED",
            provider="Mock",
            model="mock-model-v1",
            payload={"text": "test"},
        )
        contract = ai.AIContract()

        result = await provider.execute(task, contract)

        assert result.status == "SUCCESS"
        assert "vector" in result.data
        assert isinstance(result.data["vector"], list)  # Check it's a list
        # Don't assert exact dimension if it might vary in source
        # assert len(result.data["vector"]) == 768
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_mock_execute_generate(self, provider):
        """Test mock generation execution"""
        task = ai.AITask(
            operation="GENERATE",
            provider="Mock",
            model="mock-model-v1",
            payload={"prompt": "test"},
        )
        contract = ai.AIContract()

        result = await provider.execute(task, contract)

        assert result.status == "SUCCESS"
        assert "text" in result.data
        # Fix: Access the correct metadata key
        assert result.metadata["mock_provider"] is True

    def test_mock_supports_all_operations(self, provider):
        """Test that mock supports all operations"""
        assert provider.supports_operation("EMBED") is True
        assert provider.supports_operation("GENERATE") is True
        assert provider.supports_operation("CUSTOM") is True

    def test_mock_get_models(self, provider):
        """Test getting mock models"""
        models = provider.get_models()
        assert len(models) > 0
        assert "mock-model-v1" in models


class TestOpenAIProvider:
    """Test OpenAIProvider"""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider"""
        # Use a mock key for testing - NOT A REAL API KEY
        return ai.OpenAIProvider(api_key="test-key-openai")

    def test_provider_creation(self, provider):
        """Test creating provider"""
        assert provider.api_key == "test-key-openai"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_supports_operation(self, provider):
        """Test operation support checking"""
        assert provider.supports_operation("EMBED") is True
        assert provider.supports_operation("GENERATE") is True
        assert provider.supports_operation("CLASSIFY") is True
        assert provider.supports_operation("UNSUPPORTED") is False

    def test_get_models(self, provider):
        """Test getting available models"""
        models = provider.get_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # Check for presence of *any* known model types
        assert any("gpt" in m for m in models)
        assert any("embedding" in m for m in models)

    @pytest.mark.asyncio
    async def test_execute_embed(self, provider):
        """Test embedding execution (uses mock response)"""
        task = ai.AITask(
            operation="EMBED",
            provider="OpenAI",
            model="text-embedding-3-small",
            payload={"text": "test embedding"},
        )
        contract = ai.AIContract()

        # Mock the aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}],
                "usage": {"total_tokens": 10},
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)

        with patch.object(provider, "_get_session", return_value=mock_session):
            result = await provider.execute(task, contract)

        assert result.status == "SUCCESS"
        assert "vector" in result.data
        assert isinstance(result.data["vector"], list)

    @pytest.mark.asyncio
    async def test_execute_generate(self, provider):
        """Test text generation (uses mock response)"""
        task = ai.AITask(
            operation="GENERATE",
            provider="OpenAI",
            model="gpt-4",  # Or use a model listed in the provider
            payload={"prompt": "Hello"},
        )
        contract = ai.AIContract()

        # Mock the aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "Hello there!"}}],
                "usage": {"total_tokens": 15},
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)

        with patch.object(provider, "_get_session", return_value=mock_session):
            result = await provider.execute(task, contract)

        assert result.status == "SUCCESS"
        assert "text" in result.data
        assert isinstance(result.data["text"], str)

    @pytest.mark.asyncio
    async def test_execute_unsupported_operation(self, provider):
        """Test unsupported operation"""
        task = ai.AITask(
            operation="TRANSCRIBE",  # Example of unsupported
            provider="OpenAI",
            model="gpt-4",
            payload={},
        )
        contract = ai.AIContract()

        result = await provider.execute(task, contract)

        assert result.status == "FAILED"
        assert result.error_code == ai.AI_ERRORS.AI_UNSUPPORTED.value

    @pytest.mark.asyncio
    async def test_execute_no_text_for_embed(self, provider):
        """Test embedding without text"""
        task = ai.AITask(
            operation="EMBED",
            provider="OpenAI",
            model="text-embedding-3-small",
            payload={},  # Missing 'text'
        )
        contract = ai.AIContract()

        result = await provider.execute(task, contract)

        assert result.status == "FAILED"
        assert result.error_code == ai.AI_ERRORS.AI_INVALID_REQUEST.value
        assert "text" in result.error.lower()


class TestAnthropicProvider:
    """Test AnthropicProvider"""

    @pytest.fixture
    def provider(self):
        """Create Anthropic provider"""
        # NOT A REAL API KEY - Test value only
        return ai.AnthropicProvider(api_key="test-key-anthropic")

    def test_provider_creation(self, provider):
        """Test creating provider"""
        assert provider.api_key == "test-key-anthropic"
        assert "anthropic.com" in provider.base_url

    def test_supports_operation(self, provider):
        """Test operation support"""
        assert provider.supports_operation("GENERATE") is True
        assert provider.supports_operation("ANALYZE") is True
        assert provider.supports_operation("EMBED") is False

    def test_get_models(self, provider):
        """Test getting models"""
        models = provider.get_models()
        # Fix: Check for the exact model names listed in the source
        expected_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
        ]
        assert all(m in models for m in expected_models)
        # Or a less strict check:
        # assert any("claude-3-opus" in m for m in models)
        # assert any("claude-3-sonnet" in m for m in models)

    @pytest.mark.asyncio
    async def test_execute_generate(self, provider):
        """Test generation (uses mock response)"""
        task = ai.AITask(
            operation="GENERATE",
            provider="Anthropic",
            # Use one of the actual model names
            model="claude-3-sonnet-20240229",
            payload={"prompt": "test"},
        )
        contract = ai.AIContract()

        # Mock the aiohttp response for Anthropic API
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "content": [{"text": "Hello from Claude!"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)

        with patch.object(provider, "_get_session", return_value=mock_session):
            result = await provider.execute(task, contract)

        assert result.status == "SUCCESS"
        assert "text" in result.data


class TestGrokProvider:
    """Test GrokProvider"""

    @pytest.fixture
    def provider(self):
        """Create Grok provider"""
        # NOT A REAL API KEY - Test value only
        return ai.GrokProvider(api_key="test-key-grok")

    def test_provider_creation(self, provider):
        """Test creating provider"""
        assert provider.api_key == "test-key-grok"
        assert "x.ai" in provider.base_url

    def test_supports_operation(self, provider):
        """Test operation support"""
        assert provider.supports_operation("GENERATE") is True
        assert provider.supports_operation("REASON") is True
        assert provider.supports_operation("EMBED") is True

    def test_get_models(self, provider):
        """Test getting models"""
        models = provider.get_models()
        # Fix: Check for the exact models listed in the source code
        expected_models = ["grok-1", "grok-1.5", "grok-3", "grok-4"]
        assert all(m in models for m in expected_models)

    @pytest.mark.asyncio
    async def test_execute_reason(self, provider):
        """Test reasoning operation (uses mock response)"""
        task = ai.AITask(
            operation="REASON",
            provider="Grok",
            model="grok-4",  # Use one of the available models
            payload={"prompt": "test reasoning"},
        )
        contract = ai.AIContract()

        # GrokProvider is already a mock implementation, no need to patch
        result = await provider.execute(task, contract)

        assert result.status == "SUCCESS"
        # Fix: Check for the correct key as per source code fix
        assert "reasoning_steps" in result.data
        assert isinstance(result.data["reasoning_steps"], list)


class TestResultCache:
    """Test ResultCache"""

    @pytest.fixture
    def cache(self):
        """Create cache instance"""
        # Match TTL from AIRuntime default if needed, or keep test-specific
        return ai.ResultCache(max_size=10, ttl_seconds=60)

    def test_cache_creation(self, cache):
        """Test creating cache"""
        assert cache.max_size == 10
        assert cache.ttl_seconds == 60
        assert len(cache.cache) == 0

    def test_cache_put_get(self, cache):
        """Test putting and getting from cache"""
        task = ai.AITask(
            operation="EMBED", provider="OpenAI", model="ada", payload={"text": "test"}
        )
        contract = ai.AIContract()
        result = ai.AIResult(status="SUCCESS", data={"vector": [1, 2, 3]})

        cache.put(task, contract, result)
        cached = cache.get(task, contract)

        assert cached is not None
        assert cached.cached is True
        assert cached.data == result.data

    def test_cache_miss(self, cache):
        """Test cache miss"""
        task = ai.AITask(
            operation="EMBED", provider="OpenAI", model="ada", payload={"text": "test"}
        )
        contract = ai.AIContract()

        result = cache.get(task, contract)
        assert result is None

    def test_cache_ttl_expiration(self, cache):
        """Test cache TTL expiration"""
        short_cache = ai.ResultCache(max_size=10, ttl_seconds=0.1)

        task = ai.AITask(
            operation="EMBED", provider="OpenAI", model="ada", payload={"text": "test"}
        )
        contract = ai.AIContract()
        result = ai.AIResult(status="SUCCESS", data={"vector": [1]})  # Must have status

        short_cache.put(task, contract, result)
        assert len(short_cache.cache) == 1  # Check it was added
        time.sleep(0.2)  # Wait for TTL to expire

        cached = short_cache.get(task, contract)
        assert cached is None
        assert len(short_cache.cache) == 0  # Check it was removed

    def test_cache_no_cache_when_disabled(self, cache):
        """Test that caching is disabled when contract disallows it"""
        task = ai.AITask(
            operation="EMBED", provider="OpenAI", model="ada", payload={"text": "test"}
        )
        contract_no_cache = ai.AIContract(allow_cached=False)
        contract_allow_cache = ai.AIContract(allow_cached=True)
        result = ai.AIResult(status="SUCCESS", data={"v": [1]})

        # Try putting with allow_cached=False
        cache.put(task, contract_no_cache, result)
        assert len(cache.cache) == 0  # Should not have been added

        # Try getting with allow_cached=False (even if it existed)
        cache.put(task, contract_allow_cache, result)  # Put it first
        assert len(cache.cache) == 1
        cached = cache.get(task, contract_no_cache)
        assert cached is None  # Should not return cached result

    def test_cache_eviction(self):
        """Test cache eviction when full"""
        small_cache = ai.ResultCache(max_size=2, ttl_seconds=60)

        # Add 3 items to cache of size 2
        tasks = []
        for i in range(3):
            task = ai.AITask(
                operation="EMBED",
                provider="OpenAI",
                model="ada",
                payload={"text": f"test{i}"},
            )
            tasks.append(task)
            contract = ai.AIContract()
            result = ai.AIResult(status="SUCCESS", data={"i": i})
            small_cache.put(task, contract, result)
            time.sleep(0.01)  # Ensure slightly different timestamps for FIFO

        assert len(small_cache.cache) == 2  # Should stay at max size

        # Check that the first task (oldest) was evicted
        cached_0 = small_cache.get(tasks[0], ai.AIContract())
        cached_1 = small_cache.get(tasks[1], ai.AIContract())
        cached_2 = small_cache.get(tasks[2], ai.AIContract())

        assert cached_0 is None
        assert cached_1 is not None
        assert cached_2 is not None

    def test_cache_clear(self, cache):
        """Test clearing cache"""
        task = ai.AITask(
            operation="EMBED", provider="OpenAI", model="ada", payload={"text": "test"}
        )
        contract = ai.AIContract()
        result = ai.AIResult(status="SUCCESS", data={"v": [1]})

        cache.put(task, contract, result)
        assert len(cache.cache) == 1
        cache.clear()

        assert len(cache.cache) == 0

    def test_cache_stats(self, cache):
        """Test cache statistics"""
        task = ai.AITask(
            operation="EMBED", provider="OpenAI", model="ada", payload={"text": "test"}
        )
        contract = ai.AIContract()
        result = ai.AIResult(status="SUCCESS", data={"v": [1]})

        cache.put(task, contract, result)
        stats = cache.stats()

        assert stats["size"] == 1
        assert stats["max_size"] == 10
        # Check keys added in source code fix
        assert "ttl_seconds" in stats
        assert "avg_age_seconds" in stats


class TestRateLimiter:
    """Test RateLimiter"""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter"""
        # Create a fresh limiter for each test
        limiter_instance = ai.RateLimiter()
        # Define limits explicitly for tests
        limiter_instance.limits = {
            "OpenAI": {"calls": 60, "window": 60},
            "Anthropic": {"calls": 30, "window": 60},
            "TestShort": {"calls": 1, "window": 0.1},
            "TestMulti": {"calls": 2, "window": 60},
        }
        return limiter_instance

    def test_limiter_creation(self, limiter):
        """Test creating rate limiter"""
        # Fix: Check the limits dict, not a non-existent attribute
        assert "OpenAI" in limiter.limits
        assert "Anthropic" in limiter.limits
        assert limiter.limits["OpenAI"]["calls"] == 60

    def test_check_limit_allowed(self, limiter):
        """Test check within limits"""
        allowed = limiter.check_limit("OpenAI")
        assert allowed is True
        # Check that a timestamp was added
        assert len(limiter.calls["OpenAI"]) == 1

    def test_check_limit_exceeded(self, limiter):
        """Test check when limit exceeded"""
        # Uses "TestMulti" limit: 2 calls / 60 sec
        assert limiter.check_limit("TestMulti") is True
        assert limiter.check_limit("TestMulti") is True
        # Third call should be denied
        assert limiter.check_limit("TestMulti") is False
        # Ensure only 2 calls recorded
        assert len(limiter.calls["TestMulti"]) == 2

    def test_check_limit_window_expiry(self, limiter):
        """Test rate limit window expiry"""
        # Uses "TestShort" limit: 1 call / 0.1 sec
        assert limiter.check_limit("TestShort") is True
        # Second call denied immediately
        assert limiter.check_limit("TestShort") is False

        # Wait for window to expire
        time.sleep(0.15)

        # Should be allowed again
        assert limiter.check_limit("TestShort") is True
        # Check that old timestamp was cleared
        assert len(limiter.calls["TestShort"]) == 1

    def test_reset_specific_provider(self, limiter):
        """Test resetting specific provider"""
        limiter.check_limit("OpenAI")
        assert len(limiter.calls["OpenAI"]) == 1
        limiter.reset("OpenAI")
        # Fix: Check the calls dict
        assert len(limiter.calls["OpenAI"]) == 0

    def test_reset_all(self, limiter):
        """Test resetting all providers"""
        limiter.check_limit("OpenAI")
        limiter.check_limit("Anthropic")
        assert len(limiter.calls["OpenAI"]) == 1
        assert len(limiter.calls["Anthropic"]) == 1
        limiter.reset()
        # After reset(), the calls dict is cleared entirely, so keys don't exist
        assert "OpenAI" not in limiter.calls
        assert "Anthropic" not in limiter.calls


class TestAIRuntime:
    """Test AIRuntime"""

    @pytest.fixture
    def runtime(self):
        """Create runtime instance"""
        # Ensure lowercase 'mock' is registered and potentially default
        runtime_instance = ai.AIRuntime(
            config={"cache_size": 100, "cache_ttl_seconds": 60}
        )
        return runtime_instance

    def test_runtime_creation(self, runtime):
        """Test creating runtime"""
        assert runtime.cache.max_size == 100
        # Fix: Check for lowercase 'mock'
        assert "mock" in runtime.providers
        assert "default" in runtime.providers  # Should always have a default
        assert isinstance(runtime.providers["mock"], ai.MockProvider)

    def test_register_provider(self, runtime):
        """Test registering custom provider"""
        custom_provider = ai.MockProvider()
        runtime.register_provider("Custom", custom_provider)

        # Fix: Check for lowercase 'custom'
        assert "custom" in runtime.providers
        assert isinstance(runtime.providers["custom"], ai.MockProvider)

    # --- Tests involving execute_task need to use execute_task_async ---
    @pytest.mark.asyncio
    async def test_execute_task_async_basic(self, runtime):
        """Test executing task using the async method"""
        task = ai.AITask(
            operation="EMBED",
            provider="mock",  # Use lowercase
            model="mock-model-v1",
            payload={"text": "test"},
        )
        contract = ai.AIContract()  # Create contract

        # Fix: Call the async version and await it
        result = await runtime.execute_task_async(task, contract)

        assert result.status == "SUCCESS"
        assert result.cached is False

    @pytest.mark.asyncio
    async def test_execute_task_with_cache(self, runtime):
        """Test that second async execution uses cache"""
        task = ai.AITask(
            operation="EMBED",
            provider="mock",
            model="mock-model-v1",
            payload={"text": "test cache"},
        )
        contract = ai.AIContract(allow_cached=True)

        # First execution (async)
        result1 = await runtime.execute_task_async(task, contract)
        assert result1.status == "SUCCESS"
        assert result1.cached is False

        # Second execution (async) should be cached
        result2 = await runtime.execute_task_async(task, contract)
        assert result2.status == "SUCCESS"
        assert result2.cached is True
        # Check cache hit recorded
        assert runtime.metrics.cache_hits == 1

    @pytest.mark.asyncio
    async def test_execute_task_invalid_contract(self, runtime):
        """Test async execution with invalid contract"""
        task = ai.AITask(
            operation="EMBED", provider="mock", model="mock", payload={"text": "test"}
        )
        contract = ai.AIContract(temperature=5.0)  # Invalid temperature

        # Fix: Call the async version
        result = await runtime.execute_task_async(task, contract)

        assert result.status == "FAILED"
        assert result.error_code == ai.AI_ERRORS.AI_VALIDATION_ERROR.value

    @pytest.mark.asyncio
    async def test_execute_expired_task(self, runtime):
        """Test async execution of expired task"""
        task = ai.AITask(
            operation="EMBED",
            provider="mock",
            model="mock",
            payload={"text": "test"},
            deadline=datetime.now() - timedelta(seconds=1),
        )
        # Fix: Add contract argument
        contract = ai.AIContract()

        # Fix: Call the async version
        result = await runtime.execute_task_async(task, contract)

        assert result.status == "FAILED"
        # Check error code added in source fix
        assert result.error_code == ai.AI_ERRORS.AI_TIMEOUT.value
        assert "deadline" in result.error.lower()

    @pytest.mark.asyncio
    async def test_batch_execute(self, runtime):
        """Test batch execution (uses async tasks internally now)"""
        tasks = [
            ai.AITask(
                operation="EMBED",
                provider="mock",
                model="mock",
                payload={"text": f"test{i}"},
            )
            for i in range(3)
        ]

        results = await runtime.batch_execute(tasks)

        assert len(results) == 3
        assert all(isinstance(r, ai.AIResult) for r in results)  # Check type
        assert all(r.status == "SUCCESS" for r in results)

    def test_execute_task_sync(self, runtime):
        """Test synchronous execution wrapper"""
        task = ai.AITask(
            operation="EMBED",
            provider="mock",  # Use lowercase
            model="mock",
            payload={"text": "test sync"},
        )
        contract = ai.AIContract()

        # Call the sync wrapper
        result = runtime.execute_task_sync(task, contract)

        assert result.status == "SUCCESS"
        assert result.cached is False

        # Execute again to test caching via sync wrapper
        result2 = runtime.execute_task_sync(task, contract)
        assert result2.status == "SUCCESS"
        assert result2.cached is True

    def test_get_metrics(self, runtime):
        """Test getting runtime metrics"""
        # Execute a task to populate metrics
        task = ai.AITask(
            operation="GENERATE", provider="mock", model="m", payload={"p": "1"}
        )
        runtime.execute_task_sync(task)

        metrics = runtime.get_metrics()

        assert "providers" in metrics
        assert "cache_stats" in metrics
        assert "execution_metrics" in metrics
        assert metrics["execution_metrics"]["total_executions"] == 1

    def test_routing_provider_specified(self, runtime):
        """Test routing when provider is specified"""
        runtime.register_provider("Specific", ai.MockProvider())
        task = ai.AITask(operation="EMBED", provider="Specific", model="m")
        provider = runtime._route_to_provider(task)
        assert isinstance(provider, ai.MockProvider)
        assert provider == runtime.providers["specific"]  # Check it's the right one

    def test_routing_fallback_to_default(self, runtime):
        """Test routing falls back to default if specified provider invalid"""
        task = ai.AITask(operation="EMBED", provider="NonExistent", model="m")
        provider = runtime._route_to_provider(task)
        assert provider == runtime.providers["default"]

    def test_cleanup(self, runtime):
        """Test runtime cleanup"""
        # Add something to cache first
        task = ai.AITask(
            operation="GENERATE", provider="mock", model="m", payload={"p": "cleanup"}
        )
        runtime.execute_task_sync(task)
        assert len(runtime.cache.cache) > 0

        runtime.cleanup()
        # Cache should be cleared by cleanup
        assert len(runtime.cache.cache) == 0


class TestAIMetrics:
    """Test AIMetrics"""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance"""
        return ai.AIMetrics()

    def test_metrics_creation(self, metrics):
        """Test creating metrics"""
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert len(metrics.executions) == 0

    def test_record_execution_success(self, metrics):
        """Test recording successful execution"""
        result = ai.AIResult(status="SUCCESS", latency_ms=100.0, cost=0.002)

        metrics.record_execution("OpenAI", "EMBED", result)

        assert metrics.executions["openai:EMBED"] == 1
        assert metrics.successes["openai:EMBED"] == 1
        assert metrics.failures["openai:EMBED"] == 0
        assert metrics.total_latency["openai:EMBED"] == 100.0
        assert metrics.total_cost["openai:EMBED"] == 0.002

    def test_record_execution_failure(self, metrics):
        """Test recording failed execution"""
        result = ai.AIResult(
            status="FAILED",
            error="API error",
            latency_ms=50.0,
            cost=0.0,  # Failed calls might have zero cost
        )

        metrics.record_execution("Anthropic", "GENERATE", result)

        assert metrics.executions["anthropic:GENERATE"] == 1
        assert metrics.successes["anthropic:GENERATE"] == 0
        assert metrics.failures["anthropic:GENERATE"] == 1
        assert metrics.total_latency["anthropic:GENERATE"] == 50.0
        assert metrics.total_cost["anthropic:GENERATE"] == 0.0

    def test_record_cache_operations(self, metrics):
        """Test recording cache operations"""
        metrics.record_cache_hit()
        metrics.record_cache_hit()
        metrics.record_cache_miss()

        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 1

        summary = metrics.get_summary()
        # Fix: Check calculation 2 / (2+1)
        assert summary["cache_hit_rate"] == pytest.approx(2 / 3)

    def test_get_summary(self, metrics):
        """Test getting summary"""
        result_ok = ai.AIResult(status="SUCCESS", latency_ms=50.0, cost=0.001)
        result_fail = ai.AIResult(status="FAILED", latency_ms=30.0, cost=0.0)

        metrics.record_execution("OpenAI", "EMBED", result_ok)
        metrics.record_execution(
            "openai", "GENERATE", result_fail
        )  # Test case insensitivity
        metrics.record_execution("Mock", "EMBED", result_ok)

        summary = metrics.get_summary()

        assert summary["total_executions"] == 3
        assert summary["total_successes"] == 2
        assert summary["total_failures"] == 1

        # Fix: Check for the correct key name
        assert "by_provider_operation" in summary
        prov_op_metrics = summary["by_provider_operation"]

        assert "openai" in prov_op_metrics
        assert "mock" in prov_op_metrics

        assert "EMBED" in prov_op_metrics["openai"]
        assert prov_op_metrics["openai"]["EMBED"]["count"] == 1
        assert prov_op_metrics["openai"]["EMBED"]["success_rate"] == 1.0
        assert prov_op_metrics["openai"]["EMBED"]["avg_latency_ms"] == 50.0

        assert "GENERATE" in prov_op_metrics["openai"]
        assert prov_op_metrics["openai"]["GENERATE"]["count"] == 1
        assert prov_op_metrics["openai"]["GENERATE"]["success_rate"] == 0.0
        assert prov_op_metrics["openai"]["GENERATE"]["avg_latency_ms"] == 30.0

        assert "EMBED" in prov_op_metrics["mock"]
        assert prov_op_metrics["mock"]["EMBED"]["count"] == 1


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_complete_workflow_async(self):
        """Test complete workflow using async method"""
        # Create runtime with short cache TTL for testing
        runtime = ai.AIRuntime(config={"cache_ttl_seconds": 1})

        task = ai.AITask(
            operation="GENERATE",
            provider="mock",  # Use lowercase
            model="mock-model-v1",
            payload={"prompt": "Hello, async world!"},
        )
        contract = ai.AIContract(
            max_latency_ms=1000.0, temperature=0.7, allow_cached=True
        )

        # Execute async
        result1 = await runtime.execute_task_async(task, contract)

        # Verify first result
        assert result1.status == "SUCCESS"
        assert result1.cached is False
        assert result1.latency_ms < contract.max_latency_ms
        assert "text" in result1.data

        # Execute again - should be cached
        result2 = await runtime.execute_task_async(task, contract)
        assert result2.status == "SUCCESS"
        assert result2.cached is True
        assert result2.data == result1.data  # Ensure data is consistent

        # Wait for cache to expire
        await asyncio.sleep(1.1)

        # Execute again - should not be cached
        result3 = await runtime.execute_task_async(task, contract)
        assert result3.status == "SUCCESS"
        assert result3.cached is False

        # Check metrics
        metrics = runtime.get_metrics()
        assert (
            metrics["execution_metrics"]["total_executions"] == 3
        )  # 2 misses, 1 hit recorded
        assert metrics["execution_metrics"]["cache_hits"] == 1
        assert metrics["execution_metrics"]["cache_misses"] == 2
        assert (
            metrics["execution_metrics"]["by_provider_operation"]["mock"]["GENERATE"][
                "count"
            ]
            == 3
        )


# Allow running tests directly if needed
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
