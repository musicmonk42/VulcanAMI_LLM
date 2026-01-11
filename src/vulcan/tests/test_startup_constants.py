"""
Unit tests for startup constants.

Tests that all constants are properly defined and documented.
"""

import pytest

from vulcan.server.startup.constants import (
    DEFAULT_THREAD_POOL_SIZE,
    THREAD_NAME_PREFIX,
    MEMORY_GUARD_THRESHOLD_PERCENT,
    MEMORY_GUARD_CHECK_INTERVAL_SECONDS,
    REDIS_WORKER_TTL_SECONDS,
    SELF_OPTIMIZER_TARGET_LATENCY_MS,
    SELF_OPTIMIZER_TARGET_MEMORY_MB,
    SELF_OPTIMIZER_INTERVAL_SECONDS,
    SHUTDOWN_TIMEOUT_SECONDS,
    MAX_STARTUP_INFO_LOGS,
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_CHECKPOINT_DIR,
    LLM_CONFIG_PATH,
    StartupPhaseConfig,
)


class TestThreadPoolConstants:
    """Test thread pool configuration constants."""
    
    def test_default_thread_pool_size(self):
        """Test default thread pool size is reasonable."""
        assert DEFAULT_THREAD_POOL_SIZE > 0
        assert DEFAULT_THREAD_POOL_SIZE == 32
        assert isinstance(DEFAULT_THREAD_POOL_SIZE, int)
    
    def test_thread_name_prefix(self):
        """Test thread name prefix is set."""
        assert THREAD_NAME_PREFIX
        assert isinstance(THREAD_NAME_PREFIX, str)
        assert THREAD_NAME_PREFIX == "vulcan_"


class TestMemoryConstants:
    """Test memory management constants."""
    
    def test_memory_guard_threshold(self):
        """Test memory guard threshold is reasonable."""
        assert MEMORY_GUARD_THRESHOLD_PERCENT > 0
        assert MEMORY_GUARD_THRESHOLD_PERCENT <= 100
        assert MEMORY_GUARD_THRESHOLD_PERCENT == 85.0
        assert isinstance(MEMORY_GUARD_THRESHOLD_PERCENT, float)
    
    def test_memory_guard_check_interval(self):
        """Test memory guard check interval is reasonable."""
        assert MEMORY_GUARD_CHECK_INTERVAL_SECONDS > 0
        assert MEMORY_GUARD_CHECK_INTERVAL_SECONDS <= 60
        assert MEMORY_GUARD_CHECK_INTERVAL_SECONDS == 5.0
        assert isinstance(MEMORY_GUARD_CHECK_INTERVAL_SECONDS, float)


class TestRedisConstants:
    """Test Redis configuration constants."""
    
    def test_redis_worker_ttl(self):
        """Test Redis worker TTL is reasonable."""
        assert REDIS_WORKER_TTL_SECONDS > 0
        assert REDIS_WORKER_TTL_SECONDS == 3600  # 1 hour
        assert isinstance(REDIS_WORKER_TTL_SECONDS, int)
        
        # Should be at least a few minutes
        assert REDIS_WORKER_TTL_SECONDS >= 300


class TestSelfOptimizerConstants:
    """Test self-optimizer configuration constants."""
    
    def test_target_latency(self):
        """Test target latency is reasonable."""
        assert SELF_OPTIMIZER_TARGET_LATENCY_MS > 0
        assert SELF_OPTIMIZER_TARGET_LATENCY_MS == 100
        assert isinstance(SELF_OPTIMIZER_TARGET_LATENCY_MS, int)
    
    def test_target_memory(self):
        """Test target memory is reasonable."""
        assert SELF_OPTIMIZER_TARGET_MEMORY_MB > 0
        assert SELF_OPTIMIZER_TARGET_MEMORY_MB == 2000
        assert isinstance(SELF_OPTIMIZER_TARGET_MEMORY_MB, int)
    
    def test_optimization_interval(self):
        """Test optimization interval is reasonable."""
        assert SELF_OPTIMIZER_INTERVAL_SECONDS > 0
        assert SELF_OPTIMIZER_INTERVAL_SECONDS == 60
        assert isinstance(SELF_OPTIMIZER_INTERVAL_SECONDS, int)


class TestShutdownConstants:
    """Test shutdown configuration constants."""
    
    def test_shutdown_timeout(self):
        """Test shutdown timeout is reasonable."""
        assert SHUTDOWN_TIMEOUT_SECONDS > 0
        assert SHUTDOWN_TIMEOUT_SECONDS <= 10  # Should be quick
        assert SHUTDOWN_TIMEOUT_SECONDS == 2.0
        assert isinstance(SHUTDOWN_TIMEOUT_SECONDS, float)


class TestLoggingConstants:
    """Test logging configuration constants."""
    
    def test_max_startup_info_logs(self):
        """Test max startup info logs is reasonable."""
        assert MAX_STARTUP_INFO_LOGS > 0
        assert MAX_STARTUP_INFO_LOGS == 10
        assert isinstance(MAX_STARTUP_INFO_LOGS, int)


class TestDirectoryConstants:
    """Test directory path constants."""
    
    def test_config_dir(self):
        """Test config directory is set."""
        assert DEFAULT_CONFIG_DIR
        assert isinstance(DEFAULT_CONFIG_DIR, str)
        assert DEFAULT_CONFIG_DIR == "configs"
    
    def test_data_dir(self):
        """Test data directory is set."""
        assert DEFAULT_DATA_DIR
        assert isinstance(DEFAULT_DATA_DIR, str)
        assert DEFAULT_DATA_DIR == "data"
    
    def test_checkpoint_dir(self):
        """Test checkpoint directory is set."""
        assert DEFAULT_CHECKPOINT_DIR
        assert isinstance(DEFAULT_CHECKPOINT_DIR, str)
        assert DEFAULT_CHECKPOINT_DIR == "checkpoints"
    
    def test_llm_config_path(self):
        """Test LLM config path is set."""
        assert LLM_CONFIG_PATH
        assert isinstance(LLM_CONFIG_PATH, str)
        assert LLM_CONFIG_PATH == "configs/llm_config.yaml"
        assert LLM_CONFIG_PATH.endswith(".yaml")


class TestStartupPhaseConfig:
    """Test StartupPhaseConfig class."""
    
    def test_phase_config_exists(self):
        """Test that StartupPhaseConfig class exists."""
        assert StartupPhaseConfig is not None
    
    def test_phase_timeouts_exist(self):
        """Test that phase timeouts are defined."""
        assert hasattr(StartupPhaseConfig, 'CONFIGURATION_TIMEOUT')
        assert hasattr(StartupPhaseConfig, 'CORE_SERVICES_TIMEOUT')
        assert hasattr(StartupPhaseConfig, 'REASONING_SYSTEMS_TIMEOUT')
        assert hasattr(StartupPhaseConfig, 'MEMORY_SYSTEMS_TIMEOUT')
        assert hasattr(StartupPhaseConfig, 'PRELOADING_TIMEOUT')
        assert hasattr(StartupPhaseConfig, 'MONITORING_TIMEOUT')
    
    def test_phase_timeout_values(self):
        """Test that phase timeout values are reasonable."""
        assert StartupPhaseConfig.CONFIGURATION_TIMEOUT > 0
        assert StartupPhaseConfig.CORE_SERVICES_TIMEOUT > 0
        assert StartupPhaseConfig.REASONING_SYSTEMS_TIMEOUT > 0
        assert StartupPhaseConfig.MEMORY_SYSTEMS_TIMEOUT > 0
        assert StartupPhaseConfig.PRELOADING_TIMEOUT > 0
        assert StartupPhaseConfig.MONITORING_TIMEOUT > 0
        
        # Preloading should have longest timeout (model loading)
        assert StartupPhaseConfig.PRELOADING_TIMEOUT >= 60
    
    def test_critical_phases_set_exists(self):
        """Test that critical phases set exists."""
        assert hasattr(StartupPhaseConfig, 'CRITICAL_PHASES')
        assert isinstance(StartupPhaseConfig.CRITICAL_PHASES, set)
    
    def test_critical_phases_content(self):
        """Test critical phases content."""
        critical = StartupPhaseConfig.CRITICAL_PHASES
        
        # Should have at least configuration and core services
        assert "CONFIGURATION" in critical
        assert "CORE_SERVICES" in critical
        
        # Should be a reasonable number (not all phases)
        assert len(critical) <= 3
    
    def test_parallel_groups_exist(self):
        """Test that parallel groups exist."""
        assert hasattr(StartupPhaseConfig, 'PARALLEL_GROUPS')
        assert isinstance(StartupPhaseConfig.PARALLEL_GROUPS, dict)
    
    def test_parallel_groups_content(self):
        """Test parallel groups content."""
        groups = StartupPhaseConfig.PARALLEL_GROUPS
        
        # Should have some groups defined
        assert len(groups) > 0
        
        # Each group should be a list
        for group_name, group_items in groups.items():
            assert isinstance(group_items, list)
            assert len(group_items) > 0


class TestConstantDocumentation:
    """Test that constants have proper documentation."""
    
    def test_constants_have_docstrings(self):
        """Test that major constants have documentation."""
        import vulcan.server.startup.constants as constants_module
        
        # Check module has docstring
        assert constants_module.__doc__
        assert len(constants_module.__doc__) > 0


class TestConstantConsistency:
    """Test consistency between related constants."""
    
    def test_memory_threshold_reasonable(self):
        """Test memory threshold is below 100%."""
        assert MEMORY_GUARD_THRESHOLD_PERCENT < 100
    
    def test_check_interval_shorter_than_ttl(self):
        """Test memory check interval is shorter than Redis TTL."""
        assert MEMORY_GUARD_CHECK_INTERVAL_SECONDS < REDIS_WORKER_TTL_SECONDS
    
    def test_thread_pool_size_reasonable(self):
        """Test thread pool size is reasonable for modern systems."""
        # Should be at least as many as typical CPU count + headroom
        assert DEFAULT_THREAD_POOL_SIZE >= 8
        # But not excessively large
        assert DEFAULT_THREAD_POOL_SIZE <= 256
    
    def test_config_path_uses_config_dir(self):
        """Test LLM config path uses the config directory."""
        assert LLM_CONFIG_PATH.startswith(DEFAULT_CONFIG_DIR)
