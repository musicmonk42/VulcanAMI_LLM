# ============================================================
# Tests for Simple Mode Configuration
# ============================================================
"""
Tests for the simple_mode.py performance configuration module.
"""

import os
import pytest
from importlib import reload


class TestSimpleModeConfiguration:
    """Test cases for simple mode configuration."""

    def test_default_mode(self):
        """Test that default mode has expected values."""
        # Remove env vars to test defaults
        os.environ.pop("VULCAN_SIMPLE_MODE", None)
        os.environ.pop("SKIP_BERT_EMBEDDINGS", None)
        os.environ.pop("MIN_AGENTS", None)
        os.environ.pop("MAX_AGENTS", None)
        
        # Reload module with clean env
        import src.vulcan.simple_mode as sm
        reload(sm)
        
        assert sm.SIMPLE_MODE is False
        assert sm.SKIP_BERT_EMBEDDINGS is False
        assert sm.DEFAULT_MIN_AGENTS == 10
        assert sm.DEFAULT_MAX_AGENTS == 100
        assert sm.MAX_PROVENANCE_RECORDS == 1000
        assert sm.AGENT_CHECK_INTERVAL == 30

    def test_simple_mode_enabled(self):
        """Test that simple mode sets expected values."""
        os.environ["VULCAN_SIMPLE_MODE"] = "true"
        
        import src.vulcan.simple_mode as sm
        reload(sm)
        
        assert sm.SIMPLE_MODE is True
        assert sm.SKIP_BERT_EMBEDDINGS is True
        assert sm.DEFAULT_MIN_AGENTS == 1
        assert sm.DEFAULT_MAX_AGENTS == 5
        assert sm.MAX_PROVENANCE_RECORDS == 50
        assert sm.AGENT_CHECK_INTERVAL == 300
        
        # Clean up
        os.environ.pop("VULCAN_SIMPLE_MODE", None)

    def test_individual_flags(self):
        """Test that individual flags can override simple mode defaults."""
        os.environ.pop("VULCAN_SIMPLE_MODE", None)
        os.environ["SKIP_BERT_EMBEDDINGS"] = "true"
        os.environ["MIN_AGENTS"] = "2"
        os.environ["MAX_AGENTS"] = "20"
        
        import src.vulcan.simple_mode as sm
        reload(sm)
        
        assert sm.SIMPLE_MODE is False
        assert sm.SKIP_BERT_EMBEDDINGS is True
        assert sm.DEFAULT_MIN_AGENTS == 2
        assert sm.DEFAULT_MAX_AGENTS == 20
        
        # Clean up
        os.environ.pop("SKIP_BERT_EMBEDDINGS", None)
        os.environ.pop("MIN_AGENTS", None)
        os.environ.pop("MAX_AGENTS", None)

    def test_get_simple_mode_config(self):
        """Test the get_simple_mode_config function."""
        os.environ["VULCAN_SIMPLE_MODE"] = "true"
        
        import src.vulcan.simple_mode as sm
        reload(sm)
        
        config = sm.get_simple_mode_config()
        
        assert isinstance(config, dict)
        assert config["simple_mode"] is True
        assert config["skip_bert_embeddings"] is True
        assert config["min_agents"] == 1
        assert config["max_agents"] == 5
        
        # Clean up
        os.environ.pop("VULCAN_SIMPLE_MODE", None)

    def test_helper_functions(self):
        """Test is_simple_mode and should_skip_bert helper functions."""
        os.environ["VULCAN_SIMPLE_MODE"] = "true"
        
        import src.vulcan.simple_mode as sm
        reload(sm)
        
        assert sm.is_simple_mode() is True
        assert sm.should_skip_bert() is True
        
        # Clean up
        os.environ.pop("VULCAN_SIMPLE_MODE", None)

    def test_str_to_bool_variants(self):
        """Test various true/false string values."""
        import src.vulcan.simple_mode as sm
        
        # Test true values
        assert sm._str_to_bool("true") is True
        assert sm._str_to_bool("TRUE") is True
        assert sm._str_to_bool("True") is True
        assert sm._str_to_bool("1") is True
        assert sm._str_to_bool("yes") is True
        assert sm._str_to_bool("on") is True
        
        # Test false values
        assert sm._str_to_bool("false") is False
        assert sm._str_to_bool("FALSE") is False
        assert sm._str_to_bool("0") is False
        assert sm._str_to_bool("no") is False
        assert sm._str_to_bool("off") is False
        
        # Test None with default
        assert sm._str_to_bool(None, default=True) is True
        assert sm._str_to_bool(None, default=False) is False
