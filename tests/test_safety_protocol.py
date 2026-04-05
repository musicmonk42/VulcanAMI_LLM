"""Tests for canonical SafetyProtocol."""
import pytest
from src.protocols.safety import SafetyValidator, SafetyLevel


class TestSafetyValidator:
    def setup_method(self):
        self.validator = SafetyValidator()

    def test_safe_content(self):
        result = self.validator.validate("Hello, world!")
        assert result.is_safe is True
        assert result.level == SafetyLevel.SAFE

    def test_eval_detected(self):
        result = self.validator.validate("eval('malicious')")
        assert result.is_safe is False
        assert result.level == SafetyLevel.BLOCKED
        assert any("eval" in v.pattern for v in result.violations)

    def test_blacklist(self):
        v = SafetyValidator(blacklist={"forbidden_word"})
        result = v.validate("This contains forbidden_word here")
        assert result.is_safe is False

    def test_add_pattern(self):
        self.validator.add_pattern(r"DROP\s+TABLE", "high")
        result = self.validator.validate("DROP TABLE users")
        assert result.is_safe is False

    def test_technical_context_suppresses_medium(self):
        self.validator.add_pattern(r"debug_flag", "medium")
        result = self.validator.validate(
            "debug_flag = True", context={"source": "internal"}
        )
        assert result.is_safe is True

    def test_get_config(self):
        config = self.validator.get_config()
        assert "pattern_count" in config
        assert config["validation_count"] == 0
        self.validator.validate("test")
        assert self.validator.get_config()["validation_count"] == 1
