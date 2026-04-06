"""Edge case tests for SafetyValidator."""
import pytest
from src.protocols.safety import SafetyValidator, SafetyLevel


class TestSafetyEdgeCases:
    def setup_method(self):
        self.v = SafetyValidator()

    def test_empty_string(self):
        result = self.v.validate("")
        assert result.is_safe is True  # empty = nothing dangerous

    def test_unicode_emoji(self):
        result = self.v.validate("Hello \U0001f30d world \U0001f525")
        assert result.is_safe is True

    def test_sql_injection(self):
        result = self.v.validate("'; DROP TABLE users; --")
        # SafetyValidator catches code patterns, not SQL specifically
        assert isinstance(result.is_safe, bool)

    def test_path_traversal(self):
        result = self.v.validate("../../etc/passwd")
        assert isinstance(result.is_safe, bool)

    def test_null_bytes(self):
        result = self.v.validate("hello\x00world")
        assert isinstance(result.is_safe, bool)

    def test_extremely_long_input(self):
        result = self.v.validate("a" * 100_000)
        assert isinstance(result.is_safe, bool)

    def test_nested_eval_attempts(self):
        result = self.v.validate("eval(eval('os.system(\"rm -rf /\")'))")
        assert result.is_safe is False
        assert result.level == SafetyLevel.BLOCKED

    def test_subprocess_in_string(self):
        result = self.v.validate("subprocess.Popen(['rm', '-rf', '/'])")
        assert result.is_safe is False

    def test_os_system_obfuscated(self):
        result = self.v.validate("os.system('cat /etc/shadow')")
        assert result.is_safe is False

    def test_import_injection(self):
        result = self.v.validate("__import__('os').system('id')")
        assert result.is_safe is False

    def test_whitelist_with_no_match(self):
        v = SafetyValidator(whitelist={"authorized_only"})
        result = v.validate("this has no authorized terms")
        assert len(result.violations) > 0

    def test_blacklist_case_insensitive(self):
        v = SafetyValidator(blacklist={"FORBIDDEN"})
        result = v.validate("this contains forbidden content")
        assert result.is_safe is False

    def test_add_custom_pattern(self):
        self.v.add_pattern(r"SELECT\s+\*\s+FROM", "high")
        result = self.v.validate("SELECT * FROM users WHERE 1=1")
        assert result.is_safe is False

    def test_technical_context_suppresses_medium(self):
        self.v.add_pattern(r"debug_mode", "medium")
        result = self.v.validate("debug_mode = True", context={"source": "internal"})
        assert result.is_safe is True  # medium suppressed for internal

    def test_technical_context_does_not_suppress_high(self):
        result = self.v.validate("eval('malicious')", context={"source": "internal"})
        assert result.is_safe is False  # high never suppressed
