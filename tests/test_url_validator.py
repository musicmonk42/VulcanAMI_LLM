"""
Tests for URL Security Validation Module
=========================================

Comprehensive test suite for src.utils.url_validator to ensure proper
validation of URL schemes and prevention of security vulnerabilities.

Test Categories:
    - Valid URL schemes (http, https)
    - Invalid/dangerous schemes (file, ftp, data, javascript, etc.)
    - Malformed URLs
    - Edge cases
"""

import pytest
import urllib.error

from src.utils.url_validator import (
    ALLOWED_SCHEMES,
    URLValidationError,
    validate_url_scheme,
    safe_urlopen,
)


class TestValidateURLScheme:
    """Test suite for validate_url_scheme function."""

    def test_valid_http_url(self):
        """Test that valid HTTP URLs pass validation."""
        validate_url_scheme("http://example.com")
        validate_url_scheme("http://example.com/path")
        validate_url_scheme("http://example.com:8080/path?query=value")
        # Should not raise

    def test_valid_https_url(self):
        """Test that valid HTTPS URLs pass validation."""
        validate_url_scheme("https://example.com")
        validate_url_scheme("https://api.example.com/v1/users")
        validate_url_scheme("https://example.com:443/secure")
        # Should not raise

    def test_file_scheme_blocked(self):
        """Test that file:// URLs are blocked (CWE-22 prevention)."""
        with pytest.raises(URLValidationError) as exc_info:
            validate_url_scheme("file:///etc/passwd")
        assert "file" in str(exc_info.value).lower()
        assert "allowed" in str(exc_info.value).lower()

    def test_ftp_scheme_blocked(self):
        """Test that ftp:// URLs are blocked."""
        with pytest.raises(URLValidationError) as exc_info:
            validate_url_scheme("ftp://ftp.example.com/file.txt")
        assert "ftp" in str(exc_info.value).lower()

    def test_data_scheme_blocked(self):
        """Test that data:// URLs are blocked."""
        with pytest.raises(URLValidationError) as exc_info:
            validate_url_scheme("data:text/plain;base64,SGVsbG8=")
        assert "data" in str(exc_info.value).lower()

    def test_javascript_scheme_blocked(self):
        """Test that javascript:// URLs are blocked."""
        with pytest.raises(URLValidationError) as exc_info:
            validate_url_scheme("javascript:alert('XSS')")
        assert "javascript" in str(exc_info.value).lower()

    def test_empty_url(self):
        """Test that empty URLs are rejected."""
        with pytest.raises(URLValidationError) as exc_info:
            validate_url_scheme("")
        assert "empty" in str(exc_info.value).lower()

    def test_none_url(self):
        """Test that None is rejected."""
        with pytest.raises(URLValidationError):
            validate_url_scheme(None)

    def test_non_string_url(self):
        """Test that non-string URLs are rejected."""
        with pytest.raises(URLValidationError) as exc_info:
            validate_url_scheme(123)
        assert "string" in str(exc_info.value).lower()

    def test_url_without_scheme(self):
        """Test that URLs without schemes are rejected."""
        with pytest.raises(URLValidationError) as exc_info:
            validate_url_scheme("example.com/path")
        assert "scheme" in str(exc_info.value).lower()

    def test_url_without_netloc(self):
        """Test that URLs without network location are rejected."""
        with pytest.raises(URLValidationError) as exc_info:
            validate_url_scheme("http://")
        assert "host" in str(exc_info.value).lower()

    def test_custom_allowed_schemes(self):
        """Test that custom allowed schemes can be specified."""
        # Should pass with custom scheme
        validate_url_scheme("http://example.com", allowed_schemes={'http'})
        
        # Should fail with https when only http is allowed
        with pytest.raises(URLValidationError):
            validate_url_scheme("https://example.com", allowed_schemes={'http'})

    def test_case_sensitivity(self):
        """Test URL scheme case handling."""
        # Python's urlparse converts schemes to lowercase
        validate_url_scheme("HTTP://example.com")
        validate_url_scheme("HTTPS://example.com")

    def test_url_with_userinfo(self):
        """Test URLs with username:password in them."""
        validate_url_scheme("https://user:pass@example.com/path")
        # Should not raise

    def test_url_with_ipv4(self):
        """Test URLs with IPv4 addresses."""
        validate_url_scheme("http://192.168.1.1")
        validate_url_scheme("https://192.168.1.1:8080/api")

    def test_url_with_ipv6(self):
        """Test URLs with IPv6 addresses."""
        validate_url_scheme("http://[::1]")
        validate_url_scheme("https://[2001:db8::1]:8080/")

    def test_url_with_fragment(self):
        """Test URLs with fragments."""
        validate_url_scheme("https://example.com/page#section")


class TestSafeURLOpen:
    """Test suite for safe_urlopen function."""

    def test_safe_urlopen_validates_scheme(self):
        """Test that safe_urlopen validates URL schemes."""
        with pytest.raises(URLValidationError):
            safe_urlopen("file:///etc/passwd")

    def test_safe_urlopen_blocks_ftp(self):
        """Test that safe_urlopen blocks FTP URLs."""
        with pytest.raises(URLValidationError):
            safe_urlopen("ftp://example.com/file")

    def test_safe_urlopen_blocks_javascript(self):
        """Test that safe_urlopen blocks javascript URLs."""
        with pytest.raises(URLValidationError):
            safe_urlopen("javascript:alert(1)")

    def test_safe_urlopen_accepts_http(self):
        """Test that safe_urlopen accepts HTTP URLs (validation only, no network call)."""
        # We only test that validation passes, not actual network connection
        # This will fail with network error but not validation error
        try:
            safe_urlopen("http://0.0.0.0:1/", timeout=0.001)
        except URLValidationError:
            pytest.fail("Valid HTTP URL should not raise URLValidationError")
        except (urllib.error.URLError, OSError, TimeoutError):
            # Expected - network errors are fine, we just want to ensure validation passes
            pass

    def test_safe_urlopen_custom_schemes(self):
        """Test that safe_urlopen respects custom allowed schemes."""
        with pytest.raises(URLValidationError):
            safe_urlopen("https://example.com", allowed_schemes={'http'})


class TestSecurityScenarios:
    """Test suite for security-specific scenarios."""

    def test_prevents_local_file_access(self):
        """Test that local file access is prevented."""
        dangerous_urls = [
            "file:///etc/passwd",
            "file:///c:/windows/system32/config/sam",
            "file://localhost/etc/passwd",
        ]
        for url in dangerous_urls:
            with pytest.raises(URLValidationError):
                validate_url_scheme(url)

    def test_prevents_ssrf_attacks(self):
        """Test that SSRF-prone schemes are blocked."""
        ssrf_urls = [
            "ftp://internal.server/file",
            "gopher://localhost:70/",
            "dict://localhost:11211/",
        ]
        for url in ssrf_urls:
            with pytest.raises(URLValidationError):
                validate_url_scheme(url)

    def test_prevents_code_injection(self):
        """Test that code injection schemes are blocked."""
        injection_urls = [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "vbscript:msgbox(1)",
        ]
        for url in injection_urls:
            with pytest.raises(URLValidationError):
                validate_url_scheme(url)


class TestConstants:
    """Test suite for module constants."""

    def test_allowed_schemes_constant(self):
        """Test that ALLOWED_SCHEMES contains expected values."""
        assert ALLOWED_SCHEMES == {'http', 'https'}
        assert isinstance(ALLOWED_SCHEMES, set)


class TestCustomException:
    """Test suite for URLValidationError exception."""

    def test_url_validation_error_is_value_error(self):
        """Test that URLValidationError is a subclass of ValueError."""
        assert issubclass(URLValidationError, ValueError)

    def test_url_validation_error_message(self):
        """Test that URLValidationError preserves message."""
        message = "Test error message"
        error = URLValidationError(message)
        assert str(error) == message
