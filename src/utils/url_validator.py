"""
URL Security Validation Module
===============================

This module provides secure URL validation utilities to prevent path traversal (CWE-22)
and scheme-based attacks. It ensures that only HTTP/HTTPS URLs are processed, blocking
potentially dangerous schemes like file://, ftp://, data://, etc.

Security Considerations:
    - Validates URL schemes against an allowlist (http, https only)
    - Ensures the presence of a valid network location (netloc)
    - Prevents file:// access and other local file system attacks
    - Guards against SSRF attacks using non-standard schemes

Version: 1.0.0
Author: VulcanAMI Security Team
"""

import logging
import urllib.parse
import urllib.request
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Allowlist of permitted URL schemes
ALLOWED_SCHEMES: Set[str] = {'http', 'https'}


class URLValidationError(ValueError):
    """Exception raised when URL validation fails.
    
    This is a specialized ValueError that provides clear context
    about URL validation failures for security logging and monitoring.
    """
    pass


def validate_url_scheme(url: str, allowed_schemes: Optional[Set[str]] = None) -> None:
    """
    Validate that a URL uses only allowed schemes (default: http, https).
    
    This function prevents security vulnerabilities related to improper URL handling:
    - CWE-22: Path Traversal (prevents file:// scheme access)
    - CWE-918: Server-Side Request Forgery (blocks non-HTTP schemes)
    
    Args:
        url: The URL string to validate
        allowed_schemes: Optional set of allowed schemes. Defaults to ALLOWED_SCHEMES (http, https)
        
    Raises:
        URLValidationError: If the URL scheme is not in the allowlist
        URLValidationError: If the URL is malformed or missing required components
        
    Example:
        >>> validate_url_scheme("https://api.example.com/data")
        >>> validate_url_scheme("file:///etc/passwd")  # Raises URLValidationError
        
    Note:
        This function is designed to be called before any urllib.request.urlopen()
        or similar network operations to ensure URL safety.
    """
    if allowed_schemes is None:
        allowed_schemes = ALLOWED_SCHEMES
    
    if not url:
        raise URLValidationError("URL cannot be empty")
    
    if not isinstance(url, str):
        raise URLValidationError(f"URL must be a string, got {type(url).__name__}")
    
    try:
        parsed = urllib.parse.urlparse(url)
        
        # Check scheme is present and allowed
        if not parsed.scheme:
            raise URLValidationError("URL must include a scheme (e.g., http:// or https://)")
        
        if parsed.scheme not in allowed_schemes:
            logger.warning(
                f"Blocked URL with disallowed scheme: {parsed.scheme}",
                extra={"url_scheme": parsed.scheme, "url_netloc": parsed.netloc}
            )
            raise URLValidationError(
                f"Unsupported URL scheme: '{parsed.scheme}'. "
                f"Only {', '.join(sorted(allowed_schemes))} are allowed."
            )
        
        # Ensure netloc (host) is present
        if not parsed.netloc:
            raise URLValidationError(
                "URL must include a valid host/domain. "
                f"Got: {url}"
            )
        
        logger.debug(f"URL validation passed: {parsed.scheme}://{parsed.netloc}")
            
    except URLValidationError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Catch any other parsing errors and wrap them
        logger.error(f"URL parsing failed: {e}", extra={"url": url})
        raise URLValidationError(f"Invalid URL format: {e}") from e


def safe_urlopen(url: str, allowed_schemes: Optional[Set[str]] = None, **kwargs):
    """
    Safely open a URL after validating its scheme.
    
    This is a drop-in replacement for urllib.request.urlopen() that adds
    security validation before opening the connection.
    
    Args:
        url: The URL to open
        allowed_schemes: Optional set of allowed schemes. Defaults to ALLOWED_SCHEMES
        **kwargs: Additional arguments to pass to urllib.request.urlopen
                 (e.g., timeout, context, data, headers)
        
    Returns:
        The response object from urllib.request.urlopen
        
    Raises:
        URLValidationError: If the URL scheme is not allowed
        urllib.error.URLError: For network-related errors (from urlopen)
        urllib.error.HTTPError: For HTTP protocol errors (from urlopen)
        
    Example:
        >>> response = safe_urlopen("https://api.example.com/data", timeout=10)
        >>> data = response.read()
        
    Note:
        This function should be used instead of urllib.request.urlopen() directly
        to ensure all URL access is properly validated.
    """
    validate_url_scheme(url, allowed_schemes)
    
    try:
        return urllib.request.urlopen(url, **kwargs)
    except urllib.error.HTTPError as e:
        logger.warning(
            f"HTTP error for URL: {url}",
            extra={"status_code": e.code, "reason": e.reason}
        )
        raise
    except urllib.error.URLError as e:
        logger.error(
            f"URL error for URL: {url}",
            extra={"reason": str(e.reason)}
        )
        raise
    except Exception as e:
        logger.error(f"Unexpected error opening URL: {url}", exc_info=True)
        raise
