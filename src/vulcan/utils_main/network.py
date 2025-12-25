# ============================================================
# VULCAN-AGI Network Utilities Module
# Network-related utility functions
# ============================================================
#
# This module provides:
#     - Port availability checking
#     - Port scanning utilities
#     - Network configuration helpers
#
# USAGE:
#     from vulcan.utils_main.network import find_available_port
#     
#     # Find an available port starting from 8080
#     port = find_available_port("127.0.0.1", 8080)
#
# VERSION HISTORY:
#     1.0.0 - Extracted from main.py for modular architecture
#     1.0.1 - Added comprehensive documentation and additional utilities
# ============================================================

import logging
import socket
from typing import List, Optional, Tuple

# Module metadata
__version__ = "1.0.1"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Maximum number of ports to try when searching for available port
MAX_PORT_SEARCH_ATTEMPTS = 100

# Default timeout for connection tests (seconds)
DEFAULT_CONNECTION_TIMEOUT = 1.0


# ============================================================
# PORT AVAILABILITY FUNCTIONS
# ============================================================


def find_available_port(host: str, port: int, max_attempts: int = MAX_PORT_SEARCH_ATTEMPTS) -> int:
    """
    Find an available port starting from the specified port.
    
    Checks if a port is in use. If it is, increments until a free port is found.
    
    Args:
        host: The host address to bind to (e.g., "127.0.0.1", "0.0.0.0")
        port: The starting port number to check
        max_attempts: Maximum number of ports to try (default: 100)
        
    Returns:
        An available port number
        
    Raises:
        RuntimeError: If no available port is found within max_attempts
        OSError: If an unexpected socket error occurs
        
    Example:
        >>> port = find_available_port("127.0.0.1", 8080)
        >>> print(f"Using port: {port}")
    """
    original_port = port
    
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                logger.info(f"Port {port} is available.")
                return port
        except OSError as e:
            # Check for common "address in use" errors across platforms
            if (
                e.errno == 98  # EADDRINUSE on Linux
                or e.errno == 48  # EADDRINUSE on macOS
                or e.errno == 10048  # WSAEADDRINUSE on Windows
                or "Address already in use" in str(e)
                or "only one usage" in str(e)
            ):
                logger.warning(f"Port {port} is already in use. Trying next port...")
                port += 1
                if port - original_port > max_attempts:
                    error_msg = (
                        f"Could not find an available port after {max_attempts} "
                        f"attempts from base port {original_port}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                logger.error(f"Unexpected socket error: {e}")
                raise


def is_port_available(host: str, port: int) -> bool:
    """
    Check if a specific port is available.
    
    Args:
        host: The host address to check
        port: The port number to check
        
    Returns:
        True if the port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def is_port_open(host: str, port: int, timeout: float = DEFAULT_CONNECTION_TIMEOUT) -> bool:
    """
    Check if a port is open and accepting connections.
    
    This is different from is_port_available - this checks if something
    is listening on the port.
    
    Args:
        host: The host address to check
        port: The port number to check
        timeout: Connection timeout in seconds
        
    Returns:
        True if the port is open and accepting connections
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            return result == 0
    except (socket.timeout, socket.error):
        return False


def find_available_port_range(
    host: str,
    start_port: int,
    count: int,
    max_attempts: int = MAX_PORT_SEARCH_ATTEMPTS
) -> List[int]:
    """
    Find multiple consecutive available ports.
    
    Args:
        host: The host address to bind to
        start_port: The starting port number
        count: Number of consecutive ports needed
        max_attempts: Maximum number of starting positions to try
        
    Returns:
        List of available port numbers
        
    Raises:
        RuntimeError: If no suitable port range is found
    """
    attempts = 0
    current_start = start_port
    
    while attempts < max_attempts:
        ports = list(range(current_start, current_start + count))
        if all(is_port_available(host, p) for p in ports):
            logger.info(f"Found available port range: {ports[0]}-{ports[-1]}")
            return ports
        current_start += 1
        attempts += 1
    
    raise RuntimeError(
        f"Could not find {count} consecutive available ports starting from {start_port}"
    )


# ============================================================
# NETWORK INFORMATION FUNCTIONS
# ============================================================


def get_local_ip() -> str:
    """
    Get the local IP address of this machine.
    
    Returns:
        Local IP address string (e.g., "192.168.1.100")
    """
    try:
        # Create a socket to determine the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to an external address (doesn't actually send data)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def get_hostname() -> str:
    """
    Get the hostname of this machine.
    
    Returns:
        Hostname string
    """
    return socket.gethostname()


def resolve_hostname(hostname: str) -> Optional[str]:
    """
    Resolve a hostname to an IP address.
    
    Args:
        hostname: The hostname to resolve
        
    Returns:
        IP address string, or None if resolution fails
    """
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        logger.warning(f"Could not resolve hostname: {hostname}")
        return None


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Port availability
    "find_available_port",
    "is_port_available",
    "is_port_open",
    "find_available_port_range",
    # Network info
    "get_local_ip",
    "get_hostname",
    "resolve_hostname",
    # Constants
    "MAX_PORT_SEARCH_ATTEMPTS",
    "DEFAULT_CONNECTION_TIMEOUT",
]


# Log module initialization
logger.debug(f"Network utilities module v{__version__} loaded")
