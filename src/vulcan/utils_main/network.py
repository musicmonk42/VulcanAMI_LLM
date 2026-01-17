# ============================================================
# VULCAN-AGI Network Utilities Module
# Network-related utility functions
# ============================================================
#
# This module provides:
#     - Port availability checking
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
#     1.0.2 - Removed unused network utilities (is_port_open, find_available_port_range, 
#             get_local_ip, get_hostname, resolve_hostname, is_port_available)
# ============================================================

import logging
import socket

# Module metadata
__version__ = "1.0.2"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Maximum number of ports to try when searching for available port
MAX_PORT_SEARCH_ATTEMPTS = 100


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





# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Port availability
    "find_available_port",
    # Constants
    "MAX_PORT_SEARCH_ATTEMPTS",
]


# Log module initialization
logger.debug(f"Network utilities module v{__version__} loaded")
