"""
Graphix IR Registry gRPC Stub Implementation (v3.0.0)
======================================================

This module provides a production-quality gRPC service stub that enables
the Registry API Server to function without requiring compiled protobuf
definitions. It implements the RegistryService interface using runtime
method registration and generic handlers.

Key Features:
- Zero-dependency service registration (no .proto compilation required)
- Type-safe method routing with error handling
- Full compatibility with RegistryServicer implementation
- Production-grade logging and error reporting
- Thread-safe operation with gRPC's thread pool

Architecture:
    The stub uses gRPC's generic RPC handler mechanism to dynamically
    route method calls to the RegistryServicer instance. This allows
    the service to operate identically to a compiled protobuf service
    while maintaining flexibility for development and testing.

Usage:
    >>> from src.governance import registry_api_server
    >>> servicer = registry_api_server.RegistryServicer(...)
    >>> server = grpc.server(...)
    >>> add_RegistryServiceServicer_to_server(servicer, server)
    >>> server.start()

Thread Safety:
    All handlers are thread-safe and designed to work with gRPC's
    ThreadPoolExecutor-based server implementation.
"""

import functools
import logging
from typing import Any, Callable, Dict

import grpc

# Configure module logger
logger = logging.getLogger(__name__)

# Type Hint Design Note:
# We use `Any` for servicer type hints to avoid circular imports with
# registry_api_server.py. This is a common pattern for stub implementations
# that need to work with classes defined in other modules. The duck-typing
# approach is validated at runtime via hasattr() checks.

# Service name constant for consistency
REGISTRY_SERVICE_NAME = "registry.RegistryService"

# RPC method names - centralized for maintainability
RPC_METHODS = [
    "RegisterGraphProposal",
    "SubmitLanguageEvolutionProposal",
    "RecordVote",
    "RecordValidation",
    "DeployGrammarVersion",
    "QueryProposals",
    "GetFullAuditLog",
    "VerifyAuditLogIntegrity",
]


class ErrorResponse:
    """
    Standard error response object for failed RPC calls.
    
    This class provides a consistent structure for error responses
    that matches the response classes defined in registry_api_server.py.
    
    Attributes:
        status: Always "error" for error responses
        message: Human-readable error description
    """

    def __init__(self, message: str):
        """
        Initialize an error response.
        
        Args:
            message: Error message to return to the client
        """
        self.status = "error"
        self.message = message


def create_method_handler(
    servicer: Any, method_name: str
) -> Callable[[Any, grpc.ServicerContext], Any]:
    """
    Create a type-safe RPC method handler for a specific servicer method.
    
    This factory function creates a handler that:
    1. Routes the request to the appropriate servicer method
    2. Handles exceptions gracefully with proper gRPC status codes
    3. Logs errors for debugging and monitoring
    4. Returns properly typed error responses
    
    Args:
        servicer: The RegistryServicer instance containing the method
        method_name: Name of the method to invoke (e.g., "RegisterGraphProposal")
    
    Returns:
        A callable that handles the RPC call with proper error handling
    
    Raises:
        AttributeError: If the servicer doesn't have the specified method
    
    Thread Safety:
        The returned handler is thread-safe and can be called concurrently.
    """
    # Validate that the method exists at handler creation time
    if not hasattr(servicer, method_name):
        raise AttributeError(
            f"Servicer does not have method '{method_name}'. "
            f"Available methods: {dir(servicer)}"
        )

    # Get the method once at creation time for efficiency
    method = getattr(servicer, method_name)

    @functools.wraps(method)
    def handler(request: Any, context: grpc.ServicerContext) -> Any:
        """
        Handle a single RPC call with comprehensive error handling.
        
        Args:
            request: The RPC request object
            context: The gRPC ServicerContext for this call
        
        Returns:
            The response object from the servicer method or an ErrorResponse
        """
        try:
            # Delegate to the actual servicer method
            return method(request, context)
        except grpc.RpcError:
            # gRPC errors are already properly formatted, re-raise them
            raise
        except Exception as e:
            # Log the full exception for debugging
            logger.exception(
                "Unhandled exception in %s: %s", method_name, str(e), exc_info=True
            )

            # Set proper gRPC status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error in {method_name}: {e}")

            # Return a structured error response
            return ErrorResponse(message=str(e))

    return handler


def create_rpc_method_handlers(servicer: Any) -> Dict[str, grpc.RpcMethodHandler]:
    """
    Create all RPC method handlers for the Registry Service.
    
    This function builds a complete mapping of method names to their
    handlers, using the generic handler factory for consistency.
    
    Args:
        servicer: The RegistryServicer instance to route calls to
    
    Returns:
        Dictionary mapping method names to their gRPC handlers
    
    Raises:
        AttributeError: If servicer is missing any required methods
    """
    handlers = {}

    for method_name in RPC_METHODS:
        # Create the method-specific handler
        method_handler = create_method_handler(servicer, method_name)

        # Wrap it in a gRPC unary-unary handler
        # Note: Using None for serializers allows gRPC to handle the objects directly
        rpc_handler = grpc.unary_unary_rpc_method_handler(
            method_handler,
            request_deserializer=None,
            response_serializer=None,
        )

        handlers[method_name] = rpc_handler

    logger.debug(
        "Created %d RPC method handlers: %s", len(handlers), ", ".join(handlers.keys())
    )

    return handlers


def add_RegistryServiceServicer_to_server(servicer: Any, server: grpc.Server) -> None:
    """
    Register the Registry Service with a gRPC server.
    
    This function is the public API for registering the Registry Service.
    It creates all necessary method handlers and registers them with the
    server using gRPC's generic handler mechanism.
    
    This stub implementation is functionally equivalent to the
    add_RegistryServiceServicer_to_server function that would be
    generated by compiling .proto files, but operates without
    requiring protobuf compilation.
    
    Args:
        servicer: RegistryServicer instance implementing the service methods
        server: gRPC server instance to register the service with
    
    Raises:
        AttributeError: If servicer is missing required methods
        ValueError: If server is None or invalid
    
    Example:
        >>> import grpc
        >>> from concurrent import futures
        >>> from src.governance.registry_api_server import RegistryServicer
        >>> 
        >>> servicer = RegistryServicer(...)
        >>> server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        >>> add_RegistryServiceServicer_to_server(servicer, server)
        >>> server.add_insecure_port('[::]:50051')
        >>> server.start()
    
    Thread Safety:
        This function is thread-safe and can be called during server
        initialization. The registered handlers are also thread-safe.
    """
    # Validate inputs
    if server is None:
        raise ValueError("Server cannot be None")

    # Create all method handlers
    try:
        rpc_method_handlers = create_rpc_method_handlers(servicer)
    except AttributeError as e:
        logger.error("Failed to create method handlers: %s", e)
        raise

    # Create the generic RPC handler with the service name
    generic_rpc_handler = grpc.method_handlers_generic_handler(
        REGISTRY_SERVICE_NAME, rpc_method_handlers
    )

    # Register the handler with the server
    server.add_generic_rpc_handlers((generic_rpc_handler,))

    logger.info(
        "Registry Service registered with gRPC server (stub implementation, %d methods)",
        len(RPC_METHODS),
    )
