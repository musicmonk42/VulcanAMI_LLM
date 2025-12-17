"""
Minimal gRPC stub for Registry Service.

This module provides a minimal implementation to allow the registry_api_server
to start without requiring compiled protobuf files. It creates a basic gRPC
service registration that works with the simplified Python classes in
registry_api_server.py.
"""

import grpc
import logging
from concurrent import futures

logger = logging.getLogger(__name__)


def add_RegistryServiceServicer_to_server(servicer, server):
    """
    Register the Registry Service with the gRPC server.
    
    This is a minimal stub implementation that allows the server to start
    without compiled .proto files. The actual RPC methods are defined in
    the RegistryServicer class in registry_api_server.py.
    
    Args:
        servicer: The RegistryServicer instance
        server: The gRPC server instance
    """
    # Create generic handlers for each RPC method
    # These handlers route incoming RPC calls to the servicer methods
    
    def generic_handler(request, context, method_name):
        """Generic handler that routes calls to servicer methods."""
        try:
            method = getattr(servicer, method_name)
            return method(request, context)
        except Exception as e:
            logger.error(f"Error in {method_name}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            # Return a simple error response
            return type('ErrorResponse', (), {'status': 'error', 'message': str(e)})()
    
    # Create RPC method handlers
    rpc_method_handlers = {
        'RegisterGraphProposal': grpc.unary_unary_rpc_method_handler(
            lambda request, context: generic_handler(request, context, 'RegisterGraphProposal'),
            request_deserializer=None,
            response_serializer=None,
        ),
        'SubmitLanguageEvolutionProposal': grpc.unary_unary_rpc_method_handler(
            lambda request, context: generic_handler(request, context, 'SubmitLanguageEvolutionProposal'),
            request_deserializer=None,
            response_serializer=None,
        ),
        'RecordVote': grpc.unary_unary_rpc_method_handler(
            lambda request, context: generic_handler(request, context, 'RecordVote'),
            request_deserializer=None,
            response_serializer=None,
        ),
        'RecordValidation': grpc.unary_unary_rpc_method_handler(
            lambda request, context: generic_handler(request, context, 'RecordValidation'),
            request_deserializer=None,
            response_serializer=None,
        ),
        'DeployGrammarVersion': grpc.unary_unary_rpc_method_handler(
            lambda request, context: generic_handler(request, context, 'DeployGrammarVersion'),
            request_deserializer=None,
            response_serializer=None,
        ),
        'QueryProposals': grpc.unary_unary_rpc_method_handler(
            lambda request, context: generic_handler(request, context, 'QueryProposals'),
            request_deserializer=None,
            response_serializer=None,
        ),
        'GetFullAuditLog': grpc.unary_unary_rpc_method_handler(
            lambda request, context: generic_handler(request, context, 'GetFullAuditLog'),
            request_deserializer=None,
            response_serializer=None,
        ),
        'VerifyAuditLogIntegrity': grpc.unary_unary_rpc_method_handler(
            lambda request, context: generic_handler(request, context, 'VerifyAuditLogIntegrity'),
            request_deserializer=None,
            response_serializer=None,
        ),
    }
    
    # Create generic RPC handler
    generic_rpc_handler = grpc.method_handlers_generic_handler(
        'registry.RegistryService',
        rpc_method_handlers
    )
    
    # Add the handler to the server
    server.add_generic_rpc_handlers((generic_rpc_handler,))
    
    logger.info("Registry Service registered with gRPC server (using stub implementation)")
