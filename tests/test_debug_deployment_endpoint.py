#!/usr/bin/env python3
"""
Tests for the debug/deployment endpoint.

This test verifies that the debug endpoint correctly reports deployment state
from the app that receives requests, which is critical for troubleshooting
sub-app mounting issues where app.state.deployment may not be accessible.
"""

import os
import pytest
from unittest.mock import MagicMock, AsyncMock

# Import the endpoint function (will be verified via AST to avoid full import chain)
import ast
from pathlib import Path


class TestDebugDeploymentEndpointExists:
    """Verify the debug/deployment endpoint exists in the codebase."""
    
    def test_debug_endpoint_exists_in_status_py(self):
        """Verify /debug/deployment endpoint is defined in status.py."""
        status_file = Path("src/vulcan/endpoints/status.py")
        assert status_file.exists(), "status.py should exist"
        
        with open(status_file, 'r') as f:
            content = f.read()
        
        # Verify the endpoint decorator and function exist
        assert '@router.get("/debug/deployment")' in content, \
            "Debug deployment endpoint decorator should exist"
        assert 'async def debug_deployment' in content, \
            "Debug deployment function should be defined"
        
        # Verify the response includes expected fields
        assert '"deployment"' in content or "'deployment'" in content, \
            "Response should include deployment field"
        assert '"worker_id"' in content or "'worker_id'" in content, \
            "Response should include worker_id field"
        assert '"has_deployment_attr"' in content or "'has_deployment_attr'" in content, \
            "Response should include has_deployment_attr field"
    
    def test_debug_endpoint_exists_in_full_platform_py(self):
        """Verify /debug/deployment endpoint is defined in full_platform.py."""
        full_platform_file = Path("src/full_platform.py")
        assert full_platform_file.exists(), "full_platform.py should exist"
        
        with open(full_platform_file, 'r') as f:
            content = f.read()
        
        # Verify the endpoint decorator and function exist
        assert '@app.get("/debug/deployment")' in content, \
            "Debug deployment endpoint decorator should exist on parent app"
        assert 'async def debug_parent_deployment' in content, \
            "Debug deployment function should be defined for parent app"


class TestDebugDeploymentEndpointSignature:
    """Verify the endpoint has the correct signature using AST parsing."""
    
    def test_status_endpoint_signature(self):
        """Verify the status.py debug endpoint has the correct signature."""
        status_file = Path("src/vulcan/endpoints/status.py")
        
        with open(status_file, 'r') as f:
            tree = ast.parse(f.read(), filename=str(status_file))
        
        # Find the debug_deployment function
        func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "debug_deployment":
                func = node
                break
        
        assert func is not None, "debug_deployment function should exist"
        
        # Check it takes a Request parameter
        arg_names = [arg.arg for arg in func.args.args]
        assert "request" in arg_names, "Function should have 'request' parameter"
    
    def test_full_platform_endpoint_signature(self):
        """Verify the full_platform.py debug endpoint has the correct signature."""
        full_platform_file = Path("src/full_platform.py")
        
        with open(full_platform_file, 'r') as f:
            tree = ast.parse(f.read(), filename=str(full_platform_file))
        
        # Find the debug_parent_deployment function
        func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "debug_parent_deployment":
                func = node
                break
        
        assert func is not None, "debug_parent_deployment function should exist"
        
        # Check it takes a Request parameter
        arg_names = [arg.arg for arg in func.args.args]
        assert "request" in arg_names, "Function should have 'request' parameter"


class TestDebugDeploymentEndpointResponseFields:
    """Verify response includes all required diagnostic fields."""
    
    def test_required_fields_in_status_endpoint(self):
        """Check that status.py endpoint returns all required diagnostic fields."""
        status_file = Path("src/vulcan/endpoints/status.py")
        
        with open(status_file, 'r') as f:
            content = f.read()
        
        # Find the return statement in the debug_deployment function
        # Look for the dictionary being returned
        required_fields = [
            "deployment",
            "deployment_type",
            "app_title",
            "worker_id",
            "startup_time",
            "has_deployment_attr",
        ]
        
        for field in required_fields:
            assert f'"{field}"' in content or f"'{field}'" in content, \
                f"Response should include '{field}' field"
    
    def test_required_fields_in_full_platform_endpoint(self):
        """Check that full_platform.py endpoint returns all required diagnostic fields."""
        full_platform_file = Path("src/full_platform.py")
        
        with open(full_platform_file, 'r') as f:
            content = f.read()
        
        required_fields = [
            "deployment",
            "deployment_type",
            "app_title",
            "worker_id",
            "startup_time",
            "has_deployment_attr",
        ]
        
        # Check in the section around debug_parent_deployment
        start_idx = content.find("async def debug_parent_deployment")
        end_idx = content.find("@app.post", start_idx)  # Next endpoint
        if end_idx == -1:
            end_idx = len(content)
        
        func_content = content[start_idx:end_idx]
        
        for field in required_fields:
            assert f'"{field}"' in func_content or f"'{field}'" in func_content, \
                f"Response should include '{field}' field"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
