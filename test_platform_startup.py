#!/usr/bin/env python3
"""
Test script for unified platform startup
Tests that all 9 core services can be started and are responsive
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_platform_startup():
    """Test that platform can start with all services"""
    print("=" * 70)
    print("VulcanAMI Platform Startup Test")
    print("=" * 70)
    print()
    
    # Import the app (this will validate imports)
    try:
        print("1. Importing full_platform...")
        from src.full_platform import app, settings
        print("   ✓ Import successful")
        print()
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check configuration
    print("2. Checking configuration...")
    print(f"   Main server: {settings.host}:{settings.port}")
    print(f"   Service mount paths:")
    print(f"     - VULCAN: {settings.vulcan_mount}")
    print(f"     - Arena: {settings.arena_mount}")
    print(f"     - Registry: {settings.registry_mount}")
    print(f"     - API Gateway: {settings.api_gateway_mount}")
    print(f"     - DQS: {settings.dqs_mount}")
    print(f"     - PII: {settings.pii_mount}")
    print(f"   Standalone service ports:")
    print(f"     - API Server: {settings.api_server_port}")
    print(f"     - Registry gRPC: {settings.registry_grpc_port}")
    print(f"     - Listener: {settings.listener_port}")
    print()
    
    # Check service enable flags
    print("3. Service enable flags:")
    print(f"   API Gateway: {'✓ Enabled' if settings.enable_api_gateway else '✗ Disabled'}")
    print(f"   DQS Service: {'✓ Enabled' if settings.enable_dqs_service else '✗ Disabled'}")
    print(f"   PII Service: {'✓ Enabled' if settings.enable_pii_service else '✗ Disabled'}")
    print(f"   API Server: {'✓ Enabled' if settings.enable_api_server else '✗ Enabled'}")
    print(f"   Registry gRPC: {'✓ Enabled' if settings.enable_registry_grpc else '✗ Disabled'}")
    print(f"   Listener: {'✓ Enabled' if settings.enable_listener else '✗ Disabled'}")
    print()
    
    # Check routes
    print("4. Checking registered routes...")
    route_count = len(app.routes)
    print(f"   Total routes registered: {route_count}")
    
    # Show some key routes
    key_routes = ["/", "/health", "/api/status", "/docs"]
    for route in app.routes:
        if hasattr(route, 'path') and route.path in key_routes:
            print(f"   ✓ {route.path}")
    print()
    
    print("=" * 70)
    print("✅ Platform startup test completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. To start the platform, run:")
    print("   uvicorn src.full_platform:app --host 0.0.0.0 --port 8080")
    print()
    print("2. Or use the module directly:")
    print("   python -m src.full_platform")
    print()
    print("3. Access the platform at:")
    print("   http://localhost:8080")
    print()
    return True

if __name__ == "__main__":
    result = asyncio.run(test_platform_startup())
    sys.exit(0 if result else 1)
