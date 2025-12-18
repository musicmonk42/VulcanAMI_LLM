#!/bin/bash
# =============================================================================
# VULCAN Chat Interface Startup Script
# =============================================================================
# This script starts the VulcanAMI platform and opens the chat interface
# in the default browser.
# =============================================================================

set -e

echo "========================================"
echo "  VULCAN Chat Interface Startup Script"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to repository root
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "Using Python: $($PYTHON --version)"
echo ""

# Check for required dependencies
echo "Checking dependencies..."
$PYTHON -c "import fastapi, uvicorn, pydantic" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install fastapi uvicorn pydantic pydantic-settings
}
echo "✓ Dependencies OK"
echo ""

# Start the platform
echo "Starting VulcanAMI platform..."
echo "Platform will be available at http://localhost:8080"
echo ""

# Start platform in background
$PYTHON src/full_platform.py &
PLATFORM_PID=$!

# Save PID for cleanup
echo $PLATFORM_PID > /tmp/vulcan_platform.pid

# Wait for platform to initialize
echo "Waiting for platform to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✓ Platform is ready!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 1
done

# Check if platform started successfully
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "⚠️  Platform may still be starting. Please wait a moment before using the chat interface."
fi

echo ""
echo "========================================"
echo "  VULCAN Chat Interface Running!"
echo "========================================"
echo ""
echo "Backend URLs:"
echo "  Main Platform:  http://localhost:8080"
echo "  Platform Docs:  http://localhost:8080/docs"
echo "  Health Check:   http://localhost:8080/health"
echo ""
echo "Chat Endpoints:"
echo "  VULCAN Chat:    http://localhost:8080/vulcan/v1/chat"
echo "  Chat Endpoint:  http://localhost:8080/chat/v1/chat"
echo ""
echo "Frontend:"
echo "  Open vulcan_chat.html in your browser"
echo ""
echo "Press Ctrl+C to stop the platform..."
echo ""

# Open the HTML file in browser (platform-specific)
if [ -f "vulcan_chat.html" ]; then
    echo "Opening chat interface in browser..."
    if command -v xdg-open > /dev/null; then
        # Linux
        xdg-open vulcan_chat.html 2>/dev/null &
    elif command -v open > /dev/null; then
        # macOS
        open vulcan_chat.html 2>/dev/null &
    elif command -v start > /dev/null; then
        # Windows (Git Bash)
        start vulcan_chat.html 2>/dev/null &
    else
        echo "Please open vulcan_chat.html manually in your browser"
    fi
else
    echo "Note: vulcan_chat.html not found in current directory"
    echo "Please open your chat HTML file in a browser"
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down platform..."
    if [ -f /tmp/vulcan_platform.pid ]; then
        kill $(cat /tmp/vulcan_platform.pid) 2>/dev/null || true
        rm /tmp/vulcan_platform.pid
    fi
    # Kill any remaining processes
    kill $PLATFORM_PID 2>/dev/null || true
    echo "✓ Platform stopped"
    exit 0
}

# Set up signal handlers
trap cleanup INT TERM

# Wait for platform process
wait $PLATFORM_PID
