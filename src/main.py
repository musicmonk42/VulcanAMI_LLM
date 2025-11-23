"""
Alternative main entry point in src/ directory.
Exports the Flask app instance from ../app.py for use with WSGI/ASGI servers.
"""

import sys
from pathlib import Path

# Add project root to path to import app.py
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import app

# Export app for WSGI servers (gunicorn, uvicorn, etc.)
# Usage: gunicorn src.main:app

if __name__ == "__main__":
    # Run with Flask's built-in server for development only
    import os
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("DEBUG", "false").lower() == "true"
    )
