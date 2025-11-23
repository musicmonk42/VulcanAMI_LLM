"""
Main entry point module for WSGI/ASGI deployment.
Exports the Flask app instance from app.py for use with gunicorn, uvicorn, etc.
"""

from app import app

# Export app for WSGI servers (gunicorn, uwsgi, etc.)
# Usage: gunicorn main:app
# or: uwsgi --http :5000 --wsgi-file main.py --callable app

if __name__ == "__main__":
    # Run with Flask's built-in server for development only
    import os
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("DEBUG", "false").lower() == "true"
    )
