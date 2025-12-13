import os
import socket
import sys
import time
import urllib.request

# Import URL validation utility
from src.utils.url_validator import validate_url_scheme

HOST = os.getenv("ARENA_HOST", "127.0.0.1")
PORT = int(os.getenv("ARENA_PORT", "8181"))
URL  = f"http://{HOST}:{PORT}/api/metrics"

def wait_port(host, port, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False

def main():
    # If the server is already running (recommended), just probe it.
    if not wait_port(HOST, PORT, timeout=5.0):
        print(f"warn: port {HOST}:{PORT} not open; skipping live probe")
        sys.exit(0)
    try:
        # Validate URL scheme before making request
        validate_url_scheme(URL)
        
        with urllib.request.urlopen(URL, timeout=5) as r:
            if r.status != 200:
                print(f"error: metrics status {r.status}")
                sys.exit(2)
            body = r.read().decode("utf-8", errors="ignore")
            if "python_info" not in body:
                print("error: metrics body missing python_info")
                sys.exit(3)
        print("ok: metrics healthy")
        sys.exit(0)
    except Exception as e:
        print(f"error: metrics probe failed: {e}")
        sys.exit(4)

if __name__ == "__main__":
    main()
