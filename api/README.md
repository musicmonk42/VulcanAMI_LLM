# Swagger UI (local viewer)

Serve the static Swagger UI and the bundled `swagger.yml` from this folder:

1) From the `api/` directory, start a simple HTTP server:
   ```bash
   python3 -m http.server 8000
   ```
2) Open the UI in your browser:
   ```
   http://localhost:8000/index.html
   ```

Notes:
- `index.html` loads `swagger.yml` from the same directory; keep them together or update the URL inside `index.html`.
- Stop the server with `Ctrl+C` when done.

