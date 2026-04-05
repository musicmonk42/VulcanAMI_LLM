"""
Health check route handlers extracted from full_platform.py.

Provides:
- root (GET /)
- status_page (GET /status)
- health_live (GET /health/live)
- health_ready (GET /health/ready)
- health_check (GET /health)
"""

import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

router = APIRouter()


@router.get("/")
async def root():
    """
    Root endpoint serves the chat interface (static/index.html).
    For platform status, visit /status instead.
    """
    static_index = Path(__file__).parent.parent.parent / "static" / "index.html"
    if static_index.exists():
        return FileResponse(static_index, media_type="text/html")
    return HTMLResponse(
        content="<h1>Chat interface not found</h1><p><code>static/index.html</code> was not found. Please check your installation.</p>",
        status_code=404
    )


@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request):
    """
    Status page with flash messaging, live service health, and API explorer.
    """
    # Late imports to avoid circular dependencies at module load time
    from src.full_platform import flash_manager, service_manager, settings

    base_url = f"http://{request.url.hostname}:{request.url.port or settings.port}"

    service_status = await service_manager.get_service_status()
    recent_messages = await flash_manager.get_recent_messages(limit=5)

    health_checks = {}
    if settings.enable_health_checks:
        health_checks = await service_manager.check_all_health(base_url)

    flash_html = ""
    if recent_messages:
        flash_html = '<div class="flash-section">'
        for msg in reversed(recent_messages):
            level_colors = {
                "error": "#dc3545",
                "warning": "#ffc107",
                "info": "#17a2b8",
                "success": "#28a745",
            }
            color = level_colors.get(msg["level"], "#6c757d")
            flash_html += f"""
            <div class="flash-message" style="border-left: 4px solid {color};">
                <strong style="color: {color};">{msg["level"].upper()}</strong>: {msg["message"]}
                {f'<div class="flash-details">{msg["details"]}</div>' if msg.get("details") else ""}
                <div class="flash-timestamp">{msg["timestamp"]}</div>
            </div>
            """
        flash_html += "</div>"

    try:
        prometheus_available = settings.enable_metrics
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    except Exception:
        prometheus_available = False

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Graphix Vulcan Unified Platform v2.1</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .flash-section {{
                margin: 20px 0;
            }}
            .flash-message {{
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .flash-details {{
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }}
            .flash-timestamp {{
                font-size: 0.8em;
                color: #999;
                margin-top: 5px;
            }}
            .service-card {{
                background: white;
                padding: 20px;
                margin: 10px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .status-mounted {{ color: #28a745; }}
            .status-failed {{ color: #dc3545; }}
            .status-healthy {{ color: #28a745; }}
            .status-unhealthy {{ color: #ffc107; }}
            .links {{ margin-top: 10px; }}
            .links a {{
                display: inline-block;
                margin-right: 15px;
                padding: 8px 15px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }}
            .error {{ color: #dc3545; font-size: 0.9em; }}
            .badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 0.85em;
                font-weight: bold;
            }}
            .badge-success {{ background: #28a745; color: white; }}
            .badge-danger {{ background: #dc3545; color: white; }}
            .badge-warning {{ background: #ffc107; color: black; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Graphix Vulcan Unified Platform</h1>
            <p>Production-Hardened Enterprise Platform (v2.1)</p>
            <p><strong>Worker PID:</strong> {os.getpid()} | <strong>Workers:</strong> {settings.workers}</p>
            <p><strong>Auth:</strong> {settings.auth_method.value.upper()} |
               <strong>Metrics:</strong> {"Enabled" if settings.enable_metrics and prometheus_available else "Disabled"}</p>
        </div>

        {flash_html}

        <div class="service-card">
            <h2>Platform Status</h2>
            <p><strong>Status:</strong> <span class="badge badge-success">ACTIVE</span></p>
            <div class="links">
                <a href="/docs">API Documentation</a>
                <a href="/health">Health Check</a>
                <a href="/api/status">JSON Status</a>
                {f'<a href="{settings.metrics_path}">Metrics</a>' if settings.enable_metrics and prometheus_available else ""}
                <a href="/auth/token">Get Token</a>
            </div>
        </div>
    """

    for name, status in service_status.items():
        mounted = status.get("mounted", False)
        health = health_checks.get(name, {})

        status_class = "status-mounted" if mounted else "status-failed"
        status_text = "MOUNTED" if mounted else "FAILED"

        health_status = ""
        if mounted and health:
            health_class = f"status-{health.get('status', 'unknown')}"
            health_status = f'<p class="{health_class}"><strong>Health:</strong> {health.get("status", "unknown").upper()}</p>'
            if health.get("latency_ms"):
                health_status += (
                    f"<p><strong>Latency:</strong> {health['latency_ms']:.2f}ms</p>"
                )

        html_content += f"""
        <div class="service-card">
            <h2 class="{status_class}">{name.upper()}</h2>
            <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
        """

        if mounted:
            html_content += f"""
            <p><strong>Mount Path:</strong> <code>{status.get("mount_path")}</code></p>
            <p><strong>Import Path:</strong> <code>{status.get("import_path", "N/A")}</code></p>
            {health_status}
            <div class="links">
                <a href="{status.get("mount_path")}">Service Root</a>
                {f'<a href="{status.get("docs_url")}">API Docs</a>' if status.get("docs_url") else ""}
                {f'<a href="{status.get("health_path")}">Health</a>' if status.get("health_path") else ""}
            </div>
            """
        elif status.get("error"):
            html_content += f'<p class="error">Error: {status["error"]}</p>'

        html_content += "</div>"

    html_content += """
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get("/health/live", response_model=None)
async def health_live():
    """
    Kubernetes/Docker liveness probe endpoint.

    This is a fast, lightweight endpoint that only checks if the process is alive.
    It does NOT check service dependencies - use /health for comprehensive checks.

    Returns:
        200 OK with {"status": "alive"} if the server is responding
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/ready", response_model=None)
async def health_ready():
    """
    Kubernetes readiness probe endpoint.

    Checks if the application is ready to receive traffic. This performs a
    lightweight check that critical services are available without doing
    full health checks on all components.

    Returns:
        200 OK with {"status": "ready"} if ready to serve requests
        503 Service Unavailable if not ready
    """
    from src.full_platform import (
        _services_init_complete,
        _services_init_failed,
        app,
        service_manager,
    )

    try:
        # Check if services initialization is complete and successful
        services_init_complete = _services_init_complete
        services_init_failed = _services_init_failed

        # Check if service manager is initialized and has services mounted
        service_status = await service_manager.get_service_status()

        # Count how many services are mounted (indicates ready state)
        mounted_services = sum(
            1 for status in service_status.values()
            if status.get("mounted", False)
        )
        total_services = len(service_status)

        # Check if models are loaded (for full readiness)
        models_loaded = getattr(app.state, "models_loaded", False)

        # Determine status note
        if services_init_failed:
            status_note = "Services initialization failed - running in degraded mode"
        elif not services_init_complete:
            status_note = "Services are still initializing in background."
        elif not models_loaded:
            status_note = "Models are still loading in background."
        else:
            status_note = None

        # Ready if services init is complete (even if failed) OR we have at least one mounted service
        # Note: We return 200 even if models aren't loaded yet, because
        # the server can still handle basic requests.
        if services_init_complete or mounted_services > 0:
            return {
                "status": "ready" if not services_init_failed else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "mounted_services": mounted_services,
                "total_services": total_services,
                "services_init_complete": services_init_complete,
                "services_init_failed": services_init_failed,
                "models_loaded": models_loaded,
                "note": status_note
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "no_services_mounted",
                    "total_services": total_services
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(e)}
        )


@router.get("/health", response_model=None)
async def health_check(request: Request):
    """Comprehensive health check for all services."""
    from src.full_platform import service_manager, settings

    base_url = f"http://{request.url.hostname}:{request.url.port or settings.port}"

    service_status = await service_manager.get_service_status()

    health_checks = {}
    if settings.enable_health_checks:
        health_checks = await service_manager.check_all_health(base_url)

    all_healthy = all(
        h.get("status") == "healthy"
        for h in health_checks.values()
        if isinstance(h, dict) and "status" in h
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "worker_pid": os.getpid(),
        "services": {
            name: {
                "mounted": status.get("mounted", False),
                "health": health_checks.get(name, {"status": "unknown"}),
            }
            for name, status in service_status.items()
        },
    }
