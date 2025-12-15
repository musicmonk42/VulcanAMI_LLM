"""
API Gateway Service Entry Point
Wraps src/vulcan/api_gateway.py for containerized deployment
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="VulcanAMI API Gateway",
    description="Unified API Gateway for VulcanAMI Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "service": "api-gateway"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {"status": "ready", "service": "api-gateway"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "VulcanAMI API Gateway",
        "version": "1.0.0",
        "status": "running"
    }

# Import and register additional routes from vulcan.api_gateway if available
try:
    from src.vulcan import api_gateway as vulcan_gateway
    logger.info("Successfully imported vulcan API gateway module")
    # Register routes from vulcan gateway if they exist
except ImportError as e:
    logger.warning(f"Could not import vulcan API gateway: {e}")

logger.info("API Gateway service initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
