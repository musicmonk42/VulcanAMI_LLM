"""
PII Detection Service Entry Point
Wraps PII detection functionality for containerized deployment
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="VulcanAMI PII Service",
    description="PII Detection and Protection Service for VulcanAMI Platform",
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
    return {"status": "healthy", "service": "pii"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {"status": "ready", "service": "pii"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "VulcanAMI PII Detection Service",
        "version": "1.0.0",
        "status": "running"
    }

# Import PII detection functionality
try:
    from src import security_nodes
    logger.info("Successfully imported security nodes module")
except ImportError as e:
    logger.warning(f"Could not import security nodes: {e}")

try:
    from src.generation import safe_generation
    logger.info("Successfully imported safe generation module")
except ImportError as e:
    logger.warning(f"Could not import safe generation: {e}")

logger.info("PII service initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
