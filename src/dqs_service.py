"""
Data Quality System (DQS) Service Entry Point
Wraps data quality functionality for containerized deployment
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="VulcanAMI DQS Service",
    description="Data Quality System for VulcanAMI Platform",
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
    return {"status": "healthy", "service": "dqs"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {"status": "ready", "service": "dqs"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "VulcanAMI DQS Service",
        "version": "1.0.0",
        "status": "running"
    }

# Import DQS functionality
try:
    from src.gvulcan import dqs
    logger.info("Successfully imported DQS module")
except ImportError as e:
    logger.warning(f"Could not import DQS module: {e}")

logger.info("DQS service initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
