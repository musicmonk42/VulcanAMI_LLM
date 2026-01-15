"""
Memory Search Endpoint

This module provides the endpoint for searching VULCAN's long-term memory
with semantic similarity and metadata filtering.

Endpoints:
    POST /v1/memory/search - Search memory with filters
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from vulcan.endpoints.utils import require_deployment
from vulcan.metrics import error_counter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memory"])


@router.post("/v1/memory/search", response_model=None)
async def search_memory(request: Request) -> dict:
    """
    Search memory with filters.
    
    Performs semantic similarity search over VULCAN's long-term memory
    using vector embeddings. Supports metadata filtering to narrow results
    based on specific attributes.
    
    The search process:
    1. Processes the query through multimodal processor to get embedding
    2. Searches long-term memory using the embedding vector
    3. Filters results based on provided metadata filters
    4. Returns top-k results sorted by similarity score
    
    Args:
        request: FastAPI request with MemorySearchRequest body containing:
            - query: Query string to search for
            - k: Number of results to return (default 10)
            - filters: Optional dict of metadata filters (key=value exact match)
    
    Returns:
        Dict containing:
            - results: List of search results, each with:
                - id: Memory item identifier
                - score: Similarity score (higher = more similar)
                - metadata: Associated metadata dict
            - total: Total number of results after filtering
    
    Raises:
        HTTPException: 503 if system not initialized or memory unavailable
        HTTPException: 500 if search fails
    
    Example:
        Request:
        {
            "query": "What is machine learning?",
            "k": 5,
            "filters": {"category": "technical", "verified": true}
        }
        
        Response:
        {
            "results": [
                {
                    "id": "mem_123",
                    "score": 0.92,
                    "metadata": {"category": "technical", "verified": true, "date": "2026-01-10"}
                }
            ],
            "total": 1
        }
    """
    app = request.app
    
    deployment = require_deployment(request)

    try:
        memory = deployment.collective.deps.ltm
        processor = deployment.collective.deps.multimodal

        if memory is None or processor is None:
            raise HTTPException(status_code=503, detail="Memory system not available")

        # Get request body
        from vulcan.api.models import MemorySearchRequest
        body = await request.json()
        memory_request = MemorySearchRequest(**body)

        # Process query to get embedding
        loop = asyncio.get_running_loop()
        query_result = await loop.run_in_executor(
            None, processor.process_input, memory_request.query
        )

        # Search memory with embedding
        results = memory.search(query_result.embedding, k=memory_request.k)

        # Apply metadata filters if provided
        if memory_request.filters:
            filtered_results = []
            for result in results:
                metadata = result[2] if len(result) > 2 else {}
                # Check if all filter conditions match
                match = all(
                    metadata.get(key) == value
                    for key, value in memory_request.filters.items()
                )
                if match:
                    filtered_results.append(result)
            results = filtered_results

        return {
            "results": [
                {
                    "id": r[0],
                    "score": r[1],
                    "metadata": r[2] if len(r) > 2 else {}
                }
                for r in results
            ],
            "total": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="memory").inc()
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
