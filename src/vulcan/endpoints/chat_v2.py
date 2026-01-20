"""
VULCAN Tool-Based Chat Endpoint (v2)

Simplified chat endpoint that uses LLM function calling to invoke
specialized tools, replacing the complex regex routing architecture.

Architecture:
    Query → SecurityFilter → LLM with Tools → [Tool Calls] → Response

Benefits:
    - Single decision maker (LLM decides when to use tools)
    - ~90% less code than regex routing
    - No fragile regex classification
    - Natural language understanding preserved
    - Tools for precise computation (SAT, math, hashing)

Industry Standards:
    - OpenAI function calling format
    - Structured tool responses
    - Comprehensive error handling
    - Request/response logging
    - Configurable timeouts

Version History:
    1.0.0 - Initial implementation with tool-based architecture
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from vulcan.security import SecurityFilter, SECURITY_FILTER_AVAILABLE
from vulcan.tools import get_tools_for_llm, execute_tool, get_tool_by_name

# Module metadata
__version__ = "1.0.0"
__author__ = "VULCAN-AGI Team"

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat-v2"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """
    Chat request model.
    
    Attributes:
        message: The user's message
        conversation_history: Optional conversation context
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        enable_tools: Whether to allow tool usage
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="The user's message"
    )
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages in the conversation"
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum tokens in response"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    enable_tools: bool = Field(
        default=True,
        description="Whether to enable tool usage"
    )


class ToolUsage(BaseModel):
    """Record of a tool usage."""
    name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    computation_time_ms: float


class ChatResponse(BaseModel):
    """
    Chat response model.
    
    Attributes:
        response: The assistant's response
        tools_used: List of tools that were called
        request_id: Unique request identifier
        computation_time_ms: Total processing time
    """
    response: str = Field(..., description="The assistant's response")
    tools_used: List[ToolUsage] = Field(
        default_factory=list,
        description="Tools that were called during processing"
    )
    request_id: str = Field(..., description="Unique request identifier")
    computation_time_ms: float = Field(..., description="Total processing time in ms")


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum conversation history to include
MAX_HISTORY_MESSAGES = 10

# Maximum tool call iterations
MAX_TOOL_ITERATIONS = 5

# System prompt for VULCAN
SYSTEM_PROMPT = """You are VULCAN, an advanced AI assistant with access to specialized tools.

## Available Tools

You have access to tools for tasks that require precise computation:

1. **sat_solver**: For formal logic, satisfiability checking, and proofs
   - Use when: checking if logical formulas are satisfiable, proving validity
   - Example: "Is P ∧ ¬P satisfiable?" → Call sat_solver

2. **math_engine**: For symbolic mathematics using SymPy
   - Use when: computing integrals, derivatives, solving equations
   - Example: "What is the integral of x^2?" → Call math_engine

3. **hash_compute**: For cryptographic hash computation
   - Use when: computing SHA-256, MD5, Base64, or other hashes
   - Example: "What is the SHA-256 of 'hello'?" → Call hash_compute

## When to Use Tools

Use tools when:
- The task requires EXACT computation (math, hashes, formal proofs)
- The user explicitly asks for computation
- Approximate reasoning is insufficient

Do NOT use tools when:
- You can answer from knowledge
- The question is conversational
- The task is conceptual/explanatory

## Response Guidelines

1. If a tool is needed, call it and explain the result
2. If no tool is needed, respond directly
3. Always explain your reasoning
4. Be concise but thorough"""


# =============================================================================
# SECURITY FILTER
# =============================================================================

# Initialize security filter
_security_filter: Optional[SecurityFilter] = None

def get_security_filter() -> Optional[SecurityFilter]:
    """Get or create the security filter singleton."""
    global _security_filter
    if _security_filter is None and SECURITY_FILTER_AVAILABLE:
        _security_filter = SecurityFilter()
    return _security_filter


# =============================================================================
# CHAT ENDPOINT
# =============================================================================


@router.post("/v2/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    VULCAN tool-based chat endpoint.
    
    This endpoint uses LLM function calling to invoke specialized tools
    when precise computation is needed. The LLM decides when to use tools
    based on the query, eliminating the need for regex-based routing.
    
    Flow:
        1. Security check (block attacks before LLM)
        2. Build messages with conversation history
        3. Call LLM with tool definitions
        4. Execute any tool calls
        5. Return final response
    
    Args:
        request: FastAPI request object (for accessing app state)
        body: Chat request with message and options
        
    Returns:
        ChatResponse with assistant's response and tool usage info
        
    Raises:
        HTTPException: 400 if security check fails, 503 if LLM unavailable
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{request_id}] Chat request: {body.message[:100]}...")
    
    # 1. Security check
    security_filter = get_security_filter()
    if security_filter:
        security_result = security_filter.check(body.message)
        if not security_result.safe:
            logger.warning(
                f"[{request_id}] Security block: {security_result.reason} "
                f"(risk={security_result.risk_level})"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Request blocked by security filter: {security_result.reason}"
            )
    
    # 2. Get LLM client from app state
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        # Try to get OpenAI client as fallback
        try:
            from vulcan.llm.openai_client import get_openai_client
            llm = get_openai_client()
        except Exception as e:
            logger.error(f"[{request_id}] Failed to get LLM client: {e}")
    
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not available. Please check configuration."
        )
    
    # 3. Build messages
    messages = _build_messages(body.message, body.conversation_history)
    
    # 4. Get tools (if enabled)
    tools = get_tools_for_llm() if body.enable_tools else []
    
    # 5. Call LLM with tool loop
    tools_used: List[ToolUsage] = []
    
    try:
        response_text = await _chat_with_tools(
            llm=llm,
            messages=messages,
            tools=tools,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            tools_used=tools_used,
            request_id=request_id,
        )
    except Exception as e:
        logger.error(f"[{request_id}] LLM call failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"LLM processing failed: {str(e)}"
        )
    
    # 6. Calculate timing and return
    computation_time = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        f"[{request_id}] Chat complete: tools_used={[t.name for t in tools_used]}, "
        f"time_ms={computation_time:.2f}"
    )
    
    return ChatResponse(
        response=response_text,
        tools_used=tools_used,
        request_id=request_id,
        computation_time_ms=computation_time,
    )


def _build_messages(
    user_message: str,
    history: List[ChatMessage],
) -> List[Dict[str, str]]:
    """
    Build message list for LLM.
    
    Args:
        user_message: Current user message
        history: Conversation history
        
    Returns:
        List of message dicts for LLM
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (last N messages)
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        messages.append({"role": msg.role, "content": msg.content})
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    return messages


async def _chat_with_tools(
    llm,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    tools_used: List[ToolUsage],
    request_id: str,
) -> str:
    """
    Chat with tool calling loop.
    
    The LLM can call tools, receive results, and continue reasoning
    until it produces a final response.
    
    Args:
        llm: LLM client (OpenAI or compatible)
        messages: Conversation messages
        tools: Tool definitions in OpenAI format
        max_tokens: Max response tokens
        temperature: Sampling temperature
        tools_used: List to populate with tool usage records
        request_id: Request ID for logging
        
    Returns:
        Final response text from LLM
    """
    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.debug(f"[{request_id}] Tool iteration {iteration + 1}")
        
        # Call LLM
        try:
            if tools:
                response = llm.chat.completions.create(
                    model="gpt-4-turbo-preview",  # Or configured model
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                response = llm.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
        except Exception as e:
            logger.error(f"[{request_id}] LLM API call failed: {e}")
            raise
        
        # Get the response message
        message = response.choices[0].message
        
        # Check if LLM wants to call tools
        if not message.tool_calls:
            # No tool calls - return the response
            return message.content or ""
        
        # Process tool calls
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
        })
        
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            
            # Parse arguments
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"[{request_id}] Invalid tool arguments: {e}")
                tool_args = {}
            
            logger.info(f"[{request_id}] Tool call: {tool_name}({list(tool_args.keys())})")
            
            # Execute tool
            tool_start = time.perf_counter()
            try:
                result = execute_tool(tool_name, tool_args)
                tool_response = result.result if result.success else f"Error: {result.error}"
                tool_success = result.success
            except Exception as e:
                logger.error(f"[{request_id}] Tool execution failed: {e}")
                tool_response = f"Tool error: {str(e)}"
                tool_success = False
            tool_time = (time.perf_counter() - tool_start) * 1000
            
            # Record tool usage
            tools_used.append(ToolUsage(
                name=tool_name,
                arguments=tool_args,
                result=tool_response,
                success=tool_success,
                computation_time_ms=tool_time,
            ))
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_response) if isinstance(tool_response, dict) else str(tool_response),
            })
    
    # Max iterations reached - get final response without tools
    logger.warning(f"[{request_id}] Max tool iterations reached, getting final response")
    
    response = llm.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    return response.choices[0].message.content or ""


# =============================================================================
# HEALTH CHECK
# =============================================================================


@router.get("/v2/chat/health")
async def chat_health() -> Dict[str, Any]:
    """
    Health check for the v2 chat endpoint.
    
    Returns:
        Dict with status and available tools
    """
    tools = get_tools_for_llm()
    security = get_security_filter()
    
    return {
        "status": "healthy",
        "version": __version__,
        "tools_available": len(tools),
        "tool_names": [t["function"]["name"] for t in tools],
        "security_filter_enabled": security is not None,
    }
