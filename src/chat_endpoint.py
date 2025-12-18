#!/usr/bin/env python3
"""
VULCAN Chat Endpoint - Backend for vulcan_chat.html

Provides a chat interface that integrates all VULCAN platform components:
- World Model (causal reasoning)
- Unified Reasoner (5 modes)
- Memory (Graph RAG)
- Safety Validator (CSIU protocol)
- Planning Engine
- LLM Core
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# LAZY COMPONENT IMPORTS
# =============================================================================

# World Model
VULCANWorldModel = None


def _get_world_model():
    """Lazy load VULCANWorldModel to avoid import issues."""
    global VULCANWorldModel
    if VULCANWorldModel is None:
        try:
            from src.vulcan.world_model.world_model_core import (
                VULCANWorldModel as WM,
            )

            VULCANWorldModel = WM
        except (ImportError, AttributeError, TypeError, Exception) as e:
            logger.warning(f"Could not import VULCANWorldModel: {e}")
            VULCANWorldModel = False
    return VULCANWorldModel if VULCANWorldModel else None


# Unified Reasoner
UnifiedReasoner = None


def _get_unified_reasoner():
    """Lazy load UnifiedReasoner to avoid import issues."""
    global UnifiedReasoner
    if UnifiedReasoner is None:
        try:
            from src.vulcan.reasoning.unified_reasoning import (
                UnifiedReasoner as UR,
            )

            UnifiedReasoner = UR
        except (ImportError, AttributeError, TypeError, Exception) as e:
            logger.warning(f"Could not import UnifiedReasoner: {e}")
            UnifiedReasoner = False
    return UnifiedReasoner if UnifiedReasoner else None


# Hierarchical Memory
HierarchicalMemory = None


def _get_hierarchical_memory():
    """Lazy load HierarchicalMemory to avoid import issues."""
    global HierarchicalMemory
    if HierarchicalMemory is None:
        try:
            from src.vulcan.memory.hierarchical import (
                HierarchicalMemory as HM,
            )

            HierarchicalMemory = HM
        except (ImportError, AttributeError, TypeError, Exception) as e:
            logger.warning(f"Could not import HierarchicalMemory: {e}")
            HierarchicalMemory = False
    return HierarchicalMemory if HierarchicalMemory else None


# Safety Validator
SafetyValidator = None


def _get_safety_validator():
    """Lazy load SafetyValidator to avoid import issues."""
    global SafetyValidator
    if SafetyValidator is None:
        try:
            from src.vulcan.safety.safety_validator import (
                EnhancedSafetyValidator as SV,
            )

            SafetyValidator = SV
        except (ImportError, AttributeError, TypeError, Exception) as e:
            logger.warning(f"Could not import SafetyValidator: {e}")
            SafetyValidator = False
    return SafetyValidator if SafetyValidator else None


# Planning Engine
PlanningEngine = None


def _get_planning_engine():
    """Lazy load PlanningEngine to avoid import issues."""
    global PlanningEngine
    if PlanningEngine is None:
        try:
            from src.vulcan.planning import (
                EnhancedHierarchicalPlanner as PE,
            )

            PlanningEngine = PE
        except (ImportError, AttributeError, TypeError, Exception) as e:
            logger.warning(f"Could not import PlanningEngine: {e}")
            PlanningEngine = False
    return PlanningEngine if PlanningEngine else None


# GraphixTransformer (LLM)
GraphixTransformer = None


def _get_graphix_transformer():
    """Lazy load GraphixTransformer to avoid import issues."""
    global GraphixTransformer
    if GraphixTransformer is None:
        try:
            from src.llm_core.graphix_transformer import (
                GraphixTransformer as GT,
            )

            GraphixTransformer = GT
        except (ImportError, AttributeError, TypeError, Exception) as e:
            logger.warning(f"Could not import GraphixTransformer: {e}")
            GraphixTransformer = False
    return GraphixTransformer if GraphixTransformer else None


# Graph RAG
GraphRAG = None


def _get_graph_rag():
    """Lazy load GraphRAG to avoid import issues."""
    global GraphRAG
    if GraphRAG is None:
        try:
            from src.persistant_memory_v46.graph_rag import GraphRAG as GR

            GraphRAG = GR
        except (ImportError, AttributeError, TypeError, Exception) as e:
            logger.warning(f"Could not import GraphRAG: {e}")
            GraphRAG = False
    return GraphRAG if GraphRAG else None


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="VULCAN Chat Endpoint",
    description="Chat endpoint for vulcan_chat.html frontend",
    version="1.0.0",
)

# CORS middleware - allow all origins for the chat interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODELS
# =============================================================================


class HistoryItem(BaseModel):
    """Single message in conversation history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model matching vulcan_chat.html frontend expectations."""

    message: str = Field(..., description="User's message/question")
    max_tokens: int = Field(default=1024, description="Maximum tokens in response")
    enable_reasoning: bool = Field(
        default=True, description="Enable unified reasoning"
    )
    enable_memory: bool = Field(default=True, description="Enable memory retrieval")
    enable_safety: bool = Field(default=True, description="Enable safety validation")
    enable_planning: bool = Field(
        default=True, description="Enable planning engine for complex tasks"
    )
    enable_causal: bool = Field(
        default=True, description="Enable world model causal reasoning"
    )
    history: List[HistoryItem] = Field(
        default_factory=list, description="Conversation history"
    )


class ChatResponse(BaseModel):
    """Chat response model matching vulcan_chat.html frontend expectations."""

    response: str = Field(..., description="AI's response text")
    systems_used: List[str] = Field(
        default_factory=list, description="List of systems used to generate response"
    )


# =============================================================================
# VULCAN CHAT ENGINE
# =============================================================================


class VulcanChatEngine:
    """
    Main chat engine that orchestrates all VULCAN platform components.

    Follows the EXAMINE → SELECT → APPLY → REMEMBER pattern:
    - EXAMINE: Retrieve context from memory and world model
    - SELECT: Apply reasoning and planning
    - APPLY: Generate response with LLM
    - REMEMBER: Store interaction in memory
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern to avoid re-initializing components."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize VULCAN Chat Engine components."""
        if self._initialized:
            return

        logger.info("🚀 Initializing VULCAN Chat Engine...")

        # Initialize components (will be loaded lazily on first use)
        self._world_model = None
        self._reasoner = None
        self._memory = None
        self._safety = None
        self._planner = None
        self._llm = None
        self._graph_rag = None

        self._initialized = True
        logger.info("✅ VULCAN Chat Engine initialized")

    @property
    def world_model(self):
        """Lazy-load world model."""
        if self._world_model is None:
            WMClass = _get_world_model()
            if WMClass:
                try:
                    self._world_model = WMClass()
                    logger.info("✓ World Model loaded")
                except Exception as e:
                    logger.warning(f"Failed to initialize World Model: {e}")
        return self._world_model

    @property
    def reasoner(self):
        """Lazy-load unified reasoner."""
        if self._reasoner is None:
            URClass = _get_unified_reasoner()
            if URClass:
                try:
                    self._reasoner = URClass()
                    logger.info("✓ Unified Reasoner loaded")
                except Exception as e:
                    logger.warning(f"Failed to initialize Unified Reasoner: {e}")
        return self._reasoner

    @property
    def memory(self):
        """Lazy-load hierarchical memory."""
        if self._memory is None:
            HMClass = _get_hierarchical_memory()
            if HMClass:
                try:
                    self._memory = HMClass()
                    logger.info("✓ Hierarchical Memory loaded")
                except Exception as e:
                    logger.warning(f"Failed to initialize Hierarchical Memory: {e}")
        return self._memory

    @property
    def safety(self):
        """Lazy-load safety validator."""
        if self._safety is None:
            SVClass = _get_safety_validator()
            if SVClass:
                try:
                    self._safety = SVClass()
                    logger.info("✓ Safety Validator loaded")
                except Exception as e:
                    logger.warning(f"Failed to initialize Safety Validator: {e}")
        return self._safety

    @property
    def planner(self):
        """Lazy-load planning engine."""
        if self._planner is None:
            PEClass = _get_planning_engine()
            if PEClass:
                try:
                    self._planner = PEClass()
                    logger.info("✓ Planning Engine loaded")
                except Exception as e:
                    logger.warning(f"Failed to initialize Planning Engine: {e}")
        return self._planner

    @property
    def llm(self):
        """Lazy-load LLM (GraphixTransformer)."""
        if self._llm is None:
            GTClass = _get_graphix_transformer()
            if GTClass:
                try:
                    self._llm = GTClass()
                    logger.info("✓ GraphixTransformer LLM loaded")
                except Exception as e:
                    logger.warning(f"Failed to initialize GraphixTransformer: {e}")
        return self._llm

    @property
    def graph_rag(self):
        """Lazy-load Graph RAG."""
        if self._graph_rag is None:
            GRClass = _get_graph_rag()
            if GRClass:
                try:
                    self._graph_rag = GRClass()
                    logger.info("✓ Graph RAG loaded")
                except Exception as e:
                    logger.warning(f"Failed to initialize Graph RAG: {e}")
        return self._graph_rag

    def process(self, request: ChatRequest) -> ChatResponse:
        """
        Process chat message through VULCAN platform.

        Follows EXAMINE → SELECT → APPLY → REMEMBER pattern.
        """
        systems_used = []
        start_time = time.time()

        try:
            # =========================================================
            # PHASE 1: EXAMINE - Gather context and assess state
            # =========================================================
            contexts = []

            # Memory retrieval
            if request.enable_memory and self.graph_rag:
                try:
                    retrieved = self.graph_rag.retrieve(request.message, k=10)
                    if retrieved:
                        contexts = [r.content if hasattr(r, "content") else str(r) for r in retrieved]
                        systems_used.append("memory")
                        logger.debug(f"Retrieved {len(contexts)} memory contexts")
                except Exception as e:
                    logger.warning(f"Memory retrieval failed: {e}")

            # World model state assessment
            state = None
            if request.enable_causal and self.world_model:
                try:
                    if hasattr(self.world_model, "assess_state"):
                        state = self.world_model.assess_state(
                            observation=request.message, contexts=contexts
                        )
                    elif hasattr(self.world_model, "get_state"):
                        state = self.world_model.get_state()
                    systems_used.append("world_model")
                    logger.debug("World model state assessed")
                except Exception as e:
                    logger.warning(f"World model assessment failed: {e}")

            # =========================================================
            # PHASE 2: SELECT - Apply reasoning and make decisions
            # =========================================================
            reasoning_result = None

            # Unified reasoning
            if request.enable_reasoning and self.reasoner:
                try:
                    if hasattr(self.reasoner, "reason"):
                        reasoning_result = self.reasoner.reason(
                            query=request.message,
                            contexts=contexts,
                            state=state,
                        )
                    elif hasattr(self.reasoner, "process"):
                        reasoning_result = self.reasoner.process(request.message)
                    systems_used.append("reasoner")
                    logger.debug("Reasoning completed")
                except Exception as e:
                    logger.warning(f"Reasoning failed: {e}")

            # Planning for complex tasks
            plan = None
            if request.enable_planning and self.planner:
                try:
                    if self._needs_planning(request.message):
                        if hasattr(self.planner, "generate_plan"):
                            plan = self.planner.generate_plan(
                                goal=request.message,
                                context=reasoning_result if isinstance(reasoning_result, dict) else {},
                            )
                        elif hasattr(self.planner, "create_plan"):
                            plan = self.planner.create_plan(
                                goal=request.message,
                                context={},
                            )
                        systems_used.append("planner")
                        logger.debug("Planning completed")
                except Exception as e:
                    logger.warning(f"Planning failed: {e}")

            # Safety validation
            if request.enable_safety and self.safety:
                try:
                    proposal = reasoning_result if isinstance(reasoning_result, dict) else {"query": request.message}
                    context = state if isinstance(state, dict) else {}

                    if hasattr(self.safety, "validate_proposal"):
                        safety_check = self.safety.validate_proposal(
                            proposal=proposal, context=context
                        )
                    elif hasattr(self.safety, "validate"):
                        safety_check = self.safety.validate(proposal)
                    elif hasattr(self.safety, "validate_action"):
                        is_safe, reason, _ = self.safety.validate_action(proposal, context)
                        safety_check = {"approved": is_safe, "reason": reason}
                    else:
                        safety_check = {"approved": True}

                    systems_used.append("safety")

                    if not safety_check.get("approved", True):
                        return ChatResponse(
                            response="I cannot provide that response due to safety constraints.",
                            systems_used=systems_used,
                        )
                    logger.debug("Safety check passed")
                except Exception as e:
                    logger.warning(f"Safety validation failed: {e}")

            # =========================================================
            # PHASE 3: APPLY - Generate response
            # =========================================================
            response_text = self._generate_response(
                request=request,
                contexts=contexts,
                reasoning_result=reasoning_result,
                plan=plan,
                state=state,
            )
            systems_used.append("llm")

            # =========================================================
            # PHASE 4: REMEMBER - Store interaction in memory
            # =========================================================
            if request.enable_memory:
                try:
                    # Store in hierarchical memory
                    if self.memory and hasattr(self.memory, "store"):
                        self.memory.store(
                            content=f"Q: {request.message}\nA: {response_text}",
                            metadata={"type": "chat", "timestamp": time.time()},
                        )

                    # Store in Graph RAG
                    if self.graph_rag and hasattr(self.graph_rag, "store"):
                        self.graph_rag.store(
                            content=response_text,
                            metadata={"query": request.message, "timestamp": time.time()},
                        )

                    logger.debug("Interaction stored in memory")
                except Exception as e:
                    logger.warning(f"Memory storage failed: {e}")

                # Update world model
                if request.enable_causal and self.world_model and state:
                    try:
                        if hasattr(self.world_model, "update_from_interaction"):
                            self.world_model.update_from_interaction(
                                query=request.message,
                                response=response_text,
                                success=True,
                            )
                    except Exception as e:
                        logger.warning(f"World model update failed: {e}")

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Chat request processed in {elapsed_ms:.1f}ms using systems: {systems_used}"
            )

            return ChatResponse(response=response_text, systems_used=systems_used)

        except Exception as e:
            logger.error(f"Error processing chat request: {e}", exc_info=True)
            return ChatResponse(
                response=f"I encountered an error while processing your request: {str(e)}",
                systems_used=systems_used,
            )

    def _needs_planning(self, message: str) -> bool:
        """Check if message requires planning capabilities."""
        planning_keywords = [
            "plan",
            "strategy",
            "steps",
            "how to",
            "roadmap",
            "approach",
            "guide",
            "process",
            "procedure",
            "walkthrough",
        ]
        message_lower = message.lower()
        return any(kw in message_lower for kw in planning_keywords)

    def _generate_response(
        self,
        request: ChatRequest,
        contexts: List[str],
        reasoning_result: Any,
        plan: Any,
        state: Any,
    ) -> str:
        """Generate response using LLM with all gathered context."""
        # Build context for the LLM
        context_parts = []

        # Add reasoning context
        if reasoning_result:
            if isinstance(reasoning_result, dict):
                context_parts.append(f"Reasoning: {reasoning_result}")
            elif hasattr(reasoning_result, "conclusion"):
                context_parts.append(f"Reasoning: {reasoning_result.conclusion}")

        # Add plan context
        if plan:
            if isinstance(plan, dict):
                context_parts.append(f"Plan: {plan}")
            elif hasattr(plan, "to_dict"):
                context_parts.append(f"Plan: {plan.to_dict()}")

        # Add state context
        if state:
            if isinstance(state, dict):
                context_parts.append(f"State: {state}")

        # Add memory contexts
        if contexts:
            context_parts.append(f"Relevant context: {' | '.join(contexts[:3])}")

        # Add conversation history
        history_text = ""
        if request.history:
            history_entries = []
            for h in request.history[-5:]:  # Last 5 messages
                history_entries.append(f"{h.role}: {h.content}")
            history_text = "\n".join(history_entries)
            context_parts.append(f"History:\n{history_text}")

        # Build prompt
        full_context = "\n".join(context_parts) if context_parts else ""
        prompt = f"{full_context}\n\nUser: {request.message}\nAssistant:" if full_context else request.message

        # Try to generate with LLM
        if self.llm:
            try:
                if hasattr(self.llm, "generate"):
                    result = self.llm.generate(
                        prompt=prompt,
                        max_tokens=request.max_tokens,
                        temperature=0.7,
                    )
                    if hasattr(result, "text"):
                        return result.text
                    elif isinstance(result, str):
                        return result
                    elif isinstance(result, dict):
                        return result.get("text", result.get("response", str(result)))
                    return str(result)
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # Fallback response
        return self._fallback_response(request, reasoning_result, plan, contexts)

    def _fallback_response(
        self,
        request: ChatRequest,
        reasoning_result: Any,
        plan: Any,
        contexts: List[str],
    ) -> str:
        """Generate fallback response when LLM is unavailable."""
        response_parts = []

        # Add reasoning insights
        if reasoning_result:
            if isinstance(reasoning_result, dict):
                if "conclusion" in reasoning_result:
                    response_parts.append(f"Based on my analysis: {reasoning_result['conclusion']}")
                elif "result" in reasoning_result:
                    response_parts.append(f"Analysis result: {reasoning_result['result']}")
            elif hasattr(reasoning_result, "conclusion"):
                response_parts.append(f"Based on my analysis: {reasoning_result.conclusion}")

        # Add plan summary
        if plan:
            if hasattr(plan, "steps") and plan.steps:
                step_count = len(plan.steps)
                response_parts.append(f"I've created a plan with {step_count} steps.")
            elif isinstance(plan, dict) and "steps" in plan:
                step_count = len(plan["steps"])
                response_parts.append(f"I've created a plan with {step_count} steps.")

        # Add relevant context
        if contexts:
            response_parts.append(f"Relevant information found: {contexts[0][:200]}...")

        # Default response
        if not response_parts:
            response_parts.append(
                f"I received your message: '{request.message}'. "
                "However, the full LLM generation is currently unavailable. "
                "Please check that all VULCAN components are properly initialized."
            )

        return " ".join(response_parts)


# =============================================================================
# GLOBAL ENGINE INSTANCE
# =============================================================================

engine = VulcanChatEngine()


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """
    Main chat endpoint for vulcan_chat.html frontend.

    Uses all enabled VULCAN systems:
    - World Model (causal reasoning)
    - Unified Reasoner (5 modes)
    - Memory (Graph RAG)
    - Safety Validator (CSIU)
    - Planning Engine
    - LLM Core

    Request:
    ```json
    {
        "message": "user's question here",
        "max_tokens": 1024,
        "enable_reasoning": true,
        "enable_memory": true,
        "enable_safety": true,
        "enable_planning": true,
        "enable_causal": true,
        "history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    ```

    Response:
    ```json
    {
        "response": "AI's answer here",
        "systems_used": ["world_model", "reasoner", "memory", "safety", "planner"]
    }
    ```
    """
    try:
        return engine.process(request)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    systems = {}

    # Check each component
    if engine.world_model:
        systems["world_model"] = "active"
    else:
        systems["world_model"] = "unavailable"

    if engine.reasoner:
        systems["reasoner"] = "active"
    else:
        systems["reasoner"] = "unavailable"

    if engine.memory:
        systems["memory"] = "active"
    else:
        systems["memory"] = "unavailable"

    if engine.safety:
        systems["safety"] = "active"
    else:
        systems["safety"] = "unavailable"

    if engine.planner:
        systems["planner"] = "active"
    else:
        systems["planner"] = "unavailable"

    if engine.llm:
        systems["llm"] = "active"
    else:
        systems["llm"] = "unavailable"

    if engine.graph_rag:
        systems["graph_rag"] = "active"
    else:
        systems["graph_rag"] = "unavailable"

    # Determine overall status
    active_count = sum(1 for s in systems.values() if s == "active")
    total_count = len(systems)

    if active_count == total_count:
        status = "healthy"
    elif active_count > 0:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "service": "vulcan-chat",
        "active_systems": f"{active_count}/{total_count}",
        "systems": systems,
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "VULCAN Chat Endpoint",
        "version": "1.0.0",
        "description": "Backend API for vulcan_chat.html",
        "endpoints": {
            "chat": "/v1/chat",
            "health": "/health",
        },
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
