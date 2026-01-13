"""
Core apply_reasoning implementation for reasoning integration.

Contains the main apply_reasoning method that orchestrates all reasoning
components and implements the primary interface.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .types import (
    ReasoningResult,
    ReasoningStrategyType,
    RoutingDecision,
    LOG_PREFIX,
    FAST_PATH_COMPLEXITY_THRESHOLD,
    DECOMPOSITION_COMPLEXITY_THRESHOLD,
)
from .query_router import get_reasoning_type_from_route
from .selection_strategies import (
    select_with_tool_selector,
    determine_strategy_from_query,
    get_fallback_tools,
)
from .decomposition import process_with_decomposition
from .query_analysis import (
    is_self_referential,
    is_ethical_query,
    consult_world_model_introspection,
)
from .cross_domain import apply_cross_domain_transfer
from .learning import learn_from_outcome, learn_from_reasoning_outcome

logger = logging.getLogger(__name__)

# Optional answer validator import
try:
    from vulcan.reasoning.answer_validator import validate_reasoning_result
    ANSWER_VALIDATOR_AVAILABLE = True
except ImportError:
    ANSWER_VALIDATOR_AVAILABLE = False
    validate_reasoning_result = None


def apply_reasoning(
        self,
        query: str,
        query_type: str,
        complexity: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Apply reasoning to select tools and determine strategy for a query.

        This is the main entry point for applying reasoning-based tool selection.
        It analyzes the query characteristics and uses the ToolSelector (if
        available) to determine the best tools and reasoning strategy.

        Args:
            query: The user query text to process.
            query_type: Type of query from the router. Valid types include:
                - "general": General knowledge queries
                - "reasoning": Logical reasoning queries
                - "execution": Action/task execution queries
                - "perception": Pattern recognition queries
                - "planning": Multi-step planning queries
                - "learning": Knowledge acquisition queries
            complexity: Query complexity score (0.0 to 1.0).
                - 0.0-0.3: Simple queries (fast path)
                - 0.3-0.7: Medium complexity
                - 0.7-1.0: High complexity (full analysis)
            context: Optional context dictionary containing:
                - conversation_id: ID of the conversation
                - history: Previous messages in conversation
                - user_preferences: User-specific settings

        Returns:
            ReasoningResult with selected tools, strategy, and metadata.

        Raises:
            No exceptions are raised. Errors result in fallback to default strategy.

        Example:
            >>> integration = ReasoningIntegration()
            >>> result = integration.apply_reasoning(
            ...     query="Explain quantum entanglement",
            ...     query_type="reasoning",
            ...     complexity=0.8,
            ...     context={"conversation_id": "conv_123"}
            ... )
            >>> print(result.selected_tools)
            ['causal', 'probabilistic']
        """
        # Check shutdown state
        with self._shutdown_lock:
            if self._shutdown:
                logger.warning(f"{LOG_PREFIX} Called after shutdown, returning default")
                return self._create_default_result(query_type, complexity)

        # Initialize components if needed
        self._init_components()

        # Track invocation
        selection_start = time.perf_counter()
        with self._stats_lock:
            self._stats.invocations += 1

        try:
            # =================================================================
            # Note: Check for self-referential queries FIRST
            # Note: Also check for ethical queries that need world model
            # =================================================================
            # World model handles queries about VULCAN's self directly.
            # For ALL queries that are about VULCAN itself (capabilities,
            # preferences, self-awareness, etc.), consult world model first.
            # Ethical queries also benefit from world model's ethical framework.
            # =================================================================
            is_self_ref = self._is_self_referential(query)
            is_ethical = self._is_ethical_query(query)
            
            # Initialize wm_result to None - will be populated if world_model is consulted
            wm_result = None
            
            if is_self_ref or is_ethical:
                # Handle overlap: prioritize self-referential if both
                query_type_label = 'self-referential' if is_self_ref else 'ethical'
                if is_self_ref and is_ethical:
                    query_type_label = 'self-referential and ethical'
                logger.info(f"{LOG_PREFIX} Note: {query_type_label.capitalize()} query detected - consulting world model first")
                wm_result = self._consult_world_model_introspection(query)
                
                # ═══════════════════════════════════════════════════════════════════
                # Note: Handle World Model Delegation
                # The world model now has delegation intelligence - it can detect when
                # a query LOOKS self-referential but actually needs another reasoner.
                # Example: "Trolley problem - you must choose" is ethical reasoning
                #          posed TO the AI, not a question ABOUT the AI.
                # ═══════════════════════════════════════════════════════════════════
                
                if wm_result is not None and wm_result.get("needs_delegation", False):
                    recommended_tool = wm_result.get("recommended_tool")
                    delegation_reason = wm_result.get("delegation_reason", "")
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: World model recommends DELEGATION to "
                        f"'{recommended_tool}' - {delegation_reason}"
                    )
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # CRITICAL FIX (Jan 6 2026): EXECUTE DELEGATION IMMEDIATELY
                    # ═══════════════════════════════════════════════════════════════════
                    # PROBLEM: Previous code set context flags and continued to normal
                    # processing, but Note in tool_selector.py was overriding
                    # the delegation because it checks for formal logic BEFORE checking
                    # delegation context.
                    #
                    # Note: Execute the delegated tool HERE and return immediately.
                    # This prevents any downstream code from overriding the delegation.
                    #
                    # Evidence from logs:
                    #   Line 2853: [ReasoningIntegration] LLM Classification: category=SELF_INTROSPECTION
                    #   Line 2854: [WorldModel] DELEGATION RECOMMENDED: 'mathematical'
                    #   Line 2855: [ReasoningIntegration] SELF_INTROSPECTION detected - using world_model tool
                    #   ^ CONTRADICTION: Says delegation active but uses world_model
                    # ═══════════════════════════════════════════════════════════════════
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: World model delegation ACTIVE - "
                        f"executing '{recommended_tool}' immediately (NOT falling through)"
                    )
                    
                    # Set up context with delegation info
                    if context is None:
                        context = {}
                    
                    context['world_model_delegation'] = True
                    context['world_model_recommended_tool'] = recommended_tool
                    context['world_model_delegation_reason'] = delegation_reason
                    context['classifier_suggested_tools'] = [recommended_tool]
                    context['classifier_is_authoritative'] = True
                    context['prevent_router_tool_override'] = True
                    context['skip_task3_fix'] = True  # Tell tool_selector to skip formal logic check
                    
                    # Map tool name to query_type for proper routing
                    if recommended_tool == 'philosophical':
                        query_type = 'ethical'
                    elif recommended_tool == 'mathematical':
                        query_type = 'mathematical'
                    elif recommended_tool == 'causal':
                        query_type = 'causal'
                    elif recommended_tool == 'probabilistic':
                        query_type = 'probabilistic'
                    
                    # Execute with the delegated tool directly via _select_with_tool_selector
                    # This is the EARLY RETURN that was missing
                    selection_time_start = time.perf_counter()
                    result = self._select_with_tool_selector(
                        query, query_type, complexity, context
                    )
                    selection_time = (time.perf_counter() - selection_time_start) * 1000
                    
                    # FIX: Verify delegation actually happened - log warning if not
                    actual_tool = result.selected_tools[0] if result.selected_tools else "none"
                    if actual_tool != recommended_tool:
                        logger.warning(
                            f"{LOG_PREFIX} DELEGATION MISMATCH: World model recommended "
                            f"'{recommended_tool}' but tool_selector returned '{actual_tool}'. "
                            f"This may indicate tool_selector is overriding delegation."
                        )
                        # Still return result - but flag the mismatch
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata["delegation_mismatch"] = True
                        result.metadata["expected_tool"] = recommended_tool
                        result.metadata["actual_tool"] = actual_tool
                    
                    # Add delegation metadata to result (with safety check)
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["world_model_delegation"] = True
                    result.metadata["delegated_tool"] = recommended_tool
                    result.metadata["delegation_reason"] = delegation_reason
                    result.metadata["selection_time_ms"] = selection_time
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: Delegation complete - executed '{recommended_tool}' "
                        f"with confidence={result.confidence:.2f} (EARLY RETURN)"
                    )
                    
                    # EARLY RETURN - Do NOT fall through to normal processing
                    return result
                
                # Note: Lower threshold from 0.7 to 0.5 for world model
                # Only use world model result if NOT delegating
                elif wm_result is not None and wm_result.get("confidence", 0) >= 0.5:
                    # World model can handle this directly
                    selection_time = (time.perf_counter() - selection_start) * 1000
                    
                    # Note: Include conclusion in metadata for proper extraction
                    # The world model's response IS the conclusion for self-introspection queries
                    world_model_response = wm_result.get("response", "")
                    
                    logger.info(
                        f"{LOG_PREFIX} Note: World model returned confidence "
                        f"{wm_result['confidence']:.2f}. Using this result directly "
                        f"without other engines."
                    )
                    
                    # Determine reasoning type: self-referential takes priority
                    reasoning_type = "meta_reasoning" if is_self_ref else "philosophical_reasoning"
                    strategy_type = ReasoningStrategyType.META_REASONING.value if is_self_ref else ReasoningStrategyType.PHILOSOPHICAL_REASONING.value
                    
                    return ReasoningResult(
                        selected_tools=["world_model"],
                        reasoning_strategy=strategy_type,
                        confidence=wm_result["confidence"],
                        rationale=wm_result.get("reasoning", "World model introspection"),
                        metadata={
                            "query_type": query_type,
                            "complexity": complexity,
                            "self_referential": is_self_ref,
                            "ethical_query": is_ethical,
                            "world_model_response": world_model_response,
                            # Note: Add conclusion field so main.py can extract it
                            "conclusion": world_model_response,
                            "explanation": wm_result.get("reasoning", ""),
                            "reasoning_type": reasoning_type,
                            "aspect": wm_result.get("aspect", "general"),
                            "selection_time_ms": selection_time,
                        },
                    )

            # =================================================================
            # Note: LLM-BASED QUERY CLASSIFICATION (ROOT CAUSE FIX)
            # =================================================================
            # The problem: "hello" (5 chars) was getting complexity=0.50 from
            # heuristic-based calculation, causing it to hit full reasoning.
            # 
            # The fix: Use LLM layer for LANGUAGE UNDERSTANDING to:
            # 1. Determine if query needs reasoning at all
            # 2. Suggest appropriate tools based on query intent
            # 3. Set correct complexity based on actual query meaning
            #
            # Architecture note: LLMs (cloud or internal) are interchangeable
            # for classification. The reasoning engines provide correctness.
            # =================================================================
            try:
                from vulcan.routing.query_classifier import classify_query, QueryCategory
                
                classification = classify_query(query)
                
                logger.info(
                    f"{LOG_PREFIX} LLM Classification: category={classification.category}, "
                    f"complexity={classification.complexity:.2f}, skip={classification.skip_reasoning}, "
                    f"tools={classification.suggested_tools}"
                )
                
                # If classifier says skip reasoning (greetings, chitchat, simple factual)
                # return immediately without invoking any reasoning engine
                if classification.skip_reasoning:
                    logger.info(
                        f"{LOG_PREFIX} CLASSIFIER SKIP: '{query[:30]}' classified as "
                        f"{classification.category} - skipping reasoning entirely"
                    )
                    with self._stats_lock:
                        self._stats.fast_path_count += 1
                    
                    return ReasoningResult(
                        selected_tools=classification.suggested_tools or ["general"],
                        reasoning_strategy=ReasoningStrategyType.DIRECT.value,
                        confidence=classification.confidence,
                        rationale=f"Query classified as {classification.category} - no reasoning needed",
                        metadata={
                            "fast_path": True,
                            "classifier_category": classification.category,
                            "classifier_source": classification.source,
                            "complexity": classification.complexity,
                            "query_type": classification.category.lower(),
                            "selection_time_ms": (time.perf_counter() - selection_start) * 1000,
                            "needs_reasoning": False,
                        },
                    )
                
                # Classifier identified this needs reasoning - use its suggestions
                # Override the heuristic complexity with LLM-derived complexity
                if classification.complexity != complexity:
                    logger.info(
                        f"{LOG_PREFIX} Overriding heuristic complexity {complexity:.2f} with "
                        f"classifier complexity {classification.complexity:.2f}"
                    )
                    complexity = classification.complexity
                
                # If classifier suggested specific tools, pass them to context
                # Note: DON'T override world model delegation!
                # If world_model_delegation is set, the world model has already determined
                # the correct tool. The classifier should NOT override this expert judgment.
                if classification.suggested_tools:
                    if context is None:
                        context = {}
                    
                    # Note: Check if world model delegation is active
                    if context.get('world_model_delegation'):
                        logger.info(
                            f"{LOG_PREFIX} Note: World model delegation ACTIVE - "
                            f"NOT overriding with classifier tools {classification.suggested_tools}. "
                            f"Using delegated tool: {context.get('classifier_suggested_tools')}"
                        )
                        # Keep the world model's recommended tool, just add category info
                        context['classifier_category'] = classification.category
                    else:
                        # Normal case: use classifier suggestions
                        context['classifier_suggested_tools'] = classification.suggested_tools
                        context['classifier_category'] = classification.category
                        logger.info(
                            f"{LOG_PREFIX} Using classifier suggested tools: {classification.suggested_tools}"
                        )
                
                # =============================================================
                # FIX #4: Prevent tool override for simple/factual queries
                # =============================================================
                # The LLM classifier correctly identifies simple factual queries
                # (e.g., "What is the capital of France?") but QueryRouter may
                # incorrectly override with specialized tools like ['probabilistic'].
                # Respect the LLM classifier's judgment for these categories.
                # Note: Added CREATIVE and CHITCHAT to skip reasoning
                # Note: Added SELF_INTROSPECTION - these must use world_model tool
                SIMPLE_QUERY_CATEGORIES = frozenset([
                    'FACTUAL', 'CONVERSATIONAL', 'UNKNOWN', 'GREETING',
                    'CREATIVE', 'CHITCHAT',  # Note: Creative/chitchat skip reasoning
                    'factual', 'conversational', 'unknown', 'greeting',
                    'creative', 'chitchat',  # lowercase variants
                ])
                
                # Note: Self-introspection queries MUST use world_model tool
                # These query Vulcan's sense of self (CSIU, motivations, ethics, etc.)
                SELF_INTROSPECTION_CATEGORIES = frozenset([
                    'SELF_INTROSPECTION', 'self_introspection',
                ])
                
                if classification.category in SELF_INTROSPECTION_CATEGORIES:
                    # =================================================================
                    # FIX (Jan 9 2026): Check for domain reasoning keywords FIRST
                    # =================================================================
                    # Problem: Classifier may misclassify causal/analogical queries as
                    # SELF_INTROSPECTION, causing them to be forced to world_model.
                    # Example: "Confounding vs causation (Pearl-style)" classified as
                    # SELF_INTROSPECTION but should route to causal engine.
                    #
                    # Solution: Check for domain-specific keywords before forcing
                    # world_model. If domain keywords are found, route to specialized
                    # engine instead. world_model can still observe but doesn't block.
                    # =================================================================
                    query_lower = query.lower()
                    
                    # Domain keyword sets for specialized routing
                    # Note: These keywords are consistent with CAUSAL_KEYWORDS in query_classifier.py
                    # The threshold of 2+ keywords ensures single false matches don't trigger routing
                    DOMAIN_ROUTING_KEYWORDS = {
                        'causal': frozenset([
                            'causal', 'causation', 'confound', 'confounder', 'confounding',
                            'intervention', 'counterfactual', 'randomize', 'randomized',
                            'pearl', 'dag', 'backdoor', 'frontdoor', 'collider',
                            'do-calculus', 'rct', 'observational', 'experimental',
                        ]),
                        'analogical': frozenset([
                            'analogical', 'analogy', 'analogies', 'analogous',
                            'structure mapping', 'structural alignment',
                            'domain transfer', 'cross-domain', 'source domain', 'target domain',
                            'relational similarity', 'surface similarity', 'structural similarity',
                            's→t', 'domain s', 'domain t', 'deep structure',
                        ]),
                        'probabilistic': frozenset([
                            'bayes', 'bayesian', 'probability', 'probabilistic',
                            'likelihood', 'prior', 'posterior', 'conditional probability',
                            'joint distribution', 'marginal', 'independence',
                        ]),
                    }
                    
                    # Check if query contains domain reasoning keywords
                    detected_domain = None
                    detected_count = 0
                    for domain, keywords in DOMAIN_ROUTING_KEYWORDS.items():
                        count = sum(1 for kw in keywords if kw in query_lower)
                        if count >= 2:  # Require 2+ keywords for domain detection
                            if count > detected_count:
                                detected_domain = domain
                                detected_count = count
                    
                    if detected_domain:
                        # Domain reasoning detected - route to specialized engine
                        logger.info(
                            f"{LOG_PREFIX} SELF_INTROSPECTION override: detected {detected_domain} "
                            f"reasoning ({detected_count} keywords) - routing to {detected_domain} "
                            f"engine instead of world_model"
                        )
                        classification.suggested_tools = [detected_domain]
                        if context is None:
                            context = {}
                        context['classifier_suggested_tools'] = [detected_domain]
                        context['classifier_category'] = classification.category
                        context['domain_reasoning_detected'] = detected_domain
                        context['domain_keyword_count'] = detected_count
                        # Let the specialized engine handle it - don't block with world_model
                        # Note: world_model can still observe in parallel mode
                    else:
                        # No domain keywords found - actual self-introspection
                        # For self-introspection queries, ensure we use world_model tool
                        # BUG FIX: Also update query_type to prevent type mismatch downstream
                        # Previously, query_type stayed as 'MATHEMATICAL' even after overriding tools
                        original_query_type = query_type
                        query_type = 'self_introspection'  # FIX: Update query_type to match actual query
                        
                        logger.info(
                            f"{LOG_PREFIX} SELF_INTROSPECTION detected - using world_model tool "
                            f"(classifier suggested: {classification.suggested_tools}). "
                            f"Updated query_type: {original_query_type} -> {query_type}"
                        )
                        # Ensure world_model is in the suggested tools
                        if 'world_model' not in (classification.suggested_tools or []):
                            classification.suggested_tools = ['world_model']
                        if context is None:
                            context = {}
                        context['classifier_suggested_tools'] = classification.suggested_tools
                        context['prevent_router_tool_override'] = True
                        context['classifier_is_authoritative'] = True
                        context['is_self_introspection'] = True
                        context['original_query_type'] = original_query_type  # FIX: Track original for debugging
                
                elif classification.category in SIMPLE_QUERY_CATEGORIES:
                    # For simple queries, ensure we use general tools
                    if classification.suggested_tools != ['general']:
                        logger.warning(
                            f"{LOG_PREFIX} FIX#4: LLM classifier suggested "
                            f"{classification.suggested_tools} for category "
                            f"{classification.category}, overriding to ['general'] "
                            f"for simple factual query"
                        )
                        classification.suggested_tools = ['general']
                        if context is None:
                            context = {}
                        context['classifier_suggested_tools'] = ['general']
                    
                    # Set flag to prevent router override downstream
                    if context is None:
                        context = {}
                    context['prevent_router_tool_override'] = True
                    context['classifier_is_authoritative'] = True
                    logger.info(
                        f"{LOG_PREFIX} FIX#4: Preventing router tool override - "
                        f"LLM classifier identified this as {classification.category}"
                    )
                    
            except ImportError:
                logger.debug(f"{LOG_PREFIX} QueryClassifier not available, using heuristic fallback")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} QueryClassifier failed: {e}, using heuristic fallback")
            
            # =================================================================
            # FALLBACK: Simple pattern matching for obvious cases
            # This is a safety net if QueryClassifier fails, NOT the primary path
            # =================================================================
            SIMPLE_QUERY_PATTERNS = frozenset([
                'hello', 'hi', 'hey', 'howdy', 'greetings',
                'thanks', 'thank you', 'bye', 'goodbye', 'see you',
                'good morning', 'good afternoon', 'good evening',
                'ok', 'okay', 'sure', 'yes', 'no', 'maybe',
            ])
            
            query_lower = query.lower().strip()
            is_simple_greeting = (
                query_lower in SIMPLE_QUERY_PATTERNS or
                len(query_lower) < 10 and not any(c in query_lower for c in '?∧∨→¬=')
            )
            
            if is_simple_greeting:
                logger.info(
                    f"{LOG_PREFIX} PATTERN FALLBACK: '{query[:20]}' (len={len(query)}) - "
                    f"skipping reasoning entirely"
                )
                with self._stats_lock:
                    self._stats.fast_path_count += 1
                
                return ReasoningResult(
                    selected_tools=["general"],
                    reasoning_strategy=ReasoningStrategyType.DIRECT.value,
                    confidence=0.95,
                    rationale="Simple greeting/conversational - bypassing reasoning",
                    metadata={
                        "fast_path": True,
                        "simple_query_bypass": True,
                        "complexity": 0.0,  # Override upstream complexity
                        "query_type": "conversational",
                        "selection_time_ms": 0.0,
                    },
                )
            
            # Fast path for simple queries - skip heavy tool selection
            if complexity < FAST_PATH_COMPLEXITY_THRESHOLD:
                with self._stats_lock:
                    self._stats.fast_path_count += 1

                return ReasoningResult(
                    selected_tools=["general"],
                    reasoning_strategy=ReasoningStrategyType.DIRECT.value,
                    confidence=0.9,
                    rationale="Simple query - using fast path direct response",
                    metadata={
                        "fast_path": True,
                        "complexity": complexity,
                        "query_type": query_type,
                        "selection_time_ms": 0.0,
                    },
                )

            # ================================================================
            # FIX #1: QUERY PREPROCESSING - REMOVED (architectural band-aid)
            # ================================================================
            # Query preprocessing has been removed. Root causes are now fixed
            # directly in the engines (cryptographic engine header detection,
            # symbolic reasoner query decomposition, etc.)
            # The QueryDecomposer is used directly by the SymbolicReasoner.

            # Check if we should use decomposition for complex queries
            if self._should_use_decomposition(complexity):
                # Use decomposition path for complex queries
                logger.info(
                    f"{LOG_PREFIX} Using decomposition path (complexity={complexity:.2f} >= "
                    f"{DECOMPOSITION_COMPLEXITY_THRESHOLD})"
                )
                result = process_with_decomposition(self, 
                    query, query_type, complexity, context
                )
            else:
                # Use direct tool selection for simpler queries
                result = self._select_with_tool_selector(
                    query, query_type, complexity, context
                )

            # Record timing
            selection_time = (time.perf_counter() - selection_start) * 1000
            self._record_selection_time(selection_time)

            # Add timing to metadata
            result.metadata["selection_time_ms"] = selection_time

            # ================================================================
            # SEMANTIC BRIDGE FIX: Apply cross-domain knowledge transfer
            # ================================================================
            # The semantic bridge enables knowledge learned in one domain to be
            # applied in related domains, improving reasoning quality for queries
            # that span multiple conceptual areas.
            # ================================================================
            if self._should_use_cross_domain_transfer(
                result.selected_tools, 
                {"type": query_type, "complexity": complexity}
            ):
                try:
                    logger.info(
                        f"{LOG_PREFIX} Applying cross-domain knowledge transfer for "
                        f"tools: {result.selected_tools}"
                    )
                    transfer_result = apply_cross_domain_transfer(
                        orchestrator=self,
                        query=query,
                        query_analysis={"type": query_type, "complexity": complexity},
                        selected_tools=result.selected_tools,
                    )
                    
                    if transfer_result.get("success") and transfer_result.get("transfer_count", 0) > 0:
                        logger.info(
                            f"{LOG_PREFIX} Cross-domain transfer applied: "
                            f"{transfer_result['transfer_count']} concepts transferred"
                        )
                        # Add transfer info to metadata
                        result.metadata["cross_domain_transfer"] = transfer_result
                    else:
                        logger.debug(
                            f"{LOG_PREFIX} Cross-domain transfer: {transfer_result.get('error', 'no concepts transferred')}"
                        )
                except Exception as e:
                    # Cross-domain transfer is non-critical - log but don't fail
                    logger.debug(f"{LOG_PREFIX} Cross-domain transfer failed (non-critical): {e}")

            # ================================================================
            # META-REASONING VALIDATION FIX: Validate answer coherence
            # ================================================================
            # This critical validation layer catches cases where:
            # - A mathematical result is returned for a self-introspection query
            # - A calculus answer is returned for a logical/ethical query
            # - WorldModel response is discarded and wrong cached result returned
            #
            # Example bug this fixes:
            #   Query: "what makes you different from other ai systems?"
            #   Wrong answer: "3*x**2" (derivative from cached math result)
            #   Expected: World model introspection about VULCAN's capabilities
            # ================================================================
            if ANSWER_VALIDATOR_AVAILABLE and validate_reasoning_result is not None:
                try:
                    # Extract conclusion from metadata using ordered key preference
                    conclusion = ""
                    if result.metadata:
                        # Keys to check in priority order
                        conclusion_keys = ["conclusion", "world_model_response"]
                        for key in conclusion_keys:
                            conclusion = result.metadata.get(key, "")
                            if conclusion:
                                break
                        
                        # Also check nested reasoning_output if not found
                        if not conclusion:
                            reasoning_output = result.metadata.get("reasoning_output", {})
                            if isinstance(reasoning_output, dict):
                                conclusion = reasoning_output.get("conclusion", "")
                    
                    # Only validate if we have something to validate
                    if conclusion:
                        validation_result = validate_reasoning_result(
                            query=query,
                            result={"conclusion": conclusion},
                            expected_type=query_type
                        )
                        
                        if not validation_result.valid:
                            logger.warning(
                                f"{LOG_PREFIX} META-REASONING VALIDATION FAILED: "
                                f"Answer does not match query type. "
                                f"Query: '{query[:60]}...', "
                                f"Answer type mismatch detected. "
                                f"Explanation: {validation_result.explanation}"
                            )
                            # Mark result as potentially invalid
                            result.metadata["validation_failed"] = True
                            result.metadata["validation_explanation"] = validation_result.explanation
                            # Reduce confidence to signal uncertainty
                            original_confidence = result.confidence
                            result.confidence = min(result.confidence, 0.3)
                            logger.warning(
                                f"{LOG_PREFIX} Confidence reduced from {original_confidence:.2f} "
                                f"to {result.confidence:.2f} due to validation failure"
                            )
                        else:
                            logger.debug(
                                f"{LOG_PREFIX} Meta-reasoning validation passed for query"
                            )
                except Exception as validation_err:
                    # Validation is non-critical - log but don't fail the whole request
                    logger.debug(
                        f"{LOG_PREFIX} Meta-reasoning validation failed (non-critical): {validation_err}"
                    )

            # ================================================================
            # FIX #3: LEARN FROM SUCCESSFUL REASONING
            # ================================================================
            # After successful reasoning, extract reusable principles using
            # KnowledgeCrystallizer. This enables learning patterns like:
            # "SAT queries with explicit constraints need preprocessing"
            try:
                preprocessing_applied = False
                if context and 'preprocessing' in context:
                    prep_result = context['preprocessing']
                    # Handle both dict and PreprocessingResult dataclass
                    if hasattr(prep_result, 'preprocessing_applied'):
                        preprocessing_applied = prep_result.preprocessing_applied
                    elif isinstance(prep_result, dict):
                        preprocessing_applied = prep_result.get('preprocessing_applied', False)

                # Learn from outcome if confidence is high enough
                if result.confidence >= 0.7:
                    logger.info(
                        f"{LOG_PREFIX} LEARNING TRIGGERED: confidence={result.confidence:.2f} >= 0.7"
                    )
                    learn_from_reasoning_outcome(self, 
                        query=query,
                        query_type=query_type,
                        complexity=complexity,
                        selected_tools=result.selected_tools,
                        reasoning_strategy=result.reasoning_strategy,
                        success=True,
                        confidence=result.confidence,
                        execution_time=selection_time / 1000.0,  # Convert ms to seconds
                        preprocessing_applied=preprocessing_applied,
                    )
                else:
                    logger.debug(
                        f"{LOG_PREFIX} Learning skipped: confidence={result.confidence:.2f} < 0.7"
                    )
            except Exception as e:
                # Learning is non-critical - log but don't fail
                logger.debug(f"{LOG_PREFIX} Learning step failed (non-critical): {e}")

            return result

        except Exception as e:
            # Record error and return fallback
            with self._stats_lock:
                self._stats.errors += 1
                self._stats.last_error = str(e)

            logger.error(
                f"{LOG_PREFIX} Reasoning application failed: {e}",
                exc_info=True
            )

            return self._create_default_result(query_type, complexity)
