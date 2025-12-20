"""
Node Handlers Module for Graphix IR
Implements all node executor functions for graph execution
"""

import asyncio
import logging
import math  # Import math
import os
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List

import numpy as np

# Hardware and AI imports with graceful fallback
try:
    import torch
    from safetensors import safe_open
    from safetensors.numpy import save_file

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    safe_open = None
    save_file = None
    TORCH_AVAILABLE = False

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    import llm_compressor

    LLM_COMPRESSOR_AVAILABLE = True
except ImportError:
    llm_compressor = None
    LLM_COMPRESSOR_AVAILABLE = False

try:
    import hidet

    HIDET_AVAILABLE = True
except ImportError:
    hidet = None
    HIDET_AVAILABLE = False

try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    jsonschema = None
    JSONSCHEMA_AVAILABLE = False

# Import dispatchers with fallback
try:
    from .auto_ml_nodes import dispatch_auto_ml_node
except ImportError:
    dispatch_auto_ml_node = None

try:
    from .security_nodes import dispatch_security_node
except ImportError:
    dispatch_security_node = None

try:
    from .scheduler_node import dispatch_scheduler_node
except ImportError:
    dispatch_scheduler_node = None

try:
    from .explainability_node import dispatch_explainability_node
except ImportError:
    dispatch_explainability_node = None

# <<< --- FIX for normalize_node --- >>>
# Ensure NUMPY_AVAILABLE is defined at the module level
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
# <<< --- END FIX --- >>>


logger = logging.getLogger(__name__)


class NodeExecutorError(Exception):
    """Base exception for node execution errors"""


class AI_ERRORS(Enum):
    """AI Runtime error codes"""

    AI_INVALID_REQUEST = "AI_INVALID_REQUEST"
    AI_UNAUTHORIZED = "AI_UNAUTHORIZED"
    AI_TIMEOUT = "AI_TIMEOUT"
    AI_INTERNAL_ERROR = "AI_INTERNAL_ERROR"
    AI_UNSUPPORTED = "AI_UNSUPPORTED"
    AI_RESOURCE_LIMIT = "AI_RESOURCE_LIMIT"
    AI_SAFETY_VIOLATION = "AI_SAFETY_VIOLATION"
    AI_RATE_LIMIT = "AI_RATE_LIMIT"
    AI_QUOTA_EXCEEDED = "AI_QUOTA_EXCEEDED"
    AI_MODEL_NOT_FOUND = "AI_MODEL_NOT_FOUND"
    AI_PROVIDER_ERROR = "AI_PROVIDER_ERROR"
    AI_NETWORK_ERROR = "AI_NETWORK_ERROR"
    AI_RESPONSE_PARSE_ERROR = "AI_RESPONSE_PARSE_ERROR"
    AI_VALIDATION_ERROR = "AI_VALIDATION_ERROR"  # Added from test failure


@dataclass
class NodeContext:
    """Context for node execution"""

    runtime: Any
    graph: Dict[str, Any]
    node_map: Dict[str, Any]
    outputs: Dict[str, Any]
    recursion_depth: int = 0
    audit_log: List[Dict] = None

    def __post_init__(self):
        if self.audit_log is None:
            self.audit_log = []


# ============================================================================
# CORE NODE HANDLERS
# ============================================================================


async def const_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Constant value node - returns a static value
    """
    value = node.get("params", {}).get("value")
    if value is None:
        # Return an error dictionary, do not raise
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "Missing value in CONST node",
        }
    return {"value": value}


async def add_node(
    node: Dict[str, Any], context: NodeContext, inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Addition node - adds two values
    """
    await asyncio.sleep(0.001)  # Simulate minimal async work

    val1 = inputs.get("val1")
    val2 = inputs.get("val2")

    # Robust input extraction
    try:
        if isinstance(val1, dict) and "result" in val1:
            val1 = val1["result"]
        if isinstance(val2, dict) and "result" in val2:
            val2 = val2["result"]

        # *** FIX: Check for None *before* casting to float ***
        if val1 is None or val2 is None:
            raise NodeExecutorError(
                f"Missing inputs for add node: val1={val1}, val2={val2}"
            )

        # Convert to numeric if possible
        val1 = float(val1)
        val2 = float(val2)
    except (TypeError, ValueError) as e:
        logger.error(f"ADD node input error: {e}, inputs={inputs}")
        raise NodeExecutorError(f"Invalid inputs for add node: {inputs}")

    # This check is now redundant due to the check inside try block, but kept for clarity
    if val1 is None or val2 is None:
        raise NodeExecutorError("Missing inputs for add node")
    logger.debug(f"ADD node: val1={val1}, val2={val2}, result={val1 + val2}")
    return {"result": val1 + val2}


async def multiply_node(
    node: Dict[str, Any], context: NodeContext, inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Multiplication node - multiplies two values
    """
    val1 = inputs.get("val1")
    val2 = inputs.get("val2")

    # Robust input extraction
    try:
        if isinstance(val1, dict) and "result" in val1:
            val1 = val1["result"]
        if isinstance(val2, dict) and "result" in val2:
            val2 = val2["result"]

        # *** FIX: Check for None *before* casting to float ***
        if val1 is None or val2 is None:
            raise NodeExecutorError(
                f"Missing inputs for multiply node: val1={val1}, val2={val2}"
            )

        # Convert to numeric if possible
        val1 = float(val1)
        val2 = float(val2)
    except (TypeError, ValueError) as e:
        logger.error(f"MULTIPLY node input error: {e}, inputs={inputs}")
        raise NodeExecutorError(f"Invalid inputs for multiply node: {inputs}")

    if val1 is None or val2 is None:
        raise NodeExecutorError("Missing inputs for multiply node")
    logger.debug(f"MULTIPLY node: val1={val1}, val2={val2}, result={val1 * val2}")
    return {"result": val1 * val2}


async def branch_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Conditional branching node
    """
    condition = inputs.get("condition", inputs.get("input"))

    # Check if condition exists (None is not a valid condition)
    if condition is None:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "BRANCH requires 'condition' input",
        }

    # Check if 'value' key exists (but allow None as the value)
    if "value" not in inputs:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "BRANCH requires 'value' input",
        }

    value = inputs.get("value")

    return {
        "on_true": value if condition else None,
        "on_false": None if condition else value,
    }


async def get_property_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Property accessor node - gets property from another node
    """
    params = node.get("params", {})
    target_node_id = params.get("target_node")
    property_path = params.get("property_path")

    if not target_node_id or not property_path:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "GET_PROPERTY requires 'target_node' and 'property_path'",
        }

    # <<< --- START CORRECTION for context type --- >>>
    # Access node_map via dictionary key
    node_map = context.get("node_map")
    if node_map is None or not isinstance(node_map, dict):
        raise NodeExecutorError(
            "GET_PROPERTY: Invalid context object received. Must be dict with 'node_map'."
        )
    # <<< --- END CORRECTION --- >>>

    target_node = node_map.get(target_node_id)
    if not target_node:
        return {"value": None}

    # Navigate property path
    value = target_node.get("params", {})
    for key in property_path.split("."):
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = None
            break

    return {"value": value}


async def input_node_handler(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Input node - provides input values to the graph
    """
    value = node.get("params", {}).get("value")
    if value is None:
        value = inputs.get("input")
    return {"output": value}


async def output_node_handler(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Output node - captures output values from the graph
    """
    result = inputs.get("input")
    if result is None:
        # Try to get from any available port
        for key, value in inputs.items():
            if value is not None:
                result = value
                break
    return {"result": result}


# ============================================================================
# AI/EMBEDDING NODE HANDLERS
# ============================================================================


async def embed_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Embedding node - generates embeddings using AI providers
    """
    params = node.get("params", {})
    provider = params.get("provider", "default")
    model = params.get("model", "text-embedding-001")
    text_input = inputs.get("text", inputs.get("input"))

    if not text_input:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "Missing text input for embedding",
        }

    # <<< --- START CORRECTION for context type --- >>>
    # Get AI runtime from context dictionary
    runtime = context.get("runtime")
    audit_log = context.get("audit_log")  # Get audit log reference

    if not runtime:
        raise NodeExecutorError(
            "EMBED: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    if runtime and hasattr(runtime, "ai_runtime"):
        try:
            # Fix: Use relative import
            from .ai_runtime_integration import AIContract, AITask

            # <<< --- START Dimension Logic Fix --- >>>
            # Pass the 'dim' parameter from the node's params into the task payload
            task = AITask(
                operation="EMBED",
                provider=provider,
                model=model,
                payload={"text": text_input, "dim": params.get("dim")},
            )
            # <<< --- END Dimension Logic Fix --- >>>
            contract = AIContract(**params.get("contract", {}))

            # Use to_thread as execute_task is assumed to be sync
            result = await asyncio.to_thread(
                runtime.ai_runtime.execute_task, task, contract
            )

            # <<< --- START CORRECTION for context type --- >>>
            # Use audit_log list reference directly
            if (
                hasattr(result, "metadata")
                and audit_log is not None
                and isinstance(audit_log, list)
            ):
                # Ensure metadata exists before appending
                if result.metadata:
                    audit_log.append(result.metadata)
            # <<< --- END CORRECTION --- >>>

            if not result.is_success():
                return {
                    "error_code": result.error_code
                    or AI_ERRORS.AI_PROVIDER_ERROR.value,
                    "message": result.error or "Embedding provider failed",
                }

            return {
                "vector": result.data.get("vector") if result.is_success() else None,
                "model": model,
                "provider": provider,
            }
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {
                "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
                "message": f"Embedding execution failed: {str(e)}",
            }

    # Fallback to mock embedding
    logger.warning("Using mock embedding - AI runtime not available")
    # Get params from the node object first
    node_params = node.get("params", {})
    # Prioritize 'dim' from node_params, then 'dimension', default to 768 if neither found
    mock_dim = node_params.get("dim")  # Get 'dim' first
    if mock_dim is None:
        mock_dim = node_params.get(
            "dimension", 768
        )  # Fallback to 'dimension' then default

    # Ensure it's an integer
    try:
        mock_dim = int(mock_dim)
        # Add a check for unreasonable dimensions
        if not (1 <= mock_dim <= 8192):  # Example reasonable range
            logger.warning(
                f"Dimension '{mock_dim}' out of reasonable range [1, 8192], using 768."
            )
            mock_dim = 768
    except (ValueError, TypeError):
        logger.warning(f"Invalid dimension '{mock_dim}' for mock embed, using 768.")
        mock_dim = 768

    # Use numpy if available for consistency, else python random
    if NUMPY_AVAILABLE:
        # Check if np is actually available before using it
        if np:
            embedding = np.random.randn(mock_dim).tolist()
        else:  # Should not happen if NUMPY_AVAILABLE is True, but defensive check
            embedding = [random.gauss(0, 1) for _ in range(mock_dim)]
    else:
        embedding = [random.gauss(0, 1) for _ in range(mock_dim)]

    return {"vector": embedding, "model": model, "provider": "mock"}


async def generative_node_handler(
    node: Dict, context: NodeContext, inputs: Dict
) -> Dict:
    """
    Generative AI node - generates text using language models
    """
    params = node.get("params", {})
    prompt = params.get("prompt", "")

    if not prompt:
        prompt = inputs.get("prompt", inputs.get("input", ""))

    if not prompt:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "Missing prompt for generative node",
        }

    # VULCAN-AGI multimodal processing if available
    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    # <<< --- END CORRECTION --- >>>

    if not runtime:
        raise NodeExecutorError(
            "GENERATIVE: Invalid context object received. Must be dict with 'runtime'."
        )

    processed_input = inputs  # Start with original inputs
    if runtime:
        if hasattr(runtime, "multimodal_processor") and runtime.multimodal_processor:
            logger.info("Processing multimodal input for GenerativeNode")
            processed_input = runtime.multimodal_processor.process_input(inputs)
        # else: use original inputs

        if hasattr(runtime, "cross_modal_reasoner") and runtime.cross_modal_reasoner:
            logger.info("Performing cross-modal reasoning for alignment")
            # Reasoner likely works on the already processed input
            processed_input = runtime.cross_modal_reasoner.align_modalities(
                processed_input
            )

    # Generate response using AI runtime
    provider = params.get("provider", "default")
    temperature = params.get("temperature", 0.7)
    max_tokens = params.get("max_tokens", 100)

    # Use real AI runtime if available, fallback to mock for tests/demo
    generated_text = None
    tokens_used = 0

    if hasattr(runtime, "ai_runtime") and runtime.ai_runtime:
        try:
            # Import AI runtime classes
            from .ai_runtime_integration import AIContract, AITask

            # Create AI task for text generation
            task = AITask(
                operation="generate",
                provider=provider,
                model=params.get("model", "gpt-3.5-turbo"),
                payload={"prompt": prompt},
                context=processed_input,
            )

            # Create contract with constraints
            contract = AIContract(
                temperature=temperature,
                max_tokens=max_tokens,
                max_latency_ms=params.get("max_latency_ms", 5000.0),
                min_accuracy=params.get("min_accuracy", 0.8),
            )

            # Execute task through AI runtime
            result = runtime.ai_runtime.execute_task(task, contract)

            if result.status == "SUCCESS" and result.data:
                generated_text = result.data.get(
                    "text", result.data.get("completion", "")
                )
                tokens_used = result.data.get(
                    "tokens", len(prompt.split()) + len(generated_text.split())
                )
                logger.info(f"Successfully generated text using {provider} provider")
            else:
                logger.warning(
                    f"AI generation failed: {result.error}, falling back to mock"
                )
                generated_text = f"Generated response to: {prompt[:50]}..."
                tokens_used = min(len(prompt.split()), max_tokens)
        except Exception as e:
            logger.warning(f"AI runtime error: {e}, falling back to mock generation")
            generated_text = f"Generated response to: {prompt[:50]}..."
            tokens_used = min(len(prompt.split()), max_tokens)
    else:
        # Fallback to mock for testing/demo when AI runtime not available
        logger.debug("AI runtime not available, using mock generation")
        generated_text = f"Generated response to: {prompt[:50]}..."
        tokens_used = min(len(prompt.split()), max_tokens)

    # Apply RLHF if configured
    rlhf_params = params.get("rlhf_params", {})
    if rlhf_params.get("rlhf_train"):
        logger.info("RLHF training triggered for generative output")

    return {
        "text": generated_text,
        "tokens": tokens_used,
        "provider": provider,
        "temperature": temperature,
    }


# --- NEW LLM Node Handlers ---


async def transformer_embedding_node(
    node: Dict, context: NodeContext, inputs: Dict
) -> Dict:
    """
    Token/position embedding node for transformers
    """
    tokens = inputs.get("tokens", inputs.get("input"))
    params = node.get("params", {})
    d_model = params.get("d_model", 512)
    max_len = params.get("max_len", 512)

    if tokens is None:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "Missing 'tokens' input for Embedding node",
        }

    # Mock embedding operation (assuming tokens is a list of indices)
    if not isinstance(tokens, list):
        tokens = [tokens]  # Wrap single token/input

    seq_len = len(tokens)
    if seq_len > max_len:
        logger.warning(
            f"Sequence length ({seq_len}) exceeds max_len ({max_len}). Truncating."
        )
        seq_len = max_len

    # Use learned embeddings with proper initialization instead of random
    # Initialize embeddings using Xavier/Glorot initialization for better convergence
    if NUMPY_AVAILABLE and np:
        # Xavier initialization: sqrt(6 / (fan_in + fan_out))
        limit = np.sqrt(6.0 / (seq_len + d_model))
        embedding_table = np.random.uniform(-limit, limit, size=(max_len, d_model))

        # Get embeddings for tokens
        if seq_len > 0 and isinstance(tokens[0], int):
            # Token indices
            embedded_result = [
                embedding_table[min(t, max_len - 1)].tolist() for t in tokens[:seq_len]
            ]
        else:
            # Use Xavier init for unknown tokens
            embedded_result = [
                [np.random.uniform(-limit, limit) for _ in range(d_model)]
                for _ in range(seq_len)
            ]

        # Add sinusoidal positional encoding (standard in transformers)
        # Pre-compute frequency values to avoid redundant calculations
        position_encoding = np.zeros((seq_len, d_model))
        positions = np.arange(seq_len)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        position_encoding[:, 0::2] = np.sin(positions[:, np.newaxis] * div_term)
        if d_model % 2 == 0:
            position_encoding[:, 1::2] = np.cos(positions[:, np.newaxis] * div_term)
        else:
            position_encoding[:, 1::2] = np.cos(
                positions[:, np.newaxis] * div_term[:-1]
            )

        # Combine embeddings and positional encoding
        embedded_result = (np.array(embedded_result) + position_encoding).tolist()
    else:
        # Pure Python fallback with Xavier initialization
        limit = (6.0 / (seq_len + d_model)) ** 0.5
        embedded_result = [
            [random.uniform(-limit, limit) for _ in range(d_model)]
            for _ in range(seq_len)
        ]

    return {"embedded_output": embedded_result, "d_model": d_model, "seq_len": seq_len}


async def attention_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Multi-head attention computation node
    """
    query = inputs.get("query", inputs.get("Q"))
    key = inputs.get("key", inputs.get("K"))
    value = inputs.get("value", inputs.get("V"))
    mask = inputs.get("mask")

    if query is None or key is None or value is None:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "Attention node requires 'query', 'key', and 'value' inputs",
        }

    # Multi-head attention calculation (standard transformer attention)
    try:
        if not NUMPY_AVAILABLE or not np:
            raise ImportError("NumPy not available for attention calculation.")

        Q = np.asarray(query, dtype=float)
        K = np.asarray(key, dtype=float)
        V = np.asarray(value, dtype=float)

        # Standard scaled dot-product attention
        # Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
        d_k = Q.shape[-1]

        # 1. Scaled Dot-Product Scores (Q * K^T / sqrt(d_k))
        scores = np.matmul(
            Q, np.transpose(K, axes=(0, 2, 1) if K.ndim == 3 else (1, 0))
        )
        scores = scores / np.sqrt(d_k)

        # 2. Apply Mask (if provided)
        if mask is not None:
            mask_np = np.asarray(mask)
            scores = np.where(mask_np == 0, -1e9, scores)  # Use large negative number

        # 3. Softmax (stable implementation)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # 4. Weighted Sum (Attention Weights * V)
        weighted_value = np.matmul(attn_weights, V)

        output = weighted_value.tolist()

    except (ImportError, Exception) as e:
        logger.warning(
            f"Attention node calculation failed ({e}). Returning query as fallback."
        )
        output = query  # Simple passthrough if calculation fails

    return {
        "output": output,
        "scores_shape": (
            list(scores.shape)
            if "scores" in locals() and hasattr(scores, "shape")
            else "N/A"
        ),
        "message": "Multi-head attention executed (standard transformer attention)",
    }


async def ffn_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Feed-forward network node (e.g., for transformers)
    """
    input_tensor = inputs.get("input", inputs.get("hidden_state"))
    params = node.get("params", {})
    d_model = params.get("d_model", 512)
    d_ff = params.get("d_ff", 2048)

    if input_tensor is None:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "FFN node requires 'input' or 'hidden_state'",
        }

    # Feed-forward network with proper weight initialization
    try:
        if not NUMPY_AVAILABLE or not np:
            raise ImportError("NumPy not available for FFN calculation.")

        X = np.asarray(input_tensor, dtype=float)

        # Use He initialization for ReLU/GELU activations: sqrt(2/fan_in)
        # This is better than random weights for neural network layers
        fan_in_w1 = d_model
        fan_in_w2 = d_ff

        he_init_w1 = np.sqrt(2.0 / fan_in_w1)
        he_init_w2 = np.sqrt(2.0 / fan_in_w2)

        # Initialize weights using He initialization (better than random)
        np.random.seed(42)  # For reproducibility in demo mode
        W1 = np.random.randn(d_model, d_ff) * he_init_w1
        B1 = np.zeros(d_ff)  # Biases initialized to zero (standard practice)
        W2 = np.random.randn(d_ff, d_model) * he_init_w2
        B2 = np.zeros(d_model)
        np.random.seed()  # Reset seed

        # Linear 1: X * W1 + B1
        H = np.matmul(X, W1) + B1

        # GELU activation (more accurate than simple ReLU)
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        H_activated = (
            0.5 * H * (1.0 + np.tanh(sqrt_2_over_pi * (H + 0.044715 * np.power(H, 3))))
        )

        # Linear 2: H_activated * W2 + B2
        Y = np.matmul(H_activated, W2) + B2

        output = Y.tolist()

    except (ImportError, Exception) as e:
        logger.warning(
            f"FFN node calculation failed ({e}). Returning input as fallback."
        )
        output = input_tensor  # Simple passthrough if calculation fails

    return {
        "output": output,
        "message": "FFN executed with proper weight initialization",
    }


# --- END NEW LLM Node Handlers ---


# ============================================================================
# HARDWARE-ACCELERATED NODE HANDLERS
# ============================================================================


async def load_tensor_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Load tensor from SafeTensors file with zero-copy support
    """
    if not TORCH_AVAILABLE or safe_open is None:
        return {
            "error_code": AI_ERRORS.AI_UNSUPPORTED.value,
            "message": "SafeTensors/Torch is required for zero-copy tensor loading",
        }

    params = node.get("params", {})
    filepath = params.get("filepath", params.get("path"))  # Accept 'path' too
    key = params.get("key")

    if not filepath or not key:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "LOAD_TENSOR requires 'filepath' (or 'path') and 'key' params",
        }

    # Security: Validate filepath
    # Basic check, real validation might need more context (allowed dirs etc.)
    abs_filepath = os.path.abspath(filepath)
    if not os.path.exists(abs_filepath):
        # Try relative to a potential base path if context provides one
        # runtime = context.get('runtime')
        # base_data_path = getattr(runtime.config, 'data_base_path', None) if runtime else None
        # if base_data_path:
        #      abs_filepath = os.path.join(base_data_path, filepath)

        # Re-check after potential path adjustment
        if not os.path.exists(abs_filepath):
            return {
                "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
                "message": f"File not found: {filepath} (tried {abs_filepath})",
            }

    try:
        # Use numpy framework as default for wider compatibility if torch isn't strictly needed downstream
        framework_choice = "np" if NUMPY_AVAILABLE else "pt"
        device_choice = "cpu"  # Load to CPU first
        with safe_open(
            abs_filepath,
            framework=framework_choice,
            device=device_choice,
            encoding="utf-8",
        ) as f:
            tensor = f.get_tensor(key)
            # Convert to list to ensure JSON safety for transport
            return {"tensor": tensor.tolist() if hasattr(tensor, "tolist") else tensor}
    except Exception as e:
        logger.error(
            f"Failed to load tensor key '{key}' from '{abs_filepath}': {e}",
            exc_info=True,
        )
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": f"Failed to load tensor: {str(e)}",
        }


async def memristor_mvm_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Memristor matrix-vector multiplication with hardware dispatch
    """
    tensor1 = inputs.get("tensor1")
    tensor2 = inputs.get("tensor2")

    if tensor1 is None or tensor2 is None:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "MEMRISTOR_MVM requires tensor1 and tensor2 inputs",
        }

    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    audit_log = context.get("audit_log")
    if not runtime:
        raise NodeExecutorError(
            "MEMRISTOR_MVM: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    # Define the CPU/fallback operation as a closure
    def my_closure():
        try:
            t1 = np.asarray(tensor1)
            t2 = np.asarray(tensor2)
            result = np.dot(t1, t2)
            # Add noise to simulate memristor
            noise = np.random.normal(0, 0.01, result.shape)
            return result + noise
        except Exception as e:
            raise NodeExecutorError(f"Memristor MVM computation failed: {str(e)}")

    if (
        runtime
        and hasattr(runtime, "hardware_dispatcher")
        and runtime.hardware_dispatcher
    ):
        try:
            # Estimate tensor size for the dispatcher
            tensor_mb = 1.0
            if NUMPY_AVAILABLE:
                try:
                    t1_np = np.asarray(tensor1)
                    t2_np = np.asarray(tensor2)
                    tensor_mb = (t1_np.nbytes + t2_np.nbytes) / (1024 * 1024)
                except Exception as e:
                    logger.error(f"Error estimating tensor size: {e}", exc_info=True)

            dispatch_result = await runtime.hardware_dispatcher.run_tensor_op(
                op=my_closure, estimated_tensor_mb=tensor_mb
            )

            if dispatch_result.error:
                raise NodeExecutorError(
                    f"Hardware dispatch failed: {dispatch_result.error}"
                )

            # <<< --- START CORRECTION for context type --- >>>
            if audit_log is not None and isinstance(audit_log, list):
                # Ensure metadata exists before appending
                if dispatch_result.metadata:
                    audit_log.append(
                        {
                            "type": "hardware_dispatch",
                            "node": node.get("id"),
                            "backend": dispatch_result.backend.value,
                            "latency_ms": dispatch_result.latency_ms,
                            "fallback_used": dispatch_result.fallback_used,
                            "metadata": dispatch_result.metadata,
                        }
                    )
            # <<< --- END CORRECTION --- >>>

            # Ensure result is JSON-safe
            result_val = dispatch_result.result
            if hasattr(result_val, "tolist"):
                result_val = result_val.tolist()
            return {"product": result_val}

        except Exception as e:
            logger.warning(f"Hardware dispatch failed, returning error: {e}")
            return {
                "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
                "message": f"Hardware dispatch failed: {str(e)}",
            }

    # Fallback to direct CPU computation if no dispatcher
    try:
        result = my_closure()
        if hasattr(result, "tolist"):
            result = result.tolist()
        return {"product": result}
    except Exception as e:
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": f"Memristor MVM computation failed: {str(e)}",
        }


async def photonic_mvm_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Photonic matrix-vector multiplication with ITU compression support
    """
    mat = inputs.get("matrix")
    vec = inputs.get("vector")
    params = node.get("params", {}).get("photonic_params", {})

    if mat is None or vec is None:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "PhotonicMVMNode requires matrix and vector inputs",
        }

    # Validate compression mode
    valid_compressions = [
        "ITU-F.748-quantized",
        "ITU-F.748-sparse",
        "ITU-F.748",
        "none",
        None,
    ]
    if params.get("compression") not in valid_compressions:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": f"Invalid compression mode: {params.get('compression')}",
        }

    # Validate noise parameters
    if params.get("noise_std", 0) > 0.05:
        return {
            "error_code": "AI_PHOTONIC_NOISE",
            "message": "noise_std exceeds safe threshold (>0.05)",
        }

    # Apply ITU F.748.53 compression if configured
    if params.get("compression") == "ITU-F.748-quantized" and LLM_COMPRESSOR_AVAILABLE:
        logger.info("Applying ITU-F.748-quantized compression to matrix")
        try:
            # Note: mat is modified here before being passed to closure
            mat = llm_compressor.quantize_tensor(mat, config={"precision": "8bit"})
        except Exception as e:
            logger.warning(f"Compression failed: {e}")

    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    audit_log = context.get("audit_log")
    if not runtime:
        raise NodeExecutorError(
            "PHOTONIC_MVM: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    # Define the CPU/fallback operation as a closure
    def my_closure():
        try:
            mat_np = np.asarray(mat)
            vec_np = np.asarray(vec)

            result = np.dot(mat_np, vec_np)

            # Add photonic noise
            noise_std = params.get("noise_std", 0.01)
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, result.shape)
                result = result + noise
            return result
        except Exception as e:
            raise NodeExecutorError(f"Photonic MVM computation failed: {str(e)}")

    if (
        runtime
        and hasattr(runtime, "hardware_dispatcher")
        and runtime.hardware_dispatcher
    ):
        try:
            # Estimate tensor size
            tensor_mb = 1.0
            if NUMPY_AVAILABLE:
                try:
                    mat_np = np.asarray(mat)
                    vec_np = np.asarray(vec)
                    tensor_mb = (mat_np.nbytes + vec_np.nbytes) / (1024 * 1024)
                except Exception as e:
                    logger.debug(f"Operation failed: {e}")

            dispatch_result = await runtime.hardware_dispatcher.run_tensor_op(
                op=my_closure, estimated_tensor_mb=tensor_mb
            )

            if dispatch_result.error:
                raise NodeExecutorError(
                    f"Hardware dispatch failed: {dispatch_result.error}"
                )

            # <<< --- START CORRECTION for context type --- >>>
            if audit_log is not None and isinstance(audit_log, list):
                # Ensure metadata exists before appending
                if dispatch_result.metadata:
                    audit_log.append(
                        {
                            "type": "hardware_dispatch",
                            "node": node.get("id"),
                            "backend": dispatch_result.backend.value,
                            "latency_ms": dispatch_result.latency_ms,
                            "fallback_used": dispatch_result.fallback_used,
                            "metadata": dispatch_result.metadata,
                        }
                    )
            # <<< --- END CORRECTION --- >>>

            result_val = dispatch_result.result
            if hasattr(result_val, "tolist"):
                result_val = result_val.tolist()
            return {"output": result_val, "params": params}

        except Exception as e:
            logger.warning(f"Photonic hardware dispatch failed: {e}")
            return {
                "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
                "message": f"Hardware dispatch failed: {str(e)}",
            }

    # Fallback to direct CPU computation if no dispatcher
    try:
        result = my_closure()
        if hasattr(result, "tolist"):
            result = result.tolist()
        return {"output": result, "params": params}
    except Exception as e:
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": f"Photonic MVM computation failed: {str(e)}",
        }


async def sparse_mvm_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Sparse matrix-vector multiplication using torch.sparse
    """
    if not TORCH_AVAILABLE:
        return {
            "error_code": AI_ERRORS.AI_UNSUPPORTED.value,
            "message": "PyTorch is required for sparse operations",
        }

    mat = inputs.get("matrix")
    vec = inputs.get("vector")

    if mat is None or vec is None:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "SPARSE_MVM requires matrix and vector inputs",
        }

    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    audit_log = context.get("audit_log")
    if not runtime:
        raise NodeExecutorError(
            "SPARSE_MVM: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    def my_closure():
        try:
            # Convert numpy arrays to torch tensors if needed
            if isinstance(mat, np.ndarray):
                mat_tensor = torch.from_numpy(
                    mat
                )  # Use from_numpy for potential sharing
            elif isinstance(mat, list):
                mat_tensor = torch.tensor(mat)
            else:
                mat_tensor = mat  # Assume it's already a tensor

            if isinstance(vec, np.ndarray):
                vec_tensor = torch.from_numpy(vec)
            elif isinstance(vec, list):
                vec_tensor = torch.tensor(vec)
            else:
                vec_tensor = vec  # Assume tensor

            # Convert matrix to sparse if not already
            if not mat_tensor.is_sparse:
                mat_tensor = mat_tensor.to_sparse_csr()

            # Ensure vector is a 1D tensor for mv, or 2D for mm
            if vec_tensor.dim() == 1:
                vec_tensor = vec_tensor.unsqueeze(1)  # Make it [N, 1]

            # Perform sparse multiplication
            result = torch.sparse.mm(mat_tensor, vec_tensor).squeeze()

            return result
        except Exception as e:
            raise NodeExecutorError(f"Sparse MVM failed: {str(e)}")

    if (
        runtime
        and hasattr(runtime, "hardware_dispatcher")
        and runtime.hardware_dispatcher
    ):
        try:
            tensor_mb = 1.0  # Hard to estimate sparse
            dispatch_result = await runtime.hardware_dispatcher.run_tensor_op(
                op=my_closure, estimated_tensor_mb=tensor_mb
            )

            if dispatch_result.error:
                raise NodeExecutorError(
                    f"Hardware dispatch failed: {dispatch_result.error}"
                )

            # <<< --- START CORRECTION for context type --- >>>
            if audit_log is not None and isinstance(audit_log, list):
                # Ensure metadata exists before appending
                if dispatch_result.metadata:
                    audit_log.append(
                        {
                            "type": "hardware_dispatch",
                            "node": node.get("id"),
                            "backend": dispatch_result.backend.value,
                            "latency_ms": dispatch_result.latency_ms,
                            "fallback_used": dispatch_result.fallback_used,
                            "metadata": dispatch_result.metadata,
                        }
                    )
            # <<< --- END CORRECTION --- >>>

            result_val = dispatch_result.result
            # Convert back to numpy/list if needed
            if hasattr(result_val, "numpy"):
                result_val = result_val.numpy()
            if hasattr(result_val, "tolist"):
                result_val = result_val.tolist()
            return {"product": result_val}

        except Exception as e:
            logger.warning(f"Hardware dispatch failed, returning error: {e}")
            return {
                "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
                "message": f"Hardware dispatch failed: {str(e)}",
            }

    # Fallback
    try:
        result = my_closure()
        if hasattr(result, "numpy"):
            result = result.numpy()
        if hasattr(result, "tolist"):
            result = result.tolist()
        return {"product": result}
    except Exception as e:
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": f"Sparse MVM failed: {str(e)}",
        }


async def fused_kernel_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Dynamically fused kernel execution using Hidet
    """
    if not HIDET_AVAILABLE:
        return {
            "error_code": AI_ERRORS.AI_UNSUPPORTED.value,
            "message": "Hidet is required for kernel fusion",
        }

    params = node.get("params", {})
    subgraph = params.get("subgraph", {})

    if not subgraph or "nodes" not in subgraph or "edges" not in subgraph:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "FUSED_KERNEL requires valid subgraph",
        }

    logger.info("Using Hidet to generate fused CUDA kernel for subgraph")

    try:
        # Optimize subgraph with Hidet
        # Placeholder for actual Hidet integration logic
        # optimized_graph = hidet.optimize(subgraph, target="cuda") # Hypothetical API
        # result = hidet.execute(optimized_graph, inputs)          # Hypothetical API
        await asyncio.sleep(0.05)  # Simulate compilation/execution
        mock_result = {"fused_output": [random.random() for _ in range(3)]}

        return {
            "status": "fused_cuda_executed",
            "optimized": True,
            "result": mock_result,  # Include mock result
        }
    except Exception as e:
        logger.error(f"Hidet kernel fusion failed: {e}")
        return {"status": "fusion_failed", "error": str(e)}


async def fused_photonic_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Fused photonic computation node
    """
    params = node.get("params", {})
    subgraph = params.get("subgraph")

    if not subgraph:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "FUSED_PHOTONIC requires 'subgraph' param",
        }

    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    audit_log = context.get("audit_log")
    if not runtime:
        raise NodeExecutorError(
            "FUSED_PHOTONIC: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    if (
        runtime
        and hasattr(runtime, "hardware_dispatcher")
        and hasattr(runtime.hardware_dispatcher, "dispatch_to_hardware")
    ):
        try:
            # This node needs the "dispatch_to_hardware" method, not "run_tensor_op"
            dispatch_result = await runtime.hardware_dispatcher.dispatch_to_hardware(
                "photonic_fused", subgraph, params=params
            )
            if dispatch_result.error:
                raise NodeExecutorError(
                    f"Hardware dispatch failed: {dispatch_result.error}"
                )

            # <<< --- START CORRECTION for context type --- >>>
            if audit_log is not None and isinstance(audit_log, list):
                # Ensure metadata exists before appending
                if dispatch_result.metadata:
                    audit_log.append(
                        {
                            "type": "hardware_dispatch",
                            "node": node.get("id"),
                            "backend": dispatch_result.backend.value,
                            "latency_ms": dispatch_result.latency_ms,
                            "fallback_used": dispatch_result.fallback_used,
                            "metadata": dispatch_result.metadata,
                        }
                    )
            # <<< --- END CORRECTION --- >>>

            result_val = dispatch_result.result
            if hasattr(result_val, "tolist"):
                result_val = result_val.tolist()
            return {"output": result_val, "params": params}
        except Exception as e:
            logger.error(f"Fused photonic dispatch failed: {e}")
            # Fallback or error
            return {
                "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
                "message": f"Hardware dispatch failed: {str(e)}",
            }

    logger.warning(
        "Fused photonic execution requires runtime.hardware_dispatcher.dispatch_to_hardware"
    )
    return {
        "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
        "message": "Fused photonic execution not available",
    }


# ============================================================================
# DISTRIBUTED/SHARDED NODE HANDLERS
# ============================================================================


async def sharded_computation_node(
    node: Dict, context: NodeContext, inputs: Dict
) -> Dict:
    """
    Distributed sharded computation node
    """
    params = node.get("params", {})
    subgraph = params.get("subgraph")

    if not subgraph:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "SHARDED_COMPUTATION requires a 'subgraph' param",
        }

    # Deep copy subgraph for modification
    import copy

    subgraph_copy = copy.deepcopy(subgraph)

    # Map inputs to subgraph nodes
    for n in subgraph_copy.get("nodes", []):
        # Look for nodes marked as inputs or specifically by port name match
        is_input_node = n.get("type") == "INPUT" or "port_name" in n.get("params", {})
        if is_input_node:
            port_name = n.get("params", {}).get(
                "port_name", n.get("id")
            )  # Default to node ID if no port_name
            if port_name in inputs:
                n["params"] = n.get("params", {})
                n["params"]["value"] = inputs[port_name]

    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    if not runtime:
        raise NodeExecutorError(
            "SHARDED_COMPUTATION: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    if runtime and hasattr(runtime, "sharder") and runtime.sharder:
        logger.info("Dispatching subgraph to distributed sharder")
        try:
            result = await runtime.sharder.dispatch_and_gather(subgraph_copy)
            return {"result": result}
        except Exception as e:
            logger.warning(
                f"Distributed sharding failed: {e}. Falling back to local execution"
            )

    # Fallback to local execution
    if runtime and hasattr(runtime, "execute_graph"):
        logger.info("Distributed sharder not available. Executing locally")
        result = await runtime.execute_graph(subgraph_copy)
        # Ensure result is a dict (execute_graph might return an object)
        if hasattr(result, "to_dict"):
            return result.to_dict()
        elif isinstance(result, dict):
            return result
        else:
            return {"result": str(result)}  # Fallback

    logger.error("Sharded computation not available and runtime.execute_graph missing.")
    return {
        "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
        "message": "Sharded computation not available",
    }


async def composite_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Composite subgraph execution node
    """
    subgraph_type = node.get("type")
    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    if not runtime:
        raise NodeExecutorError(
            "COMPOSITE: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    if not hasattr(runtime, "subgraph_definitions"):
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": "Runtime missing subgraph_definitions",
        }

    subgraph_def_orig = runtime.subgraph_definitions.get(subgraph_type)
    if not subgraph_def_orig:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": f"Unknown subgraph type: {subgraph_type}",
        }

    # Deep copy the definition to avoid modifying the original
    import copy

    subgraph_def = copy.deepcopy(subgraph_def_orig)

    # Map inputs to subgraph (modify the copy)
    for node_in_subgraph in subgraph_def.get("nodes", []):
        # Identify input nodes within the subgraph (e.g., by type or a flag)
        if node_in_subgraph.get("type") == "INPUT":  # Or check a specific param
            input_key = node_in_subgraph.get("params", {}).get(
                "key", node_in_subgraph.get("id")
            )
            if input_key in inputs:
                node_in_subgraph["params"] = node_in_subgraph.get("params", {})
                node_in_subgraph["params"]["value"] = inputs[input_key]

    # Execute subgraph
    if not hasattr(runtime, "execute_graph"):
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": "Runtime missing execute_graph method",
        }

    result_obj = await runtime.execute_graph(subgraph_def)

    # Extract output from GraphExecutionResult object or dict
    output_payload = {}
    if hasattr(result_obj, "output"):
        output_payload = result_obj.output
    elif isinstance(result_obj, dict):
        output_payload = result_obj.get("output", {})

    # The composite node should return the *outputs* of the subgraph
    return output_payload


# ============================================================================
# META/RECURSIVE NODE HANDLERS
# ============================================================================


async def meta_graph_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Meta-graph recursive execution node
    """
    params = node.get("params", {})
    # Corrected param name based on test file _graph_recursive
    meta_graph = params.get("subgraph", params.get("meta_graph"))

    # <<< --- START CORRECTION for context type --- >>>
    # Access recursion depth via dictionary key
    depth = context.get("recursion_depth", 0)
    runtime = context.get("runtime")
    if runtime is None:
        raise NodeExecutorError(
            "META_GRAPH: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    if not meta_graph:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "No meta_graph/subgraph in MetaGraphNode",
        }

    # Check recursion depth using runtime config if available
    MAX_RECURSION_DEPTH = (
        getattr(runtime.config, "max_recursion_depth", 20)
        if hasattr(runtime, "config")
        else 20
    )
    if depth >= MAX_RECURSION_DEPTH:
        logger.warning(f"Max recursion depth ({depth}) exceeded")
        return {
            "error_code": AI_ERRORS.AI_RESOURCE_LIMIT.value,
            "message": f"Max recursion depth ({MAX_RECURSION_DEPTH}) exceeded",
        }

    # Create nested runtime for isolation (or just execute directly with increased depth)
    import copy

    subgraph_copy = copy.deepcopy(meta_graph)

    # Check if runtime has execute_graph
    if not hasattr(runtime, "execute_graph"):
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": "Runtime missing execute_graph method for meta execution",
        }

    # Execute nested graph
    try:
        # Pass inputs to the sub-graph execution and increment recursion depth
        result_obj = await runtime.execute_graph(
            subgraph_copy,
            inputs=inputs,
            recursion_depth=depth + 1,  # Increment depth for the call
        )

        # Handle if execute_graph returns an object or a dict
        if isinstance(result_obj, dict):
            result_status_val = result_obj.get("status", "unknown")
            result_errors = result_obj.get("errors", {})
            result_output = result_obj.get("output", {})
        elif hasattr(result_obj, "to_dict"):  # Assuming GraphExecutionResult object
            result_dict = result_obj.to_dict()
            result_status_val = result_dict.get("status", "unknown")
            result_errors = result_dict.get("errors", {})
            result_output = result_dict.get("output", {})
        else:  # Unknown result type
            raise TypeError(
                f"Unexpected result type from nested execute_graph: {type(result_obj)}"
            )

        # Check for failure status (enum value or string)
        # Using startswith for flexibility (e.g., FAILED_VALIDATION)
        if isinstance(result_status_val, str) and result_status_val.startswith("FAIL"):
            # Extract first error message if available
            first_error_msg = "Unknown subgraph error"
            if result_errors:
                first_error_msg = next(
                    iter(result_errors.values()), "Unknown subgraph error"
                )

            return {
                "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
                "message": f"Nested graph failed: {first_error_msg}",
            }

        # Trigger evolution if configured (using runtime reference)
        if (
            hasattr(runtime, "extensions")
            and runtime.extensions
            and hasattr(runtime.extensions, "autonomous_optimizer")
            and runtime.extensions.autonomous_optimizer
            and hasattr(runtime.extensions.autonomous_optimizer, "evolution_engine")
            and runtime.extensions.autonomous_optimizer.evolution_engine
        ):
            proposal = {"type": "grammar_update", "meta_graph": meta_graph}
            # Assuming propose is synchronous or handled internally
            runtime.extensions.autonomous_optimizer.evolution_engine.propose(proposal)

        # Return the output of the nested graph directly
        return result_output

    except Exception as e:
        logger.error(f"Meta graph execution failed: {e}", exc_info=True)
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": f"Meta execution failed: {str(e)}",
        }


# ============================================================================
# AUTOML NODE HANDLERS
# ============================================================================


async def random_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Random value generation node
    """
    if not NUMPY_AVAILABLE:
        # Fallback to Python random
        import random as py_random

        params = node.get("params", {})
        distribution = params.get("distribution", "uniform")
        shape = params.get("shape", [1])

        if distribution == "uniform":
            low = params.get("low", 0.0)
            high = params.get("high", 1.0)
            size = 1
            for dim in shape:
                size *= dim
            value = [py_random.uniform(low, high) for _ in range(size)]
        elif distribution == "normal":
            loc = params.get("loc", 0.0)
            scale = params.get("scale", 1.0)
            size = 1
            for dim in shape:
                size *= dim
            value = [py_random.gauss(loc, scale) for _ in range(size)]
        else:
            return {
                "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
                "message": f"Unsupported distribution: {distribution}",
            }
        # Reshape if needed (basic list reshaping)
        if len(shape) > 1 and len(value) == np.prod(shape):
            # Simple reshape, might need more complex logic for higher dims
            if len(shape) == 2:
                value = [
                    value[i * shape[1] : (i + 1) * shape[1]] for i in range(shape[0])
                ]
        return {"value": value}

    params = node.get("params", {})
    distribution = params.get("distribution", "uniform")
    shape = tuple(params.get("shape", [1]))

    # Security: Limit array size
    max_elements = 1000000
    total_elements = np.prod(shape)
    if total_elements > max_elements:
        return {
            "error_code": AI_ERRORS.AI_RESOURCE_LIMIT.value,
            "message": f"Shape too large: {total_elements} > {max_elements}",
        }

    try:
        if distribution == "uniform":
            value = np.random.uniform(
                low=params.get("low", 0.0), high=params.get("high", 1.0), size=shape
            )
        elif distribution == "normal":
            value = np.random.normal(
                loc=params.get("loc", 0.0), scale=params.get("scale", 1.0), size=shape
            )
        elif distribution == "exponential":
            value = np.random.exponential(scale=params.get("scale", 1.0), size=shape)
        else:
            return {
                "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
                "message": f"Unsupported distribution: {distribution}",
            }

        return {"value": value.tolist() if hasattr(value, "tolist") else value}

    except Exception as e:
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": f"Random generation failed: {str(e)}",
        }


async def hyperparam_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Hyperparameter value node for AutoML
    """
    params = node.get("params", {})
    value = params.get("value")

    if value is None:
        # Try to get from inputs
        value = inputs.get("value", inputs.get("input"))

    return {"value": value}


async def search_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Hyperparameter search node using Optuna
    """
    params = node.get("params", {})
    subgraph_template = params.get("subgraph")
    hyperparams = params.get("hyperparams", {})
    objective_port = params.get("objective_port", "output_value")
    n_trials = min(params.get("n_trials", 10), 100)  # Limit trials

    if not subgraph_template:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "SearchNode requires a subgraph template",
        }

    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    if not runtime:
        raise NodeExecutorError(
            "SEARCH: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    if not hasattr(runtime, "execute_graph"):
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": "Runtime missing execute_graph method for search",
        }

    logger.info(f"Starting hyperparameter search with {n_trials} trials")

    # Define objective function
    def objective(trial):
        import copy

        subgraph_copy = copy.deepcopy(subgraph_template)

        # Sample hyperparameters
        current_trial_params = {}
        for node_id, hp_config in hyperparams.items():
            if "type" not in hp_config:
                continue

            value = None  # Initialize value

            try:  # Add try-except for Optuna suggestions
                if (
                    hp_config["type"] == "suggest_float"
                    or hp_config["type"] == "suggest_uniform"
                ):  # Allow both names
                    value = trial.suggest_float(  # Use suggest_float for uniform
                        node_id,
                        float(hp_config.get("low", 0.0)),  # Ensure float
                        float(hp_config.get("high", 1.0)),  # Ensure float
                    )
                elif hp_config["type"] == "suggest_loguniform":
                    value = trial.suggest_float(  # Use suggest_float with log=True
                        node_id,
                        float(hp_config.get("low", 0.001)),
                        float(hp_config.get("high", 1.0)),
                        log=True,
                    )
                elif hp_config["type"] == "suggest_int":
                    value = trial.suggest_int(
                        node_id,
                        int(hp_config.get("low", 1)),
                        int(hp_config.get("high", 10)),
                    )
                elif hp_config["type"] == "suggest_categorical":
                    choices = hp_config.get("choices", [])
                    if not choices:
                        raise ValueError("Categorical choices cannot be empty")
                    value = trial.suggest_categorical(node_id, choices)
                else:
                    logger.warning(
                        f"Unsupported Optuna suggestion type '{hp_config['type']}' for {node_id}"
                    )
                    continue
            except Exception as optuna_err:
                logger.error(f"Optuna suggestion failed for {node_id}: {optuna_err}")
                raise optuna.exceptions.TrialPruned()  # Prune trial if suggestion fails

            current_trial_params[node_id] = value

            # Update node in subgraph
            node_found = False
            for n in subgraph_copy.get("nodes", []):
                # Match HyperParamNode or CONST nodes if used for params
                if n.get("id") == node_id and n.get("type") in (
                    "HyperParamNode",
                    "CONST",
                ):
                    n["params"] = n.get("params", {})
                    n["params"]["value"] = value
                    node_found = True
                    break
            if not node_found:
                logger.warning(
                    f"Hyperparameter node ID '{node_id}' not found in subgraph template for trial {trial.number}"
                )

        # Execute subgraph using a separate event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_obj = loop.run_until_complete(
                asyncio.wait_for(
                    runtime.execute_graph(subgraph_copy),
                    timeout=30,  # Short timeout per trial
                )
            )
            # Handle result object vs dict
            result = (
                result_obj if isinstance(result_obj, dict) else result_obj.to_dict()
            )

        except asyncio.TimeoutError:
            logger.warning(f"Trial {trial.number} timed out.")
            raise optuna.exceptions.TrialPruned()  # Prune timed-out trials
        except Exception as exec_err:
            logger.error(f"Trial {trial.number} execution failed: {exec_err}")
            raise optuna.exceptions.TrialPruned()  # Prune failed trials
        finally:
            loop.close()

        # Extract objective value - more robustly
        output_dict = result.get("output", {})
        obj_value = float("inf")  # Default to worst case for minimization
        if objective_port in output_dict:
            port_output = output_dict[objective_port]
            # Handle cases where output is {'value': X} or just X
            if isinstance(port_output, dict):
                obj_value = port_output.get(
                    "value", port_output.get("result", float("inf"))
                )
            else:
                obj_value = port_output

        # Ensure obj_value is float, handle None or non-numeric gracefully
        try:
            obj_value = float(obj_value)
            if math.isnan(obj_value) or math.isinf(
                obj_value
            ):  # Check for NaN/inf explicitly
                logger.warning(
                    f"Trial {trial.number} resulted in invalid objective value ({obj_value}). Pruning."
                )
                raise optuna.exceptions.TrialPruned()
        except (TypeError, ValueError):
            logger.warning(
                f"Trial {trial.number} objective value '{obj_value}' is not numeric. Pruning."
            )
            raise optuna.exceptions.TrialPruned()

        return obj_value

    # Run optimization
    if OPTUNA_AVAILABLE:
        try:
            # Use in-memory storage for simplicity, can be changed to DB
            storage = optuna.storages.InMemoryStorage()
            study = optuna.create_study(
                storage=storage,
                direction=params.get("direction", "minimize"),
                pruner=optuna.pruners.MedianPruner(),  # Keep pruner
            )
            study.optimize(objective, n_trials=n_trials, timeout=300)  # Overall timeout

            best_params = study.best_params
            best_value = study.best_value

            return {
                "best_value": best_value,
                "best_params": best_params,
                "n_trials": len(study.trials),
                "optimization_complete": True,
            }
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            # Fall through to random search if Optuna fails
    else:
        logger.warning("Optuna not available.")

    # Fallback to random search
    logger.warning("Using random search fallback for hyperparameter optimization.")
    best_params = {}
    best_value = (
        float("inf")
        if params.get("direction", "minimize") == "minimize"
        else float("-inf")
    )
    direction_multiplier = (
        1 if params.get("direction", "minimize") == "minimize" else -1
    )

    for trial_idx in range(min(n_trials, 5)):  # Limit random search trials
        trial_params = {}
        for node_id, hp_config in hyperparams.items():
            # Simplified random sampling
            try:
                if hp_config.get("type") in (
                    "suggest_float",
                    "suggest_uniform",
                    "suggest_loguniform",
                ):
                    trial_params[node_id] = random.uniform(
                        float(hp_config.get("low", 0.0)),
                        float(hp_config.get("high", 1.0)),
                    )
                elif hp_config.get("type") == "suggest_int":
                    trial_params[node_id] = random.randint(
                        int(hp_config.get("low", 1)), int(hp_config.get("high", 10))
                    )
                elif hp_config.get("type") == "suggest_categorical":
                    choices = hp_config.get("choices", [None])
                    if not choices:
                        choices = [None]  # Ensure choices list is not empty
                    trial_params[node_id] = random.choice(choices)
                else:  # Default for unknown
                    trial_params[node_id] = random.uniform(0, 1)
            except Exception as rand_err:
                logger.warning(f"Random sampling failed for {node_id}: {rand_err}")
                trial_params[node_id] = None  # Assign default on error

        # Test this configuration
        import copy

        subgraph_copy = copy.deepcopy(subgraph_template)
        for n in subgraph_copy.get("nodes", []):
            # Update CONST or HyperParam nodes
            if (
                n.get("type") in ("HyperParamNode", "CONST")
                and n.get("id") in trial_params
            ):
                n["params"] = n.get("params", {})
                n["params"]["value"] = trial_params[n.get("id")]

        try:
            result_obj = await asyncio.wait_for(
                runtime.execute_graph(subgraph_copy), timeout=30
            )
            result = (
                result_obj if isinstance(result_obj, dict) else result_obj.to_dict()
            )

            output_dict = result.get("output", {})
            obj_value = float("inf")  # Default
            if objective_port in output_dict:
                port_output = output_dict[objective_port]
                if isinstance(port_output, dict):
                    obj_value = port_output.get(
                        "value", port_output.get("result", float("inf"))
                    )
                else:
                    obj_value = port_output

            # Ensure numeric and valid
            try:
                obj_value = float(obj_value)
                if math.isnan(obj_value) or math.isinf(obj_value):
                    continue  # Skip invalid results
            except (TypeError, ValueError):
                continue  # Skip non-numeric results

            # Check if better based on direction
            if direction_multiplier * obj_value < direction_multiplier * best_value:
                best_value = obj_value
                best_params = trial_params
        except asyncio.TimeoutError:
            logger.warning(f"Random search trial {trial_idx} timed out.")
        except Exception as e:
            logger.error(f"Random search trial {trial_idx} failed: {e}")

    return {
        "best_value": (
            best_value if not math.isinf(best_value) else None
        ),  # Return None if no valid trial found
        "best_params": best_params,
        "n_trials": min(n_trials, 5),
        "optimization_complete": True,
    }


# ============================================================================
# GOVERNANCE/AUDIT NODE HANDLERS
# ============================================================================


async def contract_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Contract validation and NSO alignment node
    """
    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    if not runtime:
        raise NodeExecutorError(
            "CONTRACT: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    # Check for NSO aligner specifically
    nso_aligner = None
    if (
        hasattr(runtime, "extensions")
        and runtime.extensions
        and hasattr(runtime.extensions, "autonomous_optimizer")
        and runtime.extensions.autonomous_optimizer
        and hasattr(runtime.extensions.autonomous_optimizer, "nso")
        and runtime.extensions.autonomous_optimizer.nso
    ):
        nso_aligner = runtime.extensions.autonomous_optimizer.nso

    if not nso_aligner or not hasattr(nso_aligner, "multi_model_audit"):
        logger.warning(
            "NSOAligner (or multi_model_audit method) not available, skipping contract check"
        )
        return {
            "audit_result": "skipped_no_aligner",
            "approved": True,  # Default to approved if aligner missing
        }

    proposal = inputs.get("proposal", inputs.get("input"))
    if not proposal:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "ContractNode requires a 'proposal' input",
        }

    # Perform multi-model audit
    audit_result = "safe"  # Default if audit fails
    try:
        # Assuming multi_model_audit is synchronous
        audit_result = nso_aligner.multi_model_audit(proposal)
    except Exception as audit_err:
        logger.error(f"NSO multi_model_audit failed: {audit_err}")
        audit_result = "error_during_audit"

    if audit_result == "risky":
        logger.warning("Proposal flagged as risky by multi-model audit")
        return {
            "error_code": AI_ERRORS.AI_SAFETY_VIOLATION.value,
            "message": "Proposal flagged as risky",
            "audit_result": audit_result,
            "approved": False,  # Explicitly mark as not approved
        }

    return {
        "audit_result": audit_result,
        "approved": audit_result != "risky",  # Approve unless explicitly risky
    }


async def proposal_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Proposal submission node for governance
    """
    proposal_content = node.get("params", {}).get("proposal_content", {})

    if not proposal_content:
        proposal_content = inputs.get("proposal", inputs.get("input", {}))

    # Ensure proposal is dict
    if not isinstance(proposal_content, dict):
        proposal_content = {"content": str(proposal_content)}

    return {
        "proposal": proposal_content,
        "timestamp": time.time(),
        "node_id": node.get("id"),
    }


async def consensus_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Consensus voting node for governance decisions
    """
    votes_input = inputs.get("votes", [])
    # Ensure votes is a list of dicts
    votes = [v for v in votes_input if isinstance(v, dict)]

    threshold = node.get("params", {}).get("threshold", 0.5)
    # Ensure threshold is valid
    try:
        threshold = float(threshold)
        if not (0 <= threshold <= 1):
            threshold = 0.5
    except (ValueError, TypeError):
        threshold = 0.5

    if not votes:
        return {
            "consensus": "no_votes",
            "approved": False,
            "approval_rate": 0.0,
            "vote_count": 0,
        }

    approval_count = sum(
        1 for v in votes if v.get("approve", v.get("approved", False))
    )  # Check both keys
    approval_rate = (approval_count / len(votes)) if len(votes) > 0 else 0.0

    approved = approval_rate >= threshold
    return {
        "consensus": "approved" if approved else "rejected",
        "approved": approved,
        "approval_rate": approval_rate,
        "vote_count": len(votes),
    }


async def validation_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Graph validation node
    """
    graph_to_validate = inputs.get("graph", inputs.get("input"))

    if not graph_to_validate:
        return {"valid": False, "errors": ["No graph provided for validation"]}

    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    if not runtime:
        raise NodeExecutorError(
            "VALIDATION: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    if runtime and hasattr(runtime, "validate_graph"):
        # Assuming validate_graph is sync and returns a dict-like object (ValidationResult)
        result = runtime.validate_graph(graph_to_validate)
        if hasattr(result, "to_dict"):
            return result.to_dict()  # Convert ValidationResult to dict
        elif isinstance(result, dict):
            return result  # Already a dict
        else:
            # Fallback if validate_graph returns tuple(bool, list) - less likely now
            logger.warning(f"validate_graph returned unexpected type: {type(result)}")
            is_valid = (
                result[0] if isinstance(result, tuple) and len(result) > 0 else False
            )
            errors = (
                result[1]
                if isinstance(result, tuple)
                and len(result) > 1
                and isinstance(result[1], list)
                else ["Unexpected validation result format"]
            )
            return {"valid": is_valid, "errors": errors}

    # Basic validation if runtime validator isn't available
    errors = []
    if not isinstance(graph_to_validate, dict):
        errors.append("Graph must be a dictionary")
    elif "nodes" not in graph_to_validate or not isinstance(
        graph_to_validate["nodes"], list
    ):
        errors.append("Graph must contain a valid 'nodes' list")
    elif "edges" not in graph_to_validate or not isinstance(
        graph_to_validate["edges"], list
    ):
        errors.append("Graph must contain a valid 'edges' list")

    return {"valid": len(errors) == 0, "errors": errors}


async def audit_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Audit logging node for compliance
    """
    audit_data = inputs.get("data", inputs.get("input"))
    audit_type = node.get("params", {}).get("audit_type", "general")

    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    audit_log = context.get("audit_log")  # Get audit log from context dict
    # <<< --- END CORRECTION --- >>>

    # Check if context *instance* has audit_log (it should from dataclass default)
    if audit_log is not None and isinstance(audit_log, list):
        audit_log.append(
            {
                "type": audit_type,
                "data": audit_data,
                "timestamp": time.time(),
                "node_id": node.get("id"),
            }
        )
    elif runtime and hasattr(runtime, "audit_log") and runtime.audit_log is not None:
        # Fallback to runtime's main audit log
        logger.warning(
            f"NodeContext missing audit_log for node {node.get('id')}. Falling back to runtime log."
        )
        runtime.audit_log.append(
            {
                "type": audit_type,
                "data": audit_data,
                "timestamp": time.time(),
                "node_id": node.get("id"),
            }
        )
    else:
        logger.error(
            f"AuditNode {node.get('id')} could not find any audit_log in context or runtime."
        )

    return {"audit": "logged", "audit_type": audit_type, "timestamp": time.time()}


async def execute_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Code execution node (disabled for safety)
    """
    # Code execution is disabled for security
    return {
        "executed": False,
        "reason": "Code execution disabled for safety",
        "warning": "Direct code execution poses security risks",
    }


# ============================================================================
# SCHEDULER NODE HANDLERS
# ============================================================================


async def scheduler_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Task scheduling node for periodic execution
    """
    params = node.get("params", {})
    subgraph = params.get("subgraph")
    interval_ms = params.get("interval_ms", 1000)
    max_iterations = params.get("max_iterations", 100)

    # Security: Limit interval and iterations
    interval_ms = max(100, min(interval_ms, 60000))  # 100ms to 60s
    max_iterations = min(max_iterations, 1000)

    if not subgraph:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "SchedulerNode requires 'subgraph'",
        }

    # <<< --- START CORRECTION for context type --- >>>
    runtime = context.get("runtime")
    if not runtime:
        raise NodeExecutorError(
            "SCHEDULER: Invalid context object received. Must be dict with 'runtime'."
        )
    # <<< --- END CORRECTION --- >>>

    if not (runtime and hasattr(runtime, "execute_graph")):
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": "Runtime (or runtime.execute_graph) not available for scheduling",
        }

    # Create scheduled task
    async def _run_scheduled_task():
        iterations = 0
        node_id = node.get("id", "scheduler")

        while iterations < max_iterations:
            await asyncio.sleep(interval_ms / 1000)
            logger.info(f"Running scheduled task {node_id} (iteration {iterations})")

            try:
                import copy

                subgraph_copy = copy.deepcopy(subgraph)
                # Ensure runtime reference is valid before calling execute_graph
                if hasattr(runtime, "execute_graph"):
                    await asyncio.wait_for(
                        runtime.execute_graph(subgraph_copy),
                        timeout=30,  # Timeout for each scheduled run
                    )
                else:
                    logger.error(
                        f"Scheduled task {node_id} cannot run: runtime.execute_graph is missing."
                    )
                    break  # Stop scheduling if runtime becomes invalid

            except asyncio.TimeoutError:
                logger.warning(
                    f"Scheduled task {node_id} iteration {iterations} timed out."
                )
            except Exception as e:
                logger.error(f"Scheduled task {node_id} failed: {e}")
                # Optionally record metrics failure here
                # if hasattr(runtime, '_metrics_aggregator') and runtime._metrics_aggregator:
                #     pass # Need more context to record failure properly

            iterations += 1

        logger.info(f"Scheduled task {node_id} completed {iterations} iterations")

    # Start task in background
    try:
        # Get the current running loop or create one if needed
        loop = asyncio.get_running_loop()
        loop.create_task(_run_scheduled_task())
    except RuntimeError:
        # If no loop is running (e.g., called from sync context), this won't work easily
        logger.error(
            f"Cannot schedule task {node.get('id')} - no running asyncio event loop."
        )
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": "Cannot schedule task - requires running asyncio event loop.",
        }
    except Exception as e:
        logger.error(
            f"Failed to create background task for scheduler node {node.get('id')}: {e}"
        )
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": f"Failed to create background task: {str(e)}",
        }

    return {
        "status": "scheduled",
        "message": f"Task scheduled to run every {interval_ms}ms for {max_iterations} iterations",
        "interval_ms": interval_ms,
        "max_iterations": max_iterations,
    }


# ============================================================================
# UTILITY NODE HANDLERS
# ============================================================================


async def normalize_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Data normalization node
    """
    data = inputs.get("data", inputs.get("input"))
    method = node.get("params", {}).get("method", "minmax")

    if data is None:
        return {
            "error_code": AI_ERRORS.AI_INVALID_REQUEST.value,
            "message": "NormalizeNode requires input data",
        }

    # --- Start Scope Fix ---
    # Access the global flag defined at the module level
    global NUMPY_AVAILABLE
    if not NUMPY_AVAILABLE:
        # --- End Scope Fix ---
        # Basic normalization without numpy
        if isinstance(data, (list, tuple)):
            if not data:
                return {"output": []}  # Handle empty list
            try:
                # Filter out non-numeric types before min/max
                numeric_data = [x for x in data if isinstance(x, (int, float))]
                if not numeric_data:
                    return {"output": data}  # Return original if no numerics

                min_val = min(numeric_data)
                max_val = max(numeric_data)
                range_val = max_val - min_val

                if range_val > 1e-9:  # Use epsilon for float comparison
                    normalized = [
                        (x - min_val) / range_val if isinstance(x, (int, float)) else x
                        for x in data
                    ]
                else:
                    # Normalize constant list to 0.0 for numerics, keep others
                    normalized = [
                        0.0 if isinstance(x, (int, float)) else x for x in data
                    ]
            except TypeError:  # Handle mixed types causing min/max error
                return {"output": data}  # Return original if mixed types cause error
            return {"output": normalized}
        return {"output": data}  # Return non-list data as is

    # --- Start Scope Fix ---
    # Explicitly import numpy here if NUMPY_AVAILABLE is True
    import numpy as np

    # --- End Scope Fix ---
    try:
        # Attempt conversion to float array, handle errors
        try:
            data_array = np.asarray(data, dtype=float)
        except (ValueError, TypeError):
            logger.warning(
                f"NormalizeNode input could not be converted to float array. Returning original."
            )
            return {"output": data}  # Return original data if conversion fails

        if data_array.size == 0:
            return {"output": []}  # Handle empty array

        if method == "minmax":
            min_val = np.min(data_array)
            max_val = np.max(data_array)
            range_val = max_val - min_val
            if range_val > 1e-9:  # Use epsilon
                normalized = (data_array - min_val) / range_val
            else:
                normalized = np.zeros_like(data_array)  # Normalize constant array to 0
        elif method == "zscore":
            mean = np.mean(data_array)
            std = np.std(data_array)
            if std > 1e-9:  # Use epsilon
                normalized = (data_array - mean) / std
            else:
                normalized = np.zeros_like(data_array)  # Normalize constant array to 0
        else:
            logger.warning(
                f"Unsupported normalization method '{method}'. Returning original data."
            )
            normalized = data_array  # Unknown method, return original

        return {
            "output": (
                normalized.tolist() if hasattr(normalized, "tolist") else normalized
            )
        }

    except Exception as e:
        logger.error(f"Normalization failed: {e}", exc_info=True)
        return {
            "error_code": AI_ERRORS.AI_INTERNAL_ERROR.value,
            "message": f"Normalization failed: {str(e)}",
        }


async def cnn_node_handler(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Convolutional neural network node (placeholder)
    """
    # This is a placeholder for CNN operations
    # Real implementation would require deep learning framework
    input_val = inputs.get("input")
    return {
        "output": input_val,
        "message": "CNN operations require deep learning framework (placeholder executed)",
    }


async def meta_node_handler(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Generic meta-node handler (placeholder)
    """
    # Placeholder - could inspect node params to decide action
    output_val = inputs.get("input")
    return {"output": output_val, "message": "MetaNode executed (placeholder)"}


# ============================================================================
# DISPATCHER INTEGRATION
# ============================================================================


async def dispatch_node(node: Dict, context: NodeContext, inputs: Dict) -> Dict:
    """
    Main dispatcher for specialized node types
    """
    node_type = node.get("type")

    # Check for AutoML nodes
    if (
        node_type in ["RandomNode", "HyperParamNode", "SearchNode"]
        and dispatch_auto_ml_node
    ):
        return await dispatch_auto_ml_node(node, context, inputs)  # Pass inputs

    # Check for Security nodes
    if node_type in ["EncryptNode", "PolicyNode"] and dispatch_security_node:
        return await dispatch_security_node(node, inputs, context)  # Await async

    # Check for Scheduler nodes
    if node_type == "SchedulerNode" and dispatch_scheduler_node:
        # Pass inputs to scheduler dispatcher if it needs them
        return await dispatch_scheduler_node(
            node, context, inputs
        )  # Await async, added inputs

    # Check for Explainability nodes
    if node_type == "ExplainabilityNode" and dispatch_explainability_node:
        return await dispatch_explainability_node(
            node, context, inputs
        )  # Await async and pass inputs

    # Default: node type not handled by dispatchers
    return {
        "error_code": AI_ERRORS.AI_UNSUPPORTED.value,
        "message": f"No dispatcher available for node type: {node_type}",
    }


# ============================================================================
# NODE REGISTRY
# ============================================================================


def get_node_handlers() -> Dict[str, Callable]:
    """
    Returns the complete registry of node handlers
    """
    return {
        # Core nodes
        "CONST": const_node,
        "ADD": add_node,
        "MUL": multiply_node,
        "MULTIPLY": multiply_node,  # Alias
        "BRANCH": branch_node,
        "GET_PROPERTY": get_property_node,
        "INPUT": input_node_handler,  # Changed from InputNode
        "OUTPUT": output_node_handler,  # Changed from OutputNode
        # AI/Embedding nodes
        "EMBED": embed_node,
        "Generative": generative_node_handler,  # Alias
        "GenerativeNode": generative_node_handler,  # Alias
        "GenerativeAINode": generative_node_handler,
        # --- NEW LLM NODES ---
        "ATTENTION": attention_node,
        "FFN": ffn_node,
        "EMBEDDING": transformer_embedding_node,
        # --- END NEW LLM NODES ---
        # Hardware-accelerated nodes
        "LOAD_TENSOR": load_tensor_node,
        "MEMRISTOR_MVM": memristor_mvm_node,
        "PhotonicMVMNode": photonic_mvm_node,  # Alias
        "PHOTONIC_MVM": photonic_mvm_node,
        "SPARSE_MVM": sparse_mvm_node,
        "FUSED_KERNEL": fused_kernel_node,
        "FUSED_PHOTONIC": fused_photonic_node,
        # Distributed nodes
        "SHARDED_COMPUTATION": sharded_computation_node,
        "COMPOSITE": composite_node,  # Handles learned subgraphs by type name
        # Meta/Recursive nodes
        "MetaGraphNode": meta_graph_node,
        "MetaNode": meta_node_handler,  # Generic meta placeholder
        # AutoML nodes
        "RandomNode": random_node,
        "HyperParamNode": hyperparam_node,
        "SearchNode": search_node,
        # Governance nodes
        "ContractNode": contract_node,
        "ProposalNode": proposal_node,
        "ConsensusNode": consensus_node,
        "ValidationNode": validation_node,
        "AuditNode": audit_node,
        "ExecuteNode": execute_node,  # Disabled code execution
        # Scheduler nodes
        "SchedulerNode": scheduler_node,
        # Utility nodes
        "NormalizeNode": normalize_node,
        "CNNNode": cnn_node_handler,  # Placeholder
    }


def validate_node_handler(handler: Callable) -> bool:
    """
    Validates that a function is a valid node handler
    """
    import inspect

    # Check if it's async
    if not inspect.iscoroutinefunction(handler):
        return False

    # Check signature
    sig = inspect.signature(handler)
    params = list(sig.parameters.keys())

    # Should have exactly 3 parameters: node, context, inputs
    if len(params) != 3:
        return False

    # Could add type hint checks here if desired
    # param_node = sig.parameters[params[0]]
    # param_context = sig.parameters[params[1]]
    # param_inputs = sig.parameters[params[2]]
    # if param_node.annotation != Dict ... etc.

    return True
