"""
FAISS Configuration and Initialization Module

Handles FAISS import with proper CPU capability detection and informative
messaging about instruction set usage (AVX512, AVX2, etc.)

This module provides centralized FAISS initialization with:
- CPU capability detection (AVX512, AVX2, AVX, etc.)
- Informative logging about instruction set usage
- Graceful fallback when FAISS is not available
- Warning suppression for expected internal FAISS fallbacks
- Thread-safe singleton pattern for initialization

Usage:
    from src.utils.faiss_config import initialize_faiss, get_faiss
    
    faiss, is_available, instruction_set = initialize_faiss()
    if is_available:
        # Use FAISS for vector operations
        index = faiss.IndexFlatL2(dimension)
"""

import logging
import threading
import warnings
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Global flags for FAISS availability and configuration
FAISS_AVAILABLE: bool = False
FAISS_MODULE: Optional[Any] = None
FAISS_INSTRUCTION_SET: Optional[str] = None
_faiss_lock = threading.Lock()


def initialize_faiss() -> Tuple[Optional[Any], bool, Optional[str]]:
    """
    Initialize FAISS with proper CPU capability detection and informative logging.
    
    This function:
    1. Attempts to import the FAISS library
    2. Detects CPU capabilities (AVX512, AVX2, AVX, etc.)
    3. Logs informative messages about instruction set usage
    4. Suppresses expected internal FAISS warnings about AVX512 fallback
    5. Provides fallback information when FAISS is not available
    
    Thread-safe: Uses double-checked locking pattern.
    
    Returns:
        tuple: A tuple containing:
            - faiss_module (module or None): The FAISS module if available
            - is_available (bool): True if FAISS was successfully imported
            - instruction_set_info (str or None): Detected instruction set 
              (AVX512, AVX2, AVX, SCALAR, UNKNOWN)
    
    Example:
        >>> faiss, available, instr_set = initialize_faiss()
        >>> if available:
        ...     index = faiss.IndexFlatL2(128)
        ... else:
        ...     # Use numpy-based fallback
        ...     pass
    """
    global FAISS_AVAILABLE, FAISS_MODULE, FAISS_INSTRUCTION_SET
    
    # Fast path: already initialized
    if FAISS_MODULE is not None:
        return FAISS_MODULE, FAISS_AVAILABLE, FAISS_INSTRUCTION_SET
    
    # Thread-safe initialization with double-checked locking
    with _faiss_lock:
        # Double-check inside lock
        if FAISS_MODULE is not None:
            return FAISS_MODULE, FAISS_AVAILABLE, FAISS_INSTRUCTION_SET
        
        try:
            # Suppress FAISS internal warnings about swigfaiss_avx512 BEFORE import
            # These are expected when AVX512 is not available and FAISS falls back to AVX2
            warnings.filterwarnings(
                'ignore',
                message='.*swigfaiss_avx512.*',
                category=UserWarning
            )
            warnings.filterwarnings(
                'ignore', 
                message='.*swigfaiss_avx512.*',
                category=RuntimeWarning
            )
            # Note: ModuleNotFoundError cannot be filtered via warnings.filterwarnings
            # as it's an exception, not a warning. The try/except below handles it.
            
            # Attempt to import FAISS
            import faiss
            
            FAISS_MODULE = faiss
            FAISS_AVAILABLE = True
            
            # Detect which instruction set FAISS is using
            try:
                from src.utils.cpu_capabilities import get_cpu_capabilities
                
                caps = get_cpu_capabilities()
                best_instr = caps.get_best_vector_instruction_set()
                perf_tier = caps.get_performance_tier()
                
                # Build informative message based on CPU capabilities
                if caps.has_avx512f:
                    msg = (
                        f"✓ FAISS initialized with {best_instr} support "
                        f"({perf_tier})"
                    )
                    FAISS_INSTRUCTION_SET = "AVX512"
                    logger.info(msg)
                elif caps.has_avx2:
                    msg = (
                        f"✓ FAISS initialized with AVX2 support ({perf_tier}). "
                        f"Note: AVX512 not available on this CPU. "
                        f"Performance is optimal for this hardware."
                    )
                    FAISS_INSTRUCTION_SET = "AVX2"
                    logger.info(msg)
                elif caps.has_avx:
                    msg = (
                        f"ℹ FAISS initialized with AVX support ({perf_tier}). "
                        f"Note: AVX2/AVX512 not available. "
                        f"Consider upgrading hardware for better vector search performance."
                    )
                    FAISS_INSTRUCTION_SET = "AVX"
                    logger.info(msg)
                else:
                    msg = (
                        f"⚠ FAISS initialized with {best_instr} ({perf_tier}). "
                        f"Vector operations will use scalar fallback. "
                        f"Performance may be limited."
                    )
                    FAISS_INSTRUCTION_SET = "SCALAR"
                    logger.warning(msg)
                    
            except ImportError as cpu_err:
                # Fallback if cpu_capabilities module not available
                logger.info(
                    f"✓ FAISS library imported successfully "
                    f"(CPU capability detection unavailable: {cpu_err})"
                )
                FAISS_INSTRUCTION_SET = "UNKNOWN"
            except Exception as cpu_err:
                # Catch any other errors during CPU detection
                logger.debug(
                    f"CPU capability detection failed: {cpu_err}. "
                    f"FAISS imported successfully."
                )
                FAISS_INSTRUCTION_SET = "UNKNOWN"
            
        except (ImportError, ModuleNotFoundError) as e:
            FAISS_AVAILABLE = False
            FAISS_MODULE = None
            FAISS_INSTRUCTION_SET = None
            
            logger.info(
                f"FAISS not available ({e.__class__.__name__}). "
                f"Falling back to NumPy-based vector search. "
                f"Install faiss-cpu or faiss-gpu for better performance: "
                f"pip install faiss-cpu"
            )
        except Exception as e:
            # Catch any other unexpected errors during import
            FAISS_AVAILABLE = False
            FAISS_MODULE = None
            FAISS_INSTRUCTION_SET = None
            
            logger.warning(
                f"Unexpected error initializing FAISS: {e}. "
                f"Falling back to NumPy-based vector search.",
                exc_info=True
            )
        
        return FAISS_MODULE, FAISS_AVAILABLE, FAISS_INSTRUCTION_SET


def get_faiss() -> Optional[Any]:
    """
    Get FAISS module, initializing if necessary.
    
    This is a convenience function that ensures FAISS is initialized
    before returning the module.
    
    Returns:
        faiss module or None if not available
        
    Example:
        >>> faiss = get_faiss()
        >>> if faiss:
        ...     index = faiss.IndexFlatL2(128)
    """
    if FAISS_MODULE is None:
        initialize_faiss()
    return FAISS_MODULE


def is_faiss_available() -> bool:
    """
    Check if FAISS is available.
    
    This function will initialize FAISS if it hasn't been initialized yet.
    
    Returns:
        bool: True if FAISS is available and successfully imported
        
    Example:
        >>> if is_faiss_available():
        ...     faiss = get_faiss()
        ...     # Use FAISS
        ... else:
        ...     # Use numpy fallback
        ...     pass
    """
    if FAISS_MODULE is None:
        initialize_faiss()
    return FAISS_AVAILABLE


def get_faiss_instruction_set() -> Optional[str]:
    """
    Get the instruction set being used by FAISS.
    
    Returns the detected CPU instruction set that FAISS is using
    for vector operations.
    
    Returns:
        str or None: One of the following:
            - "AVX512": FAISS using AVX-512 instructions
            - "AVX2": FAISS using AVX2 instructions  
            - "AVX": FAISS using AVX instructions
            - "SCALAR": FAISS using scalar operations (no vector extensions)
            - "UNKNOWN": Detection unavailable or failed
            - None: FAISS not available
            
    Example:
        >>> instr_set = get_faiss_instruction_set()
        >>> if instr_set == "AVX512":
        ...     print("Optimal performance available")
        ... elif instr_set == "AVX2":
        ...     print("Good performance available")
    """
    if FAISS_MODULE is None:
        initialize_faiss()
    return FAISS_INSTRUCTION_SET


def get_faiss_config_info() -> dict:
    """
    Get comprehensive FAISS configuration information.
    
    Returns:
        dict: Configuration information including:
            - available: Whether FAISS is available
            - instruction_set: Detected instruction set
            - cpu_capabilities: Detailed CPU capabilities (if available)
            - recommendations: Performance recommendations
            
    Example:
        >>> config = get_faiss_config_info()
        >>> print(f"FAISS available: {config['available']}")
        >>> print(f"Using: {config['instruction_set']}")
    """
    if FAISS_MODULE is None:
        initialize_faiss()
    
    info = {
        'available': FAISS_AVAILABLE,
        'instruction_set': FAISS_INSTRUCTION_SET,
        'cpu_capabilities': None,
        'recommendations': []
    }
    
    # Add CPU capability details if available
    try:
        from src.utils.cpu_capabilities import get_cpu_capabilities
        
        caps = get_cpu_capabilities()
        info['cpu_capabilities'] = caps.to_dict()
        
        # Generate recommendations
        if not caps.has_avx512f and FAISS_AVAILABLE:
            info['recommendations'].append(
                "Consider hardware with AVX-512 support for optimal FAISS performance"
            )
        if not FAISS_AVAILABLE:
            info['recommendations'].append(
                "Install faiss-cpu for better vector search performance: pip install faiss-cpu"
            )
            
    except ImportError:
        pass
    
    return info
