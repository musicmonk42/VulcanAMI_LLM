# src/vulcan/world_model/meta_reasoning/serialization_mixin.py
"""
Serialization support for meta-reasoning components.

This module provides a standardized approach to pickle serialization for
classes that contain unpickleable objects like threading locks, module
references, or bound methods.

Industry-Standard Features:
- Thread-safe state capture and restoration
- Explicit handling of unpickleable attributes
- Post-deserialization hooks for dependency injection
- Comprehensive logging for debugging and auditing
- Type-safe attribute restoration
- Backward compatibility support

Problem Addressed (Persistence Firewall):
    Standard Python pickle cannot serialize:
    - threading.Lock / threading.RLock objects
    - Module references (e.g., numpy)
    - Bound methods and lambdas
    - Certain defaultdict configurations
    
    Without proper handling, attempting to pickle objects containing
    these items raises TypeError, preventing state persistence.

Usage:
    class MyComponent(SerializationMixin):
        # Define which attributes cannot be pickled
        _unpickleable_attrs = ['lock', '_np', 'external_service']
        
        # Define how to restore unpickleable attributes
        def _restore_unpickleable_attrs(self) -> None:
            self.lock = threading.RLock()
            self._np = np if NUMPY_AVAILABLE else FakeNumpy
            self.external_service = None  # Must be re-injected

Example:
    >>> component = MyComponent()
    >>> pickled = pickle.dumps(component)
    >>> restored = pickle.loads(pickled)
    >>> restored.set_external_service(service)  # Re-inject dependency

Thread Safety:
    The mixin acquires locks before capturing state to ensure
    consistency in multi-threaded environments.

Version Compatibility:
    State includes a version marker to support future schema migrations.

See Also:
    - PEP 307: Extensions to the pickle protocol
    - Python docs: object.__getstate__ / object.__setstate__
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Type, Union

logger = logging.getLogger(__name__)

# Serialization version for future compatibility
SERIALIZATION_VERSION = "1.0.0"


class SerializationMixin(ABC):
    """
    Mixin providing industry-standard pickle serialization support.
    
    This mixin handles the "persistence firewall" problem where threading
    locks, module references, and bound methods cannot be pickled.
    
    Subclasses must define:
        _unpickleable_attrs: List of attribute names that cannot be pickled
        _restore_unpickleable_attrs(): Method to restore those attributes
    
    Attributes:
        _serialization_version: Version string for compatibility checking
        _serialized_at: Timestamp when state was captured
        _restored_at: Timestamp when state was restored
    
    Example:
        class MyClass(SerializationMixin):
            _unpickleable_attrs = ['lock', '_np']
            
            def __init__(self):
                self.lock = threading.RLock()
                self._np = numpy
                self.data = {}
            
            def _restore_unpickleable_attrs(self) -> None:
                self.lock = threading.RLock()
                self._np = numpy
    """
    
    # Subclasses should override this with their specific unpickleable attributes
    _unpickleable_attrs: List[str] = ['lock']
    
    @abstractmethod
    def _restore_unpickleable_attrs(self) -> None:
        """
        Restore unpickleable attributes after deserialization.
        
        This method MUST be implemented by subclasses to restore
        any attributes listed in _unpickleable_attrs.
        
        The method is called automatically by __setstate__ after
        the pickleable state has been restored.
        
        Example:
            def _restore_unpickleable_attrs(self) -> None:
                self.lock = threading.RLock()
                self._np = np if NUMPY_AVAILABLE else FakeNumpy
                self.external_service = None
        """
        pass
    
    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare instance state for pickle serialization.
        
        This method:
        1. Acquires the lock (if present) for thread safety
        2. Creates a shallow copy of the instance dict
        3. Removes unpickleable attributes
        4. Adds serialization metadata
        5. Converts defaultdicts to regular dicts
        
        Returns:
            Dictionary containing pickleable state
            
        Thread Safety:
            Acquires self.lock if present before capturing state
            
        Metadata Added:
            _serialization_version: Version for compatibility
            _serialized_at: Timestamp of serialization
            _class_name: Fully qualified class name for debugging
        """
        # Acquire lock if present for thread-safe state capture
        lock = getattr(self, 'lock', None) or getattr(self, '_lock', None)
        if lock is not None:
            lock.acquire()
        
        try:
            state = self.__dict__.copy()
            
            # Remove all unpickleable attributes
            for attr in self._unpickleable_attrs:
                state.pop(attr, None)
            
            # Also check for common lock attribute names
            for lock_attr in ['lock', '_lock']:
                state.pop(lock_attr, None)
            
            # Convert defaultdicts to regular dicts for pickle compatibility
            state = self._convert_defaultdicts(state)
            
            # Add serialization metadata
            state['_serialization_version'] = SERIALIZATION_VERSION
            state['_serialized_at'] = time.time()
            state['_class_name'] = f"{self.__class__.__module__}.{self.__class__.__name__}"
            
            logger.debug(
                f"Serializing {self.__class__.__name__}: "
                f"removed {len(self._unpickleable_attrs)} unpickleable attrs, "
                f"state has {len(state)} keys"
            )
            
            return state
            
        finally:
            if lock is not None:
                lock.release()
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore instance state after pickle deserialization.
        
        This method:
        1. Extracts and validates serialization metadata
        2. Restores the pickleable state
        3. Restores defaultdicts from regular dicts
        4. Calls _restore_unpickleable_attrs() to restore locks, etc.
        5. Logs restoration for debugging
        
        Args:
            state: Dictionary from __getstate__
            
        Raises:
            Warning logged if version mismatch detected
            
        Post-Conditions:
            - All unpickleable attrs restored via _restore_unpickleable_attrs()
            - Lock is available for thread-safe operations
            - _restored_at timestamp is set
        """
        # Extract metadata
        serialization_version = state.pop('_serialization_version', '0.0.0')
        serialized_at = state.pop('_serialized_at', None)
        class_name = state.pop('_class_name', 'unknown')
        
        # Version compatibility check
        if serialization_version != SERIALIZATION_VERSION:
            logger.warning(
                f"Deserializing {self.__class__.__name__} from version "
                f"{serialization_version} (current: {SERIALIZATION_VERSION}). "
                f"Some state may not be restored correctly."
            )
        
        # Restore pickleable state
        self.__dict__.update(state)
        
        # Restore defaultdicts
        self._restore_defaultdicts()
        
        # Restore unpickleable attributes (locks, module refs, etc.)
        self._restore_unpickleable_attrs()
        
        # Add restoration metadata
        self._restored_at = time.time()
        
        logger.debug(
            f"Restored {self.__class__.__name__} from serialization "
            f"(serialized at: {serialized_at}, class: {class_name})"
        )
    
    def _convert_defaultdicts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert defaultdict instances to regular dicts for pickling.
        
        defaultdict can have issues with pickle when the default_factory
        is a lambda or other unpickleable callable.
        
        Args:
            state: State dictionary to process
            
        Returns:
            State with defaultdicts converted to regular dicts, with
            factory type information preserved for restoration
        """
        result = {}
        defaultdict_info = {}  # Store factory type info
        
        for key, value in state.items():
            if isinstance(value, defaultdict):
                result[key] = dict(value)
                # Store factory type for proper restoration
                factory = value.default_factory
                if factory is list:
                    defaultdict_info[key] = 'list'
                elif factory is dict:
                    defaultdict_info[key] = 'dict'
                elif factory is set:
                    defaultdict_info[key] = 'set'
                elif factory is int:
                    defaultdict_info[key] = 'int'
                elif factory is float:
                    defaultdict_info[key] = 'float'
                elif factory is str:
                    defaultdict_info[key] = 'str'
                else:
                    # Default to list for unknown factories
                    defaultdict_info[key] = 'list'
                    logger.debug(
                        f"Unknown defaultdict factory for {key}, defaulting to list"
                    )
            else:
                result[key] = value
        
        # Include factory info in state if we found any defaultdicts
        if defaultdict_info:
            result['_defaultdict_factories'] = defaultdict_info
            
        return result
    
    def _restore_defaultdicts(self) -> None:
        """
        Restore defaultdict instances after deserialization.
        
        This method restores regular dicts back to defaultdicts using
        the factory type information stored during serialization.
        """
        # Factory type mapping
        factory_map = {
            'list': list,
            'dict': dict,
            'set': set,
            'int': int,
            'float': float,
            'str': str,
        }
        
        defaultdict_factories = getattr(self, '_defaultdict_factories', {})
        for attr, factory_name in defaultdict_factories.items():
            if hasattr(self, attr) and isinstance(getattr(self, attr), dict):
                factory = factory_map.get(factory_name, list)
                setattr(self, attr, defaultdict(factory, getattr(self, attr)))
        
        # Clean up tracking attribute
        if hasattr(self, '_defaultdict_factories'):
            delattr(self, '_defaultdict_factories')


class ThreadSafeSerializationMixin(SerializationMixin):
    """
    Extended serialization mixin with explicit thread-safe lock management.
    
    This mixin adds:
    - Guaranteed lock restoration as RLock (reentrant)
    - Lock attribute name flexibility (lock or _lock)
    - Post-restoration validation
    
    Use this for classes where thread safety is critical.
    """
    
    # Standard lock attribute name (subclasses can override)
    _lock_attr_name: str = 'lock'
    
    def _restore_unpickleable_attrs(self) -> None:
        """
        Restore the threading lock.
        
        Creates a new RLock instance. Subclasses should call
        super()._restore_unpickleable_attrs() and then restore
        their own unpickleable attributes.
        """
        setattr(self, self._lock_attr_name, threading.RLock())
    
    def _validate_lock(self) -> bool:
        """
        Validate that the lock was properly restored.
        
        Returns:
            True if lock exists and is an RLock, False otherwise
        """
        lock = getattr(self, self._lock_attr_name, None)
        return lock is not None and isinstance(lock, type(threading.RLock()))


def make_pickleable(
    obj: Any,
    unpickleable_attrs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Utility function to make any object's state pickleable.
    
    This is useful for objects that don't inherit from SerializationMixin
    but need to be serialized.
    
    Args:
        obj: Object to extract state from
        unpickleable_attrs: List of attribute names to exclude
        
    Returns:
        Dictionary of pickleable state
        
    Example:
        >>> state = make_pickleable(some_obj, ['lock', '_np'])
        >>> pickle.dumps(state)
    """
    unpickleable_attrs = unpickleable_attrs or ['lock', '_lock', '_np']
    
    state = {}
    for key, value in obj.__dict__.items():
        if key not in unpickleable_attrs:
            # Type-based check first (more efficient than pickle test)
            value_type = type(value)
            
            # Known unpickleable types - skip immediately
            if value_type.__module__ == 'builtins' and value_type.__name__ == 'module':
                logger.debug(f"Skipping module reference: {key}")
                continue
            if hasattr(value_type, '__self__'):  # Bound method
                logger.debug(f"Skipping bound method: {key}")
                continue
            if value_type.__name__ in ('RLock', 'Lock', 'Semaphore'):
                logger.debug(f"Skipping threading primitive: {key}")
                continue
            
            # For remaining types, try pickle test only if uncertain
            try:
                import pickle
                pickle.dumps(value)
                state[key] = value
            except (TypeError, pickle.PicklingError) as e:
                logger.debug(f"Skipping unpickleable attribute {key}: {e}")
                continue
    
    return state


# Export all public symbols
__all__ = [
    'SerializationMixin',
    'ThreadSafeSerializationMixin',
    'make_pickleable',
    'SERIALIZATION_VERSION',
]
