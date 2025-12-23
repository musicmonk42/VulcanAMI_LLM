# module_b.py - Module B functionality
# This module demonstrates proper handling of circular imports using local imports.
# The import of module_a is done inside the function to avoid circular import errors.


def function_b(depth: int = 0):
    """
    Function B implementation that optionally calls function_a.
    
    Uses lazy import to avoid circular dependency. The depth parameter prevents
    infinite recursion by limiting the call chain depth.
    
    Args:
        depth: Current recursion depth. Stops calling function_a when depth >= 1.
    """
    print(f"Function B is called (depth={depth})")
    if depth < 1:  # Prevent infinite recursion
        # Use local import to avoid circular import at module load time
        # This is the recommended pattern when two modules need to call each other
        from .module_a import function_a  # noqa: E402
        function_a(depth + 1)