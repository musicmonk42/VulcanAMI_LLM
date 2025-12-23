# module_a.py - Module A functionality
# This module demonstrates proper handling of circular imports using local imports.
# The import of module_b is done inside the function to avoid circular import errors.


def function_a(depth: int = 0):
    """
    Call function_a and optionally function_b.
    
    Uses lazy import to avoid circular dependency. The depth parameter prevents
    infinite recursion by limiting the call chain depth.
    
    Args:
        depth: Current recursion depth. Stops calling function_b when depth >= 1.
    """
    print(f"Function A is called (depth={depth})")
    if depth < 1:  # Prevent infinite recursion
        # Use local import to avoid circular import at module load time
        # This is the recommended pattern when two modules need to call each other
        from src.module_b import function_b  # noqa: E402
        function_b(depth + 1)

