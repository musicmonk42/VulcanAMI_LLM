# module_a.py - Module A functionality
from module_b import function_b  # Avoid circular import by using local imports

def function_a():
    print("Function A is called")
    function_b()
# module_b.py - Module B functionality
from module_a import function_a  # Avoid circular import by using local imports

def function_b():
    print("Function B is called")
    function_a()