import ast
from pathlib import Path


def test_constitution_primitives_have_no_high_level_imports():
    source = Path("src/vulcan/constitution/primitives.py").read_text()
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    forbidden = (
        "vulcan.runtime",
        "vulcan.graphix",
        "vulcan.memory",
        "vulcan.deployment",
    )
    assert not any(name.startswith(forbidden) for name in imports)
