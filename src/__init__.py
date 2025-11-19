# src/__init__.py
try:
    from .stdio_policy import install
    install(replace_builtins=True, patch_ray=True, patch_colorama=True)
except Exception:
    pass
