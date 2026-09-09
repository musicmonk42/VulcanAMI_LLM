"""Compatibility facade for pre-convergence semantic imports.

The canonical contracts and deterministic operations live in
:mod:`vulcan.graphix.runtime`.  This module owns no authority and is removed
when downstream tests and research callers import the Graphix dialect directly.
"""
from vulcan.graphix.runtime import *  # noqa: F401,F403
