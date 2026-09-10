"""Canonical Phase-A runtime package.

The serving package root intentionally exports no owner, store, registry, model,
or application singleton. Import explicit request contracts and ``RuntimeAPI``
from :mod:`vulcan.runtime.api`.
"""

__all__: tuple[str, ...] = ()
