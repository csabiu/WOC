"""Public API for the pywoc package.

Exposes the main entry points:

- `woc`: compute the weighted overlap coefficient
- `radial_profile`: compute radial statistics used by `woc`
"""

from .woc import woc  # noqa: F401

def radial_profile(*args, **kwargs):  # noqa: D401
    """Lazy import wrapper for `pywoc.radial_profile.radial_profile`."""
    from .radial_profile import radial_profile as _radial_profile

    return _radial_profile(*args, **kwargs)

__all__ = ["woc", "radial_profile"]

# Keep a package version here for runtime introspection; the packaging version
# is still defined in setup.py.
__version__ = "0.4.2"
