"""Test-time compatibility helpers."""
from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    # dataclasses.dataclass gained the ``slots`` parameter in Python 3.10. When
    # running under an older interpreter (e.g. the user's Python 3.9 test
    # environment) importing modules that use ``@dataclass(slots=True)`` raises
    # ``TypeError``. Rather than touching every declaration, provide a small
    # shim that gracefully ignores the ``slots`` argument when unsupported.
    from dataclasses import dataclass as _stdlib_dataclass
    import dataclasses as _stdlib_module

    def _compat_dataclass(*args, **kwargs):
        kwargs.pop("slots", None)
        return _stdlib_dataclass(*args, **kwargs)

    _stdlib_module.dataclass = _compat_dataclass  # type: ignore[assignment]
