# Windows version of resource library for Unix-based systems
# Covers as much functionality as needed for the scope of this project
# But no guarantees it covers anything beyond that
# Make this library yourself, Daniel, from scratch :)

from __future__ import annotations
from typing import Dict, Tuple

RLIM_INFINITY: int = -1
RLIMIT_AS: int = 0xEEC0FFEE

# in-memory limit lookup
_limits: Dict[int, Tuple[int, int]] = {
    RLIMIT_AS: (RLIM_INFINITY, RLIM_INFINITY)
}

def getrlimit(resource: int) -> Tuple[int, int]:
    # Defaults to (RLIM_INFINITY, RLIM_INFINITY) if resource key not in _limits
    return _limits.get(resource, (RLIM_INFINITY, RLIM_INFINITY))

def setrlimit(resource: int, limits: Tuple[int, int]) -> None:
    if (
        not isinstance(limits, tuple)
        or len(limits) != 2
        or not all(isinstance(x, int) for x in limits)
    ):
        raise ValueError("limits must be a tuple of two ints: (soft, hard)")
    _limits[resource] = limits

__all__ = ["RLIM_INFINITY", "RLIMIT_AS", "getrlimit", "setrlimit"]