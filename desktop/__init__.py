"""desktop: boot desktop VMs, and apply resolved input to them.

Receives ``Operation``s already resolved to absolute screen pixels and applies
them to a real desktop.

One runtime dependency, Pillow, used in ``desktop.vm.readiness`` to downsample
and measure a screenshot.  HTTP is ``urllib.request``, and screenshots cross
every other boundary here as raw ``bytes``.  Two external binaries are assumed:
``qemu-system-x86_64`` and ``apptainer``.
"""

from .geometry import (
    DisplayGeometry,
    anthropic_scale_coordinates,
    clamp_to_desktop,
    geometry_from_screen_size,
    resolve_relative,
    scale_normalized_coordinate,
)
from .ir import CANONICAL_KINDS, Operation, drag, glide_to, move_to, scroll_deltas

__all__ = [
    "CANONICAL_KINDS",
    "DisplayGeometry",
    "Operation",
    "anthropic_scale_coordinates",
    "clamp_to_desktop",
    "drag",
    "geometry_from_screen_size",
    "glide_to",
    "move_to",
    "resolve_relative",
    "scale_normalized_coordinate",
    "scroll_deltas",
]

__version__ = "0.1.0"
