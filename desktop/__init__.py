"""desktop: boot desktop VMs, and apply resolved input to them.

Receives ``Operation``s already resolved to absolute screen pixels and applies
them to a real desktop.  Nothing in this package imports a grammar or a codec:
whichever coordinate convention a model emits is consumed inside that team's own
``Codec.compile`` before an ``Operation`` exists, so an absolute-coordinate model
and a relative-delta model both use this package unchanged.

One runtime dependency, Pillow, used in ``desktop.vm.readiness`` to downsample
and measure a screenshot.  HTTP is ``urllib.request``, and screenshots cross
every other boundary here as raw ``bytes``.  Two external binaries are assumed:
``qemu-system-x86_64`` and ``apptainer``.
"""

from .codec_protocol import ActionSet, ActionSpec, Codec, Handler, ParsedCall, parse_calls
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
    "ActionSet",
    "ActionSpec",
    "Codec",
    "DisplayGeometry",
    "Handler",
    "Operation",
    "ParsedCall",
    "anthropic_scale_coordinates",
    "clamp_to_desktop",
    "drag",
    "geometry_from_screen_size",
    "glide_to",
    "move_to",
    "parse_calls",
    "resolve_relative",
    "scale_normalized_coordinate",
    "scroll_deltas",
]

__version__ = "0.1.0"
