"""desktop-env: apply resolved input to a real desktop, and reset it.

THE INVARIANT.  Nothing in this package imports a grammar or a codec, and nothing
here knows what an action grammar is.  It receives ``Operation``s already resolved
to absolute screen pixels and applies them.  A team training an absolute-coordinate
model and a team training a relative-delta model both use it unchanged, because the
difference between them is fully consumed inside their own ``Codec.compile`` before
an ``Operation`` exists.

ONE runtime dependency: Pillow, used in exactly one place
(``desktop_env.vm.readiness``) to downsample and measure a screenshot.  A
hand-written PNG decoder was written and then deleted, because a decoder bug is
silent -- it makes readiness say "not ready", which is indistinguishable from a
slow VM.  HTTP is ``urllib.request``, and screenshots cross every other boundary
here as raw ``bytes``.  Two external binaries are assumed:
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
