"""Display geometry and coordinate rescaling.

``DisplayGeometry``, ``scale_normalized_coordinate`` and
``anthropic_scale_coordinates`` are copied VERBATIM from Harbor
(``harbor-framework/harbor``, Apache-2.0,
``src/harbor/agents/computer_1/runtime.py``).  They are reproduced rather than
imported so this package keeps a zero-dependency floor; keeping them byte-equal
to upstream is deliberate, so a future upstream fix is a diff and not an
archaeology exercise.

WHAT IS DELIBERATELY *NOT* COPIED: Harbor's ``CoordinateSpace`` StrEnum.  A
closed enumeration of coordinate conventions is exactly the wrong shape for this
package.  Which convention a model emits is a fact about a *codec*, recorded as
an open field inside that codec, and it is fully consumed by the codec's
``compile(text, geometry, cursor)`` before an ``Operation`` exists.  The
functions below are therefore offered to codecs as resolution *tools*; this
package never inspects, stores, or branches on a coordinate space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# BEGIN verbatim copy -- Harbor, src/harbor/agents/computer_1/runtime.py
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DisplayGeometry:
    """Geometry of the desktop and the computer window inside it."""

    desktop_width: int
    desktop_height: int
    window_x: int = 0
    window_y: int = 0
    window_width: int = 0
    window_height: int = 0


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def scale_normalized_coordinate(
    model_x: int, model_y: int, geometry: DisplayGeometry
) -> tuple[int, int]:
    """Scale 0..999 normalized coordinates to desktop-space pixels."""
    x = round(model_x * (geometry.desktop_width - 1) / 999)
    y = round(model_y * (geometry.desktop_height - 1) / 999)
    return (
        _clamp(x, 0, geometry.desktop_width - 1),
        _clamp(y, 0, geometry.desktop_height - 1),
    )


ANTHROPIC_MAX_LONG_EDGE = 1568
ANTHROPIC_MAX_TOTAL_PIXELS = 1_150_000


def anthropic_scale_coordinates(
    x: int, y: int, desktop_width: int, desktop_height: int
) -> tuple[int, int]:
    """Map Anthropic model-space coordinates back to desktop pixels.

    Anthropic may internally downscale screenshots above its long-edge or total
    pixel limits. For small Harbor defaults this is a no-op; for larger
    desktops it reverses that downscale.
    """
    long_edge = max(desktop_width, desktop_height)
    total_pixels = desktop_width * desktop_height
    long_edge_scale = (
        ANTHROPIC_MAX_LONG_EDGE / long_edge
        if long_edge > ANTHROPIC_MAX_LONG_EDGE
        else 1.0
    )
    total_pixels_scale = (
        math.sqrt(ANTHROPIC_MAX_TOTAL_PIXELS / total_pixels)
        if total_pixels > ANTHROPIC_MAX_TOTAL_PIXELS
        else 1.0
    )
    scale = min(1.0, long_edge_scale, total_pixels_scale)
    if scale >= 1.0:
        return (x, y)
    return (int(x / scale), int(y / scale))


# --------------------------------------------------------------------------- #
# END verbatim copy
# --------------------------------------------------------------------------- #


def clamp_to_desktop(x: int, y: int, geometry: DisplayGeometry) -> tuple[int, int]:
    """Clamp an absolute pixel pair into the desktop's addressable range.

    The guest's own pointer backend clamps too, but a codec that clamps *before*
    emitting an ``Operation`` gets an honest record of the intended target in the
    operation itself, which is what a receipt needs in order to distinguish "the
    model aimed off-screen" from "the guest refused the move".
    """
    return (
        _clamp(int(x), 0, geometry.desktop_width - 1),
        _clamp(int(y), 0, geometry.desktop_height - 1),
    )


def geometry_from_screen_size(width: int, height: int) -> DisplayGeometry:
    """A full-screen geometry, which is the only case a bare VM has.

    The window fields stay zero: there is no computer *window* inset inside the
    desktop when the model drives the whole framebuffer.
    """
    return DisplayGeometry(desktop_width=int(width), desktop_height=int(height))


def resolve_relative(
    cursor: tuple[int, int], dx: int, dy: int, geometry: DisplayGeometry
) -> tuple[int, int]:
    """Resolve a relative delta against a cursor into an absolute pixel pair.

    Offered here so that a relative-grammar codec has one obvious place to do
    the resolution, and so the resolution context (``cursor``, ``geometry``)
    stays visible as *data* in its signature rather than hiding in a mode flag.
    """
    return clamp_to_desktop(int(cursor[0]) + int(dx), int(cursor[1]) + int(dy), geometry)
