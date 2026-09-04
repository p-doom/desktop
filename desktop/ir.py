"""The one thing every layer of this package agrees on: an ``Operation``.

An ``Operation`` is a *resolved* input event.  Every coordinate it carries is an
absolute screen pixel in the guest's own framebuffer.  Whatever grammar produced
it -- absolute clicks, relative deltas, a 0..999 normalized grid, a
downscaled-screenshot convention -- has already been applied and discarded by the
time an ``Operation`` exists.

``kind`` is an OPEN vocabulary, a ``str`` and never an ``Enum``.  A codec may
introduce a kind this package has never heard of; the executor rejects the kinds
it cannot lower, and adding a kind is a change in one handler table rather than a
change to a closed type that every importer shares.

``CANONICAL_KINDS`` below lists the kinds this package's own executor lowers.  It
is a lowering contract, not a validator: ``Operation("my_new_kind", ())`` is a
legal value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Operation:
    """One resolved input event, in absolute guest screen pixels.

    Attributes:
        kind: What to do.  An open string vocabulary -- never an ``Enum``.
        args: Positional payload, per-kind.  Frozen so an ``Operation`` can be
            hashed, compared, and safely shared across a dispatch table.
    """

    kind: str
    args: tuple

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe view, for receipts and cross-process transport."""
        return {"kind": self.kind, "args": list(self.args)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Operation:
        """The inverse of ``as_dict``.

        Both keys are required, so a truncated receipt cannot become a
        well-formed ``Operation`` of the wrong arity -- a ``move_to`` with no
        destination, rejected later and somewhere else, or not at all.
        """
        return cls(str(payload["kind"]), tuple(payload["args"]))


# Argument shapes serialized by ``desktop.execute.protocol``. There is no
# relative member: callers resolve coordinates before constructing this IR.

GLIDE_MAXIMUM_SECONDS = 10.0

CANONICAL_KINDS: dict[str, str] = {
    "move_to": "(x: int, y: int) -- absolute pixel destination",
    "glide_to": (
        "(x: int, y: int, seconds: float) -- timed absolute move; the stroke of "
        "a drag, where the intermediate motion is itself observable to the app"
    ),
    "drag": (
        "(x0: int, y0: int, x1: int, y1: int) -- press at (x0,y0), move to "
        "(x1,y1), release; a zero-extent drag survives as a real press/release"
    ),
    "click": "(button: str) -- one press+release of 'left'|'middle'|'right'",
    "mouse_down": "(button: str)",
    "mouse_up": "(button: str)",
    "key_down": "(key: str) -- a keymap.py key name",
    "key_up": "(key: str)",
    "scroll": "(dx: int, dy: int) -- wheel ticks, +dy up, +dx right",
    "coalesced_type": "(text: str) -- exact Unicode input through direct XTEST key events",
    "ascii_type": "(text: str) -- ASCII only, no newlines, per-keystroke",
    "wait": "(seconds: float) -- clamped to [0, 10]",
    "raise_for_test": "(message: str) -- fault injection for the test suite",
}


def move_to(x: int, y: int) -> Operation:
    """An absolute cursor move."""
    return Operation("move_to", (int(x), int(y)))


def drag(x0: int, y0: int, x1: int, y1: int) -> Operation:
    """A press-move-release drag between two absolute points.

    This exists as its own kind rather than as ``mouse_down``/``move_to``/
    ``mouse_up`` so that a *genuine* zero-extent drag -- ``drag(x, y, x, y)``,
    which some applications use to place a caret or clear a selection -- still
    carries a press and a release after resolution.  Expressed as a triple it
    would be indistinguishable from a click whose move was optimized away, and
    a delta-resolving codec that produced ``dx=dy=0`` would collapse it into a
    no-op before the executor ever saw it.
    """
    return Operation("drag", (int(x0), int(y0), int(x1), int(y1)))


def glide_to(x: int, y: int, seconds: float) -> Operation:
    """A timed absolute move.

    Distinct from ``move_to`` because the intermediate motion is observable: a
    drag stroke that teleports is not the same gesture as one that sweeps, and
    some applications (selection handles, canvas tools, scrollbar thumbs) only
    respond to the sweep.
    """
    return Operation("glide_to", (int(x), int(y), glide_seconds(seconds)))


def glide_seconds(seconds: float) -> float:
    """Validate and preserve one requested glide duration."""
    value = float(seconds)
    if not math.isfinite(value) or not 0.0 < value <= GLIDE_MAXIMUM_SECONDS:
        raise ValueError(
            f"glide seconds must be finite and in (0, {GLIDE_MAXIMUM_SECONDS}], got {value!r}"
        )
    return value


def click(button: str = "left") -> Operation:
    return Operation("click", (str(button),))


def mouse_down(button: str = "left") -> Operation:
    return Operation("mouse_down", (str(button),))


def mouse_up(button: str = "left") -> Operation:
    return Operation("mouse_up", (str(button),))


def key_down(key: str) -> Operation:
    return Operation("key_down", (str(key),))


def key_up(key: str) -> Operation:
    return Operation("key_up", (str(key),))


def scroll(dx: int, dy: int) -> Operation:
    """Wheel ticks: ``+dy`` scrolls up, ``+dx`` scrolls right."""
    return Operation("scroll", (int(dx), int(dy)))


def scroll_deltas(args: tuple) -> tuple[int, int]:
    """Read a ``scroll`` operation's args.  Exactly ``(dx, dy)``, nothing else.

    The one place that reads them, so a producer and a consumer cannot disagree
    silently: reading ``args[0]`` as vertical ticks turns a codec's
    ``scroll(0, 3)`` into ``scroll(0)`` -- a silent no-op on every scroll, with no
    error anywhere.  A one-axis ``(clicks,)`` form is deliberately not accepted,
    so a grammar that emits a bare tick count gets an error rather than a second,
    differently-shaped contract to keep in agreement.
    """
    if len(args) != 2:
        raise ValueError(f"scroll requires exactly (dx, dy), got {args!r}")
    return int(args[0]), int(args[1])


def coalesced_type(text: str) -> Operation:
    return Operation("coalesced_type", (str(text),))


def ascii_type(text: str) -> Operation:
    return Operation("ascii_type", (str(text),))


def wait(seconds: float) -> Operation:
    return Operation("wait", (max(0.0, min(10.0, float(seconds))),))
