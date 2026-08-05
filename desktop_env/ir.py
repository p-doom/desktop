"""The one thing every layer of this package agrees on: an ``Operation``.

An ``Operation`` is a *resolved* input event.  Every coordinate it carries is an
absolute screen pixel in the guest's own framebuffer.  Whatever grammar produced
it -- absolute clicks, relative deltas, a 0..999 normalized grid, a
downscaled-screenshot convention -- that grammar has already been applied and
discarded by the time an ``Operation`` exists.  Nothing in this package can tell
which grammar it came from, and nothing here is allowed to care.

``kind`` is an OPEN vocabulary, deliberately a ``str`` and never an ``Enum``.  A
codec may introduce a kind this package has never heard of; the executor will
reject the kinds it cannot lower, and adding a kind is a change in exactly one
handler table rather than a change to a closed type that every importer shares.

The canonical kinds this package's own executor lowers are listed in
``CANONICAL_KINDS`` below.  That list is documentation and a lowering contract,
not a validator: ``Operation("my_new_kind", ())`` is a legal value.
"""

from __future__ import annotations

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
    def from_dict(cls, payload: dict[str, Any]) -> "Operation":
        return cls(str(payload["kind"]), tuple(payload.get("args", ())))


# --------------------------------------------------------------------------- #
# The canonical kinds, with their argument shapes.
#
# Every entry here is lowered by ``desktop_env.execute.guest_program``.  The
# coordinates in ``move_to`` and ``drag`` are absolute pixels; there is no
# relative member, because resolution happens inside a codec's ``compile`` and
# never inside this package.
# --------------------------------------------------------------------------- #

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
    "scroll": (
        "(dx: int, dy: int) -- wheel ticks, +dy up, +dx right. A one-argument "
        "(clicks,) form is also accepted and means (0, clicks)"
    ),
    "coalesced_type": (
        "(text: str) -- exact Unicode becomes input; the executor picks the "
        "mechanism (keystrokes or a clipboard paste) from the payload"
    ),
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
    return Operation("glide_to", (int(x), int(y), float(seconds)))


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
    """Read a ``scroll`` operation's args, accepting either arity.

    BOTH ARITIES ARE LIVE, and this function is the only place that decides
    which is which -- so a producer and a consumer cannot disagree silently.

    * ``(dx, dy)`` -- the two-axis form every codec emits.
    * ``(clicks,)`` -- the one-axis form the lifted guest program and its tests
      used, meaning vertical only.

    Arity disambiguates them completely, which is why both can be accepted
    without a mode flag.  The failure this guards against is real and was
    observed as a contract mismatch between layers: reading ``args[0]`` as
    vertical ticks turns a codec's ``scroll(0, 3)`` into ``scroll(0)`` -- a
    silent no-op on every scroll, with no error anywhere.
    """
    if len(args) == 1:
        return 0, int(args[0])
    if len(args) >= 2:
        return int(args[0]), int(args[1])
    raise ValueError("scroll requires (dx, dy) or (clicks,)")


def coalesced_type(text: str) -> Operation:
    return Operation("coalesced_type", (str(text),))


def ascii_type(text: str) -> Operation:
    return Operation("ascii_type", (str(text),))


def wait(seconds: float) -> Operation:
    return Operation("wait", (max(0.0, min(10.0, float(seconds))),))
