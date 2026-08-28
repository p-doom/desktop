"""One key and pointer-button name table, and the operations built from it.

The tables below are the union of the spellings different trajectory recorders
emit -- rdev / DOM ``KeyboardEvent.code`` names, computer-use aliases, bare X11
names -- resolved in the fixed order ``guest_key`` documents, so a trajectory
recorded under one spelling and replayed under another round-trips.

This module emits ``Operation`` values and fixed X11 keysyms.  Guest keycodes are
resolved from those keysyms at execution time because keycodes belong to the X
server's active keyboard map.
"""

from __future__ import annotations

import re

from ..ir import Operation

#: Explicit event-name -> canonical X11 key name.
KEY_NAMES: dict[str, str] = {
    # modifiers
    "ControlLeft": "ctrlleft",
    "ControlRight": "ctrlright",
    "ShiftLeft": "shiftleft",
    "ShiftRight": "shiftright",
    "Alt": "altleft",
    "AltLeft": "altleft",
    "AltGr": "altright",
    "AltRight": "altright",
    "MetaLeft": "winleft",
    "MetaRight": "winright",
    # editing / whitespace
    "Return": "enter",
    "Enter": "enter",
    "Escape": "esc",
    "Backspace": "backspace",
    "Delete": "delete",
    "Insert": "insert",
    "Tab": "tab",
    "Space": "space",
    "CapsLock": "capslock",
    # navigation
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "PageUp": "pageup",
    "PageDown": "pagedown",
    "Home": "home",
    "End": "end",
    # punctuation
    "Comma": ",",
    "Period": ".",
    "Slash": "/",
    "Backslash": "\\",
    "Semicolon": ";",
    "Quote": "'",
    "Minus": "-",
    "Equal": "=",
    "Backquote": "`",
    "BracketLeft": "[",
    "BracketRight": "]",
}

#: Case-insensitive aliases a model is likely to emit in prose-ish grammars.
#: A separate table because the two are matched DIFFERENTLY: ``KEY_NAMES``
#: exactly, this one on ``.upper()``.  Merging them would make the exact table
#: case-insensitive, and then a sided name could no longer resolve differently
#: from a loose one -- ``MetaRight`` stays right-sided while ``META`` selects the
#: canonical left-hand key.
KEY_ALIASES: dict[str, str] = {
    "CTRL": "ctrlleft",
    "CONTROL": "ctrlleft",
    "SHIFT": "shiftleft",
    "ALT": "altleft",
    "OPTION": "altleft",
    "CMD": "winleft",
    "COMMAND": "winleft",
    "META": "winleft",
    "SUPER": "winleft",
    "WIN": "winleft",
    "WINDOWS": "winleft",
    "ENTER": "enter",
    "RETURN": "enter",
    "ESC": "esc",
    "ESCAPE": "esc",
    "BACKSPACE": "backspace",
    "DELETE": "delete",
    "DEL": "delete",
    "TAB": "tab",
    "SPACE": "space",
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "PAGEUP": "pageup",
    "PAGE_UP": "pageup",
    "PAGEDOWN": "pagedown",
    "PAGE_DOWN": "pagedown",
    "HOME": "home",
    "END": "end",
}


KEYSYMS: dict[str, int] = {
    **{chr(code): code for code in range(0x20, 0x7F)},
    "space": 0x20,
    "backspace": 0xFF08,
    "tab": 0xFF09,
    "enter": 0xFF0D,
    "pause": 0xFF13,
    "scrolllock": 0xFF14,
    "esc": 0xFF1B,
    "home": 0xFF50,
    "left": 0xFF51,
    "up": 0xFF52,
    "right": 0xFF53,
    "down": 0xFF54,
    "pageup": 0xFF55,
    "pagedown": 0xFF56,
    "end": 0xFF57,
    "printscreen": 0xFF61,
    "insert": 0xFF63,
    "menu": 0xFF67,
    "numlock": 0xFF7F,
    "shiftleft": 0xFFE1,
    "shiftright": 0xFFE2,
    "ctrlleft": 0xFFE3,
    "ctrlright": 0xFFE4,
    "capslock": 0xFFE5,
    "altleft": 0xFFE9,
    "altright": 0xFFEA,
    "winleft": 0xFFEB,
    "winright": 0xFFEC,
    "delete": 0xFFFF,
    **{f"f{number}": 0xFFBD + number for number in range(1, 25)},
}

PRESSABLE_KEYS = frozenset(KEYSYMS)

_FUNCTION_KEY = re.compile(r"^F([1-9]|1[0-9]|2[0-4])$")

#: ``KEY_NAMES`` folded to upper case, consulted only after the exact and alias
#: tables have both missed.
#:
#: Exact names still win so sided names remain distinct.
_KEY_NAMES_FOLDED: dict[str, str] = {}
for _name, _guest_name in KEY_NAMES.items():
    # First declaration wins, so the table stays deterministic if a later entry
    # ever collides case-insensitively with an earlier one.
    _KEY_NAMES_FOLDED.setdefault(_name.upper(), _guest_name)
del _name, _guest_name


class KeymapError(ValueError):
    """A key or button name cannot be mapped to a guest name."""


def guest_key(name: str) -> str:
    """Map one event name to the guest backend's key name.

    Resolution order, fixed so that two call sites can never disagree:

    1. exact ``KEY_NAMES`` hit (``"Return"`` -> ``"enter"``)
    2. case-insensitive ``KEY_ALIASES`` hit (``"CTRL"`` -> ``"ctrlleft"``)
    3. case-insensitive ``KEY_NAMES`` hit (``"comma"`` -> ``","``), so a recorder
       that emits a differently-cased event name still lands on a pressable key
       rather than on a lowercased passthrough the guest ignores
    4. ``F1``..``F24`` (case-insensitive)
    5. ``Key<A>`` -> ``"a"``; ``Num<0>`` / ``Digit<0>`` -> ``"0"``
    6. otherwise the name lowercased, which covers a single character

    The resolved name must have a fixed X11 keysym.  Validation happens here so
    an unsupported key fails before a guest process is dispatched.
    """
    if not isinstance(name, str):
        raise KeymapError(f"key must be a string, got {type(name).__name__}")
    stripped = name.strip()
    if not stripped:
        raise KeymapError("key name is empty")
    upper = stripped.upper()
    if stripped in KEY_NAMES:
        resolved = KEY_NAMES[stripped]
    elif upper in KEY_ALIASES:
        resolved = KEY_ALIASES[upper]
    elif upper in _KEY_NAMES_FOLDED:
        resolved = _KEY_NAMES_FOLDED[upper]
    elif match := _FUNCTION_KEY.match(upper):
        resolved = f"f{match.group(1)}"
    elif len(stripped) == 4 and stripped.startswith("Key") and stripped[3].isalpha():
        resolved = stripped[3].lower()
    elif len(stripped) == 4 and stripped.startswith("Num") and stripped[3].isdigit():
        resolved = stripped[3]
    elif len(stripped) == 6 and stripped.startswith("Digit") and stripped[5].isdigit():
        resolved = stripped[5]
    else:
        resolved = stripped.lower()
    if resolved not in KEYSYMS:
        raise KeymapError(f"unsupported X11 key: {name!r}")
    return resolved


#: The only buttons the guest program can press.  ``BUTTON_MASKS`` in
#: ``guest_program`` must stay in lockstep with this set.
POINTER_BUTTONS: frozenset[str] = frozenset({"left", "middle", "right"})

#: X11 button number -> name, for reading a raw event stream back.
BUTTON_NUMBERS: dict[int, str] = {1: "left", 2: "middle", 3: "right"}

#: Recorder button spellings seen in trajectory corpora.
BUTTON_ALIASES: dict[str, str] = {
    "LMB": "left",
    "MMB": "middle",
    "RMB": "right",
    "BUTTONLEFT": "left",
    "BUTTONMIDDLE": "middle",
    "BUTTONRIGHT": "right",
    "LEFT": "left",
    "MIDDLE": "middle",
    "RIGHT": "right",
}


def guest_button(name: str | int) -> str:
    """Normalize a pointer button to ``left`` / ``middle`` / ``right``.

    Accepts an X11 button number, a recorder alias (``LMB``), or a bare name.
    Raises rather than defaulting to ``left``: a silently defaulted button is
    how a right-click test passes while testing a left click.
    """
    if isinstance(name, bool):
        raise KeymapError(f"unsupported pointer button: {name!r}")
    if isinstance(name, int):
        if name not in BUTTON_NUMBERS:
            raise KeymapError(f"unsupported X11 button number: {name}")
        return BUTTON_NUMBERS[name]
    if not isinstance(name, str):
        raise KeymapError(f"unsupported pointer button: {name!r}")
    stripped = name.strip()
    if stripped in POINTER_BUTTONS:
        return stripped
    upper = stripped.upper()
    if upper in BUTTON_ALIASES:
        return BUTTON_ALIASES[upper]
    raise KeymapError(f"unsupported pointer button: {name!r}")


def key_chord(keys: list[str] | tuple[str, ...]) -> tuple[Operation, ...]:
    """A chord: press in order, release in reverse.

    Reverse release order is not cosmetic -- releasing ``ctrl`` before ``a`` in
    ``ctrl+a`` delivers a bare ``a`` to the focused widget on the pinned guest.
    """
    names = [guest_key(key) for key in keys]
    if not names:
        raise KeymapError("empty key chord")
    downs = tuple(Operation("key_down", (key,)) for key in names)
    ups = tuple(Operation("key_up", (key,)) for key in reversed(names))
    return downs + ups


def key_press(key: str) -> tuple[Operation, ...]:
    """One key down then up."""
    mapped = guest_key(key)
    return (Operation("key_down", (mapped,)), Operation("key_up", (mapped,)))


def key_transition(key: str, *, pressed: bool) -> Operation:
    """A single half of a key event, for grammars that emit raw transitions."""
    return Operation("key_down" if pressed else "key_up", (guest_key(key),))


def button_transition(button: str | int, *, pressed: bool) -> Operation:
    """A single half of a pointer-button event."""
    kind = "mouse_down" if pressed else "mouse_up"
    return Operation(kind, (guest_button(button),))
