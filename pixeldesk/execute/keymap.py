"""One key/button name table, replacing three.

CONSOLIDATION (three implementations -> one):

  * ``rung1/transport.py``            ``_KEYS`` (19 entries) + ``pyautogui_key``
                                      -- 51 call sites
  * ``eval/osworld_vm_client.py``     ``_RDEV_TO_PYAUTOGUI`` (35) +
                                      ``_COMPUTER_USE_KEY_ALIASES`` (30) +
                                      ``_MOUSE_BUTTON_NAMES`` -- 44 call sites
  * RL ``rl/computer_use/actions.py`` inline lowercasing + button validation
                                      -- 22 call sites

The three disagreed in ways that mattered: the first knew ``Key<X>`` but not
``Num<N>``/``Digit<N>``, the second knew all three plus punctuation names and a
separate uppercase-alias table, and the third only lowercased.  A trajectory
recorded through one and replayed through another therefore did not round-trip.
The union is below, with the shapes tried in a fixed order.

The unified surface emits ``Operation`` values, never pyautogui code strings.
All three predecessors returned source text -- ``"pyautogui.keyDown('ctrl')"`` --
which forced every consumer to concatenate code and made the guest program a
string-splicing exercise.  The one place that legitimately needs pyautogui source
is ``guest_program.py``, which owns that translation for the whole package.
"""

from __future__ import annotations

import re

from ..ir import Operation


#: Explicit event-name -> guest key name.  Union of the three predecessor tables;
#: keys are rdev / DOM ``KeyboardEvent.code`` spellings, values are the pinned
#: guest backend's (pyautogui's) names.
KEY_NAMES: dict[str, str] = {
    # modifiers
    "ControlLeft": "ctrlleft",
    "ControlRight": "ctrlright",
    "ShiftLeft": "shiftleft",
    "ShiftRight": "shiftright",
    "Alt": "alt",
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
    # punctuation (present only in the eval client's table)
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
#: Kept as a second table because the two are matched DIFFERENTLY: ``KEY_NAMES``
#: exactly, this one on ``.upper()``.  Merging them would make the exact table
#: case-insensitive, and then ``Alt`` could no longer resolve differently from
#: any other casing -- which matters for the sided names, where an exact
#: ``MetaLeft`` must stay ``winleft`` while a loose ``META`` means ``win``.
#: (The twelve entries that currently overlap by casing agree on their value, so
#: no *present* mapping depends on the split; the resolution ORDER does.)
KEY_ALIASES: dict[str, str] = {
    "CTRL": "ctrl",
    "CONTROL": "ctrl",
    "SHIFT": "shift",
    "ALT": "alt",
    "OPTION": "alt",
    "CMD": "command",
    "COMMAND": "command",
    "META": "win",
    "SUPER": "win",
    "WIN": "win",
    "WINDOWS": "win",
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

_FUNCTION_KEY = re.compile(r"^F([1-9]|1[0-9]|2[0-4])$")

#: ``KEY_NAMES`` folded to upper case, consulted only after the exact and alias
#: tables have both missed.
#:
#: Without it, 20 of the 38 ``KEY_NAMES`` entries broke under a change of case and
#: broke SILENTLY: the fallback lowercases an unrecognised name, so
#: ``guest_key("comma")`` returned ``"comma"`` -- which pyautogui does not know, so
#: the keystroke was simply dropped -- while ``guest_key("Comma")`` returned
#: ``","``.  Every arrow, both sided control/meta modifiers, ``AltGr`` and all
#: eleven punctuation names were affected, i.e. exactly the keys a computer-use
#: model presses most.  Folding is a FALLBACK rather than a replacement for the
#: exact lookup so that a sided name stays distinguishable from a loose one:
#: ``MetaLeft`` is ``winleft`` while ``META`` is ``win``.
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
    2. case-insensitive ``KEY_ALIASES`` hit (``"CTRL"``, ``"cmd"`` -> ``"ctrl"``,
       ``"command"``)
    3. case-insensitive ``KEY_NAMES`` hit (``"comma"`` -> ``","``), so a recorder
       that emits a differently-cased event name still lands on a pressable key
       rather than on a lowercased passthrough the guest ignores
    4. ``F1``..``F24`` (case-insensitive)
    5. ``Key<A>`` -> ``"a"``; ``Num<0>`` / ``Digit<0>`` -> ``"0"``
    6. otherwise the name lowercased -- which covers a single character, and the
       guest backend accepts many lowercased X11 names verbatim, so passing
       through beats raising
    """
    if not isinstance(name, str):
        raise KeymapError(f"key must be a string, got {type(name).__name__}")
    stripped = name.strip()
    if not stripped:
        raise KeymapError("key name is empty")
    if stripped in KEY_NAMES:
        return KEY_NAMES[stripped]
    upper = stripped.upper()
    if upper in KEY_ALIASES:
        return KEY_ALIASES[upper]
    if upper in _KEY_NAMES_FOLDED:
        return _KEY_NAMES_FOLDED[upper]
    if match := _FUNCTION_KEY.match(upper):
        return f"f{match.group(1)}"
    if len(stripped) == 4 and stripped.startswith("Key") and stripped[3].isalpha():
        return stripped[3].lower()
    if len(stripped) == 4 and stripped.startswith("Num") and stripped[3].isdigit():
        return stripped[3]
    if len(stripped) == 6 and stripped.startswith("Digit") and stripped[5].isdigit():
        return stripped[5]
    return stripped.lower()


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
