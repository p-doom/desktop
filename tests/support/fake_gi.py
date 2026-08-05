"""A fake ``gi`` / PyGObject, so the GTK clipboard program can be executed.

``compile_unicode_coalesced_type`` is the package's only typing path with
production semantics, and it is a *generated string* that imports ``gi`` and runs
a GLib main loop.  Nothing about it is checkable by reading.  This double records
the clipboard text, runs the queued ``timeout_add`` callbacks in ``Gtk.main()``,
and can be configured to reproduce both failure modes the program guards:

* ``round_trip_ok=False`` -- ``wait_for_text`` disagrees with what was set, i.e.
  clipboard ownership never took.
* ``run_callbacks=False`` -- the owner expired before the paste callback ran.
"""

from __future__ import annotations

import sys
import types

_CLIPBOARD: "_Clipboard | None" = None
_TIMEOUTS: list[tuple[int, object]] = []


class _Clipboard:
    def __init__(self) -> None:
        self.text: str | None = None

    def set_text(self, value: str, length: int) -> None:
        self.text = value

    def wait_for_text(self) -> str | None:
        return self.text


def clipboard_text() -> str | None:
    return None if _CLIPBOARD is None else _CLIPBOARD.text


def install(*, round_trip_ok: bool = True, run_callbacks: bool = True):
    global _CLIPBOARD, _TIMEOUTS
    _CLIPBOARD = _Clipboard()
    _TIMEOUTS = []
    if not round_trip_ok:
        _CLIPBOARD.wait_for_text = lambda: "a value that is not what was set"

    def main() -> None:
        if not run_callbacks:
            return
        for _delay, callback in sorted(_TIMEOUTS, key=lambda item: item[0]):
            callback()

    gi = types.ModuleType("gi")
    gi.require_version = lambda *args, **kwargs: None
    repository = types.ModuleType("gi.repository")
    repository.Gtk = types.SimpleNamespace(
        Clipboard=types.SimpleNamespace(get=lambda selection: _CLIPBOARD),
        main=main,
        main_quit=lambda: None,
    )
    repository.Gdk = types.SimpleNamespace(SELECTION_CLIPBOARD="CLIPBOARD")
    repository.GLib = types.SimpleNamespace(
        timeout_add=lambda ms, callback: _TIMEOUTS.append((int(ms), callback))
    )
    gi.repository = repository
    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = repository
    return _CLIPBOARD, _TIMEOUTS
