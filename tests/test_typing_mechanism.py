"""``coalesced_type`` realisation: which mechanism, and did it actually type.

Two things are tested here that the rest of the suite did not test, and their
absence is why a total typing failure shipped green:

* **which route the payload takes.**  ``coalesced_type_mechanism`` is the one
  decision point, so it is asserted directly *and* through the compiled guest
  program, which is the artefact that runs.
* **the EFFECT, not the dispatch.**  Every previous coalesced-type test asserted
  that the program ran and reported ``ok`` -- which the clipboard route does
  faithfully while typing nothing into a terminal, because ``Ctrl-V`` there is
  readline's quoted-insert.  ``_GnomeTerminal`` below models just enough of that
  readline behaviour to answer the only question the gate cares about: did the
  command execute.  It is deliberately checked against a stream that must FAIL,
  so a probe that cannot see the defect cannot pass itself off as a green test.
"""

from __future__ import annotations

import pytest

from pixeldesk import ir
from pixeldesk.execute.guest_program import (
    GTK_CLIPBOARD_TYPING_MECHANISM,
    PYAUTOGUI_WRITE_TYPING_MECHANISM,
    coalesced_type_mechanism,
    compile_unicode_coalesced_type,
)

from .support.guest_runner import run_guest_program

ASCII_TEXT = "printf SOLV2_COMPOUND_OK > /tmp/solv2_compound.txt"
UNICODE_TEXT = "héllo ✓"


# --------------------------------------------------------------------------- #
# the predicate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    ["ls", ASCII_TEXT, "", " ", "~", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}"],
)
def test_printable_ascii_is_typed_directly(text):
    assert coalesced_type_mechanism(text) == PYAUTOGUI_WRITE_TYPING_MECHANISM


@pytest.mark.parametrize("text", [UNICODE_TEXT, "é", "日本語", "🙂", "a\nb", "a\tb"])
def test_everything_pyautogui_write_would_drop_takes_the_clipboard(text):
    assert coalesced_type_mechanism(text) == GTK_CLIPBOARD_TYPING_MECHANISM


def test_the_mechanism_still_rejects_a_non_string_payload():
    with pytest.raises(TypeError, match="must be a string"):
        coalesced_type_mechanism(b"ls")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# the compiled program follows the predicate
# --------------------------------------------------------------------------- #


def test_the_ascii_route_writes_keystrokes_and_never_touches_the_clipboard():
    source = compile_unicode_coalesced_type(ASCII_TEXT)
    assert source == f"pyautogui.write({ASCII_TEXT!r},interval=0)"
    assert "gi" not in source and "ctrl" not in source

    run = run_guest_program((ir.coalesced_type(ASCII_TEXT),), with_gi=True)
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True, run.payload["error"]
    assert ["write", ASCII_TEXT, 0] in run.pyautogui_calls
    assert not [call for call in run.pyautogui_calls if call[0] == "hotkey"]
    assert run.clipboard_text is None
    assert run.primitives("coalesced_type") == [
        {
            "kind": "coalesced_type",
            "call": PYAUTOGUI_WRITE_TYPING_MECHANISM,
            "utf8_bytes": len(ASCII_TEXT),
        }
    ]


def test_the_unicode_route_still_pastes_from_the_gtk_clipboard():
    run = run_guest_program((ir.coalesced_type(UNICODE_TEXT),), with_gi=True)
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True, run.payload["error"]
    assert run.clipboard_text == UNICODE_TEXT
    assert ["hotkey", ["ctrl", "a"]] in run.pyautogui_calls
    assert ["hotkey", ["ctrl", "v"]] in run.pyautogui_calls
    assert not [call for call in run.pyautogui_calls if call[0] == "write"]
    assert run.primitives("coalesced_type") == [
        {
            "kind": "coalesced_type",
            "call": GTK_CLIPBOARD_TYPING_MECHANISM,
            "utf8_bytes": len(UNICODE_TEXT.encode("utf-8")),
        }
    ]


# --------------------------------------------------------------------------- #
# the effect: did the command execute
# --------------------------------------------------------------------------- #


class _GnomeTerminal:
    """A readline line editor, modelled only where the two routes differ.

    Faithful in the one place that decided the gate: ``Ctrl-V`` is
    ``quoted-insert``, so it inserts no text and makes the NEXT keypress
    literal -- which is how the ``Return`` after a "paste" became a ``^M`` in
    the transcript instead of running anything.  ``Ctrl-A`` is
    ``beginning-of-line``, not select-all, so the clipboard route's re-assertion
    of it is also not the editing operation that route assumes.

    Not faithful, and it does not need to be: no cursor-relative insertion, no
    kill ring, no completion, no PTY.
    """

    PROMPT = "SOLV2-LS$ "

    def __init__(self) -> None:
        self.line = ""
        self.history: list[str] = []
        self.transcript = self.PROMPT
        self._quoted_insert_pending = False

    def _insert(self, text: str) -> None:
        for character in text:
            if self._quoted_insert_pending:
                self._quoted_insert_pending = False
            self.line += character
            self.transcript += character

    def _enter(self) -> None:
        if self._quoted_insert_pending:
            # The literal carriage return readline shows as ^M.  No execution.
            self._quoted_insert_pending = False
            self.line += "\r"
            self.transcript += "^M"
            return
        self.history.append(self.line)
        self.transcript += "\n" + self.line + "\n" + self.PROMPT
        self.line = ""

    def feed(self, calls: list[list]) -> "_GnomeTerminal":
        for call in calls:
            name = call[0]
            if name == "write":
                self._insert(str(call[1]))
            elif name == "hotkey":
                keys = [str(key).lower() for key in call[1]]
                if keys == ["ctrl", "v"]:
                    self._quoted_insert_pending = True
                # ctrl-a is beginning-of-line: no text, no execution.
            elif name == "keyDown" and str(call[1]).lower() in {"enter", "return"}:
                self._enter()
        return self

    def command_executed(self, command: str) -> bool:
        """Exact-match a history line, as ``oracle.history_has_exact`` does."""
        return any(line.strip() == command for line in self.history)


def _type_then_return(text: str) -> list[list]:
    """The guest calls for one scripted "type the text, press Enter" action."""
    run = run_guest_program(
        (ir.coalesced_type(text), ir.key_down("Return"), ir.key_up("Return")),
        with_gi=True,
    )
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True, run.payload["error"]
    return run.pyautogui_calls


@pytest.mark.parametrize("command", ["ls", ASCII_TEXT])
def test_the_terminal_actually_executes_the_typed_command(command):
    """The assertion whose absence let a 4/4 gate read 0/4 with zero errors."""
    terminal = _GnomeTerminal().feed(_type_then_return(command))
    assert terminal.command_executed(command) is True
    assert terminal.history == [command]
    assert "^M" not in terminal.transcript


def test_the_probe_can_see_the_clipboard_route_fail_in_a_terminal():
    """Negative control for the test above -- and the recorded defect itself.

    Without this, ``command_executed is True`` could be passing because the
    model executes anything.  The clipboard route is still correct for this
    payload (``pyautogui.write`` would drop it entirely), so what is asserted is
    a real property of a terminal, not a bug: ``Ctrl-V`` types nothing there and
    eats the following Return.  That is exactly the transcript observed in job
    138010 -- ``SOLV2-LS$ ^M``, ``command_executed: false``, no error.
    """
    terminal = _GnomeTerminal().feed(_type_then_return(UNICODE_TEXT))
    assert terminal.command_executed(UNICODE_TEXT) is False
    assert terminal.history == []
    assert terminal.transcript == "SOLV2-LS$ ^M"
