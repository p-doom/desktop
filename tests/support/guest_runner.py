"""Compile a guest program, RUN it in a subprocess, and parse its result marker.

This is the harness that turns ``guest_program`` from unexecuted code into tested
code.  One subprocess per action, exactly as the real transport does, so the
program's ``guest_process_count`` claim and its ``sys.exit(1)`` on failure are
both real observations rather than assertions about a string.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixeldesk.execute.guest_program import (
    ATOMIC_RESULT_PREFIX,
    compile_atomic_guest_program,
)
from pixeldesk.ir import Operation

SUPPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUPPORT_DIR.parent.parent

_DRIVER_HEADER = """
import json as _json, sys as _sys
_sys.path.insert(0, {support!r})
import fake_guest_backend
{gi_setup}
_module, _backend = fake_guest_backend.install(
    size={size!r}, cursor={cursor!r}, initial_mask={mask!r}
)
"""

_DRIVER_FOOTER = """
_sys.stderr.write('PYAUTOGUI_CALLS=' + _json.dumps(_module.calls) + chr(10))
_sys.stderr.write('X_EVENTS=' + _json.dumps([list(e) for e in _backend.events]) + chr(10))
{gi_report}
"""


@dataclass
class GuestRun:
    """One executed guest program."""

    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any] | None
    program: str
    expected_mask: int

    @property
    def marker_count(self) -> int:
        return sum(
            1
            for line in self.stdout.splitlines()
            if line.startswith(ATOMIC_RESULT_PREFIX)
        )

    def _tagged(self, tag: str) -> Any:
        for line in self.stderr.splitlines():
            if line.startswith(tag + "="):
                return json.loads(line[len(tag) + 1 :])
        return None

    @property
    def pyautogui_calls(self) -> list[list] | None:
        """Every ``pyautogui.*`` call the program made, in order."""
        return self._tagged("PYAUTOGUI_CALLS")

    @property
    def x_events(self) -> list[list] | None:
        return self._tagged("X_EVENTS")

    @property
    def clipboard_text(self) -> Any:
        return self._tagged("CLIPBOARD_TEXT")

    def trace(self) -> list[tuple[str, list]]:
        """The guest's own reported operation trace, as ``(kind, args)`` pairs."""
        assert self.payload is not None, self.stderr
        return [(item["kind"], item["args"]) for item in self.payload["operations"]]

    def primitives(self, kind: str | None = None) -> list[dict[str, Any]]:
        assert self.payload is not None, self.stderr
        rows = self.payload["backend_primitives"]
        return [row for row in rows if kind is None or row.get("kind") == kind]


def run_guest_program(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str] | None = None,
    initial_keys: set[str] | None = None,
    size: tuple[int, int] = (1920, 1080),
    cursor: tuple[int, int] = (50, 50),
    initial_mask: int = 0,
    with_gi: bool = False,
    gi_round_trip_ok: bool = True,
    gi_run_callbacks: bool = True,
    timeout_s: float = 60.0,
    **compile_kwargs: Any,
) -> GuestRun:
    """Compile ``operations`` and execute the result in a fresh interpreter."""
    program, expected_mask = compile_atomic_guest_program(
        operations,
        initial_buttons=set(initial_buttons or ()),
        initial_keys=set(initial_keys or ()),
        **compile_kwargs,
    )
    gi_setup = ""
    gi_report = ""
    if with_gi:
        gi_setup = (
            "import fake_gi\n"
            f"fake_gi.install(round_trip_ok={gi_round_trip_ok!r}, "
            f"run_callbacks={gi_run_callbacks!r})"
        )
        gi_report = (
            "_sys.stderr.write('CLIPBOARD_TEXT=' + "
            "_json.dumps(fake_gi.clipboard_text()) + chr(10))"
        )
    driver = (
        _DRIVER_HEADER.format(
            support=str(SUPPORT_DIR),
            gi_setup=gi_setup,
            size=size,
            cursor=cursor,
            mask=initial_mask,
        )
        + program
        + _DRIVER_FOOTER.format(gi_report=gi_report)
    )
    completed = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(REPO_ROOT),
    )
    markers = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(ATOMIC_RESULT_PREFIX)
    ]
    payload = None
    if len(markers) == 1:
        payload = json.loads(markers[0][len(ATOMIC_RESULT_PREFIX) :])
    return GuestRun(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=payload,
        program=program,
        expected_mask=expected_mask,
    )
