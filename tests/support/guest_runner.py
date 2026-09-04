"""Run the installed guest executor source against a fake XTEST backend."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from desktop.execute.keymap import guest_key
from desktop.execute.protocol import build_action_request
from desktop.ir import Operation
from desktop.vm.client import ACTION_EXECUTOR_PATH

SUPPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUPPORT_DIR.parent.parent

_DRIVER_HEADER = """
import sys as _sys
_sys.path.insert(0, {support!r})
import fake_guest_backend
_backend = fake_guest_backend.install(
    size={size!r}, cursor={cursor!r}, initial_mask={mask!r},
    initial_keys={keys!r}, fail_xtest_at={fail_xtest_at!r}
)
import runpy as _runpy
_runpy.run_path({executor!r}, run_name='__main__')
"""


@dataclass
class GuestRun:
    """One executed guest program."""

    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any] | None
    request: dict[str, Any]
    program: str
    expected_mask: int

    def _tagged(self, tag: str) -> Any:
        for line in self.stderr.splitlines():
            if line.startswith(tag + "="):
                return json.loads(line[len(tag) + 1 :])
        return None

    @property
    def x_events(self) -> list[list] | None:
        return self._tagged("X_EVENTS")

    @property
    def keymap_restored(self) -> bool | None:
        return self._tagged("KEYMAP_RESTORED")

    @property
    def held_keycodes(self) -> list[int] | None:
        return self._tagged("HELD_KEYCODES")

    def trace(self) -> list[tuple[str, list]]:
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
    backend_initial_keys: set[str] | None = None,
    size: tuple[int, int] = (1920, 1080),
    cursor: tuple[int, int] = (50, 50),
    initial_mask: int = 0,
    fail_xtest_at: int | None = None,
    timeout_s: float = 60.0,
) -> GuestRun:
    """Serialize ``operations`` and execute the real installed source."""
    held_keys = {guest_key(key) for key in initial_keys or ()}
    backend_keys = (
        held_keys
        if backend_initial_keys is None
        else {guest_key(key) for key in backend_initial_keys}
    )
    request, expected_mask, _ = build_action_request(
        operations,
        initial_buttons=set(initial_buttons or ()),
        initial_keys=held_keys,
    )
    driver = _DRIVER_HEADER.format(
        support=str(SUPPORT_DIR),
        size=size,
        cursor=cursor,
        mask=initial_mask,
        keys=tuple(sorted(backend_keys)),
        fail_xtest_at=fail_xtest_at,
        executor=str(ACTION_EXECUTOR_PATH),
    )
    completed = subprocess.run(
        [sys.executable, "-c", driver],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(REPO_ROOT),
    )
    payload = None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pass
    return GuestRun(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=payload,
        request=request,
        program=ACTION_EXECUTOR_PATH.read_text(),
        expected_mask=expected_mask,
    )
