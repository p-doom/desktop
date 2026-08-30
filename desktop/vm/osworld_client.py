"""A grammar-free client for the in-VM agent's HTTP surface.

The in-VM agent exposes a small Flask app.  Of it, exactly four endpoints matter:

    GET  /screenshot      -> raw cursor-composited JPEG bytes (1920x1080 RGB)
    POST /screen_size     -> {"width": int, "height": int}
    POST /execute         -> {"command": [...], "shell": false}; the server runs a
                             subprocess and does NOT eval strings
    GET  /accessibility   -> the platform accessibility tree (AT-SPI XML)

There is no dispatch here and no key tables: dispatch belongs to a codec and an
executor, and the key tables live in ``desktop.execute.keymap``.  This is a
transport.

Screenshots are validated with Pillow, then handed back unchanged as JPEG bytes.

HTTP is ``urllib.request``: no ``requests``, no session object, no dependency.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .observation import OBSERVATION_CONTRACT, OBSERVATION_SIZE

_LOG = logging.getLogger("desktop.vm.osworld_client")

#: Linux caps ONE argv element at ``MAX_ARG_STRLEN`` (32 pages) at ``execve``, and
#: a file pushed through ``/execute`` rides base64-encoded inside the single
#: ``python3 -c`` program.  This is therefore a property of the transport, not a
#: policy: a larger file is refused rather than being cut short by E2BIG.
GUEST_PROGRAM_MAX_BYTES = 131072

#: How long one control call of a detached command may take.  It has to exceed
#: ``_DETACHED_POLL_WINDOW_S``, because a collect call deliberately blocks in the
#: guest for that long rather than returning immediately and being re-issued.
_DETACHED_CONTROL_TIMEOUT_S = 30.0
_DETACHED_POLL_WINDOW_S = 5.0
_DETACHED_POLL_TICK_S = 0.1

#: A guest agent restart takes every open connection with it and answers again
#: within seconds; every call below is idempotent so that this is survivable.
_DETACHED_CALL_ATTEMPTS = 8
_DETACHED_RETRY_BACKOFF_S = 0.5

_DETACHED_LAUNCH_SOURCE = """\
base=$1
shift
if [ -e "$base.pid" ]; then cat "$base.pid"; exit 0; fi
( "$@" >"$base.out" 2>"$base.err" </dev/null; echo $? >"$base.rc" ) >/dev/null 2>&1 </dev/null &
echo $! >"$base.pid"
cat "$base.pid"
"""

_DETACHED_COLLECT_SOURCE = f"""\
base=$1
ticks=$2
while [ ! -f "$base.rc" ] && [ "$ticks" -gt 0 ]; do
    sleep {_DETACHED_POLL_TICK_S}
    ticks=$((ticks - 1))
done
test -f "$base.rc" || exit 1
cat "$base.rc"
base64 -w0 "$base.out"
echo
base64 -w0 "$base.err"
echo
"""

_DETACHED_DISCARD_SOURCE = """\
base=$1
if [ -n "$2" ]; then kill "$2" 2>/dev/null; fi
rm -f "$base.pid" "$base.rc" "$base.out" "$base.err"
"""


class GuestAgentError(RuntimeError):
    """The in-VM agent returned an unusable response, or could not be reached."""


@dataclass(frozen=True)
class GuestCommandResult:
    """What one argv-based guest command produced."""

    returncode: int
    stdout: str
    stderr: str


class OSWorldClient:
    """Thin synchronous client over the in-VM agent."""

    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, bytes, str]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_s if timeout_s is None else timeout_s
            ) as response:
                return (
                    int(response.status),
                    response.read(),
                    response.headers.get_content_type(),
                )
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read(), exc.headers.get_content_type()
        except (urllib.error.URLError, OSError) as exc:
            raise GuestAgentError(f"guest request {method} {path} failed: {exc}") from exc

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        status, body, _ = self._request(method, path, payload=payload, timeout_s=timeout_s)
        if status != 200:
            raise GuestAgentError(
                f"guest {method} {path} returned {status}: {body[:200]!r}"
            )
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuestAgentError(f"guest {method} {path} returned non-JSON") from exc

    def wait_ready(self, *, timeout_s: float = 180.0, poll_s: float = 2.0) -> None:
        """Poll ``/screenshot`` until it answers 200 with a body.

        This is the canonical guest-readiness check: it proves the agent process
        is up AND that an X server exists behind it, which a bare TCP connect
        does not.
        """
        started = time.time()
        last_log = 0.0
        while time.time() - started < timeout_s:
            try:
                status, body, content_type = self._request(
                    "GET", "/screenshot", timeout_s=5.0
                )
                if status == 200:
                    self._validate_screenshot(content_type, body)
                    _LOG.info("guest ready after %.1fs", time.time() - started)
                    return
            except GuestAgentError:
                pass
            elapsed = time.time() - started
            if elapsed - last_log >= 15:
                _LOG.info("waiting for guest /screenshot... %.0fs", elapsed)
                last_log = elapsed
            time.sleep(poll_s)
        raise TimeoutError(
            f"guest agent at {self.base_url} not ready after {timeout_s}s"
        )

    def screenshot(self) -> bytes:
        """Raw bytes in the one canonical desktop observation contract."""
        status, body, content_type = self._request("GET", "/screenshot")
        if status != 200 or not body:
            raise GuestAgentError(f"guest /screenshot returned {status} ({len(body)}B)")
        self._validate_screenshot(content_type, body)
        return body

    @staticmethod
    def _validate_screenshot(content_type: str, body: bytes) -> None:
        if content_type != "image/jpeg":
            raise GuestAgentError(
                f"guest /screenshot returned {content_type!r}, not image/jpeg"
            )
        if not body.startswith(b"\xff\xd8") or not body.endswith(b"\xff\xd9"):
            raise GuestAgentError("guest /screenshot returned invalid JPEG framing")
        try:
            with Image.open(io.BytesIO(body)) as image:
                image.verify()
            with Image.open(io.BytesIO(body)) as image:
                image.load()
                if image.mode != "RGB":
                    raise GuestAgentError(
                        f"{OBSERVATION_CONTRACT} requires RGB, got {image.mode!r}"
                    )
                if image.size != OBSERVATION_SIZE:
                    raise GuestAgentError(
                        f"{OBSERVATION_CONTRACT} requires {OBSERVATION_SIZE}, "
                        f"got {image.size}"
                    )
        except GuestAgentError:
            raise
        except Exception as exc:
            raise GuestAgentError("guest /screenshot returned an invalid JPEG") from exc

    def screen_size(self) -> tuple[int, int]:
        payload = self._request_json("POST", "/screen_size", payload={})
        if not isinstance(payload, dict) or "width" not in payload or "height" not in payload:
            raise GuestAgentError(f"invalid screen size: {payload!r}")
        return int(payload["width"]), int(payload["height"])

    def execute(
        self, argv: list[str], *, check: bool = True, timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Run an argv in the guest.

        ``shell: false`` always.  The agent runs a subprocess and does not eval,
        so a caller that wants Python in the guest passes
        ``["python", "-c", <program>]`` -- which is what
        ``desktop.execute.guest_program`` compiles.
        """
        result = self._request_json(
            "POST", "/execute", payload={"command": list(argv), "shell": False},
            timeout_s=timeout_s,
        )
        if not isinstance(result, dict):
            raise GuestAgentError("guest /execute returned a non-object")
        if check and (
            result.get("status") != "success" or result.get("returncode") != 0
        ):
            raise GuestAgentError(
                f"guest command failed: status={result.get('status')!r} "
                f"rc={result.get('returncode')!r} stderr={result.get('error')!r}"
            )
        return result

    def write_file(self, path: str, content: bytes) -> None:
        """Push bytes to an absolute guest path.

        Over ``/execute`` and base64, so nothing about the content is ever
        interpreted as program text on either side -- which is what callers that
        hand-embedded a file into a ``python3 -c`` program had to get right by
        hand, once per call site.

        The parent directory is NOT created: a caller that named a path whose
        directory does not exist has named the wrong path.
        """
        if not path.startswith("/"):
            raise ValueError(f"guest file path must be absolute, got {path!r}")
        program = (
            "import base64,pathlib;"
            f"pathlib.Path({path!r}).write_bytes("
            f"base64.b64decode({base64.b64encode(content).decode('ascii')!r},validate=True))"
        )
        if len(program.encode("utf-8")) > GUEST_PROGRAM_MAX_BYTES:
            raise ValueError(
                f"{len(content)} bytes do not fit in one guest argv element "
                f"(limit {GUEST_PROGRAM_MAX_BYTES} bytes of base64 program text)"
            )
        self.execute(["python3", "-c", program])

    def execute_detached(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
    ) -> GuestCommandResult:
        """Run an argv in the guest with its stdio redirected to files.

        ``execute`` is the right call for anything short.  This is for a command
        that may spawn a survivor: the agent runs it as a child and holds its
        pipes, so a daemon inheriting them keeps the agent's ``/execute`` handler
        blocked long after the command itself exited.  Here the guest shell
        redirects to files, backgrounds the command, and this polls for the
        ``.rc`` file, so nothing the command leaves behind holds a pipe open.
        """
        argv = list(argv)
        if not argv or not all(isinstance(value, str) and value for value in argv):
            raise ValueError("argv must contain at least one non-empty string")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        command_env = dict(env or {})
        for key, value in command_env.items():
            # ``env`` splits on the FIRST ``=``, so a key containing one silently
            # sets a different variable to a different value.
            if not key or "=" in key or "\x00" in key + value:
                raise ValueError(f"invalid guest command environment name {key!r}")
        command = [
            "env",
            *(f"{key}={value}" for key, value in sorted(command_env.items())),
            *argv,
        ]

        base = f"/tmp/desktop-guest-command-{uuid.uuid4().hex}"
        launch = self._retry_execute(
            ["sh", "-c", _DETACHED_LAUNCH_SOURCE, "sh", base, *command]
        )
        if launch.get("returncode") != 0:
            raise GuestAgentError(f"could not start detached guest command: {launch!r}")
        pid = str(launch.get("output") or "").strip()

        deadline = time.monotonic() + timeout_s
        while True:
            remaining = max(deadline - time.monotonic(), 0.0)
            ticks = int(min(remaining, _DETACHED_POLL_WINDOW_S) / _DETACHED_POLL_TICK_S)
            collected = self._retry_execute(
                ["sh", "-c", _DETACHED_COLLECT_SOURCE, "sh", base, str(ticks)]
            )
            if collected.get("returncode") == 0:
                result = _decode_detached_result(collected)
                self._discard_detached(base, pid="")
                return result
            if time.monotonic() >= deadline:
                self._discard_detached(base, pid=pid)
                raise TimeoutError(
                    f"guest command did not finish within {timeout_s:.1f}s: {argv!r}"
                )

    def _retry_execute(self, argv: list[str]) -> dict[str, Any]:
        last: GuestAgentError | None = None
        for attempt in range(_DETACHED_CALL_ATTEMPTS):
            try:
                return self.execute(
                    argv, check=False, timeout_s=_DETACHED_CONTROL_TIMEOUT_S
                )
            except GuestAgentError as exc:
                last = exc
                if attempt + 1 < _DETACHED_CALL_ATTEMPTS:
                    time.sleep(_DETACHED_RETRY_BACKOFF_S * (attempt + 1))
        raise GuestAgentError(
            f"guest command channel failed {_DETACHED_CALL_ATTEMPTS} times: {last!r}"
        )

    def _discard_detached(self, base: str, *, pid: str) -> None:
        # Never allowed to raise: it runs on both the success and the timeout
        # path, where it would replace the outcome the caller is waiting for.
        with contextlib.suppress(GuestAgentError):
            self.execute(
                ["sh", "-c", _DETACHED_DISCARD_SOURCE, "sh", base, pid],
                check=False,
                timeout_s=_DETACHED_CONTROL_TIMEOUT_S,
            )

    def accessibility(self) -> str:
        """The platform accessibility tree as returned by the guest.

        Handed back as text rather than parsed: its schema is the guest
        platform's, not ours, and every consumer so far wanted a different slice
        of it.

        Exactly one accepted shape -- the AT-SPI text the pinned guest sends.  This
        used to try JSON first and then unwrap whichever of four key spellings
        (``AT`` / ``at`` / ``accessibility_tree`` / ``tree``) happened to be present,
        falling back to ``errors="replace"`` twice.  A tree is XML that a consumer
        parses, so a silently replaced byte is a silently malformed tree, and four
        accepted spellings means a guest that renames the field keeps "working"
        while returning the wrong slice.
        """
        status, body, _ = self._request("GET", "/accessibility")
        if status != 200:
            raise GuestAgentError(f"guest /accessibility returned {status}")
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GuestAgentError(
                f"guest /accessibility returned {len(body)}B that are not UTF-8: {exc}"
            ) from exc

    def cursor_position(self) -> tuple[int, int]:
        """Where the guest thinks the pointer is, as a two-element JSON array.

        A fifth endpoint beyond the four above, kept deliberately: the codec
        contract is ``compile(text, geometry, cursor)``, so something has to
        supply the cursor as data, and the alternative -- round-tripping
        ``pyautogui.position()`` through ``/execute`` -- costs an interpreter
        start per read.

        Exactly one accepted shape, the one the guest sends, and the same shape
        ``HttpGuiTransport.cursor_position`` accepts.  Two readers of one endpoint
        disagreeing about its wire format is how a guest change gets diagnosed as
        an intermittent executor bug.
        """
        payload = self._request_json("GET", "/cursor_position")
        if not isinstance(payload, list) or len(payload) != 2:
            raise GuestAgentError(f"invalid cursor position: {payload!r}")
        return int(payload[0]), int(payload[1])

    def screenshot_settled(
        self,
        *,
        min_delay_s: float = 0.0,
        stability_timeout_s: float = 0.0,
        poll_s: float = 0.1,
    ) -> bytes:
        """A screenshot taken after the desktop has stopped repainting.

        An action returns as soon as its input events are *sent*; the application
        may not have handled them and repainted yet.  A screenshot grabbed
        immediately can therefore miss the action's visible effect, and the model
        then sees an unchanged frame and re-emits the same action -- the failure
        mode that looks like a stuck policy but is a race.

        With both delays zero this is exactly ``screenshot()``.  A permanently
        animating element (a blinking caret, a clock) prevents stability, in which
        case this waits the full ``stability_timeout_s`` and returns the last
        frame.

        Comparison is on the canonical JPEG bytes. Its real-guest encoder
        determinism must be established before using a nonzero stability wait.
        """
        if min_delay_s > 0:
            time.sleep(min_delay_s)
        if stability_timeout_s <= 0:
            return self.screenshot()
        previous = self.screenshot()
        deadline = time.time() + stability_timeout_s
        while time.time() < deadline:
            time.sleep(poll_s)
            current = self.screenshot()
            if current == previous:
                return current
            previous = current
        return previous


def _decode_detached_result(collected: Mapping[str, Any]) -> GuestCommandResult:
    lines = str(collected.get("output") or "").splitlines()
    if len(lines) != 3 or not lines[0].strip().lstrip("-").isdigit():
        raise GuestAgentError(f"invalid detached guest command result: {collected!r}")
    returncode, encoded_stdout, encoded_stderr = lines
    return GuestCommandResult(
        returncode=int(returncode),
        stdout=base64.b64decode(encoded_stdout).decode("utf-8", "replace"),
        stderr=base64.b64decode(encoded_stderr).decode("utf-8", "replace"),
    )
