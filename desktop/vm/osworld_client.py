"""A grammar-free client for the in-VM agent's HTTP surface.

The in-VM agent exposes a small Flask app.  Of it, exactly four endpoints matter:

    GET  /screenshot      -> raw PNG bytes (full desktop framebuffer)
    POST /screen_size     -> {"width": int, "height": int}
    POST /execute         -> {"command": [...], "shell": false}; the server runs a
                             subprocess and does NOT eval strings
    GET  /accessibility   -> the platform accessibility tree (AT-SPI XML)

There is no dispatch here and no key tables: dispatch belongs to a codec and an
executor, and the key tables live in ``desktop.execute.keymap``.  This is a
transport.

Screenshots are handed back as PNG bytes, never as decoded images -- only the
caller knows whether it wants a tensor, a thumbnail, or a byte-for-byte hash.

HTTP is ``urllib.request``: no ``requests``, no session object, no dependency.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

_LOG = logging.getLogger("desktop.vm.osworld_client")


class GuestAgentError(RuntimeError):
    """The in-VM agent returned an unusable response, or could not be reached."""


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
    ) -> tuple[int, bytes]:
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
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()
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
        status, body = self._request(method, path, payload=payload, timeout_s=timeout_s)
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
                status, body = self._request("GET", "/screenshot", timeout_s=5.0)
                if status == 200 and body:
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
        """Raw PNG bytes of the whole framebuffer.  Undecoded, on purpose."""
        status, body = self._request("GET", "/screenshot")
        if status != 200 or not body:
            raise GuestAgentError(f"guest /screenshot returned {status} ({len(body)}B)")
        return body

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

    def accessibility(self) -> str:
        """The platform accessibility tree as returned by the guest.

        Handed back as text rather than parsed: its schema is the guest
        platform's, not ours, and every consumer so far wanted a different slice
        of it.
        """
        status, body = self._request("GET", "/accessibility")
        if status != 200:
            raise GuestAgentError(f"guest /accessibility returned {status}")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body.decode("utf-8", errors="replace")
        if isinstance(payload, dict):
            for key in ("AT", "at", "accessibility_tree", "tree"):
                if key in payload:
                    return str(payload[key])
        return body.decode("utf-8", errors="replace")

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

        Note the comparison is on PNG *bytes*, so it is sensitive to encoder
        nondeterminism in a way a pixel comparison would not be.  On the pinned
        guest the encoder is deterministic; a caller on another guest that sees
        spurious instability should decode and compare pixels itself.
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
