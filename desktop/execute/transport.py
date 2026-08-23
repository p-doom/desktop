"""The transport seam: how operations reach a guest, and how to test without one.

``GuiTransport`` mentions no coordinate convention anywhere -- every member takes
pixels or names -- so the same protocol serves an absolute-coordinate model and a
relative one with no changes.

``RecordingTransport`` implements the same held-state machine and the same
lowering, in-process, so a test can assert on the exact operation sequence an
action produces without a VM in the loop.

HTTP is ``urllib.request``, not ``requests``, so the package has no runtime
dependency for it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from ..ir import Operation, scroll_deltas
from .guest_program import (
    ATOMIC_RESULT_PREFIX,
    ATOMIC_SCHEMA_VERSION,
    BUTTON_MASKS,
    PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    AtomicExecutionResult,
    ExecutionError,
    InputAudit,
    compile_atomic_guest_program,
    compile_unicode_coalesced_type,
    HeldStateError,
    expected_atomic_input_state,
    lower_guest_operations,
    pointer_mask_for_buttons,
)
from .keymap import guest_button, guest_key


class GuiTransport(Protocol):
    """Everything the executor needs from a guest.  No coordinate convention.

    ``glide_to`` and ``hscroll`` are required members, not optional capabilities:
    nothing here probes for them, and ``compile_atomic_guest_program`` emits
    ``pyautogui.moveTo(duration=)`` and ``pyautogui.hscroll`` unconditionally, so
    a transport without them raises inside the guest program rather than
    half-working.
    """

    audit: InputAudit

    def cursor_position(self) -> tuple[int, int]: ...
    def screen_size(self) -> tuple[int, int]: ...
    def move_to(self, x: int, y: int) -> None: ...
    def glide_to(self, x: int, y: int, seconds: float) -> None: ...
    def mouse_down(self, button: str = "left") -> None: ...
    def mouse_up(self, button: str = "left") -> None: ...
    def scroll(self, clicks: int) -> None: ...
    def hscroll(self, dx: int) -> None: ...
    def key_chord(self, keys: list[str]) -> None: ...
    def coalesced_type(self, text: str) -> None: ...
    def wait(self, seconds: float) -> None: ...
    def execute_atomic(
        self,
        operations: tuple[Operation, ...],
        *,
        click_backend: str = PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    ) -> AtomicExecutionResult: ...


class HttpGuiTransport:
    """Drives the guest's in-VM HTTP agent over ``POST /execute``.

    The guest agent runs a subprocess; it does not ``eval``.  Every input goes
    out as ``python -c <program>``, which is why the whole action has to be
    compiled into one program by ``guest_program``.
    """

    _PREFIX = "import pyautogui; pyautogui.FAILSAFE=False; pyautogui.PAUSE=0; "

    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.audit = InputAudit()

    def _request_json(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.load(response)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ExecutionError(f"guest request {method} {path} failed: {exc}") from exc

    def execute_argv(self, argv: list[str], *, check: bool = True) -> dict[str, Any]:
        result = self._request_json("POST", "/execute", {"command": argv, "shell": False})
        if not isinstance(result, dict):
            raise ExecutionError("guest /execute returned a non-object")
        if check and (
            result.get("status") != "success" or result.get("returncode") != 0
        ):
            raise ExecutionError(
                f"guest command failed: status={result.get('status')!r} "
                f"rc={result.get('returncode')!r} stderr={result.get('error')!r}"
            )
        return result

    def execute_pyautogui(self, code: str) -> None:
        self.execute_argv(["python", "-c", self._PREFIX + code])

    def execute_atomic(
        self,
        operations: tuple[Operation, ...],
        *,
        click_backend: str = PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    ) -> AtomicExecutionResult:
        """Compile, send, and validate exactly one guest process."""
        program, expected_mask = compile_atomic_guest_program(
            operations,
            initial_buttons=set(self.audit.held_buttons),
            initial_keys=set(self.audit.held_keys),
            click_backend=click_backend,
        )
        _, expected_keys = expected_atomic_input_state(
            operations,
            initial_buttons=set(self.audit.held_buttons),
            initial_keys=set(self.audit.held_keys),
        )
        result = self.execute_argv(["python", "-c", program], check=False)
        atomic_result = self._parse_atomic_payload(
            result, operations=operations, click_backend=click_backend
        )
        self._absorb(atomic_result, expected_keys=expected_keys)
        return atomic_result

    def _parse_atomic_payload(
        self,
        result: dict[str, Any],
        *,
        operations: tuple[Operation, ...],
        click_backend: str,
    ) -> AtomicExecutionResult:
        """Validate the marker, the schema, and the payload's self-consistency.

        Deliberately bounded: this layer refuses only payloads that are
        structurally unusable or self-contradictory, and does not check the
        guest's X-event ordering.  The guest still *reports* that evidence (see
        ``x_injection_evidence``) for a caller who wants to.
        """
        output = result.get("output")

        def fail(message: str, **extra: Any) -> None:
            raise ExecutionError(
                message,
                evidence={
                    "schema_version": "desktop_env_atomic_output_failure_v1",
                    "click_backend_expected": click_backend,
                    "execute_result": {
                        "status": result.get("status"),
                        "returncode": result.get("returncode"),
                        "error": result.get("error"),
                    },
                    "raw_stdout": output,
                    **extra,
                },
            )

        if not isinstance(output, str):
            fail("atomic guest action returned no stdout")
        markers = [
            line for line in str(output).splitlines() if line.startswith(ATOMIC_RESULT_PREFIX)
        ]
        if len(markers) != 1:
            fail(f"atomic guest result marker count was {len(markers)}", markers=markers)
        try:
            payload = json.loads(markers[0][len(ATOMIC_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise ExecutionError(
                f"atomic guest action returned invalid JSON: {exc}",
                evidence={"raw_marker": markers[0]},
            ) from exc
        if not isinstance(payload, dict) or payload.get("_de_schema") != ATOMIC_SCHEMA_VERSION:
            fail("atomic guest action returned an unexpected schema", raw_payload=payload)

        def pair(name: str) -> tuple[int, int]:
            value = payload.get(name)
            if not isinstance(value, list) or len(value) != 2:
                fail(f"atomic guest action returned an invalid {name}", raw_payload=payload)
            return int(value[0]), int(value[1])

        cursor = pair("cursor")
        cursor_before = pair("cursor_before")
        cursor_after = pair("cursor_after")
        if cursor != cursor_after:
            fail("atomic guest cursor alias/readback mismatch", raw_payload=payload)
        error = payload.get("error")
        failure_kind = payload.get("failure_kind")
        ok = bool(payload.get("ok"))
        if failure_kind not in {None, "verification", "infrastructure", "injected"}:
            fail("atomic guest action returned an invalid failure kind", observed=failure_kind)
        if ok != (failure_kind is None) or ok != (error is None):
            fail(
                "atomic guest action failure classification is self-contradictory",
                raw_payload=payload,
            )
        # The -1 default is load-bearing the other way round from the masks
        # below: here an ABSENT count must fail this check, so the default is the
        # failure rather than a substituted reading.  `True` is excluded because
        # `True == 1`, which would read a JSON `true` as exactly one process.
        count = payload.get("guest_process_count", -1)
        if not isinstance(count, int) or isinstance(count, bool) or count != 1:
            fail("atomic action did not use exactly one guest process", raw_payload=payload)
        if payload.get("click_backend") != click_backend:
            fail(
                "atomic guest action click backend drifted",
                observed=payload.get("click_backend"),
            )
        restore_errors = payload.get("attempt_hook_restore_errors") or []
        if restore_errors:
            fail("atomic guest action X attempt hooks were not restored", observed=restore_errors)

        def records(name: str) -> tuple[dict[str, Any], ...]:
            value = payload.get(name, [])
            if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
                fail(f"atomic guest action returned an invalid {name}", raw_payload=payload)
            return tuple(dict(row) for row in value)

        def mask(name: str) -> int:
            # -1 is the guest's own "this mask was never read" sentinel and is a
            # legitimate report -- `observed_pointer_button_mask` is -1 whenever
            # the action dies before its verification readback.  So the VALUE
            # stays legal and the ABSENCE must not quietly become it: a defaulted
            # -1 is a sentinel nobody reported, on a payload free to claim ok.
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                fail(f"atomic guest action returned an invalid {name}", raw_payload=payload)
            return int(value)

        traced = tuple(
            Operation(str(row["kind"]), tuple(row.get("args", ())))
            for row in records("operations")
        )
        return AtomicExecutionResult(
            ok=ok,
            cursor=cursor,
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            pointer_button_mask=mask("pointer_button_mask"),
            observed_pointer_button_mask=mask("observed_pointer_button_mask"),
            expected_pointer_button_mask=mask("expected_pointer_button_mask"),
            guest_process_count=1,
            guest_returncode=int(result.get("returncode") or 0),
            raw_result_marker=markers[0],
            cleanup_attempted=bool(payload.get("cleanup_attempted")),
            error=None if error is None else str(error),
            failure_kind=None if failure_kind is None else str(failure_kind),
            operations=traced,
            semantic_operations=operations,
            lowered_operations=lower_guest_operations(operations),
            backend_primitives=records("backend_primitives"),
            x_event_sync_evidence=records("x_event_sync_evidence"),
            x_sync_attempt_evidence=records("x_sync_attempt_evidence"),
            click_backend=click_backend,
            x_injection_evidence=records("x_injection_evidence"),
            x_injection_timestamps=records("x_injection_timestamps"),
            final_pointer_readback=dict(payload.get("final_pointer_readback") or {}),
            passive_x_observer=dict(payload.get("passive_x_observer") or {}),
        )

    def _absorb(
        self, result: AtomicExecutionResult, *, expected_keys: set[str]
    ) -> None:
        """Update the host-side audit from what the guest actually reported."""
        self.audit.operations.extend(result.operations)
        # -1 is the guest's "final readback never ran" sentinel, not a bitmask:
        # `-1 & mask` is truthy for every button, so deriving the held set from
        # it hands the next program an all-held initial mask it then fails
        # verification against forever.  Empty is not unchecked -- that next
        # program's initial readback re-verifies the mask against the X server.
        self.audit.held_buttons = (
            {
                button
                for button, mask in BUTTON_MASKS.items()
                if result.pointer_button_mask & mask
            }
            if result.pointer_button_mask >= 0
            else set()
        )
        self.audit.held_keys = expected_keys if result.ok else set()
        for operation in result.operations:
            if operation.kind == "scroll":
                self.audit.scroll_total += scroll_deltas(operation.args)[1]
            elif operation.kind in {"coalesced_type", "ascii_type"}:
                self.audit.typed_texts.append(str(operation.args[0]))

    def cursor_position(self) -> tuple[int, int]:
        value = self._request_json("GET", "/cursor_position")
        if not isinstance(value, list) or len(value) != 2:
            raise ExecutionError(f"invalid cursor position: {value!r}")
        return int(value[0]), int(value[1])

    def screen_size(self) -> tuple[int, int]:
        value = self._request_json("POST", "/screen_size", {})
        if not isinstance(value, dict):
            raise ExecutionError(f"invalid screen size: {value!r}")
        return int(value["width"]), int(value["height"])

    def move_to(self, x: int, y: int) -> None:
        width, height = self.screen_size()
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        self.execute_pyautogui(f"pyautogui.moveTo({x}, {y})")
        self.audit.operations.append(Operation("move_to", (x, y)))

    def mouse_down(self, button: str = "left") -> None:
        button = guest_button(button)
        if button in self.audit.held_buttons:
            raise HeldStateError(f"button already held: {button}")
        self.execute_pyautogui(f"pyautogui.mouseDown(button={button!r})")
        self.audit.held_buttons.add(button)
        self.audit.operations.append(Operation("mouse_down", (button,)))

    def mouse_up(self, button: str = "left") -> None:
        button = guest_button(button)
        if button not in self.audit.held_buttons:
            raise HeldStateError(f"button not held: {button}")
        self.execute_pyautogui(f"pyautogui.mouseUp(button={button!r})")
        self.audit.held_buttons.remove(button)
        self.audit.operations.append(Operation("mouse_up", (button,)))

    def scroll(self, clicks: int) -> None:
        """Vertical wheel ticks.  The audit records the two-axis form."""
        self.execute_pyautogui(f"pyautogui.scroll({int(clicks)})")
        self.audit.scroll_total += int(clicks)
        self.audit.operations.append(Operation("scroll", (0, int(clicks))))

    def hscroll(self, dx: int) -> None:
        """Horizontal wheel ticks; the audit records the two-axis form."""
        self.execute_pyautogui(f"pyautogui.hscroll({int(dx)})")
        self.audit.operations.append(Operation("scroll", (int(dx), 0)))

    def glide_to(self, x: int, y: int, seconds: float) -> None:
        """A timed absolute move: the stroke of a drag, not a teleport."""
        width, height = self.screen_size()
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        seconds = max(0.0, min(10.0, float(seconds)))
        self.execute_pyautogui(f"pyautogui.moveTo({x}, {y}, duration={seconds!r})")
        self.audit.operations.append(Operation("glide_to", (x, y, seconds)))

    def key_chord(self, keys: list[str]) -> None:
        if not keys:
            raise ExecutionError("empty key chord")
        mapped = [guest_key(key) for key in keys]
        presses = "; ".join(f"pyautogui.keyDown({key!r})" for key in mapped)
        releases = "; ".join(f"pyautogui.keyUp({key!r})" for key in reversed(mapped))
        self.execute_pyautogui(presses + "; " + releases)
        self.audit.operations.append(Operation("key_chord", tuple(keys)))

    def coalesced_type(self, text: str) -> None:
        self.execute_pyautogui(compile_unicode_coalesced_type(text))
        self.audit.typed_texts.append(text)
        self.audit.operations.append(Operation("coalesced_type", (text,)))

    def wait(self, seconds: float) -> None:
        seconds = max(0.0, min(10.0, float(seconds)))
        time.sleep(seconds)
        self.audit.operations.append(Operation("wait", (seconds,)))


class RecordingTransport:
    """Deterministic in-process transport: the executor's test double.

    Implements the same held-state machine, the same lowering, and the same
    cleanup-on-failure contract as the guest program, so a test can assert on the
    exact operation sequence an action produces without a VM in the loop.
    """

    def __init__(
        self,
        *,
        cursor: tuple[int, int] = (50, 50),
        screen: tuple[int, int] = (1920, 1080),
    ) -> None:
        self._cursor = cursor
        self._screen = screen
        self.audit = InputAudit()
        self.atomic_invocations = 0
        self.atomic_inputs: list[tuple[Operation, ...]] = []

    def cursor_position(self) -> tuple[int, int]:
        return self._cursor

    def screen_size(self) -> tuple[int, int]:
        return self._screen

    def move_to(self, x: int, y: int) -> None:
        x = max(0, min(self._screen[0] - 1, int(x)))
        y = max(0, min(self._screen[1] - 1, int(y)))
        self._cursor = (x, y)
        self.audit.operations.append(Operation("move_to", (x, y)))

    def mouse_down(self, button: str = "left") -> None:
        button = guest_button(button)
        if button in self.audit.held_buttons:
            raise HeldStateError(f"button already held: {button}")
        self.audit.held_buttons.add(button)
        self.audit.operations.append(Operation("mouse_down", (button,)))

    def mouse_up(self, button: str = "left") -> None:
        button = guest_button(button)
        if button not in self.audit.held_buttons:
            raise HeldStateError(f"button not held: {button}")
        self.audit.held_buttons.remove(button)
        self.audit.operations.append(Operation("mouse_up", (button,)))

    def scroll(self, clicks: int) -> None:
        self.audit.scroll_total += int(clicks)
        self.audit.operations.append(Operation("scroll", (0, int(clicks))))

    def hscroll(self, dx: int) -> None:
        """Horizontal wheel ticks; the audit records the two-axis form."""
        self.audit.operations.append(Operation("scroll", (int(dx), 0)))

    def glide_to(self, x: int, y: int, seconds: float) -> None:
        x = max(0, min(self._screen[0] - 1, int(x)))
        y = max(0, min(self._screen[1] - 1, int(y)))
        self._cursor = (x, y)
        self.audit.operations.append(
            Operation("glide_to", (x, y, max(0.0, min(10.0, float(seconds)))))
        )

    def key_chord(self, keys: list[str]) -> None:
        if not keys:
            raise ExecutionError("empty key chord")
        self.audit.operations.append(Operation("key_chord", tuple(keys)))

    def coalesced_type(self, text: str) -> None:
        self.audit.typed_texts.append(text)
        self.audit.operations.append(Operation("coalesced_type", (text,)))

    def wait(self, seconds: float) -> None:
        self.audit.operations.append(Operation("wait", (float(seconds),)))

    def execute_atomic(
        self,
        operations: tuple[Operation, ...],
        *,
        click_backend: str = PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    ) -> AtomicExecutionResult:
        self.atomic_invocations += 1
        self.atomic_inputs.append(operations)
        before = len(self.audit.operations)
        cursor_before = self._cursor
        initial_buttons = set(self.audit.held_buttons)
        initial_keys = set(self.audit.held_keys)
        final_buttons, final_keys = expected_atomic_input_state(
            operations, initial_buttons=initial_buttons, initial_keys=initial_keys
        )
        expected_mask = pointer_mask_for_buttons(final_buttons)
        observed_mask = -1
        cleanup_attempted = False
        error: str | None = None
        failure_kind: str | None = None
        lowered = lower_guest_operations(operations)
        primitives: list[dict[str, Any]] = []
        sync_evidence: list[dict[str, Any]] = []

        def synced(event: str) -> None:
            sync_evidence.append(
                {"event": event, "backend": "recording_x11", "flush": True, "sync": True}
            )

        try:
            for operation in lowered:
                kind, args = operation.kind, operation.args
                if kind == "move_to":
                    requested = (int(args[0]), int(args[1]))
                    move_before = self._cursor
                    self.move_to(*requested)
                    primitives.append(
                        {
                            "kind": "move_to",
                            "call": "recording.move_to",
                            "requested_position": list(requested),
                            "cursor_before": list(move_before),
                            "cursor_after": list(self._cursor),
                            "clamped": self._cursor != requested,
                        }
                    )
                elif kind == "drag":
                    start = (int(args[0]), int(args[1]))
                    end = (int(args[2]), int(args[3]))
                    self.move_to(*start)
                    self.mouse_down("left")
                    synced("mouse_down")
                    self.move_to(*end)
                    self.mouse_up("left")
                    synced("mouse_up")
                    primitives.append(
                        {
                            "kind": "drag",
                            "call": "recording.drag",
                            "start": list(start),
                            "end": list(end),
                            "zero_extent": start == end,
                        }
                    )
                elif kind == "glide_to":
                    requested = (int(args[0]), int(args[1]))
                    seconds = float(args[2]) if len(args) > 2 else 0.0
                    move_before = self._cursor
                    self.glide_to(requested[0], requested[1], seconds)
                    primitives.append(
                        {
                            "kind": "glide_to",
                            "call": "recording.glide_to",
                            "requested_position": list(requested),
                            "seconds": seconds,
                            "cursor_before": list(move_before),
                            "cursor_after": list(self._cursor),
                            "clamped": self._cursor != requested,
                        }
                    )
                elif kind == "scroll":
                    dx, dy = scroll_deltas(args)
                    # ONE trace operation carrying BOTH axes, exactly as the guest
                    # program records it.  Routing through self.scroll/self.hscroll
                    # would append two operations for a diagonal scroll and none at
                    # all for ``scroll(0, 0)``, diverging from the guest.
                    self.audit.scroll_total += dy
                    self.audit.operations.append(Operation("scroll", (dx, dy)))
                    primitives.append(
                        {
                            "kind": "scroll",
                            "call": "recording.scroll",
                            "dx": dx,
                            "dy": dy,
                        }
                    )
                elif kind == "mouse_down":
                    self.mouse_down(str(args[0]))
                    primitives.append(
                        {"kind": kind, "button": str(args[0]), "call": "recording.mouse_down"}
                    )
                    synced(kind)
                elif kind == "mouse_up":
                    self.mouse_up(str(args[0]))
                    primitives.append(
                        {"kind": kind, "button": str(args[0]), "call": "recording.mouse_up"}
                    )
                    synced(kind)
                elif kind == "click":
                    self.mouse_down(str(args[0]))
                    synced("mouse_down")
                    self.mouse_up(str(args[0]))
                    synced("mouse_up")
                    primitives.append(
                        {
                            "kind": "click",
                            "button": str(args[0]),
                            "call": "pyautogui.click(clicks=1, interval=0.05)",
                            "x11_per_event_sync_hooked": True,
                            "dwell_ms": 50,
                            "click_backend": click_backend,
                            "ordering": [
                                "mouse_down",
                                "flush",
                                "sync",
                                "dwell",
                                "mouse_up",
                                "flush",
                                "sync",
                            ],
                        }
                    )
                elif kind == "key_down":
                    # Held state is keyed on the MAPPED name, exactly as
                    # `expected_atomic_input_state` and the guest's own
                    # `_de_touched_keys` are, so two spellings of one key cannot
                    # look like two keys.  The recorded operation keeps the RAW
                    # name, because that is what the guest puts in its trace.
                    key = str(args[0])
                    held = guest_key(key)
                    if held in self.audit.held_keys:
                        raise HeldStateError(f"key already held: {held}")
                    self.audit.held_keys.add(held)
                    self.audit.operations.append(Operation("key_down", (key,)))
                    primitives.append(
                        {
                            "kind": kind,
                            "key": key,
                            "mapped_key": held,
                            "call": "recording.key_down",
                        }
                    )
                elif kind == "key_up":
                    key = str(args[0])
                    held = guest_key(key)
                    if held not in self.audit.held_keys:
                        raise HeldStateError(f"key not held: {held}")
                    self.audit.held_keys.remove(held)
                    self.audit.operations.append(Operation("key_up", (key,)))
                    primitives.append(
                        {
                            "kind": kind,
                            "key": key,
                            "mapped_key": held,
                            "call": "recording.key_up",
                        }
                    )
                elif kind == "coalesced_type":
                    self.coalesced_type(str(args[0]))
                    primitives.append({"kind": kind, "call": "recording.coalesced_type"})
                elif kind == "ascii_type":
                    text = str(args[0])
                    try:
                        text.encode("ascii")
                    except UnicodeEncodeError as exc:
                        raise ExecutionError("ascii_type received non-ASCII text") from exc
                    if "\n" in text or "\r" in text:
                        raise ExecutionError("ascii_type cannot embed Enter; emit a key event")
                    self.audit.typed_texts.append(text)
                    self.audit.operations.append(Operation("ascii_type", (text,)))
                    primitives.append({"kind": kind, "call": "recording.pyautogui.write"})
                elif kind == "wait":
                    self.wait(float(args[0]))
                    primitives.append(
                        {"kind": kind, "call": "recording.wait", "seconds": float(args[0])}
                    )
                elif kind == "raise_for_test":
                    failure_kind = "injected"
                    raise RuntimeError(str(args[0]))
                else:
                    raise ExecutionError(f"unsupported atomic operation: {kind}")
            observed_mask = pointer_mask_for_buttons(self.audit.held_buttons)
            if observed_mask != expected_mask:
                failure_kind = "verification"
                raise ExecutionError(
                    f"pointer button mask {observed_mask} != expected {expected_mask}"
                )
            self.audit.held_keys = final_keys
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if failure_kind is None:
                failure_kind = "infrastructure"
            cleanup_attempted = True
            # The guest's cleanup calls ``keyUp``/``mouseUp`` for everything it
            # touched WITHOUT tracing the releases, so neither does this.
            # Recording them here would put operations in the trace that no guest
            # ever reports, on the failure path a test needs to compare.
            self.audit.held_keys.clear()
            self.audit.held_buttons.clear()
        final_mask = pointer_mask_for_buttons(self.audit.held_buttons)
        return AtomicExecutionResult(
            ok=error is None,
            cursor=self._cursor,
            cursor_before=cursor_before,
            cursor_after=self._cursor,
            pointer_button_mask=final_mask,
            observed_pointer_button_mask=observed_mask,
            expected_pointer_button_mask=expected_mask,
            guest_process_count=1,
            guest_returncode=0 if error is None else 1,
            raw_result_marker=(
                f"{ATOMIC_RESULT_PREFIX}<recording:{'ok' if error is None else 'failed'}>"
            ),
            cleanup_attempted=cleanup_attempted,
            error=error,
            failure_kind=failure_kind,
            operations=tuple(self.audit.operations[before:]),
            semantic_operations=operations,
            lowered_operations=lowered,
            backend_primitives=tuple(primitives),
            x_event_sync_evidence=tuple(sync_evidence),
            click_backend=click_backend,
        )


__all__ = [
    "GuiTransport",
    "HttpGuiTransport",
    "RecordingTransport",
]
