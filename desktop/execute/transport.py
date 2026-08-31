"""Transport resolved operations to one guest process."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from ..ir import Operation, glide_seconds, scroll_deltas
from .guest_program import (
    ATOMIC_RESULT_PREFIX,
    ATOMIC_SCHEMA_VERSION,
    BUTTON_MASKS,
    AtomicExecutionResult,
    ExecutionError,
    HeldStateError,
    InputAudit,
    compile_atomic_guest_program,
    expected_atomic_input_state,
    lower_guest_operations,
    pointer_mask_for_buttons,
)
from .keymap import guest_button, guest_key, key_chord


class GuiTransport(Protocol):
    audit: InputAudit

    def cursor_position(self) -> tuple[int, int]: ...
    def screen_size(self) -> tuple[int, int]: ...
    def execute_atomic(self, operations: tuple[Operation, ...]) -> AtomicExecutionResult: ...


class HttpGuiTransport:
    """Drive the guest agent's ``POST /execute`` endpoint."""

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
        if check and (result.get("status") != "success" or result.get("returncode") != 0):
            raise ExecutionError(
                f"guest command failed: status={result.get('status')!r} "
                f"rc={result.get('returncode')!r} stderr={result.get('error')!r}"
            )
        return result

    def execute_atomic(self, operations: tuple[Operation, ...]) -> AtomicExecutionResult:
        """Compile, send, and validate exactly one guest process."""
        program, expected_mask = compile_atomic_guest_program(
            operations,
            initial_buttons=set(self.audit.held_buttons),
            initial_keys=set(self.audit.held_keys),
        )
        result = self.execute_argv(["python", "-c", program], check=False)
        atomic_result = self._parse_atomic_payload(result, operations=operations)
        if atomic_result.expected_pointer_button_mask != expected_mask:
            raise ExecutionError("atomic guest expected pointer mask drifted")
        _, expected_keys = expected_atomic_input_state(
            operations,
            initial_buttons=set(self.audit.held_buttons),
            initial_keys=set(self.audit.held_keys),
        )
        if atomic_result.ok and atomic_result.held_keys != tuple(sorted(expected_keys)):
            raise ExecutionError("atomic guest held key readback drifted")
        self._absorb(atomic_result)
        return atomic_result

    def _parse_atomic_payload(
        self, result: dict[str, Any], *, operations: tuple[Operation, ...]
    ) -> AtomicExecutionResult:
        output = result.get("output")

        def fail(message: str, **extra: Any) -> None:
            raise ExecutionError(
                message,
                evidence={
                    "schema_version": "desktop_atomic_output_failure_v2",
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

        def records(name: str) -> tuple[dict[str, Any], ...]:
            value = payload.get(name, [])
            if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
                fail(f"atomic guest action returned an invalid {name}", raw_payload=payload)
            return tuple(dict(row) for row in value)

        def mask(name: str) -> int:
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                fail(f"atomic guest action returned an invalid {name}", raw_payload=payload)
            return int(value)

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
        count = payload.get("guest_process_count")
        if not isinstance(count, int) or isinstance(count, bool) or count != 1:
            fail("atomic action did not use exactly one guest process", raw_payload=payload)
        held_keys = payload.get("held_keys")
        if not isinstance(held_keys, list) or not all(
            isinstance(key, str) for key in held_keys
        ):
            fail("atomic guest action returned invalid held_keys", raw_payload=payload)
        traced = tuple(
            Operation(str(row["kind"]), tuple(row.get("args", ())))
            for row in records("operations")
        )
        keymap_restorations = records("keymap_restorations")
        if ok and any(row.get("exact") is not True for row in keymap_restorations):
            fail(
                "atomic guest action claimed success with a drifted keymap restoration",
                raw_payload=payload,
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
            held_keys=tuple(held_keys),
            backend_primitives=records("backend_primitives"),
            x_injection_evidence=records("x_injection_evidence"),
            keymap_restorations=keymap_restorations,
            final_pointer_readback=dict(payload.get("final_pointer_readback") or {}),
        )

    def _absorb(self, result: AtomicExecutionResult) -> None:
        self.audit.operations.extend(result.operations)
        self.audit.held_buttons = (
            {
                button
                for button, button_mask in BUTTON_MASKS.items()
                if result.pointer_button_mask & button_mask
            }
            if result.pointer_button_mask >= 0
            else set()
        )
        self.audit.held_keys = set(result.held_keys)
        for operation in result.operations:
            if operation.kind == "scroll":
                self.audit.scroll_total += scroll_deltas(operation.args)[1]
            elif operation.kind in {"coalesced_type", "ascii_type"}:
                self.audit.typed_texts.append(str(operation.args[0]))

    def _execute_or_raise(self, operations: tuple[Operation, ...]) -> None:
        result = self.execute_atomic(operations)
        if not result.ok:
            raise ExecutionError(
                result.error or "guest input failed", evidence=result.as_dict()
            )

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
        self._execute_or_raise((Operation("move_to", (int(x), int(y))),))

    def glide_to(self, x: int, y: int, seconds: float) -> None:
        self._execute_or_raise(
            (Operation("glide_to", (int(x), int(y), glide_seconds(seconds))),)
        )

    def mouse_down(self, button: str = "left") -> None:
        self._execute_or_raise((Operation("mouse_down", (guest_button(button),)),))

    def mouse_up(self, button: str = "left") -> None:
        self._execute_or_raise((Operation("mouse_up", (guest_button(button),)),))

    def scroll(self, clicks: int) -> None:
        self._execute_or_raise((Operation("scroll", (0, int(clicks))),))

    def hscroll(self, dx: int) -> None:
        self._execute_or_raise((Operation("scroll", (int(dx), 0)),))

    def key_chord(self, keys: list[str]) -> None:
        self._execute_or_raise(key_chord(keys))

    def coalesced_type(self, text: str) -> None:
        self._execute_or_raise((Operation("coalesced_type", (text,)),))

    def wait(self, seconds: float) -> None:
        self._execute_or_raise((Operation("wait", (max(0.0, min(10.0, float(seconds))),)),))


__all__ = ["GuiTransport", "HttpGuiTransport"]
