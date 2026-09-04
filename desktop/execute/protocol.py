"""Build the versioned action payload executed inside a desktop VM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir import Operation, glide_seconds, scroll_deltas
from .keymap import KEYSYMS, guest_button, guest_key


class ExecutionError(RuntimeError):
    """A guest action could not be compiled, dispatched, or verified."""

    def __init__(self, message: str, *, evidence: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = evidence


class HeldStateError(ExecutionError):
    """The caller supplied an impossible held-state transition."""


@dataclass
class InputAudit:
    """Host-side record of executed operations and observed held state."""

    operations: list[Operation] = field(default_factory=list)
    held_buttons: set[str] = field(default_factory=set)
    held_keys: set[str] = field(default_factory=set)
    scroll_total: int = 0
    typed_texts: list[str] = field(default_factory=list)


BUTTON_MASKS = {"left": 1 << 8, "middle": 1 << 9, "right": 1 << 10}
BUTTON_NUMBERS = {"left": 1, "middle": 2, "right": 3}
ALL_POINTER_BUTTON_MASK = sum(BUTTON_MASKS.values())
ACTION_CONTRACT = "desktop_actions_v1"
ACTION_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
CLICK_DWELL_S = 0.05
MOTION_HZ = 60


@dataclass(frozen=True)
class ExecutionReceipt:
    """Verified account of one action request."""

    ok: bool
    requested_operations: tuple[Operation, ...]
    operations: tuple[Operation, ...]
    lowered_operations: tuple[Operation, ...]
    cursor_before: tuple[int, int]
    cursor_after: tuple[int, int]
    host_cursor_before: tuple[int, int]
    host_cursor_after: tuple[int, int]
    cursor_readback_verified: bool
    pointer_button_mask: int
    observed_pointer_button_mask: int
    expected_pointer_button_mask: int
    held_keys: tuple[str, ...]
    executor_process_count: int
    executor_returncode: int
    cleanup_attempted: bool
    error: str | None
    failure_kind: str | None
    backend_primitives: tuple[dict[str, Any], ...] = ()
    x_injection_evidence: tuple[dict[str, Any], ...] = ()
    keymap_restorations: tuple[dict[str, Any], ...] = ()
    final_pointer_readback: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "requested_operations": [item.as_dict() for item in self.requested_operations],
            "operations": [item.as_dict() for item in self.operations],
            "lowered_operations": [item.as_dict() for item in self.lowered_operations],
            "cursor_before": list(self.cursor_before),
            "cursor_after": list(self.cursor_after),
            "host_cursor_before": list(self.host_cursor_before),
            "host_cursor_after": list(self.host_cursor_after),
            "cursor_readback_verified": self.cursor_readback_verified,
            "pointer_button_mask": self.pointer_button_mask,
            "observed_pointer_button_mask": self.observed_pointer_button_mask,
            "expected_pointer_button_mask": self.expected_pointer_button_mask,
            "held_keys": list(self.held_keys),
            "executor_process_count": self.executor_process_count,
            "executor_returncode": self.executor_returncode,
            "cleanup_attempted": self.cleanup_attempted,
            "error": self.error,
            "failure_kind": self.failure_kind,
            "backend_primitives": list(self.backend_primitives),
            "x_injection_evidence": list(self.x_injection_evidence),
            "keymap_restorations": list(self.keymap_restorations),
            "final_pointer_readback": dict(self.final_pointer_readback),
        }


def expected_atomic_input_state(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str],
    initial_keys: set[str],
) -> tuple[set[str], set[str]]:
    """Simulate held-state transitions before dispatch."""
    buttons = {guest_button(button) for button in initial_buttons}
    keys = {guest_key(key) for key in initial_keys}
    for operation in operations:
        if operation.kind == "drag":
            if "left" in buttons:
                raise HeldStateError("button already held: left")
        elif operation.kind == "mouse_down":
            button = guest_button(operation.args[0])
            if button in buttons:
                raise HeldStateError(f"button already held: {button}")
            buttons.add(button)
        elif operation.kind == "mouse_up":
            button = guest_button(operation.args[0])
            if button not in buttons:
                raise HeldStateError(f"button not held: {button}")
            buttons.remove(button)
        elif operation.kind == "key_down":
            key = guest_key(operation.args[0])
            if key in keys:
                raise HeldStateError(f"key already held: {key}")
            keys.add(key)
        elif operation.kind == "key_up":
            key = guest_key(operation.args[0])
            if key not in keys:
                raise HeldStateError(f"key not held: {key}")
            keys.remove(key)
    return buttons, keys


def pointer_mask_for_buttons(buttons: set[str]) -> int:
    normalized = {guest_button(button) for button in buttons}
    return sum(BUTTON_MASKS[button] for button in normalized)


def lower_guest_operations(operations: tuple[Operation, ...]) -> tuple[Operation, ...]:
    """Lower only an adjacent same-button press/release to one click."""
    lowered: list[Operation] = []
    index = 0
    while index < len(operations):
        operation = operations[index]
        if operation.kind == "mouse_down" and index + 1 < len(operations):
            following = operations[index + 1]
            if following.kind == "mouse_up" and guest_button(following.args[0]) == guest_button(
                operation.args[0]
            ):
                lowered.append(Operation("click", (guest_button(operation.args[0]),)))
                index += 2
                continue
        lowered.append(operation)
        index += 1
    return tuple(lowered)


def _require_args(operation: Operation, count: int) -> tuple:
    if len(operation.args) != count:
        raise ExecutionError(
            f"{operation.kind} requires {count} arguments, got {operation.args!r}"
        )
    return operation.args


def _validate_text(text: Any) -> str:
    if not isinstance(text, str):
        raise ExecutionError("typing text must be a string")
    for character in text:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise ExecutionError("typing text contains a surrogate code point")
        if (codepoint < 0x20 and character not in "\b\t\n\r") or 0x7F <= codepoint < 0xA0:
            raise ExecutionError(f"typing text contains unsupported U+{codepoint:04X}")
    return text


def _compile_rows(operations: tuple[Operation, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operation in operations:
        kind = operation.kind
        if kind in {"move_to", "drag"}:
            count = 2 if kind == "move_to" else 4
            args = _require_args(operation, count)
            rows.append({"kind": kind, "args": [int(value) for value in args]})
        elif kind == "glide_to":
            x, y, seconds = _require_args(operation, 3)
            rows.append({"kind": kind, "args": [int(x), int(y), glide_seconds(seconds)]})
        elif kind in {"click", "mouse_down", "mouse_up"}:
            (raw_button,) = _require_args(operation, 1)
            button = guest_button(raw_button)
            rows.append(
                {"kind": kind, "args": [button], "button_number": BUTTON_NUMBERS[button]}
            )
        elif kind in {"key_down", "key_up"}:
            (raw_key,) = _require_args(operation, 1)
            key = guest_key(raw_key)
            rows.append(
                {"kind": kind, "args": [str(raw_key)], "key": key, "keysym": KEYSYMS[key]}
            )
        elif kind == "scroll":
            dx, dy = scroll_deltas(operation.args)
            rows.append({"kind": kind, "args": [dx, dy]})
        elif kind in {"coalesced_type", "ascii_type"}:
            (raw_text,) = _require_args(operation, 1)
            text = _validate_text(raw_text)
            if kind == "ascii_type":
                try:
                    text.encode("ascii")
                except UnicodeEncodeError as exc:
                    raise ExecutionError("ascii_type received non-ASCII text") from exc
                if "\n" in text or "\r" in text:
                    raise ExecutionError("ascii_type cannot embed Enter; emit a key event")
            rows.append({"kind": kind, "args": [text]})
        elif kind == "wait":
            (seconds,) = _require_args(operation, 1)
            rows.append({"kind": kind, "args": [max(0.0, min(10.0, float(seconds)))]})
        elif kind == "raise_for_test":
            (message,) = _require_args(operation, 1)
            rows.append({"kind": kind, "args": [str(message)]})
        else:
            raise ExecutionError(f"unsupported atomic operation: {kind}")
    return rows


def build_action_request(
    operations: tuple[Operation, ...],
    *,
    initial_buttons: set[str],
    initial_keys: set[str],
) -> tuple[dict[str, Any], int, set[str]]:
    """Validate one operation tuple and serialize the sole guest action shape."""
    if type(operations) is not tuple or not operations:
        raise ExecutionError("operations must be a non-empty tuple")
    if not all(isinstance(operation, Operation) for operation in operations):
        raise ExecutionError("operations must contain only Operation values")
    normalized_initial_buttons = {guest_button(button) for button in initial_buttons}
    normalized_initial_keys = {guest_key(key) for key in initial_keys}
    final_buttons, final_keys = expected_atomic_input_state(
        operations,
        initial_buttons=normalized_initial_buttons,
        initial_keys=normalized_initial_keys,
    )
    lowered = lower_guest_operations(operations)
    rows = _compile_rows(lowered)
    touched_keys = set(normalized_initial_keys)
    touched_buttons = set(normalized_initial_buttons)
    for row in rows:
        if row["kind"] in {"key_down", "key_up"}:
            touched_keys.add(str(row["key"]))
        elif row["kind"] in {"click", "mouse_down", "mouse_up"}:
            touched_buttons.add(str(row["args"][0]))
        elif row["kind"] == "drag":
            touched_buttons.add("left")
    keysyms = {key: KEYSYMS[key] for key in sorted(touched_keys | {"shiftleft"})}
    expected_mask = pointer_mask_for_buttons(final_buttons)
    return (
        {
            "contract": ACTION_CONTRACT,
            "schema_version": ACTION_SCHEMA_VERSION,
            "rows": rows,
            "operations": [item.as_dict() for item in operations],
            "lowered_operations": [item.as_dict() for item in lowered],
            "keysyms": keysyms,
            "expected_keys": sorted(final_keys),
            "expected_initial_keys": sorted(normalized_initial_keys),
            "expected_mask": expected_mask,
            "expected_initial_mask": pointer_mask_for_buttons(normalized_initial_buttons),
            "touched_button_numbers": sorted(
                BUTTON_NUMBERS[button] for button in touched_buttons
            ),
            "all_button_mask": ALL_POINTER_BUTTON_MASK,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "click_dwell_seconds": CLICK_DWELL_S,
            "motion_hz": MOTION_HZ,
        },
        expected_mask,
        final_keys,
    )
