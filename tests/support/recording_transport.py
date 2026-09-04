"""Deterministic in-process double for the public desktop action surface."""

from __future__ import annotations

from typing import Any

from desktop.execute.keymap import guest_button, guest_key, key_chord
from desktop.execute.protocol import (
    ExecutionError,
    ExecutionReceipt,
    HeldStateError,
    InputAudit,
    build_action_request,
    expected_atomic_input_state,
    lower_guest_operations,
    pointer_mask_for_buttons,
)
from desktop.ir import Operation, glide_seconds, scroll_deltas


class RecordingClient:
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
        self._cursor = self._clamp(x, y)
        self.audit.operations.append(Operation("move_to", self._cursor))

    def glide_to(self, x: int, y: int, seconds: float) -> None:
        self._cursor = self._clamp(x, y)
        self.audit.operations.append(
            Operation("glide_to", (*self._cursor, glide_seconds(seconds)))
        )

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
        self.audit.operations.append(Operation("scroll", (int(dx), 0)))

    def key_chord(self, keys: list[str]) -> None:
        self.execute(key_chord(keys))

    def coalesced_type(self, text: str) -> None:
        self.execute((Operation("coalesced_type", (text,)),))

    def wait(self, seconds: float) -> None:
        self.audit.operations.append(Operation("wait", (max(0.0, min(10.0, float(seconds))),)))

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        return (
            max(0, min(self._screen[0] - 1, int(x))),
            max(0, min(self._screen[1] - 1, int(y))),
        )

    def execute(self, operations: tuple[Operation, ...]) -> ExecutionReceipt:
        build_action_request(
            operations,
            initial_buttons=set(self.audit.held_buttons),
            initial_keys=set(self.audit.held_keys),
        )
        self.atomic_invocations += 1
        self.atomic_inputs.append(operations)
        before = len(self.audit.operations)
        cursor_before = self._cursor
        initial_buttons = set(self.audit.held_buttons)
        initial_keys = set(self.audit.held_keys)
        final_buttons, final_keys = expected_atomic_input_state(
            operations,
            initial_buttons=initial_buttons,
            initial_keys=initial_keys,
        )
        expected_mask = pointer_mask_for_buttons(final_buttons)
        observed_mask = -1
        cleanup_attempted = False
        error: str | None = None
        failure_kind: str | None = None
        lowered = lower_guest_operations(operations)
        primitives: list[dict[str, Any]] = []
        try:
            for operation in lowered:
                kind, args = operation.kind, operation.args
                if kind == "move_to":
                    self._cursor = self._clamp(args[0], args[1])
                    self.audit.operations.append(Operation(kind, self._cursor))
                elif kind == "glide_to":
                    self._cursor = self._clamp(args[0], args[1])
                    self.audit.operations.append(
                        Operation(kind, (*self._cursor, glide_seconds(args[2])))
                    )
                elif kind == "drag":
                    start = self._clamp(args[0], args[1])
                    end = self._clamp(args[2], args[3])
                    self._cursor = end
                    self.audit.operations.append(Operation(kind, (*start, *end)))
                    primitives.append(
                        {
                            "kind": kind,
                            "backend": "recording XTEST",
                            "start": list(start),
                            "end": list(end),
                            "zero_extent": start == end,
                        }
                    )
                    continue
                elif kind == "click":
                    button = guest_button(args[0])
                    self.audit.operations.append(Operation(kind, (button,)))
                elif kind == "mouse_down":
                    button = guest_button(args[0])
                    self.audit.held_buttons.add(button)
                    self.audit.operations.append(Operation(kind, (button,)))
                elif kind == "mouse_up":
                    button = guest_button(args[0])
                    self.audit.held_buttons.remove(button)
                    self.audit.operations.append(Operation(kind, (button,)))
                elif kind == "scroll":
                    dx, dy = scroll_deltas(args)
                    self.audit.scroll_total += dy
                    self.audit.operations.append(Operation(kind, (dx, dy)))
                elif kind == "key_down":
                    key = guest_key(args[0])
                    self.audit.held_keys.add(key)
                    self.audit.operations.append(Operation(kind, (str(args[0]),)))
                elif kind == "key_up":
                    key = guest_key(args[0])
                    self.audit.held_keys.remove(key)
                    self.audit.operations.append(Operation(kind, (str(args[0]),)))
                elif kind in {"coalesced_type", "ascii_type"}:
                    text = str(args[0])
                    if kind == "ascii_type":
                        if not text.isascii():
                            raise ExecutionError("ascii_type received non-ASCII text")
                        if "\n" in text or "\r" in text:
                            raise ExecutionError(
                                "ascii_type cannot embed Enter; emit a key event"
                            )
                    self.audit.typed_texts.append(text)
                    self.audit.operations.append(Operation(kind, (text,)))
                elif kind == "wait":
                    self.audit.operations.append(
                        Operation(kind, (max(0.0, min(10.0, float(args[0]))),))
                    )
                elif kind == "raise_for_test":
                    failure_kind = "injected"
                    raise RuntimeError(str(args[0]))
                else:
                    raise ExecutionError(f"unsupported atomic operation: {kind}")
                primitives.append({"kind": kind, "backend": "recording XTEST"})
            observed_mask = pointer_mask_for_buttons(self.audit.held_buttons)
            if observed_mask != expected_mask or self.audit.held_keys != final_keys:
                failure_kind = "verification"
                raise ExecutionError("recorded held-state readback drifted")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if failure_kind is None:
                failure_kind = "infrastructure"
            cleanup_attempted = True
            self.audit.held_keys.clear()
            self.audit.held_buttons.clear()
        final_mask = pointer_mask_for_buttons(self.audit.held_buttons)
        return ExecutionReceipt(
            ok=error is None,
            requested_operations=operations,
            operations=tuple(self.audit.operations[before:]),
            lowered_operations=lowered,
            cursor_before=cursor_before,
            cursor_after=self._cursor,
            host_cursor_before=cursor_before,
            host_cursor_after=self._cursor,
            cursor_readback_verified=True,
            pointer_button_mask=final_mask,
            observed_pointer_button_mask=observed_mask,
            expected_pointer_button_mask=expected_mask,
            held_keys=tuple(sorted(self.audit.held_keys)),
            executor_process_count=1,
            executor_returncode=0 if error is None else 1,
            cleanup_attempted=cleanup_attempted,
            error=error,
            failure_kind=failure_kind,
            backend_primitives=tuple(primitives),
        )
