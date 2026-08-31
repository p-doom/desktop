from __future__ import annotations

import pytest

from desktop import ir
from desktop.execute.protocol import (
    BUTTON_MASKS,
    BUTTON_NUMBERS,
    RESULT_SCHEMA_VERSION,
)

from .support.guest_runner import run_guest_program


def _event_shape(payload: dict) -> list[tuple[str, int]]:
    return [(event["event"], event["detail"]) for event in payload["x_injection_evidence"]]


def test_typed_newline_and_tab_match_explicit_key_events() -> None:
    text = "a\nb\tc"
    typed = run_guest_program((ir.coalesced_type(text),))
    expanded_operations = (
        ir.coalesced_type("a"),
        ir.key_down("Return"),
        ir.key_up("Return"),
        ir.coalesced_type("b"),
        ir.key_down("Tab"),
        ir.key_up("Tab"),
        ir.coalesced_type("c"),
    )
    expanded = run_guest_program(expanded_operations)

    assert typed.returncode == expanded.returncode == 0
    assert _event_shape(typed.payload) == _event_shape(expanded.payload)
    assert typed.request["operations"] == [ir.coalesced_type(text).as_dict()]
    assert expanded.request["operations"] == [
        operation.as_dict() for operation in expanded_operations
    ]


@pytest.mark.parametrize("button", ["left", "middle", "right"])
def test_click_and_explicit_transitions_keep_their_own_button(button: str) -> None:
    pair = (ir.mouse_down(button), ir.mouse_up(button))
    click = run_guest_program((ir.click(button),))
    transitions = run_guest_program(pair)
    expected_events = [
        ("button_press", BUTTON_NUMBERS[button]),
        ("button_release", BUTTON_NUMBERS[button]),
    ]

    assert _event_shape(click.payload) == expected_events
    assert _event_shape(transitions.payload) == expected_events
    assert transitions.request["operations"] == [operation.as_dict() for operation in pair]
    assert transitions.payload["lowered_operations"] == [ir.click(button).as_dict()]

    pressed = run_guest_program((ir.mouse_down(button),))
    released = run_guest_program(
        (ir.mouse_up(button),),
        initial_buttons={button},
        initial_mask=BUTTON_MASKS[button],
    )
    assert pressed.payload["pointer_button_mask"] == BUTTON_MASKS[button]
    assert released.payload["pointer_button_mask"] == 0


def test_failure_cleanup_preserves_the_atomic_receipt_and_releases_holds() -> None:
    operations = (
        ir.key_down("ControlLeft"),
        ir.mouse_down("right"),
        ir.Operation("raise_for_test", ("boom",)),
    )
    run = run_guest_program(operations)

    assert run.returncode == 1
    assert run.payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert run.payload["executor_process_count"] == 1
    assert run.payload["cleanup_attempted"] is True
    assert run.payload["failure_kind"] == "injected"
    assert run.payload["pointer_button_mask"] == 0
    assert run.payload["held_keys"] == []
    assert run.request["operations"] == [operation.as_dict() for operation in operations]
    phases = [event["phase"] for event in run.payload["x_injection_evidence"]]
    assert "cleanup_key" in phases
    assert "cleanup_button" in phases
