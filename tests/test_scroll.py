"""``ir.scroll_deltas``, and the ``scroll`` lowering.

Why this is first: every grammar emits ``scroll(0, dy)``.  A one-arity reader
would take ``args[0]`` as the vertical ticks and lower every one of them to
``pyautogui.scroll(0)`` -- a no-op wheel event, on every scroll, in every grammar,
with no error anywhere.  It would present as "the model never learned to scroll".

So the assertions here are about the DIRECTION and MAGNITUDE that reach
``pyautogui``, not merely about the shape of the tuple.

``(dx, dy)`` is the only accepted shape: anything else is a loud error, not a
second contract two layers have to keep in agreement.
"""

from __future__ import annotations

import pytest

from pixeldesk import ir
from pixeldesk.execute.guest_program import compile_atomic_guest_program
from pixeldesk.ir import Operation, scroll_deltas
from tests.support.guest_runner import run_guest_program



@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((0, 3), (0, 3)),
        ((0, -3), (0, -3)),
        ((5, 0), (5, 0)),
        ((4, 5), (4, 5)),
        ((0, 0), (0, 0)),
    ],
)
def test_two_arity_is_read_as_dx_dy(args, expected):
    assert scroll_deltas(args) == expected


@pytest.mark.parametrize("args", [(), (3,), (1, 2, 3)])
def test_any_arity_but_two_is_refused(args):
    """Refused, not guessed: a bare tick count and a truncated triple both used
    to be read as something, which is a second contract to keep in agreement."""
    with pytest.raises(ValueError, match="exactly"):
        scroll_deltas(args)


def test_constructor_and_reader_round_trip():
    assert scroll_deltas(ir.scroll(2, -9).args) == (2, -9)


def _scroll_lines(operation: Operation) -> list[str]:
    program, _ = compile_atomic_guest_program(
        (operation,), initial_buttons=set(), initial_keys=set()
    )
    return [
        line.strip()
        for line in program.splitlines()
        if "pyautogui.scroll(" in line or "pyautogui.hscroll(" in line
    ]


def test_the_grammar_shaped_call_lowers_to_a_real_vertical_scroll():
    """``scroll(0, 3)`` must NOT become ``pyautogui.scroll(0)``."""
    assert _scroll_lines(ir.scroll(0, 3)) == ["pyautogui.scroll(3)"]
    assert "pyautogui.scroll(0)" not in _scroll_lines(ir.scroll(0, 3))


def test_a_one_arity_scroll_is_refused_by_the_compiler():
    with pytest.raises(ValueError, match="exactly"):
        _scroll_lines(Operation("scroll", (3,)))


def test_horizontal_only_uses_hscroll_and_emits_no_vertical_event():
    assert _scroll_lines(ir.scroll(-2, 0)) == ["pyautogui.hscroll(-2)"]


def test_both_axes_emit_both_primitives():
    lines = _scroll_lines(ir.scroll(4, 5))
    assert "pyautogui.hscroll(4)" in lines
    assert "pyautogui.scroll(5)" in lines


def test_a_zero_scroll_emits_no_wheel_event_at_all():
    assert _scroll_lines(ir.scroll(0, 0)) == []


@pytest.mark.parametrize(("dy", "expected"), [(3, 3), (-3, -3), (1, 1)])
def test_executed_vertical_scroll_passes_the_signed_tick_count(dy, expected):
    """Positive ``dy`` is up, and the sign survives compilation and execution."""
    run = run_guest_program((ir.scroll(0, dy),))
    assert run.returncode == 0, run.stderr
    assert run.payload is not None and run.payload["ok"] is True
    assert run.pyautogui_calls == [["scroll", expected]]
    assert run.trace() == [("scroll", [0, dy])]
    assert run.primitives("scroll") == [
        {"kind": "scroll", "call": "pyautogui.scroll/hscroll", "dx": 0, "dy": dy}
    ]


def test_executed_diagonal_scroll_moves_both_axes_with_the_right_signs():
    run = run_guest_program((ir.scroll(-6, 7),))
    assert run.pyautogui_calls == [["hscroll", -6], ["scroll", 7]]
    assert run.trace() == [("scroll", [-6, 7])]


def test_executed_zero_scroll_touches_no_wheel_but_is_still_reported():
    run = run_guest_program((ir.scroll(0, 0),))
    assert run.pyautogui_calls == []
    assert run.trace() == [("scroll", [0, 0])]


@pytest.mark.parametrize("args", [(0, 3), (0, -3), (4, 5), (-6, 7), (0, 0), (5, 0)])
def test_recording_transport_trace_matches_the_guest_trace(recording, args):
    """The double's operation sequence is the reason it exists; it must match.

    A two-axis scroll used to append TWO operations here -- ``(0, dy)`` then
    ``(dx, 0)`` -- while the guest reports ONE ``(dx, dy)``, and ``scroll(0, 0)``
    used to append none at all against the guest's one.
    """
    dx, dy = scroll_deltas(args)
    guest = run_guest_program((Operation("scroll", args),))
    result = recording.execute_atomic((Operation("scroll", args),))
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()
    assert [(op.kind, op.args) for op in result.operations] == [("scroll", (dx, dy))]


def test_recording_transport_scroll_total_counts_vertical_ticks(recording):
    recording.execute_atomic((ir.scroll(0, 3),))
    recording.execute_atomic((ir.scroll(9, -1),))
    recording.execute_atomic((ir.scroll(0, 5),))
    assert recording.audit.scroll_total == 3 - 1 + 5


def test_audit_absorption_reads_the_vertical_axis_of_a_scroll_trace():
    """``HttpGuiTransport._absorb`` reads index 1, i.e. dy, of a two-axis trace."""
    from pixeldesk.execute.guest_program import AtomicExecutionResult
    from pixeldesk.execute.transport import HttpGuiTransport

    transport = HttpGuiTransport("http://127.0.0.1:1")
    result = AtomicExecutionResult(
        ok=True,
        cursor=(0, 0),
        cursor_before=(0, 0),
        cursor_after=(0, 0),
        pointer_button_mask=0,
        observed_pointer_button_mask=0,
        expected_pointer_button_mask=0,
        guest_process_count=1,
        guest_returncode=0,
        raw_result_marker="",
        cleanup_attempted=False,
        error=None,
        failure_kind=None,
        operations=(Operation("scroll", (11, -4)),),
        semantic_operations=(),
        lowered_operations=(),
    )
    transport._absorb(result, expected_keys=set())
    assert transport.audit.scroll_total == -4
