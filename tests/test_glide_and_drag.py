"""The direct-XTEST ``glide_to`` and ``drag`` paths, and drag's self-balancing.

``glide_to`` must emit a timed sequence of motion events -- a sweep, not a
teleport. ``drag`` emits press / motion / release inside one process and records
``zero_extent`` so a genuine ``drag(x, y, x, y)`` remains distinguishable.

``expected_atomic_input_state``'s drag branch must be self-balancing: the press
and the release are both inside the single operation, so the held-button set on
either side of a drag is identical, and equal numbers of downs and ups reach the
guest.
"""

from __future__ import annotations

import pytest

from desktop import ir
from desktop.execute.protocol import (
    ExecutionError,
    expected_atomic_input_state,
    lower_guest_operations,
)
from tests.support.guest_runner import run_guest_program


def test_glide_lowers_to_a_timed_moveto():
    run = run_guest_program((ir.glide_to(700, 400, 0.25),))
    assert run.returncode == 0, run.stderr
    (primitive,) = run.primitives("glide_to")
    assert primitive["seconds"] == 0.25
    assert primitive["motion_events"] >= 2


def test_glide_duration_is_the_difference_from_move_to():
    """A ``move_to`` teleports; a ``glide_to`` must carry a non-zero duration."""
    glide = run_guest_program((ir.glide_to(700, 400, 0.25),))
    move = run_guest_program((ir.move_to(700, 400),))
    assert glide.primitives("glide_to")[0]["seconds"] == 0.25
    assert glide.primitives("glide_to")[0]["motion_events"] > 1
    assert len(move.payload["x_injection_evidence"]) == 1
    assert glide.payload["cursor_after"] == move.payload["cursor_after"]


def test_glide_records_its_own_primitive_and_seconds():
    run = run_guest_program((ir.glide_to(300, 200, 1.5),))
    (primitive,) = run.primitives("glide_to")
    assert primitive["backend"] == "python-xlib XTEST"
    assert primitive["seconds"] == 1.5
    assert primitive["requested_position"] == [300, 200]
    assert primitive["clamped"] is False
    assert run.trace() == [("glide_to", [300, 200, 1.5])]


def test_glide_clamps_to_the_screen_and_says_so():
    run = run_guest_program((ir.glide_to(5000, 5000, 0.1),), size=(1920, 1080))
    (primitive,) = run.primitives("glide_to")
    assert primitive["clamped"] is True
    assert primitive["requested_position"] == [5000, 5000]
    assert primitive["cursor_after"] == [1919, 1079]
    assert run.payload["x_injection_evidence"][-1]["x"] == 1919
    assert run.payload["x_injection_evidence"][-1]["y"] == 1079


@pytest.mark.parametrize(
    ("given", "expected"), [(-1.0, None), (99.0, None), (0.01, 0.01), (10.0, 10.0)]
)
def test_glide_seconds_are_validated_without_clamping(given, expected):
    operation = ir.Operation("glide_to", (10, 10, given))
    if expected is None:
        with pytest.raises(ValueError, match="glide seconds"):
            run_guest_program((operation,))
    else:
        run = run_guest_program((operation,))
        assert run.primitives("glide_to")[0]["seconds"] == expected


def test_glide_moves_the_cursor_the_guest_reports_back():
    run = run_guest_program((ir.glide_to(640, 480, 0.05),), cursor=(1, 1))
    assert run.payload["cursor_before"] == [1, 1]
    assert run.payload["cursor_after"] == [640, 480]


# --------------------------------------------------------------------------- #
# A held-button stroke: mouse_down / glide_to / mouse_up
#
# Adopted from the grammar suite, which had a check that a tool-call grammar's
# ``left_click_drag`` reaches the guest as a timed move BETWEEN the press and the
# release.  The codec's half belongs to that suite; the lowering is this layer's,
# so the assertion lives here -- built from ``Operation``s directly, so it stays
# grammar-free.
# --------------------------------------------------------------------------- #

HELD_STROKE = (ir.mouse_down("left"), ir.glide_to(1200, 700, 0.5), ir.mouse_up("left"))


def test_a_held_stroke_keeps_the_timed_move_inside_the_held_button():
    run = run_guest_program(HELD_STROKE)
    phases = [event["phase"] for event in run.payload["x_injection_evidence"]]
    assert phases[0] == "mouse_down"
    assert set(phases[1:-1]) == {"glide"}
    assert phases[-1] == "mouse_up"


def test_a_held_stroke_is_not_coalesced_into_a_click():
    """A ``glide_to`` between the transitions must prevent coalescing, or the
    stroke would be replaced by a click and the drag would never happen."""
    lowered = lower_guest_operations(HELD_STROKE)
    assert [op.kind for op in lowered] == ["mouse_down", "glide_to", "mouse_up"]


def test_a_held_stroke_executes_with_the_button_down_across_the_move():
    """Executed, not read: the same ordering in the real call sequence."""
    run = run_guest_program(HELD_STROKE)
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True, run.payload["error"]
    assert [event["event"] for event in run.payload["x_injection_evidence"]] == [
        "button_press",
        *["motion_notify"] * 30,
        "button_release",
    ]
    assert run.trace() == [
        ("mouse_down", ["left"]),
        ("glide_to", [1200, 700, 0.5]),
        ("mouse_up", ["left"]),
    ]


def test_a_held_stroke_leaves_no_button_held():
    run = run_guest_program(HELD_STROKE)
    assert run.payload["expected_pointer_button_mask"] == 0
    assert run.payload["observed_pointer_button_mask"] == 0


def test_the_recording_double_reproduces_a_held_stroke(recording):
    guest = run_guest_program(HELD_STROKE)
    result = recording.execute(HELD_STROKE)
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()
    assert result.ok is True


def test_drag_lowers_to_press_move_release_in_one_process():
    run = run_guest_program((ir.drag(100, 100, 400, 300),))
    assert run.returncode == 0, run.stderr
    assert [event["phase"] for event in run.payload["x_injection_evidence"]] == [
        "drag_start",
        "drag_press",
        "drag_end",
        "drag_release",
    ]
    assert run.payload["executor_process_count"] == 1
    assert run.trace() == [("drag", [100, 100, 400, 300])]


def test_a_zero_extent_drag_still_produces_a_real_press_and_release():
    """The entire reason ``drag`` is its own kind rather than a triple."""
    run = run_guest_program((ir.drag(200, 200, 200, 200),))
    events = run.payload["x_injection_evidence"]
    press = [event for event in events if event["event"] == "button_press"]
    release = [event for event in events if event["event"] == "button_release"]
    assert len(press) == 1 and len(release) == 1


def test_zero_extent_is_recorded_on_the_release_primitive():
    zero = run_guest_program((ir.drag(5, 5, 5, 5),))
    real = run_guest_program((ir.drag(5, 5, 6, 6),))
    assert zero.primitives("drag")[0]["zero_extent"] is True
    assert real.primitives("drag")[0]["zero_extent"] is False


def test_a_drag_is_not_collapsed_into_a_click_by_the_lowering():
    """``lower_guest_operations`` must leave a drag alone; only adjacent
    same-button ``mouse_down``/``mouse_up`` pairs coalesce."""
    lowered = lower_guest_operations((ir.drag(1, 1, 2, 2),))
    assert [op.kind for op in lowered] == ["drag"]
    run = run_guest_program((ir.drag(1, 1, 2, 2),))
    assert run.primitives("click") == []


def test_drag_leaves_no_button_held_and_verifies_its_own_mask():
    run = run_guest_program((ir.drag(1, 1, 2, 2),))
    assert run.payload["expected_pointer_button_mask"] == 0
    assert run.payload["observed_pointer_button_mask"] == 0
    assert run.payload["ok"] is True


def test_drag_leaves_the_held_button_set_unchanged():
    buttons, keys = expected_atomic_input_state(
        (ir.drag(1, 2, 3, 4),), initial_buttons=set(), initial_keys=set()
    )
    assert buttons == set() and keys == set()


@pytest.mark.parametrize("count", [1, 2, 5])
def test_repeated_drags_stay_balanced(count):
    buttons, _ = expected_atomic_input_state(
        (ir.drag(1, 2, 3, 4),) * count, initial_buttons=set(), initial_keys=set()
    )
    assert buttons == set()


def test_a_drag_does_not_disturb_another_button_that_is_already_held():
    buttons, _ = expected_atomic_input_state(
        (ir.drag(1, 2, 3, 4),), initial_buttons={"middle"}, initial_keys=set()
    )
    assert buttons == {"middle"}


def test_a_drag_cannot_start_with_left_already_down():
    with pytest.raises(ExecutionError, match="button already held: left"):
        expected_atomic_input_state(
            (ir.drag(1, 2, 3, 4),), initial_buttons={"left"}, initial_keys=set()
        )
    with pytest.raises(ExecutionError, match="button already held: left"):
        expected_atomic_input_state(
            (ir.mouse_down("left"), ir.drag(1, 2, 3, 4)),
            initial_buttons=set(),
            initial_keys=set(),
        )


def test_the_guest_emits_equal_numbers_of_downs_and_ups_for_a_drag():
    run = run_guest_program((ir.drag(1, 1, 9, 9), ir.drag(9, 9, 1, 1)))
    events = [event["event"] for event in run.payload["x_injection_evidence"]]
    assert events.count("button_press") == events.count("button_release") == 2


def test_a_drag_after_a_held_press_of_a_different_button_executes():
    run = run_guest_program(
        (ir.mouse_down("middle"), ir.drag(3, 3, 8, 8), ir.mouse_up("middle"))
    )
    assert run.payload["ok"] is True, run.payload["error"]
    assert run.payload["observed_pointer_button_mask"] == 0


def test_recording_drag_matches_the_guest_trace(recording):
    guest = run_guest_program((ir.drag(10, 10, 40, 40),))
    result = recording.execute((ir.drag(10, 10, 40, 40),))
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()


def test_recording_drag_records_zero_extent(recording):
    result = recording.execute((ir.drag(7, 7, 7, 7),))
    (primitive,) = result.backend_primitives
    assert primitive["zero_extent"] is True


def test_recording_glide_matches_the_guest_trace_and_moves_the_cursor(recording):
    guest = run_guest_program((ir.glide_to(88, 99, 0.2),))
    result = recording.execute((ir.glide_to(88, 99, 0.2),))
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()
    assert recording.cursor_position() == (88, 99)


def test_recording_drag_is_balanced_and_holds_nothing_afterwards(recording):
    recording.execute((ir.drag(1, 1, 2, 2),))
    assert recording.audit.held_buttons == set()


# --------------------------------------------------------------------------- #
# STANDING INVARIANT: the double must agree with the guest it doubles
#
# ``RecordingClient`` exists so the executor is testable without a VM, and
# EVERY test built on it inherits its fidelity.  It was found disagreeing with the
# real guest program on `scroll`: two operations for a diagonal, none at all for
# `scroll(0, 0)`, against the guest's one.  A double that diverges silently
# invalidates every assertion made through it, so the agreement is now asserted
# directly, for every canonical kind, rather than assumed.
# --------------------------------------------------------------------------- #

#: One representative payload per kind the executor claims to lower.  Kinds are
#: grouped where a kind is only legal next to another (a release needs a press).
CANONICAL_ACTIONS: dict[str, tuple] = {
    "move_to": (ir.move_to(321, 654),),
    "move_to_clamped": (ir.move_to(99999, 99999),),
    "glide_to": (ir.glide_to(300, 400, 0.05),),
    "glide_to_clamped": (ir.glide_to(99999, 99999, 0.05),),
    "drag": (ir.drag(10, 20, 30, 40),),
    "drag_zero_extent": (ir.drag(10, 20, 10, 20),),
    "click_left": (ir.click("left"),),
    "click_right": (ir.click("right"),),
    "click_middle": (ir.click("middle"),),
    "mouse_down_up": (ir.mouse_down("left"), ir.mouse_up("left")),
    "mouse_hold_across_move": (
        ir.mouse_down("left"),
        ir.move_to(5, 5),
        ir.mouse_up("left"),
    ),
    "key_down_up": (ir.key_down("ControlLeft"), ir.key_up("ControlLeft")),
    "key_chord": (
        ir.key_down("ControlLeft"),
        ir.key_down("KeyA"),
        ir.key_up("a"),
        ir.key_up("ctrlleft"),
    ),
    "scroll_up": (ir.scroll(0, 3),),
    "scroll_down": (ir.scroll(0, -3),),
    "scroll_horizontal": (ir.scroll(5, 0),),
    "scroll_diagonal": (ir.scroll(-6, 7),),
    "scroll_zero": (ir.scroll(0, 0),),
    "ascii_type": (ir.ascii_type("hello world"),),
    "wait": (ir.wait(0.0),),
    "held_stroke": HELD_STROKE,
    "raise_for_test": (ir.Operation("raise_for_test", ("boom",)),),
    # The cleanup path, with real held state to release: the guest releases the
    # key without tracing the release, so the double must not trace one either.
    "raise_with_a_key_held": (
        ir.key_down("ControlLeft"),
        ir.Operation("raise_for_test", ("boom",)),
    ),
}


@pytest.mark.parametrize("name", sorted(CANONICAL_ACTIONS))
def test_the_double_and_the_guest_agree_on_the_operation_trace(name, recording):
    """Same kinds, same args, same COUNT -- for every canonical kind."""
    operations = CANONICAL_ACTIONS[name]
    guest = run_guest_program(operations)
    assert guest.payload is not None, guest.stderr
    result = recording.execute(operations)
    recorded = [(op.kind, list(op.args)) for op in result.operations]
    assert len(recorded) == len(guest.trace()), (
        f"{name}: double produced {len(recorded)} operations, "
        f"guest produced {len(guest.trace())}"
    )
    assert recorded == guest.trace(), name


@pytest.mark.parametrize("name", sorted(CANONICAL_ACTIONS))
def test_the_double_and_the_guest_agree_on_success_and_held_state(name, recording):
    operations = CANONICAL_ACTIONS[name]
    guest = run_guest_program(operations)
    result = recording.execute(operations)
    assert result.ok is guest.payload["ok"], name
    assert result.failure_kind == guest.payload["failure_kind"], name
    assert (
        result.expected_pointer_button_mask == guest.payload["expected_pointer_button_mask"]
    ), name
    assert (
        result.observed_pointer_button_mask == guest.payload["observed_pointer_button_mask"]
    ), name


@pytest.mark.parametrize("name", sorted(CANONICAL_ACTIONS))
def test_the_double_and_the_guest_agree_on_the_lowering(name, recording):
    operations = CANONICAL_ACTIONS[name]
    guest = run_guest_program(operations)
    result = recording.execute(operations)
    assert [op.as_dict() for op in result.lowered_operations] == guest.payload[
        "lowered_operations"
    ], name
    assert [op.as_dict() for op in result.requested_operations] == guest.request[
        "operations"
    ], name


def test_the_invariant_covers_every_kind_the_executor_lowers():
    """The table above must not fall behind ``CANONICAL_KINDS``.

    ``coalesced_type`` has its own Unicode-specific test below rather than a table
    entry.
    """
    from desktop.ir import CANONICAL_KINDS

    covered = {
        operation.kind for operations in CANONICAL_ACTIONS.values() for operation in operations
    }
    expected = set(CANONICAL_KINDS) - {"coalesced_type"}
    assert expected <= covered, f"uncovered kinds: {sorted(expected - covered)}"


def test_the_coalesced_type_kind_also_agrees(recording):
    operations = (ir.coalesced_type("héllo ✓"),)
    guest = run_guest_program(operations)
    result = recording.execute(operations)
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()
    assert result.ok is guest.payload["ok"] is True
