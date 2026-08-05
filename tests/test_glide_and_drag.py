"""ITEM 2: the ``glide_to`` and ``drag`` lowerings, and drag's self-balancing.

``glide_to`` must reach ``pyautogui.moveTo(..., duration=)`` -- a sweep, not a
teleport, because some widgets only respond to the sweep.  ``drag`` must lower to
press / move / release inside the ONE process, and must record ``zero_extent`` so a
genuine ``drag(x, y, x, y)`` is distinguishable from a click afterwards.

``expected_atomic_input_state``'s drag branch must be self-balancing: the press
and the release are both inside the single operation, so the held-button set on
either side of a drag is identical, and equal numbers of downs and ups reach the
guest.
"""

from __future__ import annotations

import pytest

from desktop_env import ir
from desktop_env.execute.guest_program import (
    ExecutionError,
    expected_atomic_input_state,
    lower_guest_operations,
)
from tests.support.guest_runner import run_guest_program

# --------------------------------------------------------------------------- #
# glide_to
# --------------------------------------------------------------------------- #


def test_glide_lowers_to_a_timed_moveto():
    run = run_guest_program((ir.glide_to(700, 400, 0.25),))
    assert run.returncode == 0, run.stderr
    assert run.pyautogui_calls == [["moveTo", 700, 400, 0.25]]


def test_glide_duration_is_the_difference_from_move_to():
    """A ``move_to`` teleports; a ``glide_to`` must carry a non-zero duration."""
    glide = run_guest_program((ir.glide_to(700, 400, 0.25),))
    move = run_guest_program((ir.move_to(700, 400),))
    assert glide.pyautogui_calls[0][3] == 0.25
    assert move.pyautogui_calls[0][3] == 0.0
    assert glide.pyautogui_calls[0][:3] == move.pyautogui_calls[0][:3]


def test_glide_records_its_own_primitive_and_seconds():
    run = run_guest_program((ir.glide_to(300, 200, 1.5),))
    (primitive,) = run.primitives("glide_to")
    assert primitive["call"] == "pyautogui.moveTo(duration=)"
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
    assert run.pyautogui_calls == [["moveTo", 1919, 1079, 0.1]]


@pytest.mark.parametrize(("given", "expected"), [(-1.0, 0.0), (99.0, 10.0), (10.0, 10.0)])
def test_glide_seconds_are_clamped_to_the_documented_range(given, expected):
    run = run_guest_program((ir.Operation("glide_to", (10, 10, given)),))
    assert run.pyautogui_calls[0][3] == expected


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
    """In the generated SOURCE: press, then the timed move, then release."""
    from desktop_env.execute.guest_program import compile_atomic_guest_program

    source, _ = compile_atomic_guest_program(
        HELD_STROKE, initial_buttons=set(), initial_keys=set()
    )
    assert "pyautogui.moveTo(_tx,_ty,duration=0.5)" in source
    press = source.index("pyautogui.mouseDown(button='left')")
    stroke = source.index("duration=0.5")
    release = source.index("pyautogui.mouseUp(button='left')")
    assert press < stroke < release


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
    assert run.pyautogui_calls == [
        ["mouseDown", "left"],
        ["moveTo", 1200, 700, 0.5],
        ["mouseUp", "left"],
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
    result = recording.execute_atomic(HELD_STROKE)
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()
    assert result.ok is True


# --------------------------------------------------------------------------- #
# drag
# --------------------------------------------------------------------------- #


def test_drag_lowers_to_press_move_release_in_one_process():
    run = run_guest_program((ir.drag(100, 100, 400, 300),))
    assert run.returncode == 0, run.stderr
    assert run.pyautogui_calls == [
        ["moveTo", 100, 100, 0.0],
        ["mouseDown", "left"],
        ["moveTo", 400, 300, 0.0],
        ["mouseUp", "left"],
    ]
    assert run.payload["guest_process_count"] == 1
    assert run.trace() == [
        ("move_to", [100, 100]),
        ("mouse_down", ["left"]),
        ("move_to", [400, 300]),
        ("mouse_up", ["left"]),
    ]


def test_a_zero_extent_drag_still_produces_a_real_press_and_release():
    """The entire reason ``drag`` is its own kind rather than a triple."""
    run = run_guest_program((ir.drag(200, 200, 200, 200),))
    calls = [call[0] for call in run.pyautogui_calls]
    assert calls.count("mouseDown") == 1
    assert calls.count("mouseUp") == 1
    events = [event for event in run.x_events if event[0] == "fake_input"]
    press = [event for event in events if event[1] == 4]
    release = [event for event in events if event[1] == 5]
    assert len(press) == 1 and len(release) == 1


def test_zero_extent_is_recorded_on_the_release_primitive():
    zero = run_guest_program((ir.drag(5, 5, 5, 5),))
    real = run_guest_program((ir.drag(5, 5, 6, 6),))
    assert zero.primitives("mouse_up")[0]["zero_extent"] is True
    assert real.primitives("mouse_up")[0]["zero_extent"] is False
    assert zero.primitives("mouse_up")[0]["drag"] is True


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


# --------------------------------------------------------------------------- #
# expected_atomic_input_state: the drag branch is self-balancing
# --------------------------------------------------------------------------- #


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
    calls = [call[0] for call in run.pyautogui_calls]
    assert calls.count("mouseDown") == calls.count("mouseUp") == 2


def test_a_drag_after_a_held_press_of_a_different_button_executes():
    run = run_guest_program(
        (ir.mouse_down("middle"), ir.drag(3, 3, 8, 8), ir.mouse_up("middle"))
    )
    assert run.payload["ok"] is True, run.payload["error"]
    assert run.payload["observed_pointer_button_mask"] == 0


# --------------------------------------------------------------------------- #
# The recording double's drag/glide must match
# --------------------------------------------------------------------------- #


def test_recording_drag_matches_the_guest_trace(recording):
    guest = run_guest_program((ir.drag(10, 10, 40, 40),))
    result = recording.execute_atomic((ir.drag(10, 10, 40, 40),))
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()


def test_recording_drag_records_zero_extent(recording):
    result = recording.execute_atomic((ir.drag(7, 7, 7, 7),))
    (primitive,) = result.backend_primitives
    assert primitive["zero_extent"] is True


def test_recording_glide_matches_the_guest_trace_and_moves_the_cursor(recording):
    guest = run_guest_program((ir.glide_to(88, 99, 0.2),))
    result = recording.execute_atomic((ir.glide_to(88, 99, 0.2),))
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()
    assert recording.cursor_position() == (88, 99)


def test_recording_drag_is_balanced_and_holds_nothing_afterwards(recording):
    recording.execute_atomic((ir.drag(1, 1, 2, 2),))
    assert recording.audit.held_buttons == set()


# --------------------------------------------------------------------------- #
# STANDING INVARIANT: the double must agree with the guest it doubles
#
# ``RecordingTransport`` exists so the executor is testable without a VM, and
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
    "scroll_one_arity": (ir.Operation("scroll", (4,)),),
    "scroll_horizontal": (ir.scroll(5, 0),),
    "scroll_diagonal": (ir.scroll(-6, 7),),
    "scroll_zero": (ir.scroll(0, 0),),
    "ascii_type": (ir.ascii_type("hello world"),),
    "wait": (ir.wait(0.0),),
    "held_stroke": HELD_STROKE,
}


@pytest.mark.parametrize("name", sorted(CANONICAL_ACTIONS))
def test_the_double_and_the_guest_agree_on_the_operation_trace(name, recording):
    """Same kinds, same args, same COUNT -- for every canonical kind."""
    operations = CANONICAL_ACTIONS[name]
    guest = run_guest_program(operations)
    assert guest.payload is not None, guest.stderr
    result = recording.execute_atomic(operations)
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
    result = recording.execute_atomic(operations)
    assert result.ok is guest.payload["ok"], name
    assert result.failure_kind == guest.payload["failure_kind"], name
    assert (
        result.expected_pointer_button_mask
        == guest.payload["expected_pointer_button_mask"]
    ), name
    assert (
        result.observed_pointer_button_mask
        == guest.payload["observed_pointer_button_mask"]
    ), name


@pytest.mark.parametrize("name", sorted(CANONICAL_ACTIONS))
def test_the_double_and_the_guest_agree_on_the_lowering(name, recording):
    operations = CANONICAL_ACTIONS[name]
    guest = run_guest_program(operations)
    result = recording.execute_atomic(operations)
    assert [op.as_dict() for op in result.lowered_operations] == guest.payload[
        "lowered_operations"
    ], name
    assert [op.as_dict() for op in result.semantic_operations] == guest.payload[
        "semantic_operations"
    ], name


def test_the_invariant_covers_every_kind_the_executor_lowers():
    """The table above must not fall behind ``CANONICAL_KINDS``.

    ``raise_for_test`` is excluded deliberately: it is fault injection, and its
    cleanup path is where the double legitimately records more than the guest
    traces (the guest releases keys without tracing the releases).  That
    divergence is documented rather than asserted away.
    """
    from desktop_env.ir import CANONICAL_KINDS

    covered = {
        operation.kind
        for operations in CANONICAL_ACTIONS.values()
        for operation in operations
    }
    expected = set(CANONICAL_KINDS) - {"raise_for_test", "coalesced_type"}
    assert expected <= covered, f"uncovered kinds: {sorted(expected - covered)}"


def test_the_coalesced_type_kind_also_agrees(recording):
    """Separate because it needs the fake GTK stack in the guest subprocess."""
    operations = (ir.coalesced_type("héllo ✓"),)
    guest = run_guest_program(operations, with_gi=True)
    result = recording.execute_atomic(operations)
    assert [(op.kind, list(op.args)) for op in result.operations] == guest.trace()
    assert result.ok is guest.payload["ok"] is True
