"""The engine's single transport contract and the receipt it builds."""

from __future__ import annotations

import inspect

import pytest

from desktop import ir
from desktop.execute.engine import Engine, StepReceipt
from desktop.execute.guest_program import (
    AtomicExecutionResult,
    ExecutionError,
    InputAudit,
)
from desktop.execute.transport import RecordingTransport


def _result(**overrides) -> AtomicExecutionResult:
    base = dict(
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
        operations=(),
        semantic_operations=(),
        lowered_operations=(),
    )
    base.update(overrides)
    return AtomicExecutionResult(**base)


class _BaseTransport:
    """Everything ``GuiTransport`` needs, minus ``execute_atomic``."""

    def __init__(self, cursor=(0, 0), screen=(1920, 1080)) -> None:
        self.audit = InputAudit()
        self._cursor = cursor
        self._screen = screen

    def cursor_position(self):
        return self._cursor

    def screen_size(self):
        return self._screen


class RaisingTypeErrorTransport(_BaseTransport):
    def execute_atomic(self, operations):
        raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")


def test_recording_transport_satisfies_the_contract():
    assert Engine(RecordingTransport()).transport is not None


def test_a_genuine_typeerror_from_inside_a_transport_propagates():
    engine = Engine(RaisingTypeErrorTransport())
    with pytest.raises(TypeError, match="unsupported operand"):
        engine.apply((ir.move_to(1, 2),))


def test_a_propagated_typeerror_leaves_no_receipt_behind():
    engine = Engine(RaisingTypeErrorTransport())
    with pytest.raises(TypeError):
        engine.apply((ir.move_to(1, 2),))
    assert engine.receipts == []


def test_the_engine_never_retries_a_transport_typeerror():
    source = inspect.getsource(Engine._execute)
    assert "except" not in source


def test_a_successful_apply_builds_a_verified_receipt(recording):
    engine = Engine(recording)
    receipt = engine.apply((ir.move_to(200, 300),))
    assert isinstance(receipt, StepReceipt)
    assert receipt.ok is True
    assert receipt.error is None and receipt.failure_kind is None
    assert receipt.cursor_before == (50, 50) and receipt.cursor_after == (200, 300)
    assert receipt.host_cursor_before == (50, 50)
    assert receipt.host_cursor_after == (200, 300)
    assert receipt.cursor_readback_verified is True
    assert receipt.requested_operations == (ir.move_to(200, 300),)
    assert engine.receipts == [receipt]


def test_the_receipt_is_json_safe():
    import json

    engine = Engine(RecordingTransport())
    receipt = engine.apply((ir.drag(1, 1, 2, 2),))
    assert json.loads(json.dumps(receipt.as_dict()))["ok"] is True


def test_the_executed_cursor_delta_is_recorded(recording):
    engine = Engine(recording)
    receipt = engine.apply((ir.move_to(60, 70),))
    assert receipt.atomic_state["executed_cursor_delta"] == [10, 20]


def test_a_host_guest_cursor_disagreement_fails_the_step():
    """The failure mode a delta-resolving grammar cannot otherwise detect."""

    class LyingTransport(_BaseTransport):
        def execute_atomic(self, operations):
            return _result(cursor=(9, 9), cursor_before=(7, 7), cursor_after=(9, 9))

    receipt = Engine(LyingTransport()).apply((ir.move_to(1, 1),))
    assert receipt.ok is False
    assert receipt.failure_kind == "verification"
    assert "cursor readback mismatch" in receipt.error
    assert receipt.atomic_state["ok"] is False


def test_readback_verification_cannot_be_switched_off():
    """It used to be, and the receipt then claimed a check that had not run:
    ``cursor_readback_verified: true`` beside two ``None`` host cursors."""
    with pytest.raises(TypeError):
        Engine(_BaseTransport(), verify_cursor_readback=False)


def test_a_guest_side_failure_keeps_its_own_failure_kind(recording):
    engine = Engine(recording)
    receipt = engine.apply((ir.Operation("raise_for_test", ("boom",)),))
    assert receipt.ok is False
    assert receipt.failure_kind == "injected"
    assert "boom" in receipt.error


def test_a_step_whose_final_readback_never_ran_still_leaves_a_receipt():
    """The receipt is the authoritative published account of an action, so the
    step whose final pointer readback never ran is exactly the one that must not
    lose it -- which is why absorbing the ``-1`` sentinel reports rather than
    raises.  The raw sentinel reaches the receipt, so it stays measurable."""

    class NoFinalReadbackTransport(_BaseTransport):
        def execute_atomic(self, operations):
            return _result(
                ok=False,
                pointer_button_mask=-1,
                error="final pointer readback failed: RuntimeError: display gone",
                failure_kind="infrastructure",
            )

    engine = Engine(NoFinalReadbackTransport())
    receipt = engine.apply((ir.move_to(1, 2),))
    assert engine.receipts == [receipt]
    assert receipt.ok is False
    assert receipt.failure_kind == "infrastructure"
    assert "final pointer readback failed" in receipt.error
    assert receipt.atomic_state["pointer_button_mask"] == -1


def test_apply_or_raise_carries_the_receipt_as_evidence(recording):
    engine = Engine(recording)
    with pytest.raises(ExecutionError) as caught:
        engine.apply_or_raise((ir.Operation("raise_for_test", ("boom",)),))
    assert caught.value.evidence["failure_kind"] == "injected"
    assert caught.value.evidence["ok"] is False


def test_apply_or_raise_returns_the_receipt_when_the_step_worked(recording):
    engine = Engine(recording)
    assert engine.apply_or_raise((ir.move_to(4, 5),)).ok is True


def test_geometry_comes_from_the_transport(recording):
    geometry = Engine(recording).geometry()
    assert (geometry.desktop_width, geometry.desktop_height) == (1920, 1080)
    assert (geometry.window_width, geometry.window_height) == (0, 0)



def test_the_engine_source_contains_no_action_name_or_coordinate_flag():
    """The engine's whole reason for existing: it must stay grammar-free."""
    source = inspect.getsource(Engine)
    for forbidden in ("moveRel", "pyautogui", "normalized", "relative", "absolute"):
        assert forbidden not in source, forbidden
