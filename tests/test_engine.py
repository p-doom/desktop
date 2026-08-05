"""ITEM 7: ``Engine._accepts_click_backend`` and the receipt it builds.

The capability is decided by INSPECTING the signature once, at construction.  The
alternative -- calling and catching ``TypeError`` -- also swallows a genuine
``TypeError`` raised from *inside* a transport, and then retries with different
arguments.  A real bug would be read as "capability absent".  The critical test
here is therefore the last group: a transport that raises ``TypeError`` from its
body must have that ``TypeError`` propagate.
"""

from __future__ import annotations

import inspect

import pytest

from desktop_env import ir
from desktop_env.execute.engine import Engine, StepReceipt
from desktop_env.execute.guest_program import (
    DIRECT_XTEST_CLICK_BACKEND,
    PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    AtomicExecutionResult,
    ExecutionError,
    InputAudit,
)
from desktop_env.execute.transport import RecordingTransport


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
        self.seen: list[dict] = []

    def cursor_position(self):
        return self._cursor

    def screen_size(self):
        return self._screen


class PinnedBackendTransport(_BaseTransport):
    """A transport that pins one click backend: no ``click_backend`` parameter."""

    def execute_atomic(self, operations):
        self.seen.append({"operations": operations})
        return _result(semantic_operations=operations)


class SwitchableBackendTransport(_BaseTransport):
    def execute_atomic(self, operations, *, click_backend="default"):
        self.seen.append({"operations": operations, "click_backend": click_backend})
        return _result(semantic_operations=operations)


class KwargsTransport(_BaseTransport):
    def execute_atomic(self, operations, **kwargs):
        self.seen.append({"operations": operations, **kwargs})
        return _result(semantic_operations=operations)


class PositionalOnlyTransport(_BaseTransport):
    def execute_atomic(self, operations, click_backend="default"):
        self.seen.append({"operations": operations, "click_backend": click_backend})
        return _result(semantic_operations=operations)


class RaisingTypeErrorTransport(_BaseTransport):
    """Accepts the keyword, then raises a GENUINE ``TypeError`` from its body."""

    def execute_atomic(self, operations, *, click_backend="default"):
        raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")


class RaisingTypeErrorPinnedTransport(_BaseTransport):
    def execute_atomic(self, operations):
        raise TypeError("a real bug deep inside the transport")


class OpaqueTransport(_BaseTransport):
    """``execute_atomic`` whose signature cannot be inspected."""

    execute_atomic = staticmethod(min)  # a C builtin: inspect.signature raises


# --------------------------------------------------------------------------- #
# Capability detection
# --------------------------------------------------------------------------- #


def test_recording_transport_accepts_a_click_backend():
    engine = Engine(RecordingTransport())
    assert engine._click_backend_supported is True


def test_a_transport_pinning_one_backend_does_not_accept_the_keyword():
    engine = Engine(PinnedBackendTransport())
    assert engine._click_backend_supported is False


def test_a_var_keyword_transport_is_treated_as_accepting_it():
    assert Engine(KwargsTransport())._click_backend_supported is True


def test_a_positional_or_keyword_parameter_also_counts():
    assert Engine(PositionalOnlyTransport())._click_backend_supported is True


def test_an_uninspectable_transport_is_assumed_not_to_accept_it():
    assert Engine(OpaqueTransport())._click_backend_supported is False


def test_the_capability_is_decided_once_at_construction_and_cached():
    transport = PinnedBackendTransport()
    engine = Engine(transport)
    assert engine._click_backend_supported is False
    # Swapping in a switchable implementation afterwards must NOT change the
    # cached answer: caching is the documented behaviour, and a per-call probe
    # would be a per-call inspect.signature on the hot path.
    transport.execute_atomic = SwitchableBackendTransport().execute_atomic
    assert engine._click_backend_supported is False


def test_detection_does_not_call_the_transport():
    transport = SwitchableBackendTransport()
    Engine(transport)
    assert transport.seen == []


# --------------------------------------------------------------------------- #
# Dispatch follows the detected capability
# --------------------------------------------------------------------------- #


def test_a_switchable_transport_receives_the_configured_backend():
    transport = SwitchableBackendTransport()
    engine = Engine(transport, click_backend=DIRECT_XTEST_CLICK_BACKEND)
    engine.apply((ir.move_to(1, 2),))
    assert transport.seen[0]["click_backend"] == DIRECT_XTEST_CLICK_BACKEND


def test_the_default_backend_is_the_release_motion_one():
    transport = SwitchableBackendTransport()
    Engine(transport).apply((ir.move_to(1, 2),))
    assert transport.seen[0]["click_backend"] == PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND


def test_a_pinned_transport_is_called_without_the_keyword():
    transport = PinnedBackendTransport()
    engine = Engine(transport, click_backend=DIRECT_XTEST_CLICK_BACKEND)
    receipt = engine.apply((ir.move_to(1, 2),))
    assert receipt.ok is True
    assert transport.seen == [{"operations": (ir.move_to(1, 2),)}]


# --------------------------------------------------------------------------- #
# THE CRITICAL CASE
# --------------------------------------------------------------------------- #


def test_a_genuine_typeerror_from_inside_a_switchable_transport_propagates():
    """It must NOT be read as "capability absent" and retried."""
    engine = Engine(RaisingTypeErrorTransport())
    with pytest.raises(TypeError, match="unsupported operand"):
        engine.apply((ir.move_to(1, 2),))


def test_a_genuine_typeerror_from_inside_a_pinned_transport_propagates():
    engine = Engine(RaisingTypeErrorPinnedTransport())
    with pytest.raises(TypeError, match="a real bug deep inside"):
        engine.apply((ir.move_to(1, 2),))


def test_a_propagated_typeerror_leaves_no_receipt_behind():
    engine = Engine(RaisingTypeErrorTransport())
    with pytest.raises(TypeError):
        engine.apply((ir.move_to(1, 2),))
    assert engine.receipts == []


def test_the_engine_never_wraps_execute_atomic_in_except_typeerror():
    """Structural guard on the mechanism, not just the behaviour."""
    source = inspect.getsource(Engine._execute)
    assert "except" not in source
    assert "TypeError" in inspect.getsource(Engine._accepts_click_backend)


# --------------------------------------------------------------------------- #
# The receipt
# --------------------------------------------------------------------------- #


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
        def execute_atomic(self, operations, *, click_backend="x"):
            return _result(cursor=(9, 9), cursor_before=(7, 7), cursor_after=(9, 9))

    receipt = Engine(LyingTransport()).apply((ir.move_to(1, 1),))
    assert receipt.ok is False
    assert receipt.failure_kind == "verification"
    assert "cursor readback mismatch" in receipt.error
    assert receipt.atomic_state["ok"] is False


def test_readback_verification_can_be_switched_off():
    class LyingTransport(_BaseTransport):
        def execute_atomic(self, operations, *, click_backend="x"):
            return _result(cursor=(9, 9), cursor_before=(7, 7), cursor_after=(9, 9))

    receipt = Engine(LyingTransport(), verify_cursor_readback=False).apply(
        (ir.move_to(1, 1),)
    )
    assert receipt.ok is True
    assert receipt.host_cursor_before is None and receipt.host_cursor_after is None
    assert receipt.cursor_readback_verified is True


def test_a_guest_side_failure_keeps_its_own_failure_kind(recording):
    engine = Engine(recording)
    receipt = engine.apply((ir.Operation("raise_for_test", ("boom",)),))
    assert receipt.ok is False
    assert receipt.failure_kind == "injected"
    assert "boom" in receipt.error


def test_apply_or_raise_carries_the_receipt_as_evidence(recording):
    engine = Engine(recording)
    with pytest.raises(ExecutionError) as caught:
        engine.apply_or_raise((ir.Operation("raise_for_test", ("boom",)),))
    assert caught.value.evidence["failure_kind"] == "injected"
    assert caught.value.evidence["ok"] is False


def test_apply_or_raise_returns_the_receipt_when_the_step_worked(recording):
    engine = Engine(recording)
    assert engine.apply_or_raise((ir.move_to(4, 5),)).ok is True


# --------------------------------------------------------------------------- #
# Geometry / resolution context
# --------------------------------------------------------------------------- #


def test_geometry_comes_from_the_transport(recording):
    geometry = Engine(recording).geometry()
    assert (geometry.desktop_width, geometry.desktop_height) == (1920, 1080)
    assert (geometry.window_width, geometry.window_height) == (0, 0)


def test_resolution_context_returns_geometry_and_cursor_together(recording):
    geometry, cursor = Engine(recording).resolution_context()
    assert cursor == (50, 50)
    assert geometry.desktop_width == 1920


def test_apply_text_passes_live_geometry_and_cursor_into_the_codec(recording):
    seen = {}

    class Codec:
        name = "probe"
        handlers: dict = {}

        def compile(self, text, geometry, cursor):
            seen.update(text=text, geometry=geometry, cursor=cursor)
            return (ir.move_to(11, 22),)

    recording.move_to(100, 100)
    receipt = Engine(recording).apply_text("anything", Codec())
    assert seen["cursor"] == (100, 100)
    assert seen["geometry"].desktop_width == 1920
    assert receipt.cursor_after == (11, 22)


def test_the_engine_source_contains_no_action_name_or_coordinate_flag():
    """The engine's whole reason for existing: it must stay grammar-free."""
    source = inspect.getsource(Engine)
    for forbidden in ("moveRel", "pyautogui", "normalized", "relative", "absolute"):
        assert forbidden not in source, forbidden
