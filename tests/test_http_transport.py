"""``HttpGuiTransport``: the real path to a guest, and its payload validator.

The validator is the interesting half.  It is deliberately BOUNDED -- the
predecessor asserted a whole preregistered X-event ordering on every action, ~900
lines belonging to a click investigation -- so what is left must refuse exactly
the payloads that are structurally unusable or self-contradictory, and accept
everything else.  Each refusal is checked, because a validator that is too strict
rejects good actions and one that is too loose lets a contradictory receipt count
as a success.

The transport is driven against a fake agent on localhost, so none of this needs
a VM.  The programs it sends are real, and one test EXECUTES the program the
transport produced against the fake guest backend, closing the loop from
``execute_atomic`` to a parsed ``AtomicExecutionResult``.
"""

from __future__ import annotations

import functools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from desktop import ir
from desktop.execute.guest_program import (
    ATOMIC_RESULT_PREFIX,
    ATOMIC_SCHEMA_VERSION,
    DIRECT_XTEST_CLICK_BACKEND,
    PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    ExecutionError,
)
from desktop.execute.transport import HttpGuiTransport
from tests.support.guest_runner import run_guest_program

STATE: dict = {}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self._respond()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        STATE.setdefault("requests", []).append(
            {"path": self.path, "body": json.loads(raw) if raw else None}
        )
        self._respond()

    def _respond(self):
        route = STATE.get("routes", {}).get(self.path)
        if route is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        status, payload = route() if callable(route) else route
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def agent_server():
    class _Server(ThreadingHTTPServer):
        daemon_threads = True

    server = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def transport(agent_server):
    STATE.clear()
    STATE["routes"] = {}
    STATE["requests"] = []
    return HttpGuiTransport(f"http://127.0.0.1:{agent_server.server_port}", timeout_s=5.0)


def ok_execute(output: str = "", returncode: int = 0):
    return (200, {"status": "success", "returncode": returncode, "output": output})


def sent_programs() -> list[str]:
    return [
        request["body"]["command"][-1]
        for request in STATE["requests"]
        if request["path"] == "/execute"
    ]


@functools.lru_cache(maxsize=None)
def _guest_stdout(operations, frozen_kwargs) -> str:
    return run_guest_program(operations, **dict(frozen_kwargs)).stdout


def guest_marker(operations, **kwargs) -> str:
    """A REAL marker line, produced by executing the real compiled program.

    Cached: each call is a subprocess, and the same action is asked for by
    several tests.
    """
    frozen = tuple(
        sorted(
            (key, tuple(sorted(value)) if isinstance(value, set) else value)
            for key, value in kwargs.items()
        )
    )
    return _guest_stdout(operations, frozen)


def test_a_trailing_slash_is_stripped_from_the_base_url():
    assert HttpGuiTransport("http://h:5000/").base_url == "http://h:5000"


def test_an_unreachable_guest_raises_an_execution_error():
    with pytest.raises(ExecutionError, match="guest request POST /execute failed"):
        HttpGuiTransport("http://127.0.0.1:1", timeout_s=1.0).execute_argv(["true"])


def test_a_failed_command_is_reported_with_its_status_and_stderr(transport):
    STATE["routes"]["/execute"] = (
        200,
        {"status": "error", "returncode": 3, "error": "no such display"},
    )
    with pytest.raises(ExecutionError, match="guest command failed"):
        transport.execute_argv(["true"])


def test_a_failed_command_can_be_returned_unchecked(transport):
    STATE["routes"]["/execute"] = (200, {"status": "error", "returncode": 3})
    assert transport.execute_argv(["true"], check=False)["returncode"] == 3


def test_a_non_object_execute_response_is_refused(transport):
    STATE["routes"]["/execute"] = (200, ["nope"])
    with pytest.raises(ExecutionError, match="non-object"):
        transport.execute_argv(["true"])


def test_an_unroutable_path_is_an_execution_error(transport):
    """A 404 from the agent arrives as an HTTPError whose body is not JSON."""
    STATE["routes"]["/execute"] = None
    with pytest.raises(ExecutionError, match="guest request POST /execute failed"):
        transport.execute_argv(["true"])


def test_every_input_goes_out_as_python_dash_c(transport):
    """The guest agent runs a subprocess; it does not eval.  That is why the whole
    action has to be compiled into one program."""
    STATE["routes"]["/execute"] = ok_execute()
    transport.execute_pyautogui("pyautogui.moveTo(1, 2)")
    (request,) = STATE["requests"]
    assert request["body"]["shell"] is False
    assert request["body"]["command"][:2] == ["python", "-c"]
    assert request["body"]["command"][2].startswith("import pyautogui;")


def test_cursor_position_is_parsed_as_a_pair(transport):
    STATE["routes"]["/cursor_position"] = (200, [11, 22])
    assert transport.cursor_position() == (11, 22)


@pytest.mark.parametrize("payload", [{"x": 1, "y": 2}, [1], [1, 2, 3], "x", None])
def test_a_malformed_cursor_position_is_refused(transport, payload):
    STATE["routes"]["/cursor_position"] = (200, payload)
    with pytest.raises(ExecutionError, match="invalid cursor position"):
        transport.cursor_position()


def test_screen_size_is_parsed(transport):
    STATE["routes"]["/screen_size"] = (200, {"width": 1920, "height": 1080})
    assert transport.screen_size() == (1920, 1080)


def test_a_malformed_screen_size_is_refused(transport):
    STATE["routes"]["/screen_size"] = (200, [1920, 1080])
    with pytest.raises(ExecutionError, match="invalid screen size"):
        transport.screen_size()


def test_move_to_clamps_against_the_live_screen_size(transport):
    STATE["routes"]["/screen_size"] = (200, {"width": 1920, "height": 1080})
    STATE["routes"]["/execute"] = ok_execute()
    transport.move_to(5000, -20)
    assert "pyautogui.moveTo(1919, 0)" in sent_programs()[-1]
    assert transport.audit.operations[-1] == ir.move_to(1919, 0)


def test_glide_to_clamps_and_records_its_duration(transport):
    STATE["routes"]["/screen_size"] = (200, {"width": 1920, "height": 1080})
    STATE["routes"]["/execute"] = ok_execute()
    transport.glide_to(5000, 500, 99.0)
    assert "duration=10.0" in sent_programs()[-1]
    assert transport.audit.operations[-1] == ir.Operation("glide_to", (1919, 500, 10.0))


def test_mouse_down_and_up_maintain_the_held_set(transport):
    STATE["routes"]["/execute"] = ok_execute()
    transport.mouse_down("LMB")
    assert transport.audit.held_buttons == {"left"}
    with pytest.raises(ExecutionError, match="button already held: left"):
        transport.mouse_down("left")
    transport.mouse_up(1)
    assert transport.audit.held_buttons == set()
    with pytest.raises(ExecutionError, match="button not held: left"):
        transport.mouse_up("left")


def test_an_unknown_button_never_reaches_the_guest(transport):
    STATE["routes"]["/execute"] = ok_execute()
    from desktop.execute.keymap import KeymapError

    with pytest.raises(KeymapError):
        transport.mouse_down("nonsense")
    assert sent_programs() == []


def test_scroll_records_the_two_axis_form(transport):
    STATE["routes"]["/execute"] = ok_execute()
    transport.scroll(-4)
    assert "pyautogui.scroll(-4)" in sent_programs()[-1]
    assert transport.audit.operations[-1] == ir.scroll(0, -4)
    assert transport.audit.scroll_total == -4


def test_hscroll_records_the_horizontal_axis(transport):
    STATE["routes"]["/execute"] = ok_execute()
    transport.hscroll(7)
    assert "pyautogui.hscroll(7)" in sent_programs()[-1]
    assert transport.audit.operations[-1] == ir.scroll(7, 0)


def test_a_key_chord_presses_in_order_and_releases_in_reverse(transport):
    STATE["routes"]["/execute"] = ok_execute()
    transport.key_chord(["CTRL", "KeyA"])
    program = sent_programs()[-1]
    assert program.index("keyDown('ctrl')") < program.index("keyDown('a')")
    assert program.index("keyDown('a')") < program.index("keyUp('a')")
    assert program.index("keyUp('a')") < program.index("keyUp('ctrl')")


def test_an_empty_key_chord_is_refused(transport):
    with pytest.raises(ExecutionError, match="empty key chord"):
        transport.key_chord([])


def test_coalesced_type_sends_the_gtk_clipboard_program(transport):
    STATE["routes"]["/execute"] = ok_execute()
    transport.coalesced_type("héllo")
    program = sent_programs()[-1]
    assert "gi.require_version('Gtk','3.0')" in program
    assert transport.audit.typed_texts == ["héllo"]


def test_wait_clamps_and_records(transport):
    transport.wait(-5)
    transport.wait(99)
    assert [op.args[0] for op in transport.audit.operations] == [0.0, 10.0]


# --------------------------------------------------------------------------- #
# execute_atomic, end to end against a real compiled program
# --------------------------------------------------------------------------- #


def test_a_real_atomic_action_round_trips_through_the_transport(transport):
    """The full loop: compile, "send", parse the guest's real marker."""
    operations = (ir.move_to(300, 400), ir.click("left"))
    stdout = guest_marker(operations)
    STATE["routes"]["/execute"] = ok_execute(stdout)
    result = transport.execute_atomic(operations)
    assert result.ok is True
    assert result.error is None and result.failure_kind is None
    assert result.cursor_after == (300, 400)
    assert result.guest_process_count == 1
    assert result.semantic_operations == operations
    assert [op.kind for op in result.lowered_operations] == ["move_to", "click"]
    assert transport.audit.held_buttons == set()


def test_the_transport_sends_exactly_one_program_per_action(transport):
    operations = (ir.move_to(1, 2),)
    STATE["routes"]["/execute"] = ok_execute(guest_marker(operations))
    transport.execute_atomic(operations)
    assert len(sent_programs()) == 1
    assert sent_programs()[0].startswith("import json, sys, traceback")


def test_a_held_button_is_absorbed_into_the_audit(transport):
    operations = (ir.mouse_down("left"),)
    STATE["routes"]["/execute"] = ok_execute(guest_marker(operations))
    result = transport.execute_atomic(operations)
    assert result.ok is True
    assert transport.audit.held_buttons == {"left"}


def test_held_keys_are_absorbed_on_success_and_dropped_on_failure(transport):
    holding = (ir.key_down("ControlLeft"),)
    STATE["routes"]["/execute"] = ok_execute(guest_marker(holding))
    transport.execute_atomic(holding)
    assert transport.audit.held_keys == {"ctrlleft"}
    failing = (ir.Operation("raise_for_test", ("boom",)),)
    STATE["routes"]["/execute"] = ok_execute(
        guest_marker(failing, initial_keys={"ctrlleft"}), returncode=1
    )
    result = transport.execute_atomic(failing)
    assert result.ok is False
    assert transport.audit.held_keys == set(), "a failed action must not leave keys held"


def test_a_failing_action_is_reported_not_raised(transport):
    operations = (ir.Operation("raise_for_test", ("injected failure",)),)
    STATE["routes"]["/execute"] = ok_execute(guest_marker(operations), returncode=1)
    result = transport.execute_atomic(operations)
    assert result.ok is False
    assert result.failure_kind == "injected"
    assert "injected failure" in result.error
    assert result.guest_returncode == 1


def test_typed_text_is_absorbed_into_the_audit(transport):
    operations = (ir.ascii_type("hello"),)
    STATE["routes"]["/execute"] = ok_execute(guest_marker(operations))
    transport.execute_atomic(operations)
    assert transport.audit.typed_texts == ["hello"]


def test_the_click_backend_is_threaded_into_the_compiled_program(transport):
    operations = (ir.click("right"),)
    stdout = guest_marker(operations, click_backend=DIRECT_XTEST_CLICK_BACKEND)
    STATE["routes"]["/execute"] = ok_execute(stdout)
    result = transport.execute_atomic(
        operations, click_backend=DIRECT_XTEST_CLICK_BACKEND
    )
    assert result.click_backend == DIRECT_XTEST_CLICK_BACKEND
    assert DIRECT_XTEST_CLICK_BACKEND in sent_programs()[-1]


def _payload(**overrides) -> dict:
    base = {
        "ok": True,
        "cursor": [1, 2],
        "cursor_before": [0, 0],
        "cursor_after": [1, 2],
        "_de_schema": ATOMIC_SCHEMA_VERSION,
        "pointer_button_mask": 0,
        "observed_pointer_button_mask": 0,
        "expected_pointer_button_mask": 0,
        "guest_process_count": 1,
        "cleanup_attempted": False,
        "error": None,
        "failure_kind": None,
        "operations": [],
        "backend_primitives": [],
        "x_event_sync_evidence": [],
        "x_sync_attempt_evidence": [],
        "click_backend": PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
        "x_injection_evidence": [],
        "x_injection_timestamps": [],
        "final_pointer_readback": {},
        "attempt_hook_restore_errors": [],
        "passive_x_observer": {},
    }
    base.update(overrides)
    return base


def _parse(transport, payload=None, *, output=None, returncode=0):
    if output is None:
        output = ATOMIC_RESULT_PREFIX + json.dumps(payload)
    return transport._parse_atomic_payload(
        {"status": "success", "returncode": returncode, "output": output},
        operations=(),
        click_backend=PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    )


def test_a_well_formed_payload_is_accepted(transport):
    assert _parse(transport, _payload()).ok is True


def test_a_payload_with_no_stdout_is_refused(transport):
    with pytest.raises(ExecutionError, match="returned no stdout") as caught:
        transport._parse_atomic_payload(
            {"status": "success", "returncode": 0, "output": None},
            operations=(),
            click_backend=PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
        )
    assert caught.value.evidence["schema_version"] == (
        "desktop_env_atomic_output_failure_v1"
    )


@pytest.mark.parametrize("count", [0, 2, 3])
def test_a_wrong_number_of_markers_is_refused(transport, count):
    line = ATOMIC_RESULT_PREFIX + json.dumps(_payload())
    output = "\n".join([line] * count) if count else "chatter with no marker"
    with pytest.raises(ExecutionError, match="marker count was"):
        _parse(transport, output=output)


def test_guest_chatter_around_the_marker_is_tolerated(transport):
    """Guest stdout is shared with GTK warnings and X11 noise."""
    line = ATOMIC_RESULT_PREFIX + json.dumps(_payload())
    output = f"Gtk-WARNING **: blah\n{line}\nlibGL error: nope\n"
    assert _parse(transport, output=output).ok is True


def test_an_invalid_json_marker_is_refused(transport):
    with pytest.raises(ExecutionError, match="invalid JSON") as caught:
        _parse(transport, output=ATOMIC_RESULT_PREFIX + "{not json}")
    assert "raw_marker" in caught.value.evidence


@pytest.mark.parametrize("schema", [0, 2, None, "1"])
def test_a_wrong_schema_version_is_refused(transport, schema):
    with pytest.raises(ExecutionError, match="unexpected schema"):
        _parse(transport, _payload(_de_schema=schema))


def test_a_non_object_payload_is_refused(transport):
    with pytest.raises(ExecutionError, match="unexpected schema"):
        _parse(transport, output=ATOMIC_RESULT_PREFIX + json.dumps([1, 2]))


@pytest.mark.parametrize("name", ["cursor", "cursor_before", "cursor_after"])
@pytest.mark.parametrize("value", [[1], [1, 2, 3], "1,2", None, {}])
def test_a_malformed_cursor_pair_is_refused(transport, name, value):
    with pytest.raises(ExecutionError, match=f"invalid {name}"):
        _parse(transport, _payload(**{name: value}))


def test_a_cursor_that_disagrees_with_its_own_alias_is_refused(transport):
    """``cursor`` and ``cursor_after`` are the same readback; disagreement means
    the payload contradicts itself."""
    with pytest.raises(ExecutionError, match="alias/readback mismatch"):
        _parse(transport, _payload(cursor=[1, 2], cursor_after=[3, 4]))


@pytest.mark.parametrize("kind", ["weird", "", "Verification", 42])
def test_an_unknown_failure_kind_is_refused(transport, kind):
    with pytest.raises(ExecutionError, match="invalid failure kind"):
        _parse(transport, _payload(ok=False, error="x", failure_kind=kind))


@pytest.mark.parametrize("kind", ["verification", "infrastructure", "injected"])
def test_every_documented_failure_kind_is_accepted(transport, kind):
    result = _parse(transport, _payload(ok=False, error="boom", failure_kind=kind))
    assert result.failure_kind == kind
    assert result.error == "boom"


@pytest.mark.parametrize(
    "overrides",
    [
        {"ok": True, "error": "boom", "failure_kind": None},
        {"ok": True, "error": None, "failure_kind": "verification"},
        {"ok": False, "error": None, "failure_kind": None},
        {"ok": False, "error": "boom", "failure_kind": None},
        {"ok": False, "error": None, "failure_kind": "injected"},
    ],
)
def test_a_self_contradictory_failure_classification_is_refused(transport, overrides):
    """``ok``, ``error`` and ``failure_kind`` must agree, or a failed action could
    be counted as a success."""
    with pytest.raises(ExecutionError, match="self-contradictory"):
        _parse(transport, _payload(**overrides))


@pytest.mark.parametrize("count", [0, 2, -1])
def test_more_than_one_guest_process_is_refused(transport, count):
    """One process per action is the module's central claim."""
    with pytest.raises(ExecutionError, match="exactly one guest process"):
        _parse(transport, _payload(guest_process_count=count))


@pytest.mark.parametrize("count", ["1", "abc", None, [1], 1.0, True])
def test_a_non_integer_guest_process_count_is_refused(transport, count):
    """It used to go through a bare ``int()``, so a non-numeric value escaped as a
    ``ValueError`` carrying no payload instead of a refusal carrying one -- and
    ``"1"`` was silently coerced into passing the module's central claim.  ``True``
    is here because ``True == 1``, so a JSON ``true`` would otherwise read as
    exactly one guest process."""
    with pytest.raises(ExecutionError, match="exactly one guest process") as caught:
        _parse(transport, _payload(guest_process_count=count))
    assert caught.value.evidence["raw_payload"]["guest_process_count"] == count


def test_an_absent_guest_process_count_is_refused(transport):
    """Unlike the masks, the ``-1`` default here is the FAILURE: an absent count
    must fall through to this check rather than be read as a real count."""
    payload = _payload()
    del payload["guest_process_count"]
    with pytest.raises(ExecutionError, match="exactly one guest process"):
        _parse(transport, payload)


def test_a_drifted_click_backend_is_refused(transport):
    with pytest.raises(ExecutionError, match="click backend drifted") as caught:
        _parse(transport, _payload(click_backend=DIRECT_XTEST_CLICK_BACKEND))
    assert caught.value.evidence["observed"] == DIRECT_XTEST_CLICK_BACKEND


def test_unrestored_x_hooks_are_refused(transport):
    """The program monkeypatches XTest inside the guest; leaving the hooks in
    place would silently corrupt every later action in that interpreter."""
    with pytest.raises(ExecutionError, match="X attempt hooks were not restored"):
        _parse(transport, _payload(attempt_hook_restore_errors=["sync restore failed"]))


@pytest.mark.parametrize(
    "name",
    [
        "operations",
        "backend_primitives",
        "x_event_sync_evidence",
        "x_sync_attempt_evidence",
        "x_injection_evidence",
        "x_injection_timestamps",
    ],
)
@pytest.mark.parametrize("value", ["not a list", [1, 2], [None], {}])
def test_a_malformed_evidence_list_is_refused(transport, name, value):
    with pytest.raises(ExecutionError, match=f"invalid {name}"):
        _parse(transport, _payload(**{name: value}))


def test_the_traced_operations_are_rebuilt_as_operations(transport):
    payload = _payload(
        operations=[
            {"kind": "move_to", "args": [4, 5]},
            {"kind": "scroll", "args": [0, 3]},
        ]
    )
    result = _parse(transport, payload)
    assert result.operations == (ir.move_to(4, 5), ir.scroll(0, 3))


def test_missing_optional_evidence_defaults_rather_than_failing(transport):
    """The validator is bounded on purpose: absent optional evidence is fine."""
    payload = _payload()
    for optional in (
        "backend_primitives",
        "x_event_sync_evidence",
        "x_injection_evidence",
        "final_pointer_readback",
        "passive_x_observer",
        "attempt_hook_restore_errors",
    ):
        payload.pop(optional)
    result = _parse(transport, payload)
    assert result.ok is True
    assert result.backend_primitives == ()
    assert result.final_pointer_readback == {}


MASK_FIELDS = (
    "pointer_button_mask",
    "observed_pointer_button_mask",
    "expected_pointer_button_mask",
)


@pytest.mark.parametrize("name", MASK_FIELDS)
def test_an_absent_pointer_mask_is_refused_rather_than_defaulted(transport, name):
    """The masks are REQUIRED, unlike the optional evidence above.

    They used to default to -1, which is the guest's "never read" sentinel: an
    absent mask became a sentinel nobody reported, on a payload still free to
    claim ``ok`` -- and the held-button audit then read every button as held with
    no failure signal anywhere in the receipt.
    """
    payload = _payload()
    del payload[name]
    with pytest.raises(ExecutionError, match=f"invalid {name}"):
        _parse(transport, payload)


@pytest.mark.parametrize("name", MASK_FIELDS)
@pytest.mark.parametrize("value", [None, "0", 0.0, True, [0]])
def test_a_non_integer_pointer_mask_is_refused(transport, name, value):
    """``True`` is in here on purpose: ``isinstance(True, int)`` holds, so an
    unguarded read turns ``true`` into 1 -- a left button nobody pressed."""
    with pytest.raises(ExecutionError, match=f"invalid {name}"):
        _parse(transport, _payload(**{name: value}))


@pytest.mark.parametrize("name", MASK_FIELDS)
def test_a_reported_minus_one_pointer_mask_is_kept(transport, name):
    """-1 is legal as a VALUE: ``observed_pointer_button_mask`` is -1 whenever the
    action dies before its verification readback, so refusing it would refuse real
    payloads.  Only its ABSENCE is refused."""
    assert getattr(_parse(transport, _payload(**{name: -1})), name) == -1


def test_the_guest_returncode_is_carried_onto_the_result(transport):
    result = _parse(
        transport, _payload(ok=False, error="x", failure_kind="injected"), returncode=1
    )
    assert result.guest_returncode == 1


def test_the_raw_marker_is_preserved_for_auditing(transport):
    payload = _payload()
    result = _parse(transport, payload)
    assert result.raw_result_marker.startswith(ATOMIC_RESULT_PREFIX)
    assert json.loads(result.raw_result_marker[len(ATOMIC_RESULT_PREFIX) :]) == payload


def test_the_result_is_json_safe(transport):
    result = _parse(transport, _payload())
    assert json.loads(json.dumps(result.as_dict()))["ok"] is True


def test_the_no_readback_sentinel_is_not_absorbed_as_held_buttons(transport):
    """``pointer_button_mask`` is -1 when the guest's final readback never ran.

    ``-1 & mask`` is truthy for EVERY button, so deriving the held set from the
    sentinel holds all three -- and the held set is what the NEXT program is
    compiled to expect, so every later action fails verification against buttons
    nobody pressed.  Absorbing it must not RAISE either: ``Engine.apply`` turns a
    failed action into a receipt, and an exception out of here would delete the
    receipt for the one step that failed.  The sentinel is not swallowed -- it
    stays on the result; that it reaches the published receipt is asserted in
    ``test_engine``.
    """
    STATE["routes"]["/execute"] = ok_execute(
        ATOMIC_RESULT_PREFIX
        + json.dumps(
            _payload(
                ok=False,
                error="final pointer readback failed: RuntimeError: display gone",
                failure_kind="infrastructure",
                pointer_button_mask=-1,
                final_pointer_readback={"attempted": True, "success": False},
            )
        ),
        returncode=1,
    )
    result = transport.execute_atomic((ir.move_to(1, 2),))
    assert transport.audit.held_buttons == set()
    assert result.pointer_button_mask == -1
    assert result.ok is False
    assert result.failure_kind == "infrastructure"

    follow_up = (ir.move_to(3, 4),)
    STATE["routes"]["/execute"] = ok_execute(guest_marker(follow_up))
    transport.execute_atomic(follow_up)
    assert "_de_expected_initial_mask=0" in sent_programs()[-1]
