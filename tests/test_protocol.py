from __future__ import annotations

import re
from pathlib import Path

import pytest

from desktop import ir
from desktop.execute.protocol import (
    ACTION_CONTRACT,
    ACTION_SCHEMA_VERSION,
    build_action_request,
    lower_guest_operations,
)
from desktop.vm.client import ACTION_EXECUTOR_PATH, GuestCommandResult
from desktop.vm.session import GUEST_JSON_MARKER, GuestScript, SessionError
from tests.support.guest_runner import run_guest_program

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "desktop"

ALL_CANONICAL_OPERATIONS = (
    ir.move_to(11, 12),
    ir.glide_to(13, 14, 0.01),
    ir.drag(15, 16, 17, 18),
    ir.click("left"),
    ir.mouse_down("right"),
    ir.mouse_up("right"),
    ir.key_down("ControlLeft"),
    ir.key_up("ControlLeft"),
    ir.scroll(0, 2),
    ir.ascii_type("abc"),
    ir.wait(0.0),
)


def test_no_old_rung1_identifier_survives():
    stale = re.compile(r"RUNG1A|_r1a_|_r1a\b|\br1a_", re.IGNORECASE)
    offenders = []
    for path in PACKAGE_ROOT.rglob("*"):
        if path.suffix not in {".py", ".def", ".md", ".patch"}:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if stale.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert offenders == []


def test_the_action_request_has_one_exact_versioned_shape():
    request, _, _ = build_action_request(
        ALL_CANONICAL_OPERATIONS, initial_buttons=set(), initial_keys=set()
    )
    assert request["contract"] == ACTION_CONTRACT
    assert request["schema_version"] == ACTION_SCHEMA_VERSION
    assert request["operations"] == [
        operation.as_dict() for operation in ALL_CANONICAL_OPERATIONS
    ]
    assert request["lowered_operations"] == [
        operation.as_dict() for operation in lower_guest_operations(ALL_CANONICAL_OPERATIONS)
    ]


def test_the_executor_is_a_real_file_not_generated_source():
    source = ACTION_EXECUTOR_PATH.read_text()
    assert "json.load(sys.stdin)" in source
    assert "from Xlib import X" in source
    assert "pyautogui" not in source
    assert "compile_atomic_guest_program" not in source


def test_every_canonical_kind_executes_through_the_installed_file():
    run = run_guest_program(ALL_CANONICAL_OPERATIONS)
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True
    assert run.request["operations"] == [
        operation.as_dict() for operation in ALL_CANONICAL_OPERATIONS
    ]


def test_guest_script_reads_only_its_marker_line():
    script = GuestScript(client=None)  # type: ignore[arg-type]
    result = GuestCommandResult(
        0,
        "noise\n" + GUEST_JSON_MARKER + '{"root":"/tmp/x"}\nmore noise',
        "",
    )
    assert script.parse(result) == {"root": "/tmp/x"}


def test_guest_script_refuses_an_ambiguous_marker():
    script = GuestScript(client=None)  # type: ignore[arg-type]
    result = GuestCommandResult(
        0,
        GUEST_JSON_MARKER + "{}\n" + GUEST_JSON_MARKER + "{}",
        "",
    )
    with pytest.raises(SessionError, match="2 result markers"):
        script.parse(result)


@pytest.mark.parametrize("operation", ALL_CANONICAL_OPERATIONS)
def test_an_operation_round_trips_through_its_json_view(operation):
    assert ir.Operation.from_dict(operation.as_dict()) == operation


@pytest.mark.parametrize("payload", [{"kind": "move_to"}, {"args": [1, 2]}])
def test_a_truncated_operation_payload_is_refused(payload):
    with pytest.raises(KeyError):
        ir.Operation.from_dict(payload)
