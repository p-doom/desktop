"""ITEM 3: the ``RUNG1A_*`` -> ``DESKTOP_ENV_*`` / ``_r1a_*`` -> ``_de_*`` rename.

~200 mechanical, unverified occurrences.  Two failure directions, and only one of
them is loud:

* a surviving old name is *usually* loud (a NameError inside the guest program),
  but a surviving old name in a MARKER string is silent -- the host then waits for
  a marker the guest never sends, forever;
* a half-renamed marker in the other direction is equally silent: the guest emits
  ``DESKTOP_ENV_x`` and the host still greps for ``RUNG1A_x``.

So this module (a) greps the tree in both directions, (b) checks marker parity by
matching every emitter against its parser, and (c) EXECUTES a program containing
every canonical kind, which is the only way a surviving ``_r1a_`` local would show
up at all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pixeldesk import ir
from pixeldesk.execute import guest_program as GP
from pixeldesk.execute.guest_program import ATOMIC_RESULT_PREFIX, compile_atomic_guest_program
from pixeldesk.vm.session import GUEST_JSON_MARKER, GuestScript
from tests.support.guest_runner import run_guest_program

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "pixeldesk"

#: Every kind the executor claims to lower, with a representative payload.
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


def _source_files() -> list[Path]:
    return sorted(
        list(PACKAGE_ROOT.rglob("*.py"))
        + list(PACKAGE_ROOT.rglob("*.def"))
        + list(PACKAGE_ROOT.rglob("*.md"))
        + [REPO_ROOT / "pyproject.toml", REPO_ROOT / "README.md"]
    )


def test_no_old_rung1a_or_r1a_name_survives_anywhere():
    stale = re.compile(r"RUNG1A|_r1a_|_r1a\b|\br1a_", re.IGNORECASE)
    offenders = []
    for path in _source_files():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if stale.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert offenders == [], "stale pre-rename names survive:\n" + "\n".join(offenders)


def test_the_generated_guest_program_uses_only_the_new_local_prefix():
    program, _ = compile_atomic_guest_program(
        ALL_CANONICAL_OPERATIONS, initial_buttons=set(), initial_keys=set()
    )
    assert "_r1a" not in program
    assert "RUNG1A" not in program
    assert program.count("_de_") > 100, "the renamed locals should dominate the program"


def test_every_desktop_env_marker_constant_is_both_emitted_and_parsed():
    """The two wire markers in the package, matched emitter to parser."""
    assert ATOMIC_RESULT_PREFIX == "DESKTOP_ENV_ATOMIC_RESULT="
    assert GUEST_JSON_MARKER == "DESKTOP_ENV_JSON="
    transport_source = (PACKAGE_ROOT / "execute" / "transport.py").read_text()
    session_source = (PACKAGE_ROOT / "vm" / "session.py").read_text()
    # The parsers reference the CONSTANT, never a re-spelled literal -- which is
    # what makes a half-rename impossible rather than merely unlikely.
    assert "ATOMIC_RESULT_PREFIX" in transport_source
    assert "DESKTOP_ENV_ATOMIC_RESULT" not in transport_source
    assert "self.marker" in session_source


def test_no_desktop_env_marker_literal_is_spelled_out_twice():
    """A marker spelled as a literal in two files is a half-rename waiting to
    happen; every one must come from a single constant."""
    literals: dict[str, list[str]] = {}
    pattern = re.compile(r"[\"'](DESKTOP_ENV_[A-Z_]*=)[\"']")
    for path in PACKAGE_ROOT.rglob("*.py"):
        for match in pattern.finditer(path.read_text()):
            literals.setdefault(match.group(1), []).append(path.name)
    for marker, files in literals.items():
        assert len(set(files)) == 1, f"{marker} is spelled literally in {sorted(set(files))}"


def test_the_atomic_result_marker_round_trips_emitter_to_parser():
    run = run_guest_program((ir.move_to(3, 4),))
    assert run.marker_count == 1, run.stdout
    marker_line = next(
        line for line in run.stdout.splitlines() if line.startswith(ATOMIC_RESULT_PREFIX)
    )
    payload = json.loads(marker_line[len(ATOMIC_RESULT_PREFIX) :])
    assert payload["_de_schema"] == GP.ATOMIC_SCHEMA_VERSION


def test_the_schema_key_the_guest_emits_is_the_key_the_host_checks():
    """``_de_schema`` is renamed payload state; a mismatch rejects every action."""
    run = run_guest_program((ir.move_to(1, 1),))
    assert "_de_schema" in run.payload
    from pixeldesk.execute.transport import HttpGuiTransport

    transport = HttpGuiTransport("http://127.0.0.1:1")
    result = transport._parse_atomic_payload(
        {"output": run.stdout, "returncode": 0, "status": "success"},
        operations=(ir.move_to(1, 1),),
        click_backend=GP.PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
    )
    assert result.ok is True
    assert result.cursor_after == (1, 1)


def test_a_wrongly_named_schema_key_is_rejected_rather_than_ignored():
    from pixeldesk.execute.guest_program import ExecutionError
    from pixeldesk.execute.transport import HttpGuiTransport

    transport = HttpGuiTransport("http://127.0.0.1:1")
    stdout = ATOMIC_RESULT_PREFIX + json.dumps({"_r1a_schema": 1, "ok": True})
    with pytest.raises(ExecutionError, match="unexpected schema"):
        transport._parse_atomic_payload(
            {"output": stdout, "returncode": 0, "status": "success"},
            operations=(),
            click_backend=GP.PYAUTOGUI_RELEASE_MOTION_CLICK_BACKEND,
        )


def test_the_guest_json_marker_round_trips_through_guest_script():
    """``GuestScript`` greps for its own marker; emitter and parser must agree."""
    script = GuestScript(client=None)  # type: ignore[arg-type]
    stdout = "Gtk-WARNING chatter\n" + GUEST_JSON_MARKER + '{"root": "/tmp/x"}\nmore noise'
    assert script.parse({"output": stdout}) == {"root": "/tmp/x"}


def test_guest_script_refuses_a_marker_it_cannot_find_exactly_once():
    from pixeldesk.vm.session import SessionError

    script = GuestScript(client=None)  # type: ignore[arg-type]
    with pytest.raises(SessionError, match="0 result markers"):
        script.parse({"output": "RUNG1A_JSON={}"})
    with pytest.raises(SessionError, match="2 result markers"):
        script.parse({"output": GUEST_JSON_MARKER + "{}\n" + GUEST_JSON_MARKER + "{}"})


def test_the_resolve_guest_root_program_prints_the_marker_the_parser_wants():
    """Generated guest source, checked against the parser without a guest."""
    script = GuestScript(client=None)  # type: ignore[arg-type]
    recorded = {}

    class _Client:
        def execute(self, argv, *, check=True, timeout_s=None):
            recorded["program"] = argv[-1]
            return {"output": GUEST_JSON_MARKER + '{"root": "/home/u/n"}'}

    script.client = _Client()  # type: ignore[assignment]
    assert str(script.resolve_guest_root("n")) == "/home/u/n"
    assert GUEST_JSON_MARKER in recorded["program"]
    compile(recorded["program"], "<guest>", "exec")  # it is valid Python


def test_a_program_containing_every_canonical_kind_executes_cleanly():
    """A surviving ``_r1a_`` local is a NameError, and only execution finds it."""
    run = run_guest_program(ALL_CANONICAL_OPERATIONS)
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True, run.payload["error"]
    assert run.payload["error"] is None
    assert run.payload["failure_kind"] is None
    assert run.payload["cleanup_attempted"] is False
    assert run.payload["attempt_hook_restore_errors"] == []


def test_the_coalesced_type_program_executes_with_its_renamed_locals():
    """``compile_unicode_coalesced_type`` carries its own ``_de_pasted`` flag."""
    # A non-ASCII payload, because that is now what selects the clipboard route.
    source = GP.compile_unicode_coalesced_type("é")
    assert "_de_pasted" in source and "_r1a" not in source
    run = run_guest_program((ir.coalesced_type("héllo ✓"),), with_gi=True)
    assert run.returncode == 0, run.stderr
    assert run.payload["ok"] is True, run.payload["error"]
    assert run.clipboard_text == "héllo ✓"


def test_every_payload_key_the_host_reads_is_a_key_the_guest_emits():
    """Field-name parity across the whole receipt, not just the marker."""
    run = run_guest_program(ALL_CANONICAL_OPERATIONS)
    emitted = set(run.payload)
    read = {
        "ok",
        "cursor",
        "cursor_before",
        "cursor_after",
        "pointer_button_mask",
        "observed_pointer_button_mask",
        "expected_pointer_button_mask",
        "guest_process_count",
        "cleanup_attempted",
        "error",
        "failure_kind",
        "operations",
        "backend_primitives",
        "x_event_sync_evidence",
        "x_sync_attempt_evidence",
        "click_backend",
        "x_injection_evidence",
        "x_injection_timestamps",
        "final_pointer_readback",
        "passive_x_observer",
        "attempt_hook_restore_errors",
        "_de_schema",
    }
    assert read <= emitted, f"host reads keys the guest never emits: {sorted(read - emitted)}"


def test_the_step_comment_marker_is_emitted_for_every_lowered_operation():
    """Not parsed by anything, but it is a renamed marker; keep it consistent.

    Note the indices track the LOWERED stream, not the semantic one: a coalesced
    ``mouse_down``/``mouse_up`` pair is one ``click`` step, so step *n* is not
    semantic operation *n*.  Both streams are in the payload, so the mapping is
    recoverable -- but do not read a step number as a semantic index.
    """
    from pixeldesk.execute.guest_program import lower_guest_operations

    program, _ = compile_atomic_guest_program(
        ALL_CANONICAL_OPERATIONS, initial_buttons=set(), initial_keys=set()
    )
    lowered = lower_guest_operations(ALL_CANONICAL_OPERATIONS)
    for index, operation in enumerate(lowered):
        assert f"# DESKTOP_ENV_ATOMIC_STEP_{index}:{operation.kind}" in program
    assert len(lowered) < len(ALL_CANONICAL_OPERATIONS), "this sample should coalesce"
    assert f"DESKTOP_ENV_ATOMIC_STEP_{len(lowered)}:" not in program


@pytest.mark.parametrize("operation", ALL_CANONICAL_OPERATIONS)
def test_an_operation_round_trips_through_its_json_view(operation):
    """``as_dict`` is what a receipt carries across a process boundary, so its
    inverse has to reconstruct the same operation for every canonical kind."""
    assert ir.Operation.from_dict(operation.as_dict()) == operation


@pytest.mark.parametrize("payload", [{"kind": "move_to"}, {"args": [1, 2]}])
def test_a_truncated_operation_payload_is_refused(payload):
    """``args`` used to default to ``()``, so a truncated receipt rebuilt into a
    well-formed operation of the wrong arity."""
    with pytest.raises(KeyError):
        ir.Operation.from_dict(payload)


def test_every_documented_environment_variable_is_read_somewhere():
    """A renamed env-var name that nothing reads is silently ignored config."""
    from pixeldesk.vm.factory import ENVIRONMENT

    sources = "\n".join(path.read_text() for path in PACKAGE_ROOT.rglob("*.py"))
    for name in ENVIRONMENT:
        assert sources.count(f'"{name}"') >= 2, f"{name} is documented but never read"


def test_every_desktop_env_environment_variable_read_is_documented():
    from pixeldesk.vm.factory import ENVIRONMENT

    pattern = re.compile(r"environ\.get\(\s*\"(DESKTOP_ENV_[A-Z_]+)\"")
    used = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        used.update(pattern.findall(path.read_text()))
    assert used <= set(ENVIRONMENT), f"undocumented: {sorted(used - set(ENVIRONMENT))}"
