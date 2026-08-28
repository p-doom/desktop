"""``ResetReceipt``, ``consume_receipt``, and ``_runtime_observation``.

A reset that silently no-ops is indistinguishable from a working one unless
something proves the guest rewound.  The attestation does three things, and each
is tested for what it would miss if it were absent:

* a NONCE FILE planted before the reset must be gone after it -- the only direct
  evidence the guest state moved;
* the runtime's observable state must CHANGE across the reset;
* the receipt is MAC'd with a per-session secret, single-use, and ordered, so a
  caller cannot report a reset it did not perform.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import os

import pytest

from desktop.vm.runtime import Checkpoint, GuestPorts, RuntimeState
from desktop.vm.session import (
    DesktopResetMode,
    DesktopSession,
    GuestScript,
    ResetReceipt,
    SessionError,
    canonical_json,
    sha256_file,
    task_unique_session_id,
    write_json_atomic,
)


class FakeRuntime:
    """A runtime whose observable state advances on every restore."""

    name = "fake"
    base_checkpoint = "base"

    def __init__(self) -> None:
        self.checkpoints: dict[str, Checkpoint] = {}
        self.restores: list[str] = []
        self.generation = 0
        self.started = False
        self.stopped = 0

    def start(self) -> RuntimeState:
        self.started = True
        return RuntimeState(
            runtime_id="rt-1",
            ports=GuestPorts(server=1),
            base_url="http://127.0.0.1:1",
            accelerator="tcg",
            detail={"pid": os.getpid()},
        )

    def stop(self) -> None:
        self.started = False
        self.stopped += 1

    def is_ready(self, *, timeout_s: float = 0.0) -> bool:
        return self.started

    def ensure_base(self) -> Checkpoint:
        record = Checkpoint("base", 1)
        self.checkpoints["base"] = record
        return record

    def checkpoint(self, name: str) -> Checkpoint:
        record = Checkpoint(name, 2 + len(self.checkpoints))
        self.checkpoints[name] = record
        return record

    def restore(self, name: str) -> Checkpoint:
        self.restores.append(name)
        self.generation += 1
        return self.checkpoints[name]

    def has_checkpoint(self, name: str) -> bool:
        return name in self.checkpoints

    def list_checkpoints(self) -> tuple[Checkpoint, ...]:
        return tuple(self.checkpoints.values())

    def delete_checkpoint(self, name: str) -> None:
        self.checkpoints.pop(name, None)


class FakeClient:
    """A guest client whose sentinel really disappears when the runtime rewinds."""

    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime
        self.programs: list[str] = []
        self.sentinel_survives_reset = False

    def execute(self, argv, *, check=True, timeout_s=None):
        program = argv[-1]
        self.programs.append(program)
        if "assert not Path(" in program and self.sentinel_survives_reset:
            raise RuntimeError("sentinel is still present after the reset")
        return {"status": "success", "returncode": 0, "output": ""}

    def cursor_position(self):
        return (self.runtime.generation, self.runtime.generation)

    def screenshot(self):
        return f"frame-{self.runtime.generation}".encode()


@pytest.fixture
def session(tmp_path):
    runtime = FakeRuntime()
    session = DesktopSession(
        runtime,
        scratch_root=tmp_path,
        metadata_path=tmp_path / "session.json",
        session_id="sess-1",
        require_single_task=False,
    )
    session.start()
    session.client = FakeClient(runtime)  # type: ignore[assignment]
    yield session
    if session._started:
        session.close()


def test_a_snapshot_reset_rewinds_the_runtime_and_settles_its_receipt(session):
    transport = session.reset(mode=DesktopResetMode.SNAPSHOT)
    assert session.runtime.restores == ["base"]
    assert session._reset_sequence == 1
    assert session._outstanding_receipt_sha256 is None
    assert transport is session.transport


def test_the_default_reset_mode_is_the_one_that_rewinds(session):
    """A caller that says nothing must get the strong guarantee, not the weak
    one: the guest keeping state across an episode is silent contamination."""
    session.reset()
    assert session.runtime.restores == ["base"]


def test_a_logical_reset_leaves_the_guest_running(session):
    """For a task whose expensive setup -- a warm browser -- must survive the
    boundary.  Nothing rewinds, so there is nothing to attest and no sequence to
    advance, and the sentinel is never planted."""
    before = session.transport
    transport = session.reset(mode=DesktopResetMode.LOGICAL)
    assert session.runtime.restores == []
    assert session._reset_sequence == 0
    assert session._outstanding_receipt_sha256 is None
    assert session.client.programs == []
    assert transport is session.transport
    assert transport is not before, "held input state must not cross the boundary"
    assert transport.base_url == before.base_url


def test_a_logical_reset_is_refused_while_a_receipt_is_outstanding(session):
    """The two modes share one reset slot, so an unconsumed snapshot receipt must
    not be stranded by a logical reset taken on top of it."""
    session.reset_with_receipt()
    with pytest.raises(SessionError, match="must be consumed before another reset"):
        session.reset(mode=DesktopResetMode.LOGICAL)


def test_a_logical_reset_on_an_unstarted_session_is_refused(tmp_path):
    session = DesktopSession(FakeRuntime(), scratch_root=tmp_path, require_single_task=False)
    with pytest.raises(SessionError, match="session is not started"):
        session.reset(mode=DesktopResetMode.LOGICAL)


def test_a_reset_mode_is_its_wire_value(session):
    """The mode crosses to a consumer as a string, so the member and the string
    a caller writes in a config have to be the same thing."""
    assert DesktopResetMode("logical") is DesktopResetMode.LOGICAL
    assert DesktopResetMode.SNAPSHOT == "snapshot"


def test_a_reset_produces_a_receipt_describing_it(session):
    transport, receipt = session.reset_with_receipt()
    assert isinstance(receipt, ResetReceipt)
    assert receipt.session_id == "sess-1"
    assert receipt.reset_sequence == 1
    assert receipt.checkpoint_name == "base"
    assert session.runtime.restores == ["base"]
    assert receipt.reset_completed_monotonic_ns >= receipt.reset_started_monotonic_ns
    assert transport is session.transport


def test_the_generation_must_advance_across_the_reset(session):
    _, receipt = session.reset_with_receipt()
    assert receipt.prior_generation_id != receipt.new_generation_id
    assert receipt.runtime_state_before_sha256 == receipt.prior_generation_id
    assert receipt.runtime_state_after_sha256 == receipt.new_generation_id


def test_a_reset_that_changes_nothing_is_refused(session):
    """The core anti-no-op check: if the runtime looks identical afterwards, the
    reset did not happen."""
    session.runtime.restore = lambda name: Checkpoint(name, 0)  # no generation bump
    with pytest.raises(SessionError, match="did not change the runtime's observed state"):
        session.reset_with_receipt()


def test_a_sentinel_that_survives_the_reset_is_refused(session):
    """Direct evidence about the GUEST, not just about the runtime object."""
    session.client.sentinel_survives_reset = True
    with pytest.raises(SessionError, match="did not rewind the pre-reset guest sentinel"):
        session.reset_with_receipt()


def test_the_sentinel_is_planted_before_the_restore_and_checked_after(session):
    session.reset_with_receipt()
    plants = [p for p in session.client.programs if "write_text" in p]
    checks = [p for p in session.client.programs if "assert not Path(" in p]
    assert len(plants) == 1 and len(checks) == 1
    assert session.client.programs.index(plants[0]) < session.client.programs.index(
        checks[0]
    )


def test_the_receipt_commits_to_the_sentinel_without_revealing_it(session):
    _, receipt = session.reset_with_receipt()
    assert len(receipt.guest_sentinel_path_sha256) == 64
    assert len(receipt.guest_sentinel_nonce_sha256) == 64
    planted = next(p for p in session.client.programs if "write_text" in p)
    # The nonce itself is in the guest program but only its digest is in the
    # receipt, so a receipt can be published without leaking the nonce.
    assert receipt.guest_sentinel_nonce_sha256 not in planted


def test_each_reset_uses_a_fresh_nonce_and_path(session):
    _, first = session.reset_with_receipt()
    session.consume_receipt(first)
    _, second = session.reset_with_receipt()
    assert first.guest_sentinel_nonce_sha256 != second.guest_sentinel_nonce_sha256
    assert first.guest_sentinel_path_sha256 != second.guest_sentinel_path_sha256
    assert first.reset_id != second.reset_id


def test_the_receipt_is_json_safe(session):
    _, receipt = session.reset_with_receipt()
    assert json.loads(json.dumps(receipt.as_dict()))["reset_sequence"] == 1


def test_the_receipt_is_immutable(session):
    _, receipt = session.reset_with_receipt()
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.reset_sequence = 99  # type: ignore[misc]


def test_a_valid_receipt_is_consumed_once(session):
    _, receipt = session.reset_with_receipt()
    session.consume_receipt(receipt)
    with pytest.raises(SessionError, match="already consumed"):
        session.consume_receipt(receipt)


def test_a_reset_cannot_be_repeated_until_its_receipt_is_consumed(session):
    session.reset_with_receipt()
    with pytest.raises(SessionError, match="must be consumed before another reset"):
        session.reset_with_receipt()


def test_consuming_the_receipt_unblocks_the_next_reset(session):
    _, first = session.reset_with_receipt()
    session.consume_receipt(first)
    _, second = session.reset_with_receipt()
    assert second.reset_sequence == 2
    session.consume_receipt(second)


@pytest.mark.parametrize(
    "field",
    [
        "session_id",
        "reset_id",
        "checkpoint_name",
        "prior_generation_id",
        "new_generation_id",
        "guest_sentinel_path_sha256",
        "guest_sentinel_nonce_sha256",
        "runtime_state_before_sha256",
        "runtime_state_after_sha256",
    ],
)
def test_tampering_with_any_payload_field_breaks_the_mac(session, field):
    _, receipt = session.reset_with_receipt()
    forged = dataclasses.replace(receipt, **{field: "tampered"})
    with pytest.raises(SessionError, match="MAC does not verify"):
        session.consume_receipt(forged)


@pytest.mark.parametrize("field", ["reset_sequence", "reset_started_monotonic_ns"])
def test_tampering_with_a_numeric_field_breaks_the_mac(session, field):
    _, receipt = session.reset_with_receipt()
    forged = dataclasses.replace(receipt, **{field: 12345})
    with pytest.raises(SessionError, match="MAC does not verify"):
        session.consume_receipt(forged)


def test_a_forged_mac_is_rejected(session):
    _, receipt = session.reset_with_receipt()
    with pytest.raises(SessionError, match="MAC does not verify"):
        session.consume_receipt(dataclasses.replace(receipt, attestor_mac="0" * 64))


def test_a_forged_digest_is_rejected(session):
    _, receipt = session.reset_with_receipt()
    with pytest.raises(SessionError, match="digest does not verify"):
        session.consume_receipt(dataclasses.replace(receipt, receipt_sha256="0" * 64))


def test_the_secret_is_per_session_so_a_receipt_does_not_travel(session, tmp_path):
    """A second session must not accept the first session's receipt."""
    _, receipt = session.reset_with_receipt()
    other = DesktopSession(
        FakeRuntime(),
        scratch_root=tmp_path / "other",
        session_id="sess-2",
        require_single_task=False,
    )
    with pytest.raises(SessionError, match="MAC does not verify"):
        other.consume_receipt(receipt)


def test_two_sessions_have_different_attestor_secrets(tmp_path):
    first = DesktopSession(FakeRuntime(), scratch_root=tmp_path, session_id="a")
    second = DesktopSession(FakeRuntime(), scratch_root=tmp_path, session_id="b")
    assert first._attestor_secret != second._attestor_secret
    assert len(first._attestor_secret) == 32


def test_a_non_receipt_is_rejected_by_type(session):
    session.reset_with_receipt()
    for bad in ({"session_id": "sess-1"}, None, "receipt", 42):
        with pytest.raises(SessionError, match="type mismatch"):
            session.consume_receipt(bad)  # type: ignore[arg-type]


def test_a_receipt_for_the_wrong_sequence_is_rejected(session):
    """Reachable only for a receipt whose MAC is genuine, so it is forged with
    the session's own secret."""
    _, receipt = session.reset_with_receipt()
    payload = receipt.as_dict()
    payload.pop("receipt_sha256")
    payload.pop("attestor_mac")
    payload["reset_sequence"] = 99
    mac = hmac.new(
        session._attestor_secret, canonical_json(payload), hashlib.sha256
    ).hexdigest()
    digest = hashlib.sha256(canonical_json({**payload, "attestor_mac": mac})).hexdigest()
    forged = ResetReceipt(**payload, attestor_mac=mac, receipt_sha256=digest)
    with pytest.raises(SessionError, match="out of order"):
        session.consume_receipt(forged)


def test_a_correctly_maced_receipt_that_is_not_outstanding_is_rejected(session):
    """Belt and braces: the sequence matches but it is not the reset in flight."""
    _, receipt = session.reset_with_receipt()
    session.consume_receipt(receipt)
    payload = receipt.as_dict()
    payload.pop("receipt_sha256")
    payload.pop("attestor_mac")
    payload["reset_id"] = "a different reset id"
    mac = hmac.new(
        session._attestor_secret, canonical_json(payload), hashlib.sha256
    ).hexdigest()
    digest = hashlib.sha256(canonical_json({**payload, "attestor_mac": mac})).hexdigest()
    forged = ResetReceipt(**payload, attestor_mac=mac, receipt_sha256=digest)
    with pytest.raises(SessionError, match="does not match the outstanding reset"):
        session.consume_receipt(forged)


def test_reset_consumes_the_receipt_for_you(session):
    session.reset()
    session.reset()  # would raise if the first receipt were left outstanding
    assert session._reset_sequence == 2
    assert session._outstanding_receipt_sha256 is None


def test_resetting_an_unstarted_session_is_an_error(tmp_path):
    session = DesktopSession(FakeRuntime(), scratch_root=tmp_path, session_id="x")
    with pytest.raises(SessionError, match="not started"):
        session.reset_with_receipt()


def test_the_observation_is_stable_when_nothing_changes(session):
    assert session._runtime_observation() == session._runtime_observation()


def test_the_observation_changes_when_the_guest_state_changes(session):
    before = session._runtime_observation()
    session.runtime.generation += 1
    assert session._runtime_observation() != before


def test_the_observation_changes_when_a_checkpoint_appears(session):
    before = session._runtime_observation()
    session.runtime.checkpoint("extra")
    assert session._runtime_observation() != before


def test_the_observation_covers_readiness_cursor_and_screenshot(session):
    payload = json.loads(session._runtime_observation())
    assert set(payload) == {
        "checkpoints",
        "ready",
        "cursor",
        "screenshot_sha256",
        "reset_sequence",
    }
    assert payload["ready"] is True
    assert payload["cursor"] == [0, 0]
    assert len(payload["screenshot_sha256"]) == 64


def test_the_observation_tolerates_a_guest_that_will_not_answer(session):
    """It must never raise: a reset of a wedged guest still needs an observation."""

    def explode(*args, **kwargs):
        raise ConnectionError("guest is wedged")

    session.client.cursor_position = explode
    session.client.screenshot = explode
    payload = json.loads(session._runtime_observation())
    assert payload["cursor"] is None
    assert payload["screenshot_sha256"] is None


def test_the_observation_does_not_introspect_provider_internals():
    """The predecessor hashed a provider's private dict, which tied the receipt to
    one implementation."""
    import inspect

    source = inspect.getsource(DesktopSession._runtime_observation)
    assert "__dict__" not in source
    assert "timings" not in source


def test_a_reset_hands_back_a_fresh_transport_with_an_empty_audit(session):
    before = session.transport
    before.audit.held_buttons.add("left")
    before.audit.held_keys.add("ctrl")
    before.audit.scroll_total = 7
    after = session.reset()
    assert after is not before
    assert after.base_url == before.base_url
    assert after.audit.held_buttons == set()
    assert after.audit.held_keys == set()
    assert after.audit.scroll_total == 0
    assert after.audit.operations == []


def test_reset_to_checkpoint_also_refreshes_the_transport(session):
    before = session.transport
    session.runtime.checkpoint("post-setup")
    after = session.reset_to_checkpoint("post-setup")
    assert after is not before
    assert session.runtime.restores == ["post-setup"]


def test_reset_to_checkpoint_creates_the_checkpoint_on_first_use(session):
    calls = []
    session.reset_to_checkpoint("tier2", setup=lambda transport: calls.append(transport))
    assert len(calls) == 1
    assert session.runtime.has_checkpoint("tier2")
    assert session.runtime.restores == ["base"]
    # The second call restores instead of re-running setup.
    session.reset_to_checkpoint("tier2", setup=lambda transport: calls.append(transport))
    assert len(calls) == 1
    assert session.runtime.restores == ["base", "tier2"]


def test_metadata_is_written_at_start_and_finalised_at_close(session, tmp_path):
    metadata = json.loads((tmp_path / "session.json").read_text())
    assert metadata["schema_version"] == "desktop_env_session_v1"
    assert metadata["session_id"] == "sess-1"
    assert metadata["closed"] is False
    assert metadata["runtime"] == "fake"
    assert metadata["one_vm_per_task"] is False
    session.close()
    finalised = json.loads((tmp_path / "session.json").read_text())
    assert finalised["closed"] is True
    assert finalised["cleanup_errors"] == []


def test_close_removes_the_scratch_directory_and_the_task_lock(session, tmp_path):
    scratch = session.scratch_dir
    lock = session._task_lock_path
    assert scratch.is_dir() and lock.is_file()
    session.close()
    assert not scratch.exists()
    assert not lock.exists()


def test_close_restores_the_environment_it_changed(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/original/tmpdir")
    session = DesktopSession(
        FakeRuntime(), scratch_root=tmp_path, session_id="env", require_single_task=False
    )
    session.start()
    assert os.environ["TMPDIR"] == str(session.scratch_dir)
    session.close()
    assert os.environ["TMPDIR"] == "/original/tmpdir"


def test_a_failed_start_closes_what_it_opened(tmp_path):
    class BrokenRuntime(FakeRuntime):
        def ensure_base(self):
            raise RuntimeError("no base checkpoint")

    session = DesktopSession(
        BrokenRuntime(), scratch_root=tmp_path, session_id="broken", require_single_task=False
    )
    with pytest.raises(RuntimeError, match="no base checkpoint"):
        session.start()
    assert session.scratch_dir is None
    assert session._task_lock_handle is None
    assert session.runtime.stopped >= 1


def test_close_is_idempotent(session):
    session.close()
    session.close()


def test_gpu_visibility_can_be_forbidden(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    session = DesktopSession(
        FakeRuntime(),
        scratch_root=tmp_path,
        session_id="gpu",
        require_single_task=False,
        forbid_gpu_visibility=True,
    )
    with pytest.raises(SessionError, match="GPU visibility is forbidden"):
        session.start()


def test_the_session_is_a_context_manager(tmp_path):
    runtime = FakeRuntime()
    with DesktopSession(
        runtime, scratch_root=tmp_path, session_id="ctx", require_single_task=False
    ) as session:
        assert session.transport is not None
    assert runtime.stopped >= 1


def test_a_session_id_is_unique_per_process_and_task():
    assert task_unique_session_id() != task_unique_session_id()


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"a": 1}) == b'{"a":1}'


def test_canonical_json_survives_a_non_serialisable_value():
    assert b"object" in canonical_json({"a": object()}) or canonical_json({"a": object()})


def test_write_json_atomic_leaves_no_temp_file(tmp_path):
    target = tmp_path / "nested" / "out.json"
    write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]


def test_write_json_atomic_is_private_to_the_user(tmp_path):
    import stat

    target = tmp_path / "out.json"
    write_json_atomic(target, {"a": 1})
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"\x00\xff" * 1000)
    assert sha256_file(path) == hashlib.sha256(b"\x00\xff" * 1000).hexdigest()


def test_a_guest_script_reads_only_its_marker_line():
    """Guest stdout is shared with GTK warnings and X11 chatter."""
    script = GuestScript(client=None)  # type: ignore[arg-type]
    noisy = (
        "Gtk-WARNING **: cannot open display\n"
        "DESKTOP_ENV_JSON={\"value\": 42}\n"
        "some trailing chatter\n"
    )
    assert script.parse({"output": noisy}) == {"value": 42}


def test_a_guest_script_refuses_ambiguous_or_absent_markers():
    script = GuestScript(client=None)  # type: ignore[arg-type]
    with pytest.raises(SessionError, match="0 result markers"):
        script.parse({"output": "nothing here"})
    with pytest.raises(SessionError, match="2 result markers"):
        script.parse({"output": "DESKTOP_ENV_JSON={}\nDESKTOP_ENV_JSON={}"})
    with pytest.raises(SessionError, match="no stdout"):
        script.parse({"output": None})


def test_a_guest_script_reports_invalid_json_as_such():
    script = GuestScript(client=None)  # type: ignore[arg-type]
    with pytest.raises(SessionError, match="invalid JSON"):
        script.parse({"output": "DESKTOP_ENV_JSON={not json}"})


