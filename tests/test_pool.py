"""``DesktopSessionPool`` lifecycle: prewarm, checkout, release, retire, reap.

Not on the original risk list, but ``pool.py`` is the largest file in the package
and was the second least-exercised.  Everything here runs without a VM because the
pool's entire requirement of a session is ``close() -> None``.

Two seams make it testable: ``session_factory`` and ``port_allocator`` are both
injected, and ``clock`` is injectable, so lease expiry is testable without
sleeping through it.  Background threads are real, so each assertion waits on a
condition with a deadline rather than sleeping a guessed interval.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from desktop.vm.pool import (
    DesktopPoolConfig,
    DesktopSessionPool,
    PortLease,
    PortRangeLease,
    WorkerPorts,
    ports_for_worker,
)


class FakeSession:
    """A pooled session: it only has to close."""

    def __init__(self, lease: PortLease) -> None:
        self.lease = lease
        self.closed = 0
        self.close_raises = False
        self.calls: list[str] = []

    def close(self) -> None:
        self.closed += 1
        if self.close_raises:
            raise RuntimeError("close exploded")

    def do_work(self, value: int) -> int:
        self.calls.append(f"do_work({value})")
        return value * 2


def fake_allocator_factory(tmp_path: Path):
    """A port allocator with no real flock or bind: pure bookkeeping."""
    counter = {"slot": 0}
    released: list[int] = []

    class FakeLease(PortLease):
        def release(self) -> None:
            if self._released:
                return
            released.append(self.slot)
            self._released = True

    def allocate(*, lock_dir, log_dir, work_dir=None) -> PortLease:
        slot = counter["slot"]
        counter["slot"] += 1
        workdir = Path(work_dir or lock_dir) / f"w{slot}"
        logdir = Path(log_dir) / f"w{slot}"
        workdir.mkdir(parents=True, exist_ok=True)
        logdir.mkdir(parents=True, exist_ok=True)
        ports = ports_for_worker(50000, slot)
        return FakeLease(
            ports=ports,
            slot=slot,
            workdir=workdir,
            logdir=logdir,
            _range_lease=PortRangeLease(
                start=ports.server,
                count=10,
                lock_dir=Path(lock_dir),
                purpose="fake",
                _lock_files=(),
            ),
        )

    allocate.released = released  # type: ignore[attr-defined]
    return allocate


@pytest.fixture
def pool_factory(tmp_path):
    """Builds pools sharing one recording factory, and closes them afterwards."""
    built: list[FakeSession] = []
    pools: list[DesktopSessionPool] = []
    failures = {"remaining": 0}

    def make(**config_kwargs) -> DesktopSessionPool:
        def session_factory(lease: PortLease) -> FakeSession:
            if failures["remaining"] > 0:
                failures["remaining"] -= 1
                raise RuntimeError("session start failed")
            session = FakeSession(lease)
            built.append(session)
            return session

        pool = DesktopSessionPool(
            config=DesktopPoolConfig(**config_kwargs),
            root_dir=tmp_path / f"pool{len(pools)}",
            session_factory=session_factory,
            port_allocator=fake_allocator_factory(tmp_path / f"pool{len(pools)}"),
        )
        pools.append(pool)
        return pool

    make.built = built  # type: ignore[attr-defined]
    make.failures = failures  # type: ignore[attr-defined]
    yield make
    for pool in pools:
        pool.close()


def wait_until(predicate, *, timeout_s: float = 10.0) -> bool:
    """Wait on a real background thread without guessing an interval."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_ready_sessions": -1}, "non-negative"),
        ({"max_sessions": 0}, "at least 1"),
        ({"min_ready_sessions": 3, "max_sessions": 2}, "cannot exceed"),
        ({"max_rollouts_per_session": 0}, "at least 1"),
        ({"checkout_timeout_s": 0}, "positive"),
        ({"lease_timeout_s": -1}, "positive"),
        ({"startup_timeout_s": 0}, "positive"),
        ({"startup_retry_backoff_s": -1}, "non-negative"),
        ({"startup_retry_backoff_max_s": 0}, "positive"),
        ({"status_heartbeat_interval_s": -1}, "non-negative"),
    ],
)
def test_an_invalid_pool_config_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DesktopPoolConfig(**kwargs)


def test_path_fields_are_coerced_to_paths():
    config = DesktopPoolConfig(status_dir="/tmp/x", port_lock_dir="/tmp/y")
    assert isinstance(config.status_dir, Path)
    assert isinstance(config.port_lock_dir, Path)


def test_nothing_boots_until_start_is_called(pool_factory):
    pool = pool_factory(min_ready_sessions=2)
    assert pool_factory.built == []
    pool.start()
    assert wait_until(lambda: len(pool_factory.built) == 2)


def test_start_creates_the_pool_directories(pool_factory):
    pool = pool_factory(min_ready_sessions=0)
    pool.start()
    for directory in (
        pool.root_dir,
        pool.status_dir,
        pool.port_lock_dir,
        pool.runtime_dir,
        pool.log_dir,
        pool.artifact_dir,
    ):
        assert directory.is_dir(), directory


def test_start_is_idempotent(pool_factory):
    pool = pool_factory(min_ready_sessions=1)
    pool.start()
    assert wait_until(lambda: len(pool_factory.built) == 1)
    pool.start()
    assert len(pool_factory.built) == 1


def test_the_ready_floor_is_maintained_but_not_exceeded(pool_factory):
    pool = pool_factory(min_ready_sessions=2, max_sessions=5)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 2)
    time.sleep(0.1)
    assert pool.snapshot()["ready"] == 2
    assert len(pool_factory.built) == 2


def test_the_pool_never_exceeds_max_sessions(pool_factory):
    pool = pool_factory(min_ready_sessions=3, max_sessions=3)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 3)
    checked_out = [pool.checkout(timeout_s=5) for _ in range(3)]
    time.sleep(0.1)
    assert len(pool_factory.built) == 3
    for handle in checked_out:
        handle.release()


def test_checkout_hands_out_a_ready_session(pool_factory):
    pool = pool_factory(min_ready_sessions=1)
    pool.start()
    handle = pool.checkout(timeout_s=5)
    assert isinstance(handle.env, FakeSession)
    assert pool.snapshot()["leased"] == 1
    handle.release()
    assert pool.snapshot()["leased"] == 0


def test_checkout_starts_the_pool_if_it_was_not_started(pool_factory):
    pool = pool_factory(min_ready_sessions=1)
    handle = pool.checkout(timeout_s=10)
    assert handle.env is not None
    handle.release()


def test_checkout_prefers_the_oldest_ready_session(pool_factory):
    pool = pool_factory(min_ready_sessions=2)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 2)
    ids = [session["session_id"] for session in pool.snapshot()["sessions"]]
    handle = pool.checkout(timeout_s=5)
    assert handle.session_id == min(ids)
    handle.release()


def test_a_released_session_is_handed_out_again(pool_factory):
    pool = pool_factory(min_ready_sessions=1, max_rollouts_per_session=10)
    pool.start()
    first = pool.checkout(timeout_s=5)
    identifier = first.session_id
    first.release()
    second = pool.checkout(timeout_s=5)
    assert second.session_id == identifier
    second.release()


def test_checkout_times_out_when_nothing_can_be_started(pool_factory):
    pool = pool_factory(min_ready_sessions=1, max_sessions=1)
    pool.start()
    handle = pool.checkout(timeout_s=5)
    try:
        with pytest.raises(TimeoutError, match="timed out waiting for a ready desktop"):
            pool.checkout(timeout_s=0.3)
    finally:
        handle.release()


def test_a_non_positive_checkout_timeout_is_refused(pool_factory):
    pool = pool_factory(min_ready_sessions=0)
    pool.start()
    with pytest.raises(ValueError, match="must be positive"):
        pool.checkout(timeout_s=0)


def test_release_is_idempotent(pool_factory):
    pool = pool_factory(min_ready_sessions=1, max_rollouts_per_session=10)
    pool.start()
    handle = pool.checkout(timeout_s=5)
    handle.release()
    handle.release()
    assert pool.snapshot()["sessions"][0]["rollouts_completed"] == 1


def test_a_checked_out_session_is_a_context_manager(pool_factory):
    pool = pool_factory(min_ready_sessions=1, max_rollouts_per_session=10)
    pool.start()
    with pool.checkout(timeout_s=5) as handle:
        assert handle.env is not None
    assert pool.snapshot()["leased"] == 0


def test_a_checked_out_session_exposes_the_block_its_lease_holds(pool_factory):
    """A caller that starts a host-side service beside the desktop needs a port
    nothing else was given, and the lease is the only thing that knows which."""
    pool = pool_factory(min_ready_sessions=1)
    pool.start()
    with pool.checkout(timeout_s=5) as handle:
        assert isinstance(handle.ports, WorkerPorts)
        assert handle.ports == handle.env.lease.ports
        assert handle.ports.auxiliary_port() == handle.ports.server + 4


def test_two_checked_out_sessions_never_share_an_auxiliary_port(pool_factory):
    pool = pool_factory(min_ready_sessions=2, max_sessions=2)
    pool.start()
    with pool.checkout(timeout_s=5) as first, pool.checkout(timeout_s=5) as second:
        assert set(first.ports.all()).isdisjoint(second.ports.all())


def test_an_exception_inside_the_context_retires_the_session(pool_factory):
    pool = pool_factory(min_ready_sessions=1)
    pool.start()
    session_ids = set()
    with pytest.raises(RuntimeError, match="rollout blew up"):
        with pool.checkout(timeout_s=5) as handle:
            session_ids.add(handle.session_id)
            raise RuntimeError("rollout blew up")
    assert wait_until(lambda: pool.snapshot()["total_failed"] >= 1)
    assert wait_until(lambda: pool_factory.built[0].closed == 1)
    assert pool.snapshot()["last_error"] is not None


def test_a_session_retires_after_its_rollout_budget(pool_factory):
    pool = pool_factory(min_ready_sessions=1, max_rollouts_per_session=2)
    pool.start()
    first = pool.checkout(timeout_s=5)
    identifier = first.session_id
    first.release()
    second = pool.checkout(timeout_s=5)
    assert second.session_id == identifier
    second.release()
    assert wait_until(lambda: pool_factory.built[0].closed == 1)
    # ... and the pool replaces it.
    assert wait_until(lambda: len(pool_factory.built) == 2)
    third = pool.checkout(timeout_s=5)
    assert third.session_id != identifier
    third.release()


def test_a_failed_release_retires_immediately(pool_factory):
    pool = pool_factory(min_ready_sessions=1, max_rollouts_per_session=50)
    pool.start()
    handle = pool.checkout(timeout_s=5)
    handle.release(failed=True, error="the guest wedged")
    assert wait_until(lambda: pool_factory.built[0].closed == 1)
    assert pool.snapshot()["total_failed"] == 1
    assert pool.snapshot()["last_error"] == "the guest wedged"


def test_retiring_a_session_releases_its_port_lease(pool_factory):
    pool = pool_factory(min_ready_sessions=1, max_rollouts_per_session=1)
    pool.start()
    handle = pool.checkout(timeout_s=5)
    lease = pool_factory.built[0].lease
    handle.release()
    assert wait_until(lambda: lease._released is True)


def test_a_session_whose_close_raises_is_still_removed(pool_factory):
    pool = pool_factory(min_ready_sessions=1, max_rollouts_per_session=1)
    pool.start()
    assert wait_until(lambda: len(pool_factory.built) == 1)
    pool_factory.built[0].close_raises = True
    handle = pool.checkout(timeout_s=5)
    handle.release()
    assert wait_until(lambda: "close failed" in (pool.snapshot()["last_error"] or ""))
    assert wait_until(lambda: pool.snapshot()["retiring"] == 0)


def test_releasing_an_unknown_session_is_a_no_op(pool_factory):
    pool = pool_factory(min_ready_sessions=0)
    pool.start()
    pool.release("session-999999")


def test_a_lease_with_no_activity_is_reaped(tmp_path):
    """A holder that dies without releasing must not strand a desktop."""
    clock = {"now": 1000.0}
    built: list[FakeSession] = []
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(
            min_ready_sessions=1, lease_timeout_s=300.0, status_heartbeat_interval_s=0
        ),
        root_dir=tmp_path / "reap",
        session_factory=lambda lease: built.append(FakeSession(lease)) or built[-1],
        port_allocator=fake_allocator_factory(tmp_path / "reap"),
        clock=lambda: clock["now"],
    )
    try:
        pool.start()
        assert wait_until(lambda: pool.snapshot()["ready"] == 1)
        pool.checkout(timeout_s=5)  # deliberately never released
        assert pool.reap_stale_leases() == 0
        clock["now"] += 301.0
        assert pool.reap_stale_leases() == 1
        assert wait_until(lambda: built[0].closed == 1)
        assert pool.snapshot()["stale_leases_retired"] == 1
        assert "lease timed out" in pool.snapshot()["last_error"]
    finally:
        pool.close()


def test_activity_refreshes_the_lease(tmp_path):
    clock = {"now": 1000.0}
    built: list[FakeSession] = []
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(
            min_ready_sessions=1, lease_timeout_s=100.0, status_heartbeat_interval_s=0
        ),
        root_dir=tmp_path / "touch",
        session_factory=lambda lease: built.append(FakeSession(lease)) or built[-1],
        port_allocator=fake_allocator_factory(tmp_path / "touch"),
        clock=lambda: clock["now"],
    )
    try:
        pool.start()
        assert wait_until(lambda: pool.snapshot()["ready"] == 1)
        handle = pool.checkout(timeout_s=5)
        clock["now"] += 90.0
        handle.touch()
        clock["now"] += 90.0
        assert pool.reap_stale_leases() == 0, "a touched lease must not be reaped"
        clock["now"] += 20.0
        assert pool.reap_stale_leases() == 1
    finally:
        pool.close()


def test_the_tracked_env_proxy_refreshes_activity_around_every_call(tmp_path):
    clock = {"now": 1000.0}
    built: list[FakeSession] = []
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(
            min_ready_sessions=1, lease_timeout_s=100.0, status_heartbeat_interval_s=0
        ),
        root_dir=tmp_path / "proxy",
        session_factory=lambda lease: built.append(FakeSession(lease)) or built[-1],
        port_allocator=fake_allocator_factory(tmp_path / "proxy"),
        clock=lambda: clock["now"],
    )
    try:
        pool.start()
        assert wait_until(lambda: pool.snapshot()["ready"] == 1)
        handle = pool.checkout(timeout_s=5)
        env = handle.tracked_env()
        clock["now"] += 90.0
        assert env.do_work(21) == 42, "the proxy must forward the return value"
        clock["now"] += 90.0
        assert pool.reap_stale_leases() == 0
        assert built[0].calls == ["do_work(21)"]
        # Non-callable attributes pass straight through.
        assert env.closed == 0
    finally:
        pool.close()


def test_a_ready_session_is_never_reaped(tmp_path):
    clock = {"now": 1000.0}
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(
            min_ready_sessions=1, lease_timeout_s=10.0, status_heartbeat_interval_s=0
        ),
        root_dir=tmp_path / "ready",
        session_factory=FakeSession,
        port_allocator=fake_allocator_factory(tmp_path / "ready"),
        clock=lambda: clock["now"],
    )
    try:
        pool.start()
        assert wait_until(lambda: pool.snapshot()["ready"] == 1)
        clock["now"] += 10_000.0
        assert pool.reap_stale_leases() == 0
    finally:
        pool.close()


def test_a_startup_failure_is_recorded_and_retried(pool_factory):
    pool = pool_factory(
        min_ready_sessions=1, startup_retry_backoff_s=0.05, startup_retry_backoff_max_s=0.1
    )
    pool_factory.failures["remaining"] = 2
    pool.start()
    assert wait_until(lambda: pool.snapshot()["total_failed"] >= 1)
    assert "session start failed" in pool.snapshot()["last_error"]
    assert wait_until(lambda: pool.snapshot()["ready"] == 1, timeout_s=15)
    assert pool.snapshot()["consecutive_start_failures"] == 0


def test_a_startup_failure_releases_the_port_lease(tmp_path):
    allocator = fake_allocator_factory(tmp_path / "leak")
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(
            min_ready_sessions=1, startup_retry_backoff_s=60.0, status_heartbeat_interval_s=0
        ),
        root_dir=tmp_path / "leak",
        session_factory=lambda lease: (_ for _ in ()).throw(RuntimeError("nope")),
        port_allocator=allocator,
    )
    try:
        pool.start()
        assert wait_until(lambda: pool.snapshot()["total_failed"] >= 1)
        assert allocator.released, "a failed start must not strand its port block"
    finally:
        pool.close()


def test_the_backoff_grows_with_consecutive_failures(tmp_path):
    clock = {"now": 1000.0}
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(
            min_ready_sessions=1,
            startup_retry_backoff_s=10.0,
            startup_retry_backoff_max_s=40.0,
            status_heartbeat_interval_s=0,
        ),
        root_dir=tmp_path / "backoff",
        session_factory=lambda lease: (_ for _ in ()).throw(RuntimeError("nope")),
        port_allocator=fake_allocator_factory(tmp_path / "backoff"),
        clock=lambda: clock["now"],
    )
    try:
        pool._started = True
        deadlines = []
        for failures in (1, 2, 3, 4, 5):
            pool._consecutive_start_failures = failures
            deadlines.append(pool._next_retry_deadline_locked() - clock["now"])
        assert deadlines == [10.0, 20.0, 40.0, 40.0, 40.0]
    finally:
        pool.close()


def test_the_cooldown_is_reported_in_the_status(tmp_path):
    clock = {"now": 1000.0}
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(
            min_ready_sessions=1,
            startup_retry_backoff_s=100.0,
            status_heartbeat_interval_s=0,
        ),
        root_dir=tmp_path / "cool",
        session_factory=lambda lease: (_ for _ in ()).throw(RuntimeError("nope")),
        port_allocator=fake_allocator_factory(tmp_path / "cool"),
        clock=lambda: clock["now"],
    )
    try:
        pool.start()
        assert wait_until(lambda: pool.snapshot()["total_failed"] >= 1)
        snapshot = pool.snapshot()
        assert snapshot["startup_cooldown_remaining_s"] > 0
        assert snapshot["next_start_attempt_at"] is not None
        assert snapshot["retry_scheduled"] is True
    finally:
        pool.close()


def test_the_status_file_is_written_and_is_valid_json(pool_factory):
    pool = pool_factory(min_ready_sessions=1)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 1)
    payload = json.loads(pool.status_path.read_text())
    assert payload["ready"] == 1
    assert payload["pid"] > 0
    assert payload["closed"] is False
    assert payload["worker_name"] == pool.status_path.stem


def _unstarted_pool(root_dir: Path) -> DesktopSessionPool:
    return DesktopSessionPool(
        config=DesktopPoolConfig(),
        root_dir=root_dir,
        session_factory=lambda lease: FakeSession(lease),
    )


def test_the_port_lock_namespace_is_node_wide_and_not_per_run(tmp_path):
    """Two runs on one node have to contend for the same slots.

    Under ``root_dir/"port_locks"`` each run had its own namespace on shared
    storage, so two runs -- or two jobs, or one job twice -- each took slot 0 and
    handed out the same four ports, each holding a lock the other could not see.

    It equally must not be ``$TMPDIR``: measured on this cluster,
    ``job_container/tmpfs`` bind-mounts a PRIVATE ``/var/tmp`` (and ``/dev/shm``)
    per job while ``/tmp`` is the node's real filesystem, visible across jobs. A
    lock under ``$TMPDIR`` would be job-private and would protect nothing, which
    is the whole failure this default exists to avoid.
    """
    first = _unstarted_pool(tmp_path / "run-a")
    second = _unstarted_pool(tmp_path / "run-b")

    assert first.port_lock_dir == second.port_lock_dir
    assert not first.port_lock_dir.is_relative_to(tmp_path)
    assert first.port_lock_dir.is_relative_to("/tmp")
    assert not first.port_lock_dir.is_relative_to("/var/tmp")


def test_the_runtime_dir_is_node_local_rather_than_on_the_shared_run_root(
    tmp_path, monkeypatch
):
    """A lease's workdir becomes its session's TMPDIR, so QEMU's ~2.7 GB unlinked
    overlay lands in it. Defaulting under ``root_dir`` put every VM's overlay on
    shared /fast. ``$TMPDIR`` is the job's private, node-local, Slurm-deleted
    directory, which is also why no prepare-wipe or shutdown-wipe is needed."""
    monkeypatch.setenv("TMPDIR", "/var/tmp")

    pool = _unstarted_pool(tmp_path / "run-a")

    assert not pool.runtime_dir.is_relative_to(tmp_path)
    assert pool.runtime_dir.is_relative_to("/var/tmp")
    # still per-process, so two pools in one job do not share one scratch root
    assert str(os.getpid()) in pool.runtime_dir.name


def test_the_status_file_describes_each_session_and_its_ports(pool_factory):
    pool = pool_factory(min_ready_sessions=1)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 1)
    (session,) = pool.snapshot()["sessions"]
    assert session["status"] == "ready"
    assert set(session["ports"]) == {"server", "chromium", "vnc", "vlc"}
    assert session["lease_slot"] == 0
    assert "workdir" in session


def test_the_status_file_is_never_seen_partially_written(pool_factory):
    """The property that matters for another process reading the status file.

    Asserting that no ``.tmp`` sibling exists is the WRONG test -- the temp file
    is a legitimate transient, and a background heartbeat writing concurrently
    makes such an assertion flaky.  What must hold is that the target itself
    always parses: ``os.replace`` swaps it in one step, so a reader either sees
    the old document or the new one, never a truncated one.
    """
    pool = pool_factory(min_ready_sessions=2, status_heartbeat_interval_s=0.001)
    pool.start()
    assert wait_until(lambda: pool.status_path.is_file())
    for _ in range(400):
        payload = json.loads(pool.status_path.read_text())
        assert payload["worker_name"] == pool.status_path.stem
    assert pool.status_path.parent.name == "status"


def test_starting_sessions_appear_in_the_status_before_they_are_ready(tmp_path):
    gate = threading.Event()
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(min_ready_sessions=1, status_heartbeat_interval_s=0),
        root_dir=tmp_path / "starting",
        session_factory=lambda lease: gate.wait(10) or FakeSession(lease),
        port_allocator=fake_allocator_factory(tmp_path / "starting"),
    )
    try:
        pool.start()
        assert wait_until(lambda: pool.snapshot()["starting"] == 1)
        (starting,) = pool.snapshot()["starting_sessions"]
        assert starting["status"] == "starting"
        assert starting["age_s"] >= 0
        assert pool.snapshot()["oldest_starting_age_s"] is not None
    finally:
        gate.set()
        pool.close()


def test_close_closes_every_session_and_releases_every_lease(pool_factory):
    pool = pool_factory(min_ready_sessions=2)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 2)
    leases = [session.lease for session in pool_factory.built]
    pool.close()
    assert all(session.closed == 1 for session in pool_factory.built)
    assert all(lease._released for lease in leases)
    assert pool.snapshot()["closed"] is True


def test_one_session_that_will_not_close_does_not_strand_the_others(pool_factory):
    """A bare loop in ``close()`` aborted on the first raising ``close()``, so
    every LATER session stayed open and kept its port block -- a leaked VM, its
    memory and four host ports, per pool shutdown -- and the exception escaped
    ``close()`` as well."""
    pool = pool_factory(min_ready_sessions=3, max_sessions=3)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 3)
    sessions = sorted(pool_factory.built, key=lambda item: item.lease.slot)
    sessions[0].close_raises = True
    leases = [session.lease for session in sessions]
    pool.close()  # must not raise
    assert all(session.closed == 1 for session in sessions), [
        session.closed for session in sessions
    ]
    assert all(lease._released for lease in leases), [
        lease._released for lease in leases
    ]
    assert "close failed" in (pool.snapshot()["last_error"] or "")


def test_every_session_that_will_not_close_is_reported(pool_factory):
    pool = pool_factory(min_ready_sessions=2, max_sessions=2)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 2)
    for session in pool_factory.built:
        session.close_raises = True
    before = pool.snapshot()["total_failed"]
    pool.close()
    assert pool.snapshot()["total_failed"] == before + 2
    assert all(session.lease._released for session in pool_factory.built)


def test_close_is_idempotent(pool_factory):
    pool = pool_factory(min_ready_sessions=1)
    pool.start()
    assert wait_until(lambda: len(pool_factory.built) == 1)
    pool.close()
    pool.close()
    assert pool_factory.built[0].closed == 1


def test_checkout_after_close_is_refused(pool_factory):
    pool = pool_factory(min_ready_sessions=1)
    pool.start()
    pool.close()
    with pytest.raises(RuntimeError, match="is closed"):
        pool.checkout(timeout_s=1)


def test_a_session_that_finishes_starting_after_close_is_closed_not_leaked(tmp_path):
    """The narrow window: the factory has ALREADY been entered when close lands.

    Waiting for ``starting == 1`` is not enough -- that is set before the factory
    is called, and closing then takes the early-return path where no session is
    built at all (covered separately below).  So the factory signals that it has
    been entered, and only then does the pool close.
    """
    entered = threading.Event()
    gate = threading.Event()
    built: list[FakeSession] = []

    def slow_factory(lease: PortLease) -> FakeSession:
        entered.set()
        gate.wait(10)
        session = FakeSession(lease)
        built.append(session)
        return session

    allocator = fake_allocator_factory(tmp_path / "race")
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(min_ready_sessions=1, status_heartbeat_interval_s=0),
        root_dir=tmp_path / "race",
        session_factory=slow_factory,
        port_allocator=allocator,
    )
    pool.start()
    assert entered.wait(10), "the session factory was never entered"
    pool.close()
    gate.set()
    assert wait_until(lambda: bool(built) and built[0].closed == 1)
    assert wait_until(lambda: allocator.released != [])


def test_closing_before_the_factory_runs_builds_no_session_and_frees_the_lease(tmp_path):
    """The other side of that window: close lands before the factory is called,
    so no session is constructed and the port block is handed straight back."""
    release_lease = threading.Event()
    built: list[FakeSession] = []
    allocator = fake_allocator_factory(tmp_path / "early")

    def gated_allocator(**kwargs):
        lease = allocator(**kwargs)
        release_lease.wait(10)
        return lease

    gated_allocator.released = allocator.released  # type: ignore[attr-defined]
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(min_ready_sessions=1, status_heartbeat_interval_s=0),
        root_dir=tmp_path / "early",
        session_factory=lambda lease: built.append(FakeSession(lease)) or built[-1],
        port_allocator=gated_allocator,
    )
    pool.start()
    assert wait_until(lambda: pool.snapshot()["starting"] == 1)
    pool.close()
    release_lease.set()
    assert wait_until(lambda: allocator.released != [])
    assert built == [], "no session should be built once the pool is closed"


def test_the_log_write_dir_is_symlinked_when_it_differs(tmp_path):
    pool = DesktopSessionPool(
        config=DesktopPoolConfig(
            min_ready_sessions=0,
            status_heartbeat_interval_s=0,
            log_runtime_dir=tmp_path / "elsewhere" / "logs",
        ),
        root_dir=tmp_path / "sym",
        session_factory=FakeSession,
        port_allocator=fake_allocator_factory(tmp_path / "sym"),
    )
    try:
        pool.start()
        assert pool.log_write_dir.is_symlink()
        assert pool.log_write_dir.resolve() == pool.log_dir.resolve()
    finally:
        pool.close()


def test_a_wrong_target_log_symlink_is_refused(tmp_path):
    from desktop.vm.pool import _ensure_symlink_dir

    (tmp_path / "real").mkdir()
    (tmp_path / "other").mkdir()
    link = tmp_path / "link"
    _ensure_symlink_dir(link, tmp_path / "real")
    _ensure_symlink_dir(link, tmp_path / "real")  # idempotent
    with pytest.raises(RuntimeError, match="refusing to replace"):
        _ensure_symlink_dir(link, tmp_path / "other")


def test_concurrent_checkouts_never_hand_the_same_session_to_two_callers(pool_factory):
    pool = pool_factory(min_ready_sessions=4, max_sessions=4, max_rollouts_per_session=50)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 4)
    seen: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def take():
        barrier.wait(10)
        handle = pool.checkout(timeout_s=10)
        with lock:
            seen.append(handle.session_id)
        time.sleep(0.05)
        handle.release()

    threads = [threading.Thread(target=take) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert len(seen) == 4
    assert len(set(seen)) == 4, f"a session was handed out twice: {seen}"


def test_every_leased_port_block_is_unique(pool_factory):
    pool = pool_factory(min_ready_sessions=3, max_sessions=3)
    pool.start()
    assert wait_until(lambda: pool.snapshot()["ready"] == 3)
    blocks = [
        tuple(sorted(session["ports"].values())) for session in pool.snapshot()["sessions"]
    ]
    assert len(set(blocks)) == 3
    flattened = [port for block in blocks for port in block]
    assert len(set(flattened)) == len(flattened)


def test_worker_ports_serialise_every_field():
    ports = WorkerPorts(server=1, chromium=2, vnc=3, vlc=4)
    assert ports.as_dict() == {"server": 1, "chromium": 2, "vnc": 3, "vlc": 4}
