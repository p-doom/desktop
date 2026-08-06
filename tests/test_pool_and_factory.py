"""ITEM 5: ``qemu_session_factory`` port pinning -- can two pooled desktops collide?

Background, and the reason this ranks above the sandbox provider: the pool's
``flock`` lease was previously DECORATIVE.  ``QemuRuntime`` allocated its own ports
with ``bind(0)`` + ``close``, so the lock guarded a block of ports nothing listened
on while QEMU bound an unrelated block.  Two pooled sessions could therefore
collide despite both holding a lease.

Four couplings make the lease mean something, and each is checked here:

1. ``lease.ports`` is threaded into the runtime;
2. two concurrent leases cannot receive overlapping blocks;
3. boot retry is pinned to ONE attempt when ports are pinned (retrying the same
   four ports hides a real conflict behind a timeout);
4. ``require_single_task`` defaults to ``False`` for pooled sessions -- ``True``
   makes the *second* session fail to start.
"""

from __future__ import annotations

import itertools
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from desktop_env.vm import factory as factory_module
from desktop_env.vm.factory import (
    ENVIRONMENT,
    ConfigError,
    build_desktop_session,
    build_qemu_runtime,
    qemu_session_factory,
)
from desktop_env.vm.pool import (
    PortLease,
    WorkerPorts,
    allocate_worker_ports,
    get_port_base,
    ports_for_worker,
)
from desktop_env.vm.qemu import QemuError, QemuRuntime
from desktop_env.vm.runtime import GuestPorts


_PORT_WINDOW = itertools.count()


@pytest.fixture
def port_base(monkeypatch):
    """A high, aligned port window unique to THIS test and THIS process.

    A single shared base made these tests contend with each other -- the
    exhaustion test needs its first two slots free, and the
    already-bound-port test binds one of them -- and made two concurrent runs of
    the suite collide. A per-test window removes both.

    The window sits ABOVE the kernel's ephemeral range
    (``/proc/sys/net/ipv4/ip_local_port_range``, 32768-60999 here). A window
    inside it was intermittently "busy" because unrelated outbound connections on
    a shared login node land there, which showed up as random skips.
    """
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("JOB_ID", raising=False)

    def window_is_free(start: int) -> bool:
        for port in range(start, start + 80):
            probe = socket.socket()
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("", port))
            except OSError:
                return False
            finally:
                probe.close()
        return True

    # Walk candidate windows rather than skipping on the first busy one: another
    # tenant of the node holding a single port must not cost a test.
    for _ in range(40):
        base = 61000 + (os.getpid() % 20) * 150 + next(_PORT_WINDOW) * 80
        base -= base % 10
        if base + 80 > 65535:
            break
        if window_is_free(base):
            monkeypatch.setenv("DESKTOP_ENV_PORT_BASE", str(base))
            return base
    pytest.skip("no free 80-port window above the ephemeral range")


@pytest.fixture
def image(tmp_path) -> Path:
    path = tmp_path / "guest.qcow2"
    path.write_bytes(b"\x00" * 64)
    return path


# --------------------------------------------------------------------------- #
# 1. lease.ports is threaded through
# --------------------------------------------------------------------------- #


def test_the_lease_ports_reach_the_runtime(port_base, tmp_path, image, monkeypatch):
    seen: dict = {}

    def recording_build(**kwargs):
        seen.update(kwargs)

        class Session:
            def start(self_inner):
                seen["started"] = True

            def close(self_inner):
                pass

        return Session()

    monkeypatch.setattr(factory_module, "build_desktop_session", recording_build)
    lease = allocate_worker_ports(
        lock_dir=tmp_path / "locks", work_dir=tmp_path / "work", log_dir=tmp_path / "logs"
    )
    try:
        qemu_session_factory(image=image)(lease)
    finally:
        lease.release()
    assert isinstance(seen["ports"], GuestPorts)
    assert seen["ports"].server == lease.ports.server
    assert seen["ports"].chromium == lease.ports.chromium
    assert seen["ports"].vnc == lease.ports.vnc
    assert seen["ports"].vlc == lease.ports.vlc
    assert seen["started"] is True


def test_the_runtime_forwards_exactly_the_pinned_ports(image, tmp_path, monkeypatch):
    """The end of the chain: the pinned block is what appears in ``-netdev``."""
    import desktop_env.vm.qemu as qemu_module

    captured: dict = {}

    class FakeProcess:
        pid = 1234
        returncode = None

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(qemu_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(QemuRuntime, "_wait_ready", lambda *a, **k: None)
    runtime = QemuRuntime(
        image=image,
        accelerator="tcg",
        ports=GuestPorts(server=41000, chromium=41001, vnc=41002, vlc=41003),
        log_dir=tmp_path / "logs",
        # AF_UNIX caps the path at ~108 bytes, so the socket cannot live under a
        # deep pytest tmp path.  That the runtime REFUSES a too-long one rather
        # than failing obscurely inside QEMU is asserted in test_qemu.py.
        qmp_dir=Path("/tmp"),
    )
    state = runtime.start()
    netdev = next(arg for arg in captured["command"] if arg.startswith("user,id=net0"))
    assert "hostfwd=tcp::41000-:5000" in netdev
    assert "hostfwd=tcp::41001-:9222" in netdev
    assert "hostfwd=tcp::41002-:5900" in netdev
    assert "hostfwd=tcp::41003-:8080" in netdev
    assert state.ports.server == 41000
    assert state.base_url == "http://127.0.0.1:41000"


def test_a_runtime_with_pinned_ports_never_calls_free_port(image, monkeypatch):
    import desktop_env.vm.qemu as qemu_module

    def explode():
        raise AssertionError("free_port must not be reached when ports are pinned")

    monkeypatch.setattr(qemu_module, "free_port", explode)
    monkeypatch.setattr(QemuRuntime, "_wait_ready", lambda *a, **k: None)

    class FakeProcess:
        pid = 1
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(qemu_module.subprocess, "Popen", lambda *a, **k: FakeProcess())
    runtime = QemuRuntime(
        image=image, accelerator="tcg", ports=GuestPorts(server=41010), qmp_dir=Path("/tmp")
    )
    assert runtime.start().ports.server == 41010


def test_an_unpinned_runtime_does_allocate_its_own_ports(image):
    runtime = QemuRuntime(image=image, accelerator="tcg")
    assert runtime.pinned_ports is None


# --------------------------------------------------------------------------- #
# 2. two leases cannot overlap
# --------------------------------------------------------------------------- #


def test_two_leases_receive_disjoint_port_blocks(port_base, tmp_path):
    first = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work")
    second = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work")
    try:
        assert first.slot != second.slot
        first_ports = set(first.ports.as_dict().values())
        assert first_ports.isdisjoint(second.ports.as_dict().values())
        assert first.workdir != second.workdir
    finally:
        first.release()
        second.release()


def test_many_concurrent_leases_are_all_pairwise_disjoint(port_base, tmp_path):
    leases = [
        allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work")
        for _ in range(6)
    ]
    try:
        blocks = [set(lease.ports.as_dict().values()) for lease in leases]
        assert len({lease.slot for lease in leases}) == len(leases)
        for index, block in enumerate(blocks):
            for other in blocks[index + 1 :]:
                assert block.isdisjoint(other)
        assert len({lease.workdir for lease in leases}) == len(leases)
    finally:
        for lease in leases:
            lease.release()


def test_a_released_slot_is_handed_out_again(port_base, tmp_path):
    first = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work")
    slot, ports = first.slot, first.ports
    first.release()
    second = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work")
    try:
        assert second.slot == slot and second.ports == ports
    finally:
        second.release()


def test_a_lease_held_by_another_PROCESS_is_not_handed_out_again(port_base, tmp_path):
    """The flock is advisory and cross-process; a same-process test would pass
    trivially because ``flock`` is per-open-file-description."""
    locks = tmp_path / "locks"
    work = tmp_path / "work"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys, time\n"
                f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"
                "from desktop_env.vm.pool import allocate_worker_ports\n"
                f"lease = allocate_worker_ports(\n"
                f"    lock_dir={str(locks)!r}, work_dir={str(work)!r}\n"
                f")\n"
                "print(lease.ports.server, flush=True)\n"
                "time.sleep(30)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "DESKTOP_ENV_PORT_BASE": str(port_base)},
    )
    try:
        held_server_port = int(holder.stdout.readline().strip())
        mine = allocate_worker_ports(lock_dir=locks, work_dir=work)
        try:
            assert mine.ports.server != held_server_port
            assert held_server_port not in set(mine.ports.as_dict().values())
        finally:
            mine.release()
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()


def test_a_port_already_bound_by_something_else_is_skipped(port_base, tmp_path):
    """The allocator probes each port in a block before accepting the slot."""
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("", port_base + 2))  # the vnc port of slot 0
    blocker.listen(1)
    try:
        lease = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work")
        try:
            assert lease.slot != 0
            assert port_base + 2 not in set(lease.ports.as_dict().values())
        finally:
            lease.release()
    finally:
        blocker.close()


def test_port_blocks_are_strided_and_aligned():
    block = ports_for_worker(20000, 3, stride=10)
    assert block == WorkerPorts(
        server=20030, chromium=20031, vnc=20032, vlc=20033
    )
    assert ports_for_worker(20000, 0).server + 10 == ports_for_worker(20000, 1).server


def test_a_block_beyond_the_tcp_range_is_refused():
    with pytest.raises(ValueError, match="exceeds the TCP port range"):
        ports_for_worker(65530, 1)


def test_the_port_base_is_per_job_so_two_jobs_on_one_node_do_not_collide():
    assert get_port_base("12345") != get_port_base("12346")
    assert get_port_base(None) == 20000
    assert get_port_base("not-a-number") == 20000
    assert get_port_base(None, configured_base="30000") == 30000


@pytest.mark.parametrize("bad", ["0", "1023", "70000"])
def test_an_out_of_range_configured_base_is_refused(bad):
    with pytest.raises(ValueError, match="between 1024 and 65535"):
        get_port_base(None, configured_base=bad)


def test_a_non_numeric_configured_base_is_refused():
    with pytest.raises(ValueError, match="must be an integer"):
        get_port_base(None, configured_base="abc")


def test_a_misaligned_configured_base_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("DESKTOP_ENV_PORT_BASE", "41005")
    with pytest.raises(ValueError, match="aligned to port stride"):
        allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work")


def test_exhausting_every_slot_raises_rather_than_reusing_one(port_base, tmp_path):
    leases = [
        allocate_worker_ports(
            lock_dir=tmp_path / "locks", work_dir=tmp_path / "work", max_slots=2
        )
        for _ in range(2)
    ]
    try:
        with pytest.raises(RuntimeError, match="no available port blocks"):
            allocate_worker_ports(
                lock_dir=tmp_path / "locks", work_dir=tmp_path / "work", max_slots=2
            )
    finally:
        for lease in leases:
            lease.release()


# --------------------------------------------------------------------------- #
# 3. boot retry is pinned to one attempt
# --------------------------------------------------------------------------- #


def test_boot_retry_is_pinned_to_one_attempt_when_ports_are_pinned(image, monkeypatch):
    """Retrying the same four ports would hide a real conflict behind a timeout."""
    attempts = []

    def failing_start(self):
        attempts.append(1)
        raise QemuError("qemu died early (rc=1)")

    monkeypatch.setattr(QemuRuntime, "_start_once", failing_start)
    runtime = QemuRuntime(image=image, accelerator="tcg", ports=GuestPorts(server=41020))
    with pytest.raises(QemuError):
        runtime.start()
    assert len(attempts) == 1


def test_an_unpinned_runtime_still_retries_the_lost_bind_race(image, monkeypatch):
    attempts = []

    def failing_start(self):
        attempts.append(1)
        raise QemuError("qemu died early (rc=1)")

    monkeypatch.setattr(QemuRuntime, "_start_once", failing_start)
    monkeypatch.setattr("desktop_env.vm.qemu.time.sleep", lambda seconds: None)
    with pytest.raises(QemuError):
        QemuRuntime(image=image, accelerator="tcg").start()
    assert len(attempts) == 3


def test_a_non_race_failure_is_not_retried_even_unpinned(image, monkeypatch):
    attempts = []

    def failing_start(self):
        attempts.append(1)
        raise QemuError("something else entirely")

    monkeypatch.setattr(QemuRuntime, "_start_once", failing_start)
    with pytest.raises(QemuError, match="something else"):
        QemuRuntime(image=image, accelerator="tcg").start()
    assert len(attempts) == 1


# --------------------------------------------------------------------------- #
# 4. require_single_task defaults to False for pooled sessions
# --------------------------------------------------------------------------- #


def test_the_pooled_factory_defaults_require_single_task_to_false():
    import inspect

    assert inspect.signature(qemu_session_factory).parameters[
        "require_single_task"
    ].default is False


def test_a_standalone_session_still_defaults_it_to_true():
    """The guarantee exists for the opposite case: a scheduler task that must own
    exactly one VM."""
    import inspect

    assert inspect.signature(build_desktop_session).parameters[
        "require_single_task"
    ].default is True


def test_the_pooled_factory_passes_the_flag_through(port_base, tmp_path, image, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        factory_module,
        "build_desktop_session",
        lambda **kwargs: seen.update(kwargs) or type("S", (), {"start": lambda s: None})(),
    )
    lease = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w")
    try:
        qemu_session_factory(image=image)(lease)
        assert seen["require_single_task"] is False
        qemu_session_factory(image=image, require_single_task=True)(lease)
        assert seen["require_single_task"] is True
    finally:
        lease.release()


class DummyRuntime:
    """The minimum a ``DesktopSession`` needs, with no VM behind it."""

    name = "dummy"
    base_checkpoint = "base"

    def start(self):
        from desktop_env.vm.runtime import RuntimeState

        return RuntimeState(
            runtime_id="r",
            ports=GuestPorts(server=1),
            base_url="http://127.0.0.1:1",
            accelerator="tcg",
        )

    def stop(self):
        pass

    def ensure_base(self):
        from desktop_env.vm.runtime import Checkpoint

        return Checkpoint("base", 0)

    def is_ready(self, *, timeout_s=0.0):
        return True

    def list_checkpoints(self):
        return ()


def test_two_sessions_sharing_a_scratch_root_collide_on_the_task_lock(tmp_path):
    """DOCUMENTS A DEFECT in the factory's stated reasoning.

    ``require_single_task`` does NOT gate the one-VM-per-task ``flock``: the lock
    is taken unconditionally in ``_prepare_isolation``, and the flag only decides
    whether a ``SLURM_NTASKS != 1`` environment is rejected.  So the second
    session fails here even with the pooled default of ``False``.
    """
    from desktop_env.vm.session import DesktopSession, SessionError

    first = DesktopSession(
        DummyRuntime(), scratch_root=tmp_path, session_id="a", require_single_task=False
    )
    first.start()
    second = DesktopSession(
        DummyRuntime(), scratch_root=tmp_path, session_id="b", require_single_task=False
    )
    try:
        with pytest.raises(SessionError, match="another VM is already live"):
            second.start()
    finally:
        first.close()


def test_what_actually_separates_two_pooled_sessions_is_the_per_lease_scratch_root(
    tmp_path,
):
    """The real mechanism, which the factory docstring attributes to the flag.

    Each lease has its own ``workdir``, the session's task lock lives INSIDE its
    scratch root, so two pooled sessions never contend -- at either setting of
    ``require_single_task``.  The pooled default of ``False`` is still right, but
    for a different reason than stated: a pool process legitimately runs under
    ``SLURM_NTASKS > 1``.
    """
    from desktop_env.vm.session import DesktopSession

    for flag in (False, True):
        first = DesktopSession(
            DummyRuntime(),
            scratch_root=tmp_path / f"w_{flag}_0",
            session_id=f"c{flag}",
            require_single_task=flag,
        )
        second = DesktopSession(
            DummyRuntime(),
            scratch_root=tmp_path / f"w_{flag}_1",
            session_id=f"d{flag}",
            require_single_task=flag,
        )
        first.start()
        try:
            second.start()  # would raise if the lock were shared
        finally:
            second.close()
            first.close()


def test_the_flag_only_gates_the_scheduler_task_count_check(tmp_path, monkeypatch):
    """What ``require_single_task`` actually controls, pinned."""
    from desktop_env.vm.session import DesktopSession, SessionError

    monkeypatch.setenv("SLURM_NTASKS", "4")
    strict = DesktopSession(
        DummyRuntime(), scratch_root=tmp_path / "s", session_id="s", require_single_task=True
    )
    with pytest.raises(SessionError, match="exactly one scheduler task is required"):
        strict.start()
    relaxed = DesktopSession(
        DummyRuntime(), scratch_root=tmp_path / "r", session_id="r", require_single_task=False
    )
    relaxed.start()
    relaxed.close()


# --------------------------------------------------------------------------- #
# Scratch is released with the lease
# --------------------------------------------------------------------------- #


def test_the_lease_workdir_becomes_the_session_scratch_root(
    port_base, tmp_path, image, monkeypatch
):
    seen: dict = {}
    monkeypatch.setattr(
        factory_module,
        "build_desktop_session",
        lambda **kwargs: seen.update(kwargs) or type("S", (), {"start": lambda s: None})(),
    )
    lease = allocate_worker_ports(
        lock_dir=tmp_path / "l", work_dir=tmp_path / "w", log_dir=tmp_path / "g"
    )
    try:
        qemu_session_factory(image=image)(lease)
    finally:
        lease.release()
    assert seen["scratch_root"] == lease.workdir
    assert seen["metadata_path"] == lease.workdir / "session.json"
    assert seen["log_dir"] == lease.logdir


def test_closing_a_pooled_session_releases_its_lease(port_base, tmp_path):
    from desktop_env.vm.pool import DesktopPoolSession, _close_session_resources

    lease = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w")
    closed = []
    session = DesktopPoolSession(
        session_id="s",
        env=type("E", (), {"close": lambda self: closed.append(1)})(),
        lease=lease,
        status="ready",
        rollouts_completed=0,
        created_at=0.0,
        updated_at=0.0,
    )
    _close_session_resources(session)
    assert closed == [1]
    assert lease._released is True
    # ... and the slot is immediately reusable.
    again = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w")
    try:
        assert again.slot == lease.slot
    finally:
        again.release()


def test_a_lease_release_is_idempotent(port_base, tmp_path):
    lease = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w")
    lease.release()
    lease.release()
    assert lease._released is True


def test_a_lease_is_a_context_manager(port_base, tmp_path):
    with allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w") as lease:
        assert isinstance(lease, PortLease)
    assert lease._released is True


def test_the_lease_describes_exactly_the_ports_the_runtime_forwards():
    """A fifth port, ``qemu_vnc``, used to be leased and probed and then never
    forwarded: the runtime boots ``-display none -nographic``, so QEMU has no VNC
    server to expose.  The lease described a service that did not exist."""
    assert set(ports_for_worker(20000, 0).as_dict()) == set(GuestPorts(server=1).as_dict())
    assert set(GuestPorts(server=1).as_dict()) == {"server", "chromium", "vnc", "vlc"}


def test_removing_the_fifth_port_did_not_move_any_slot(port_base, tmp_path):
    """The stride stays 10, so every slot's four ports keep their old numbers."""
    for slot in range(4):
        block = ports_for_worker(port_base, slot)
        assert block.server == port_base + slot * 10
        assert block.chromium == block.server + 1
        assert block.vnc == block.server + 2
        assert block.vlc == block.server + 3


def test_a_released_lease_removes_its_own_working_directory(port_base, tmp_path):
    """``runtime_dir`` used to accumulate one ``w<pid>_<slot>`` directory per
    process forever, on a shared filesystem."""
    lease = allocate_worker_ports(
        lock_dir=tmp_path / "l", work_dir=tmp_path / "w", log_dir=tmp_path / "g"
    )
    workdir, logdir = lease.workdir, lease.logdir
    assert workdir.is_dir() and logdir.is_dir()
    lease.release()
    assert not workdir.exists()
    assert not logdir.exists()
    assert (tmp_path / "w").is_dir(), "the shared root itself must survive"


def test_a_lease_whose_scratch_is_not_empty_is_left_alone(port_base, tmp_path):
    """``rmdir``, not ``rmtree``: a session that failed to clean up its own scratch
    must leave visible evidence rather than have it silently deleted."""
    lease = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w")
    leftover = lease.workdir / "desktop-env-something" / "evidence.txt"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("a session did not clean up")
    lease.release()
    assert leftover.is_file(), "evidence of a failed cleanup must not be destroyed"


# --------------------------------------------------------------------------- #
# Factory configuration hygiene
# --------------------------------------------------------------------------- #


def test_a_missing_image_is_an_error_rather_than_a_guess(monkeypatch):
    monkeypatch.delenv("DESKTOP_ENV_IMAGE", raising=False)
    with pytest.raises(ConfigError, match="no guest image"):
        build_qemu_runtime()


def test_a_nonexistent_image_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        build_qemu_runtime(image=tmp_path / "absent.qcow2")


def test_an_explicit_argument_beats_the_environment(image, monkeypatch, tmp_path):
    other = tmp_path / "other.qcow2"
    other.write_bytes(b"\x00")
    monkeypatch.setenv("DESKTOP_ENV_IMAGE", str(other))
    assert build_qemu_runtime(image=image, accelerator="tcg").image == image.resolve()


def test_the_environment_is_the_documented_fallback(image, monkeypatch):
    monkeypatch.setenv("DESKTOP_ENV_IMAGE", str(image))
    monkeypatch.setenv("DESKTOP_ENV_VM_SMP", "7")
    monkeypatch.setenv("DESKTOP_ENV_VM_MEM", "3G")
    monkeypatch.setenv("DESKTOP_ENV_ACCEL", "tcg")
    runtime = build_qemu_runtime()
    assert runtime.smp == 7 and runtime.memory == "3G" and runtime.accelerator == "tcg"


@pytest.mark.parametrize("bad", ["nonsense", "kvm2", "TCG"])
def test_an_unknown_accelerator_is_refused(image, bad):
    with pytest.raises(ConfigError, match="must be 'kvm' or 'tcg'"):
        build_qemu_runtime(image=image, accelerator=bad)


@pytest.mark.parametrize("bad", ["zero", "1.5"])
def test_a_non_integer_smp_is_refused(image, bad):
    with pytest.raises(ConfigError, match="must be an integer"):
        build_qemu_runtime(image=image, smp=bad, accelerator="tcg")


@pytest.mark.parametrize("bad", ["8GB", "eight", "8 G", "-4G"])
def test_a_malformed_memory_size_is_refused_before_anything_boots(image, bad, monkeypatch):
    """Every other value in ``ENVIRONMENT`` is checked here; this one used to go
    straight to QEMU's ``-m``, so a typo surfaced as a VM that died during
    startup -- a config error wearing a boot failure's message."""
    monkeypatch.setenv("DESKTOP_ENV_VM_MEM", bad)
    with pytest.raises(ConfigError, match="must be a QEMU -m size"):
        build_qemu_runtime(image=image, accelerator="tcg")


@pytest.mark.parametrize("good", ["8G", "4096M", "2g", "512"])
def test_a_well_formed_memory_size_is_accepted(image, good):
    assert build_qemu_runtime(image=image, memory=good, accelerator="tcg").memory == good


def test_a_zero_smp_is_refused(image):
    with pytest.raises(ConfigError, match="at least 1"):
        build_qemu_runtime(image=image, smp=0, accelerator="tcg")


def test_every_documented_variable_has_a_description():
    assert "DESKTOP_ENV_PORT_BASE" in ENVIRONMENT
    for name, description in ENVIRONMENT.items():
        assert name.startswith("DESKTOP_ENV_") and description.strip()


def test_importing_the_factory_has_no_side_effects():
    """No name registry, no plugin lookup, no import-time patching -- the whole
    point of the module, and the outage its docstring records."""
    import inspect

    source = inspect.getsource(factory_module)
    for forbidden in ("importlib", "entry_points", "register", "monkeypatch"):
        assert f"{forbidden}(" not in source
