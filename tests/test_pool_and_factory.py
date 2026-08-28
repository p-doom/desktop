"""``qemu_session_factory`` port pinning -- can two pooled desktops collide?

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

from desktop.vm import factory as factory_module
from desktop.vm.factory import (
    ENVIRONMENT,
    ConfigError,
    build_desktop_session,
    build_qemu_runtime,
    qemu_session_factory,
)
from desktop.vm.pool import (
    HUB_PORT_RANGE,
    PORT_BASE,
    DesktopPoolConfig,
    PortLease,
    WorkerPorts,
    acquire_port_range,
    allocate_worker_ports,
    ports_for_worker,
)
from desktop.vm.qemu import QemuError, QemuRuntime
from desktop.vm.runtime import GuestPorts


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
    def window_is_free(start: int) -> bool:
        for port in range(start, start + 80):
            probe = socket.socket()
            try:
                probe.bind(("", port))
            except OSError:
                return False
            finally:
                probe.close()
        return True

    # Walk candidate windows rather than skipping on the first busy one: another
    # tenant of the node holding a single port must not cost a test.  The walk
    # WRAPS inside the 55 windows that fit above the ephemeral range; it used to
    # run off 65535 and skip, so once this module held more ``port_base`` tests
    # than the per-process offset left room for, the tail of them silently
    # stopped running -- eight of them, measured.
    for _ in range(55):
        offset = (os.getpid() % 20) * 160 + next(_PORT_WINDOW) * 80
        base = 61000 + offset % 4400
        if window_is_free(base):
            return base
    pytest.skip("no free 80-port window above the ephemeral range")


@pytest.fixture
def image(tmp_path) -> Path:
    path = tmp_path / "guest.qcow2"
    path.write_bytes(b"\x00" * 64)
    return path


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
        qemu_session_factory(image=image, startup_timeout_s=1200.0)(lease)
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
    import desktop.vm.qemu as qemu_module

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


def test_a_runtime_without_a_leased_block_refuses_to_boot(image, monkeypatch):
    """The one allocator is `allocate_worker_ports`, so there is nowhere else for
    ports to come from. A self-allocating fallback would be a second allocator
    that cannot see the first's locks."""
    monkeypatch.setattr(QemuRuntime, "_wait_ready", lambda *a, **k: None)
    runtime = QemuRuntime(image=image, accelerator="tcg")
    with pytest.raises(QemuError, match="allocate_worker_ports"):
        runtime.start()


def test_two_leases_receive_disjoint_port_blocks(port_base, tmp_path):
    first = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work",
        log_dir=tmp_path / "logs", base=port_base,
    )
    second = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work",
        log_dir=tmp_path / "logs", base=port_base,
    )
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
        allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work",
        log_dir=tmp_path / "logs", base=port_base,
    )
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
    first = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work",
        log_dir=tmp_path / "logs", base=port_base,
    )
    slot, ports = first.slot, first.ports
    first.release()
    second = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work",
        log_dir=tmp_path / "logs", base=port_base,
    )
    try:
        assert second.slot == slot and second.ports == ports
    finally:
        second.release()


def test_a_lease_held_by_another_PROCESS_is_not_handed_out_again(port_base, tmp_path):
    """The flock is advisory and cross-process; a same-process test would pass
    trivially because ``flock`` is per-open-file-description."""
    locks = tmp_path / "locks"
    work = tmp_path / "work"
    logs = tmp_path / "logs"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys, time\n"
                f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"
                "from desktop.vm.pool import allocate_worker_ports\n"
                f"lease = allocate_worker_ports(\n"
                f"    lock_dir={str(locks)!r}, work_dir={str(work)!r},\n"
                f"    log_dir={str(logs)!r}, base={port_base}\n"
                f")\n"
                "print(lease.ports.server, flush=True)\n"
                "time.sleep(30)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        held_server_port = int(holder.stdout.readline().strip())
        mine = allocate_worker_ports(
            lock_dir=locks, work_dir=work, log_dir=logs, base=port_base
        )
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
        lease = allocate_worker_ports(lock_dir=tmp_path / "locks", work_dir=tmp_path / "work",
        log_dir=tmp_path / "logs", base=port_base,
    )
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
        server=20030,
        chromium=20031,
        vnc=20032,
        vlc=20033,
        auxiliary=(20034, 20035, 20036, 20037, 20038, 20039),
    )
    assert ports_for_worker(20000, 0).server + 10 == ports_for_worker(20000, 1).server


def test_a_block_beyond_the_tcp_range_is_refused():
    with pytest.raises(ValueError, match="exceeds the TCP port range"):
        ports_for_worker(65530, 1)


def test_one_span_for_the_node_rather_than_one_derived_per_job():
    """A base of `20000 + int(job_id) % 10000` collided EXACTLY for two jobs whose
    ids differ by a multiple of 10000, and no arithmetic on a monotonically
    increasing id avoids that. The span is shared and the lock arbitrates it."""
    assert PORT_BASE == 20000
    highest = ports_for_worker(PORT_BASE, 511)
    assert max(highest.as_dict().values()) < 32768, (
        "the span must stay clear of ip_local_port_range (32768-60999 here)"
    )


def test_exhausting_every_slot_raises_rather_than_reusing_one(port_base, tmp_path):
    leases = [
        allocate_worker_ports(
            lock_dir=tmp_path / "locks",
            work_dir=tmp_path / "work",
            log_dir=tmp_path / "logs",
            base=port_base,
            max_slots=2
        )
        for _ in range(2)
    ]
    try:
        with pytest.raises(RuntimeError, match="no available port blocks"):
            allocate_worker_ports(
                lock_dir=tmp_path / "locks",
            work_dir=tmp_path / "work",
            log_dir=tmp_path / "logs",
            base=port_base,
            max_slots=2
            )
    finally:
        for lease in leases:
            lease.release()


def test_an_acquired_range_is_contiguous_and_describes_itself(port_base, tmp_path):
    with acquire_port_range(
        count=3,
        purpose="unit-test",
        range_start=port_base,
        range_end=port_base + 30,
        lock_dir=tmp_path / "locks",
    ) as lease:
        assert lease.ports == (lease.start, lease.start + 1, lease.start + 2)
        assert lease.end == lease.start + 2
        assert lease.purpose == "unit-test"
        for port in lease.ports:
            assert (tmp_path / "locks" / f"port-{port}.lock").is_file()
    assert lease._released is True


def test_an_exhausted_range_raises_rather_than_reusing_a_port(port_base, tmp_path):
    held = [
        acquire_port_range(
            count=2,
            purpose="unit-test",
            range_start=port_base,
            range_end=port_base + 3,
            lock_dir=tmp_path / "locks",
            step=2,
        )
        for _ in range(2)
    ]
    try:
        assert {lease.start for lease in held} == {port_base, port_base + 2}
        with pytest.raises(RuntimeError, match="no available port blocks of 2"):
            acquire_port_range(
                count=2,
                purpose="unit-test",
                range_start=port_base,
                range_end=port_base + 3,
                lock_dir=tmp_path / "locks",
                step=2,
            )
    finally:
        for lease in held:
            lease.release()


def test_a_range_held_by_another_PROCESS_is_not_handed_out_again(port_base, tmp_path):
    """The flock is advisory and cross-process; a same-process test would pass
    trivially because ``flock`` is per-open-file-description."""
    locks = tmp_path / "locks"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys, time\n"
                f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"
                "from desktop.vm.pool import acquire_port_range\n"
                "lease = acquire_port_range(\n"
                f"    count=2, purpose='holder', range_start={port_base},\n"
                f"    range_end={port_base + 3}, lock_dir={str(locks)!r}, step=2\n"
                ")\n"
                "print(lease.start, flush=True)\n"
                "time.sleep(30)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        held_start = int(holder.stdout.readline().strip())
        mine = acquire_port_range(
            count=2,
            purpose="unit-test",
            range_start=port_base,
            range_end=port_base + 3,
            lock_dir=locks,
            step=2,
        )
        try:
            assert mine.start != held_start
            assert set(mine.ports).isdisjoint({held_start, held_start + 1})
        finally:
            mine.release()
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()


def test_a_loopback_bind_host_probes_only_that_address(port_base, tmp_path):
    """The value a hub passes, and the only one other than the default.  A probe
    on loopback must still refuse a port already held there."""
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", port_base))
    blocker.listen(1)
    try:
        with acquire_port_range(
            count=1,
            purpose="unit-test",
            range_start=port_base,
            range_end=port_base + 9,
            lock_dir=tmp_path / "locks",
            bind_host="127.0.0.1",
        ) as lease:
            assert lease.start != port_base
    finally:
        blocker.close()


def test_a_pinned_range_fails_rather_than_sliding_to_another(port_base, tmp_path):
    """A caller that pinned a base port published it somewhere, so quietly
    handing back a different one is worse than failing."""
    first = acquire_port_range(
        count=2,
        purpose="unit-test",
        range_start=port_base,
        range_end=port_base + 9,
        lock_dir=tmp_path / "locks",
        exact_start=port_base,
    )
    try:
        with pytest.raises(RuntimeError, match=f"port range {port_base}-"):
            acquire_port_range(
                count=2,
                purpose="unit-test",
                range_start=port_base,
                range_end=port_base + 9,
                lock_dir=tmp_path / "locks",
                exact_start=port_base,
            )
    finally:
        first.release()


def test_a_range_whose_port_is_bound_by_something_else_is_skipped(port_base, tmp_path):
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("", port_base + 1))
    blocker.listen(1)
    try:
        with acquire_port_range(
            count=2,
            purpose="unit-test",
            range_start=port_base,
            range_end=port_base + 9,
            lock_dir=tmp_path / "locks",
        ) as lease:
            assert port_base + 1 not in lease.ports
    finally:
        blocker.close()


def test_a_worker_block_and_a_free_range_share_one_lock_namespace(port_base, tmp_path):
    """The point of having ONE primitive.  A second allocator with its own lock
    naming would hand a hub the very ports a desktop is already forwarding."""
    locks = tmp_path / "locks"
    worker = allocate_worker_ports(
        lock_dir=locks, work_dir=tmp_path / "w", log_dir=tmp_path / "g", base=port_base
    )
    try:
        with pytest.raises(RuntimeError, match="no available port blocks"):
            acquire_port_range(
                count=1,
                purpose="unit-test",
                range_start=worker.ports.server,
                range_end=worker.ports.auxiliary[-1],
                lock_dir=locks,
            )
    finally:
        worker.release()


def test_every_new_primitive_is_reachable_from_the_subpackage():
    """The consumer imports from ``desktop.vm``, so a name that exists in a
    module but is missing from the package is not delivered."""
    import desktop.vm as package

    for name in (
        "HUB_PORT_RANGE",
        "DesktopResetMode",
        "GuestCommandResult",
        "PortRangeLease",
        "acquire_port_range",
    ):
        assert name in package.__all__
        assert getattr(package, name) is not None


def test_the_hub_span_does_not_overlap_the_desktop_span():
    """Both are leased from the same node-local lock directory, so an overlap
    would put two kinds of service in contention for one port."""
    desktop_end = max(ports_for_worker(PORT_BASE, 511).all())
    assert desktop_end < HUB_PORT_RANGE[0]
    assert HUB_PORT_RANGE == (30000, 39999)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0}, "count must be positive"),
        ({"purpose": "  "}, "purpose must not be empty"),
        ({"range_start": 0}, "must be within 1-65535"),
        ({"count": 50}, "smaller than the requested count"),
        ({"step": 0}, "step must be positive"),
        ({"exact_start": 70000}, "must lie inside the allocation range"),
    ],
)
def test_an_impossible_range_request_is_refused(port_base, tmp_path, kwargs, message):
    request = {
        "count": 2,
        "purpose": "unit-test",
        "range_start": port_base,
        "range_end": port_base + 9,
        "lock_dir": tmp_path / "locks",
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        acquire_port_range(**request)


def test_the_auxiliary_ports_are_the_rest_of_the_stride():
    block = ports_for_worker(20000, 0, stride=10)
    assert block.auxiliary == (20004, 20005, 20006, 20007, 20008, 20009)
    assert block.auxiliary_port() == 20004
    assert block.auxiliary_port(2) == 20006
    assert block.all() == (20000, 20001, 20002, 20003, *block.auxiliary)


def test_an_auxiliary_port_beyond_the_stride_is_refused():
    with pytest.raises(ValueError, match="no auxiliary worker port at index 6"):
        ports_for_worker(20000, 0, stride=10).auxiliary_port(6)


def test_an_auxiliary_port_is_leased_and_probed_like_the_forwarded_ones(
    port_base, tmp_path
):
    """A caller binds an auxiliary port for real, so a block whose auxiliary is
    taken is not usable.  Only the four forwarded ports used to be probed."""
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("", port_base + 5))  # an auxiliary port of slot 0
    blocker.listen(1)
    try:
        lease = allocate_worker_ports(
            lock_dir=tmp_path / "locks",
            work_dir=tmp_path / "w",
            log_dir=tmp_path / "g",
            base=port_base,
        )
        try:
            assert lease.slot != 0
            assert port_base + 5 not in lease.ports.all()
        finally:
            lease.release()
    finally:
        blocker.close()


def test_a_base_with_no_room_for_a_whole_block_is_refused(tmp_path):
    """65535 is a hard end, and the span is clamped to it, so a base this high
    reports that no whole block fits.  Unclamped, the span would run to 70649 and
    the complaint would be about a port number nobody asked for."""
    with pytest.raises(ValueError, match="smaller than the requested count"):
        allocate_worker_ports(
            lock_dir=tmp_path / "locks",
            work_dir=tmp_path / "w",
            log_dir=tmp_path / "g",
            base=65530,
        )


def test_a_failed_boot_is_not_retried_and_tears_the_process_down(image, monkeypatch):
    """Retrying the same four leased ports would hide a real conflict behind a
    timeout, and every unsuccessful exit has to tear the VM down: a boot timeout
    raises TimeoutError, and `__exit__` never runs if `__enter__` raised."""
    attempts: list[int] = []
    torn_down: list[int] = []

    def failing_start(self):
        attempts.append(1)
        raise QemuError("qemu died early (rc=1)")

    monkeypatch.setattr(QemuRuntime, "_start_once", failing_start)
    monkeypatch.setattr(
        QemuRuntime, "_teardown_process", lambda self: torn_down.append(1)
    )
    runtime = QemuRuntime(image=image, accelerator="tcg", ports=GuestPorts(server=41020))
    with pytest.raises(QemuError):
        runtime.start()
    assert attempts == [1]
    assert torn_down == [1]


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
    lease = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w", log_dir=tmp_path / "g",
        base=port_base,
    )
    try:
        qemu_session_factory(image=image, startup_timeout_s=1200.0)(lease)
        assert seen["require_single_task"] is False
        qemu_session_factory(image=image, startup_timeout_s=1200.0, require_single_task=True)(lease)
        assert seen["require_single_task"] is True
    finally:
        lease.release()


class DummyRuntime:
    """The minimum a ``DesktopSession`` needs, with no VM behind it."""

    name = "dummy"
    base_checkpoint = "base"

    def start(self):
        from desktop.vm.runtime import RuntimeState

        return RuntimeState(
            runtime_id="r",
            ports=GuestPorts(server=1),
            base_url="http://127.0.0.1:1",
            accelerator="tcg",
        )

    def stop(self):
        pass

    def ensure_base(self):
        from desktop.vm.runtime import Checkpoint

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
    from desktop.vm.session import DesktopSession, SessionError

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
    from desktop.vm.session import DesktopSession

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
    from desktop.vm.session import DesktopSession, SessionError

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
        lock_dir=tmp_path / "l",
        work_dir=tmp_path / "w",
        log_dir=tmp_path / "g",
        base=port_base,
    )
    try:
        qemu_session_factory(image=image, startup_timeout_s=1200.0)(lease)
    finally:
        lease.release()
    assert seen["scratch_root"] == lease.workdir
    assert seen["log_dir"] == lease.logdir
    # NOT the workdir: `release` removes that with `rmdir`, and a file we mean to
    # keep sitting in it makes the never-empty case the only case, which turns
    # "the directory is still there" from a leak signal into noise.
    assert seen["metadata_path"] == lease.logdir / "session.json"


def test_closing_a_pooled_session_releases_its_lease(port_base, tmp_path):
    from desktop.vm.pool import DesktopPoolSession, _close_session_resources

    lease = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w", log_dir=tmp_path / "g",
        base=port_base,
    )
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
    again = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w", log_dir=tmp_path / "g",
        base=port_base,
    )
    try:
        assert again.slot == lease.slot
    finally:
        again.release()


def test_a_lease_release_is_idempotent(port_base, tmp_path):
    lease = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w", log_dir=tmp_path / "g",
        base=port_base,
    )
    lease.release()
    lease.release()
    assert lease._released is True


def test_a_lease_is_a_context_manager(port_base, tmp_path):
    with allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w", log_dir=tmp_path / "g",
        base=port_base,
    ) as lease:
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
        lock_dir=tmp_path / "l",
        work_dir=tmp_path / "w",
        log_dir=tmp_path / "g",
        base=port_base,
    )
    workdir, logdir = lease.workdir, lease.logdir
    assert workdir.is_dir() and logdir.is_dir()
    lease.release()
    assert not workdir.exists()
    assert logdir.is_dir(), "the log directory exists to outlive the lease"
    assert (tmp_path / "w").is_dir(), "the shared root itself must survive"


def test_a_lease_whose_scratch_is_not_empty_is_left_alone(port_base, tmp_path):
    """``rmdir``, not ``rmtree``: a session that failed to clean up its own scratch
    must leave visible evidence rather than have it silently deleted."""
    lease = allocate_worker_ports(lock_dir=tmp_path / "l", work_dir=tmp_path / "w", log_dir=tmp_path / "g",
        base=port_base,
    )
    leftover = lease.workdir / "desktop-env-something" / "evidence.txt"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("a session did not clean up")
    lease.release()
    assert leftover.is_file(), "evidence of a failed cleanup must not be destroyed"


def test_a_pooled_session_leaves_its_scratch_empty_for_the_lease_to_remove(
    port_base, tmp_path, image, monkeypatch
):
    """The two halves of the contract have to hold at the same time.

    "``rmdir`` failed, so something leaked" is only information if the clean case
    really is empty.  Checking the leftover case alone passed while the pooled
    factory wrote ``session.json`` into the very directory it wanted removed:
    ``rmdir`` then raised ``ENOTEMPTY`` on every release, ``suppress(OSError)``
    swallowed it, and a genuine leak was indistinguishable from a clean exit.
    """
    from desktop.vm.session import DesktopSession

    lease = allocate_worker_ports(
        lock_dir=tmp_path / "l",
        work_dir=tmp_path / "w",
        log_dir=tmp_path / "g",
        base=port_base,
    )
    built: list[DesktopSession] = []

    def real_session_over_a_dummy_runtime(*, image, ports, log_dir, **kwargs):
        """Everything the factory decides, and the real metadata writer."""
        session = DesktopSession(DummyRuntime(), **kwargs)
        built.append(session)
        return session

    monkeypatch.setattr(
        factory_module, "build_desktop_session", real_session_over_a_dummy_runtime
    )
    session = qemu_session_factory(image=image, startup_timeout_s=1200.0)(lease)
    assert built == [session]
    session.close()

    assert (lease.logdir / "session.json").is_file(), "the metadata must be kept"
    assert sorted(p.name for p in lease.workdir.iterdir()) == []
    lease.release()
    assert not lease.workdir.exists()


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


def test_a_startup_budget_below_the_runtimes_own_phases_is_refused(image):
    """We publish `startup_timeout_s` for a REMOTE supervisor to judge us by.

    It shipped at 840 s against phases that legitimately sum to 960 (boot 300 +
    QMP connect 60 + snapshot 600), so a slow but healthy first boot was
    classified `stale_starting`, restarted mid-snapshot, leaked its VM, and
    repeated. Nothing enforced the relation; the number was only ever reported.
    """
    runtime = build_qemu_runtime(image=image, accelerator="tcg")
    assert runtime.start_budget_s == 960.0

    with pytest.raises(ConfigError, match="below this runtime's own worst-case"):
        qemu_session_factory(
            image=image, accelerator="tcg", startup_timeout_s=runtime.start_budget_s - 1
        )

    qemu_session_factory(
        image=image, accelerator="tcg", startup_timeout_s=runtime.start_budget_s
    )


def test_the_shipped_pool_default_satisfies_the_invariant(image):
    """The default configuration has to be a legal one, or the check is theatre."""
    qemu_session_factory(
        image=image,
        accelerator="tcg",
        startup_timeout_s=DesktopPoolConfig().startup_timeout_s,
    )


def test_a_lowered_phase_timeout_lowers_the_budget_it_is_checked_against(image):
    """The invariant is the relation, not either number: shrink a phase and a
    smaller budget becomes legal.

    `snapshot_timeout_s` is the largest term (600 of the 960) and
    `build_qemu_runtime` does not expose it, so `boot_timeout_s` is the only phase
    a caller can currently move.
    """
    assert (
        build_qemu_runtime(
            image=image, accelerator="tcg", boot_timeout_s=60.0
        ).start_budget_s
        == 720.0
    )
    qemu_session_factory(
        image=image, accelerator="tcg", boot_timeout_s=60.0, startup_timeout_s=720.0
    )
    with pytest.raises(ConfigError, match="below this runtime's own worst-case"):
        qemu_session_factory(
            image=image, accelerator="tcg", boot_timeout_s=60.0, startup_timeout_s=719.0
        )


def test_every_documented_variable_has_a_description():
    for name, description in ENVIRONMENT.items():
        assert name.startswith("DESKTOP_ENV_") and description.strip()


def test_importing_the_factory_has_no_side_effects():
    """No name registry, no plugin lookup, no import-time patching -- the whole
    point of the module, and the outage its docstring records."""
    import inspect

    source = inspect.getsource(factory_module)
    for forbidden in ("importlib", "entry_points", "register", "monkeypatch"):
        assert f"{forbidden}(" not in source
