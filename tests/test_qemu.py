"""``QemuRuntime.fork()``, ``_pick_accelerator`` / ``-accel tcg``, and QMP.

Never run before.  Split into three tiers so as much as possible runs anywhere:

* **no binary needed** -- argv construction, accelerator selection, the QMP client
  against a fake QMP server on a real unix socket, and ``fork``'s bookkeeping
  against a stub ``qemu-img`` that really creates the overlay file;
* **``needs_vm``** -- the same ``fork`` against a real ``qemu-img``, which is the
  only way to know the ``-b``/``-F`` pair actually produces the backing chain;
* **``needs_vm`` + ``kvm``** -- a real boot, real ``savevm``/``loadvm``/``delvm``.

On a node with no qemu at all, point ``DESKTOP_ENV_QEMU_BIN`` /
``DESKTOP_ENV_QEMU_IMG_BIN`` at a wrapper that ``exec``s into the KVM-tier
container; that is how the marked tests were run.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
import time
from pathlib import Path

import pytest

import desktop.vm.qemu as qemu_module
from desktop.vm.qemu import (
    BASE_CHECKPOINT,
    QemuError,
    QemuRuntime,
    QmpClient,
    free_port,
    kvm_available,
)
from desktop.vm.runtime import GuestPorts
from tests.conftest import qemu_img_binary, qemu_system_binary

STUB_QEMU_IMG = r"""#!/bin/bash
# A stub qemu-img: logs argv, and really creates the overlay file.
printf '%s\n' "$*" >> "${STUB_QEMU_IMG_LOG:-/dev/null}"
if [ "$1" = "create" ]; then
  out="${@: -1}"
  base=""
  previous=""
  for arg in "$@"; do
    [ "$previous" = "-b" ] && base="$arg"
    previous="$arg"
  done
  if [ -n "$base" ] && [ ! -f "$base" ]; then
    echo "qemu-img: backing file '$base' does not exist" >&2; exit 1
  fi
  printf 'QFI\xfb stub overlay of %s\n' "$base" > "$out"
  exit 0
fi
echo "stub qemu-img: unsupported subcommand $1" >&2
exit 2
"""


@pytest.fixture
def base_image(tmp_path) -> Path:
    path = tmp_path / "base.qcow2"
    path.write_bytes(b"\x00" * 128)
    return path


@pytest.fixture
def stub_qemu_img(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "qemu-img"
    path.write_text(STUB_QEMU_IMG)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("STUB_QEMU_IMG_LOG", str(tmp_path / "qemu_img.log"))
    return path


def _stub_log(tmp_path) -> list[str]:
    log = tmp_path / "qemu_img.log"
    return log.read_text().strip().splitlines() if log.exists() else []


def _free_guest_port_block(count: int = 4) -> GuestPorts:
    """Four consecutive free ports above the kernel's ephemeral range.

    QEMU binds these itself, so they must be genuinely free at boot time -- and
    they must not sit inside ``ip_local_port_range``, where an unrelated outbound
    connection on a shared node can take one.
    """
    for base in range(43000, 44000, 10):
        probes = []
        try:
            for offset in range(count):
                probe = socket.socket()
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("", base + offset))
                probes.append(probe)
        except OSError:
            continue
        finally:
            for probe in probes:
                probe.close()
        if len(probes) == count:
            return GuestPorts(
                server=base, chromium=base + 1, vnc=base + 2, vlc=base + 3
            )
    pytest.skip("no free 4-port block in 43000..44000")


def test_free_port_never_hands_out_the_same_port_twice_in_a_process():
    ports = {free_port() for _ in range(40)}
    assert len(ports) == 40


def test_free_ports_are_in_the_ephemeral_range():
    assert 1024 < free_port() <= 65535


def test_kvm_is_chosen_when_the_node_has_it(base_image, monkeypatch):
    monkeypatch.setattr(qemu_module, "kvm_available", lambda: True)
    assert QemuRuntime(image=base_image).accelerator == "kvm"


def test_tcg_is_chosen_when_dev_kvm_is_missing(base_image, monkeypatch):
    """A node without /dev/kvm is a SLOW node, not a dead one."""
    monkeypatch.setattr(qemu_module, "kvm_available", lambda: False)
    assert QemuRuntime(image=base_image).accelerator == "tcg"


def test_falling_back_to_tcg_warns_that_it_invalidates_timings(
    base_image, monkeypatch, caplog
):
    monkeypatch.setattr(qemu_module, "kvm_available", lambda: False)
    with caplog.at_level("WARNING", logger="desktop.vm.qemu"):
        QemuRuntime(image=base_image)
    assert "not valid for timing" in caplog.text


@pytest.mark.parametrize("choice", ["kvm", "tcg"])
def test_an_explicit_accelerator_is_never_overridden(base_image, choice, monkeypatch):
    monkeypatch.setattr(qemu_module, "kvm_available", lambda: choice == "tcg")
    assert QemuRuntime(image=base_image, accelerator=choice).accelerator == choice


def test_kvm_available_matches_the_device_permissions():
    assert kvm_available() == os.access("/dev/kvm", os.R_OK | os.W_OK)


@pytest.fixture
def captured_argv(monkeypatch):
    captured: dict = {}

    class FakeProcess:
        pid = 4242
        returncode = None

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(qemu_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(QemuRuntime, "_wait_ready", lambda *a, **k: None)
    return captured


def test_tcg_argv_asks_for_tcg_and_a_portable_cpu(base_image, captured_argv, tmp_path):
    QemuRuntime(
        image=base_image,
        accelerator="tcg",
        ports=GuestPorts(server=42000),
        log_dir=tmp_path,
        qmp_dir=Path("/tmp"),
    ).start()
    command = captured_argv["command"]
    assert "type=q35,accel=tcg" in command
    assert "-enable-kvm" not in command
    assert command[command.index("-cpu") + 1] == "max"


def test_kvm_argv_enables_kvm_and_passes_the_host_cpu(base_image, captured_argv, tmp_path):
    QemuRuntime(
        image=base_image,
        accelerator="kvm",
        ports=GuestPorts(server=42010),
        log_dir=tmp_path,
        qmp_dir=Path("/tmp"),
    ).start()
    command = captured_argv["command"]
    assert "-enable-kvm" in command
    assert "type=q35,accel=kvm" in command
    assert command[command.index("-cpu") + 1] == "host"


def test_the_image_is_opened_with_snapshot_on(base_image, captured_argv, tmp_path):
    """So neither the parent nor a fork ever writes the backing file."""
    QemuRuntime(
        image=base_image, accelerator="tcg", ports=GuestPorts(server=42020),
        log_dir=tmp_path, qmp_dir=Path("/tmp"),
    ).start()
    drive = next(arg for arg in captured_argv["command"] if arg.startswith("file="))
    assert "snapshot=on" in drive
    assert "format=qcow2" in drive


def test_a_qmp_monitor_is_always_requested(base_image, captured_argv, tmp_path):
    """Without it savevm/loadvm are unreachable."""
    QemuRuntime(
        image=base_image, accelerator="tcg", ports=GuestPorts(server=42030),
        log_dir=tmp_path, qmp_dir=Path("/tmp"),
    ).start()
    command = captured_argv["command"]
    qmp = command[command.index("-qmp") + 1]
    assert qmp.startswith("unix:") and "server=on" in qmp and "wait=off" in qmp


def test_a_too_long_qmp_path_is_refused_before_qemu_starts(base_image, tmp_path):
    """AF_UNIX caps the path length; failing here beats failing inside QEMU."""
    deep = tmp_path / ("d" * 90) / ("e" * 90)
    deep.mkdir(parents=True)
    with pytest.raises(QemuError, match="too long for AF_UNIX"):
        QemuRuntime(
            image=base_image, accelerator="tcg", ports=GuestPorts(server=42040), qmp_dir=deep
        ).start()


def test_a_missing_image_is_refused_before_qemu_starts(tmp_path):
    with pytest.raises(QemuError, match="VM image missing"):
        QemuRuntime(image=tmp_path / "absent.qcow2", accelerator="tcg").start()


def test_state_before_start_is_an_error(base_image):
    with pytest.raises(QemuError, match="not started"):
        QemuRuntime(image=base_image, accelerator="tcg").state()


def test_is_ready_is_false_before_start(base_image):
    assert QemuRuntime(image=base_image, accelerator="tcg").is_ready() is False


def test_a_boot_timeout_tears_the_process_down(base_image, tmp_path, monkeypatch):
    """``_wait_ready`` raises ``TimeoutError``, not ``QemuError``, so it matched
    no handler in ``start`` and propagated with QEMU still running -- holding its
    port block and its whole ``-m`` allocation, with the QMP socket still on
    disk.  ``__exit__`` never runs when ``__enter__`` raises, so
    ``with QemuRuntime(...) as vm:`` leaked a VM on every failed boot."""
    torn_down = []

    class FakeProcess:
        """Alive until terminated, so the teardown path is really taken."""

        pid = 5
        returncode = None
        _alive = True

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self._alive = False
            self.returncode = 0

        def kill(self):
            self.terminate()

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(qemu_module.subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr(
        QemuRuntime,
        "_wait_ready",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("guest agent not ready")),
    )
    original = QemuRuntime._teardown_process

    def traced(self):
        torn_down.append(1)
        return original(self)

    monkeypatch.setattr(QemuRuntime, "_teardown_process", traced)
    runtime = QemuRuntime(
        image=base_image, accelerator="tcg", ports=GuestPorts(server=42050),
        log_dir=tmp_path, qmp_dir=Path("/tmp"),
    )
    with pytest.raises(TimeoutError):
        with runtime:
            pass
    assert torn_down, "a failed boot must tear the process down"
    assert runtime._process is None
    assert runtime._ports is None and runtime._qmp_path is None


def test_a_non_retryable_qemu_error_also_tears_down(base_image, monkeypatch):
    torn_down = []
    monkeypatch.setattr(
        QemuRuntime,
        "_start_once",
        lambda self: (_ for _ in ()).throw(QemuError("something structural")),
    )
    monkeypatch.setattr(
        QemuRuntime, "_teardown_process", lambda self: torn_down.append(1)
    )
    with pytest.raises(QemuError, match="something structural"):
        QemuRuntime(image=base_image, accelerator="tcg").start()
    assert torn_down == [1]


def test_stop_is_idempotent(base_image):
    runtime = QemuRuntime(image=base_image, accelerator="tcg")
    runtime.stop()
    runtime.stop()


@pytest.fixture
def qmp_server(tmp_path):
    """A fake QMP server: greeting, interleaved async events, errors."""
    path = str(tmp_path / "q.sock")
    received: list[dict] = []

    def serve():
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)
        connection, _ = server.accept()
        connection.sendall(b'{"QMP":{"version":{"qemu":{"major":6}}}}\n')
        buffer = b""
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                message = json.loads(line)
                received.append(message)
                command = message["execute"]
                # An async event ALWAYS precedes the reply, so a client that does
                # not skip events would read an event as its return value.
                connection.sendall(b'{"event":"RESUME","timestamp":{}}\n')
                if command == "human-monitor-command":
                    line_in = message["arguments"]["command-line"]
                    output = "Error: no such snapshot" if "missing" in line_in else ""
                    connection.sendall(json.dumps({"return": output}).encode() + b"\n")
                elif command == "explode":
                    connection.sendall(
                        b'{"error":{"class":"GenericError","desc":"it exploded"}}\n'
                    )
                else:
                    connection.sendall(b'{"return":null}\n')

    threading.Thread(target=serve, daemon=True).start()
    return path, received


def test_the_client_completes_the_qmp_handshake(qmp_server):
    path, received = qmp_server
    client = QmpClient(path, connect_timeout_s=5)
    try:
        assert received[0]["execute"] == "qmp_capabilities"
    finally:
        client.close()


def test_async_events_are_skipped_rather_than_read_as_replies(qmp_server):
    path, _ = qmp_server
    client = QmpClient(path, connect_timeout_s=5)
    try:
        assert client.execute("stop") is None
        assert client.hmp("savevm base") == ""
    finally:
        client.close()


def test_a_qmp_error_reply_raises(qmp_server):
    path, _ = qmp_server
    client = QmpClient(path, connect_timeout_s=5)
    try:
        with pytest.raises(QemuError, match="it exploded"):
            client.execute("explode")
    finally:
        client.close()


def test_hmp_returns_the_monitor_text(qmp_server):
    path, _ = qmp_server
    client = QmpClient(path, connect_timeout_s=5)
    try:
        assert "no such snapshot" in client.hmp("loadvm missing")
    finally:
        client.close()


def test_connecting_to_a_socket_that_never_appears_times_out(tmp_path):
    with pytest.raises(QemuError, match="could not connect to QMP socket"):
        QmpClient(str(tmp_path / "never.sock"), connect_timeout_s=0.5)


def _forkable(base_image, binary, tmp_path, **kwargs) -> QemuRuntime:
    return QemuRuntime(
        image=base_image,
        qemu_img_binary=str(binary),
        overlay_dir=tmp_path / "overlays",
        log_dir=tmp_path / "logs",
        qmp_dir=Path("/tmp"),
        accelerator="tcg",
        runtime_id="parent",
        **kwargs,
    )


def test_fork_builds_the_documented_qemu_img_command(base_image, stub_qemu_img, tmp_path):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    child = parent.fork(name="alpha")
    (logged,) = _stub_log(tmp_path)
    assert logged.split() == [
        "create",
        "-f",
        "qcow2",
        "-b",
        str(base_image.resolve()),
        "-F",
        "qcow2",
        str(child.image),
    ]


def test_the_backing_format_is_always_explicit(base_image, stub_qemu_img, tmp_path):
    """Modern ``qemu-img`` refuses to probe the backing format."""
    _forkable(base_image, stub_qemu_img, tmp_path).fork(name="a")
    assert "-F qcow2" in _stub_log(tmp_path)[0]


def test_a_child_owns_its_overlay_and_the_parent_does_not(
    base_image, stub_qemu_img, tmp_path
):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    child = parent.fork(name="alpha")
    assert child.owns_image is True
    assert parent.owns_image is False
    assert child.image.exists()
    assert child.image.parent == tmp_path / "overlays"


def test_a_child_unlinks_its_overlay_when_stopped(base_image, stub_qemu_img, tmp_path):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    child = parent.fork(name="alpha")
    overlay = child.image
    child.stop()
    assert not overlay.exists()
    assert base_image.exists(), "the parent's backing file must survive"


def test_the_parent_stops_its_children_before_itself(base_image, stub_qemu_img, tmp_path):
    order: list[str] = []
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    children = [parent.fork(name=f"c{index}") for index in range(3)]
    for child in children:
        child._teardown_process = (
            lambda identifier=child.runtime_id: order.append(identifier)
        )
    parent._teardown_process = lambda: order.append("parent")
    parent.stop()
    assert order[-1] == "parent", order
    assert set(order[:-1]) == {child.runtime_id for child in children}


def test_stopping_the_parent_removes_every_child_overlay(
    base_image, stub_qemu_img, tmp_path
):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    overlays = [parent.fork(name=f"c{index}").image for index in range(3)]
    assert all(path.exists() for path in overlays)
    parent.stop()
    assert not any(path.exists() for path in overlays)
    assert parent._children == []


def test_a_child_whose_stop_raises_does_not_block_the_parent(
    base_image, stub_qemu_img, tmp_path
):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    first = parent.fork(name="boom")
    second = parent.fork(name="fine")
    first.stop = lambda: (_ for _ in ()).throw(RuntimeError("child stop exploded"))
    parent.stop()
    assert not second.image.exists()
    assert parent._children == []


def test_a_fork_name_collision_is_refused(base_image, stub_qemu_img, tmp_path):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    parent.fork(name="alpha")
    with pytest.raises(QemuError, match="fork overlay already exists"):
        parent.fork(name="alpha")


def test_an_unnamed_fork_gets_a_unique_name(base_image, stub_qemu_img, tmp_path):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    names = {parent.fork().image.name for _ in range(5)}
    assert len(names) == 5


def test_a_child_inherits_the_parents_configuration(base_image, stub_qemu_img, tmp_path):
    parent = _forkable(
        base_image, stub_qemu_img, tmp_path, smp=3, memory="2G", base_checkpoint="bc"
    )
    child = parent.fork(name="alpha")
    assert (child.smp, child.memory, child.base_checkpoint) == (3, "2G", "bc")
    assert child.accelerator == parent.accelerator
    assert child.overlay_dir == parent.overlay_dir
    assert child.runtime_id == "parent.alpha"


def test_a_child_has_its_own_ports_and_checkpoints(base_image, stub_qemu_img, tmp_path):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    child = parent.fork(name="alpha")
    assert child.pinned_ports is None
    assert child.list_checkpoints() == ()
    assert child is not parent


def test_a_failing_qemu_img_surfaces_its_stderr(base_image, stub_qemu_img, tmp_path):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    base_image.unlink()
    with pytest.raises(QemuError, match="qemu-img create failed"):
        parent.fork(name="alpha")


def test_the_fork_is_timed(base_image, stub_qemu_img, tmp_path):
    parent = _forkable(base_image, stub_qemu_img, tmp_path)
    parent.fork(name="alpha")
    assert [label for label, _ in parent.timings] == ["cow_fork[alpha]"]


def test_checkpoint_bookkeeping_is_empty_before_anything_is_saved(base_image):
    runtime = QemuRuntime(image=base_image, accelerator="tcg")
    assert runtime.list_checkpoints() == ()
    assert runtime.has_checkpoint(BASE_CHECKPOINT) is False


def test_deleting_an_unknown_checkpoint_is_a_no_op(base_image):
    """Cleanup must never fail a rollout."""
    QemuRuntime(image=base_image, accelerator="tcg").delete_checkpoint("nope")


def test_the_default_base_checkpoint_name_is_stable():
    assert BASE_CHECKPOINT == "desktop_env_base"


@pytest.mark.needs_vm
def test_fork_against_real_qemu_img_produces_the_backing_chain(tmp_path):
    """Only a real ``qemu-img`` proves the ``-b``/``-F`` pair does what is claimed."""
    import subprocess

    binary = qemu_img_binary()
    base = tmp_path / "base.qcow2"
    subprocess.run(
        [binary, "create", "-f", "qcow2", str(base), "64M"], check=True, capture_output=True
    )
    parent = QemuRuntime(
        image=base,
        qemu_img_binary=binary,
        overlay_dir=tmp_path / "overlays",
        log_dir=tmp_path,
        qmp_dir=Path("/tmp"),
        accelerator="tcg",
    )
    child = parent.fork(name="real")
    assert child.image.exists()
    info = json.loads(
        subprocess.run(
            [binary, "info", "--output=json", str(child.image)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert Path(info["backing-filename"]).resolve() == base.resolve()
    assert info["backing-filename-format"] == "qcow2"
    assert info["format"] == "qcow2"
    child.stop()
    assert not child.image.exists()
    assert base.exists()


@pytest.mark.needs_vm
def test_real_qemu_img_refuses_a_backing_file_without_an_explicit_format(tmp_path):
    """The reason ``-F qcow2`` is not optional, asserted against the real tool."""
    import subprocess

    binary = qemu_img_binary()
    base = tmp_path / "base.qcow2"
    subprocess.run(
        [binary, "create", "-f", "qcow2", str(base), "16M"], check=True, capture_output=True
    )
    result = subprocess.run(
        [binary, "create", "-f", "qcow2", "-b", str(base), str(tmp_path / "no_format.qcow2")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "backing format" in (result.stderr + result.stdout).lower()


@pytest.mark.needs_vm
def test_the_qemu_binary_supports_both_accelerators(tmp_path):
    """``_pick_accelerator``'s premise: tcg is really available as a fallback."""
    import subprocess

    result = subprocess.run(
        [qemu_system_binary(), "-accel", "help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "tcg" in result.stdout


@pytest.mark.needs_vm
@pytest.mark.kvm
def test_a_real_boot_forwards_ports_and_serves_qmp_snapshots(tmp_path, monkeypatch):
    """The measured claim of this module -- savevm/loadvm over QMP -- end to end.

    A blank qcow2 has no guest agent, so ``start`` is expected to time out; the
    QEMU process is alive at that point and everything after it is real.
    """
    import subprocess

    binary_img = qemu_img_binary()
    disk = tmp_path / "disk.qcow2"
    subprocess.run(
        [binary_img, "create", "-f", "qcow2", str(disk), "256M"],
        check=True,
        capture_output=True,
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    # A FIXED port block made this test collide with its own previous run: a
    # just-killed QEMU's forwards can linger, and the suite is run repeatedly.
    # Take a block that is actually free right now, above the ephemeral range.
    boot_ports = _free_guest_port_block()
    runtime = QemuRuntime(
        image=disk,
        qemu_binary=qemu_system_binary(),
        qemu_img_binary=binary_img,
        overlay_dir=scratch,
        log_dir=tmp_path / "logs",
        qmp_dir=Path("/tmp"),
        smp=1,
        memory="512M",
        boot_timeout_s=3.0,
        restore_timeout_s=3.0,
        ports=boot_ports,
    )
    # A blank disk has no guest agent, so readiness genuinely times out -- and the
    # leak fix then tears the process down, which is asserted by the fact that the
    # second start below succeeds in binding the same ports.
    with pytest.raises(TimeoutError):
        runtime.start()
    assert runtime._process is None, "the timed-out boot leaked a QEMU process"

    def wait_for_hostfwd(self, server_port, process, *, timeout_s):
        """Stand in for guest readiness: wait until QEMU's hostfwd accepts."""
        deadline = time.monotonic() + max(timeout_s, 10.0)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise QemuError(f"qemu died early (rc={process.returncode})")
            probe = socket.socket()
            probe.settimeout(1)
            try:
                if probe.connect_ex(("127.0.0.1", server_port)) == 0:
                    return
            finally:
                probe.close()
            time.sleep(0.2)
        raise TimeoutError(f"hostfwd never bound on :{server_port}")

    runtime._wait_ready = wait_for_hostfwd.__get__(runtime)  # type: ignore[method-assign]
    state = runtime.start()
    try:
        assert state.accelerator == "kvm"
        assert state.ports.server == boot_ports.server

        runtime.suspend()
        runtime.resume()

        checkpoint = runtime.checkpoint("probe")
        assert checkpoint.kind == "qemu_savevm"
        assert checkpoint.detail["seconds"] > 0
        assert runtime.has_checkpoint("probe")
        assert [item.name for item in runtime.list_checkpoints()] == ["probe"]
        assert "probe" in runtime._monitor().hmp("info snapshots")

        runtime.restore("probe")
        assert any(label.startswith("loadvm[") for label, _ in runtime.timings)

        runtime.delete_checkpoint("probe")
        assert runtime.list_checkpoints() == ()
        assert "no snapshot" in runtime._monitor().hmp("info snapshots").lower()

        with pytest.raises(QemuError, match="savevm .* failed"):
            runtime.checkpoint("a name with spaces")
    finally:
        process = runtime._process
        runtime.stop()
        assert process is None or process.poll() is not None
    assert not list(Path("/tmp").glob(f"deqmp_{os.getpid()}_*")), "QMP socket leaked"
