"""A local QEMU runtime with QMP snapshots, CoW forks, and a TCG fallback.

Measured on the pinned guest:

    kill+reboot revert          13.6 - 16.6 s
    savevm (RAM+disk snapshot)   3.1 s   (once per VM / per task)
    loadvm + guest-ready         4.4 - 5.2 s   (stable over 15 successive
                                                restores, no creep)

So a reset's VM phase goes 13.6-16.6 s -> ~4.5 s.

The snapshot path is validated against 6 real OSWorld TRAIN tasks: reboot,
snapshot-revert, and post-setup-snapshot revert give IDENTICAL task rewards
before and after a scripted ground-truth solve, and identical guest state (file
lists, gsettings keys, sink volume, window list, task output dir, and a guest
HTTPS fetch after 15 successive restores).  A restore provably rewinds a SOLVED
task back to unsolved and a re-solve scores again.

This is a plain object a caller constructs.  It does not reach into another
package's namespace or rewrite a third-party provider factory at import time.

Two ideas were read from trycua/cua and reimplemented; no trycua code is
imported:

  * CoW fork.  ``qemu-img create -f qcow2 -b <base> -F qcow2 <child>`` gives N
    isolated children off one warm base, for parallel rollouts, instead of
    serializing every rollout on one ``loadvm``.
  * ``-accel tcg`` fallback.  A node without ``/dev/kvm`` is a slow node rather
    than a dead one, which is still useful for plumbing tests.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .runtime import Checkpoint, GuestPorts, RuntimeState

_LOG = logging.getLogger("desktop.vm.qemu")

# Guest-side ports the in-VM agent / debug interfaces listen on.
GUEST_SERVER_PORT = 5000
GUEST_CHROMIUM_PORT = 9222
GUEST_VNC_PORT = 5900
GUEST_VLC_PORT = 8080

BASE_CHECKPOINT = "desktop_env_base"

_PORT_LOCK = threading.Lock()
_HANDED_OUT: set[int] = set()


class QemuError(RuntimeError):
    """QEMU could not be started, snapshotted, forked, or reached."""


def free_port() -> int:
    """An ephemeral port not already handed out inside this process.

    Cross-process races are still possible (the socket must be closed before QEMU
    can bind it); ``start`` retries with fresh ports when QEMU loses one.
    """
    with _PORT_LOCK:
        for _ in range(50):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            probe.close()
            if port not in _HANDED_OUT:
                _HANDED_OUT.add(port)
                return port
        raise QemuError("could not find a free host port")


class QmpClient:
    """Minimal synchronous QMP client (enough for human-monitor-command)."""

    def __init__(self, path: str, *, connect_timeout_s: float = 60.0, io_timeout_s: float = 600.0) -> None:
        self.path = path
        deadline = time.time() + connect_timeout_s
        last: Exception | None = None
        while time.time() < deadline:
            try:
                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self._sock.settimeout(io_timeout_s)
                self._sock.connect(path)
                break
            except OSError as exc:  # socket not created yet / qemu still starting
                last = exc
                try:
                    self._sock.close()
                except OSError:
                    pass
                time.sleep(0.2)
        else:
            raise QemuError(f"could not connect to QMP socket {path}: {last!r}")
        self._buf = b""
        self._read_json()  # greeting
        self.execute("qmp_capabilities")

    def _read_json(self) -> dict:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise QemuError("QMP connection closed")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return json.loads(line.decode("utf-8"))

    def execute(self, command: str, **arguments: Any) -> object:
        payload: dict = {"execute": command}
        if arguments:
            payload["arguments"] = arguments
        self._sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        while True:
            message = self._read_json()
            if "event" in message:  # async events interleave with replies
                continue
            if "error" in message:
                raise QemuError(f"QMP {command} failed: {message['error']}")
            return message.get("return")

    def hmp(self, command_line: str) -> str:
        out = self.execute("human-monitor-command", **{"command-line": command_line})
        return "" if out is None else str(out)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def kvm_available() -> bool:
    """Whether this node can accelerate with KVM at all."""
    return os.access("/dev/kvm", os.R_OK | os.W_OK)


class QemuRuntime:
    """One QEMU VM with QMP snapshots, CoW forks, and an accelerator fallback."""

    name = "qemu"

    def __init__(
        self,
        *,
        image: Path,
        qemu_binary: Path | str = "qemu-system-x86_64",
        qemu_img_binary: Path | str = "qemu-img",
        overlay_dir: Path | None = None,
        log_dir: Path | None = None,
        qmp_dir: Path | None = None,
        smp: int = 4,
        memory: str = "8G",
        accelerator: str | None = None,
        boot_timeout_s: float = 300.0,
        restore_timeout_s: float = 120.0,
        snapshot_timeout_s: float = 600.0,
        base_checkpoint: str = BASE_CHECKPOINT,
        runtime_id: str | None = None,
        owns_image: bool = False,
        ports: GuestPorts | None = None,
    ) -> None:
        self.image = Path(image)
        self.qemu_binary = str(qemu_binary)
        self.qemu_img_binary = str(qemu_img_binary)
        self.overlay_dir = Path(overlay_dir) if overlay_dir else Path(os.environ.get("TMPDIR", "/tmp"))
        self.log_dir = Path(log_dir) if log_dir else Path("/tmp")
        self.qmp_dir = Path(qmp_dir) if qmp_dir else self._default_qmp_dir()
        self.smp = int(smp)
        self.memory = str(memory)
        self.boot_timeout_s = float(boot_timeout_s)
        self.restore_timeout_s = float(restore_timeout_s)
        self.snapshot_timeout_s = float(snapshot_timeout_s)
        self.base_checkpoint = base_checkpoint
        self.runtime_id = runtime_id or f"{os.getpid()}-{uuid.uuid4().hex[:10]}"
        self.owns_image = bool(owns_image)
        # Caller-pinned ports. Set this when something ELSE owns port allocation
        # -- notably DesktopSessionPool, whose PortLease holds an flock on a port
        # block for the lease's lifetime. Leaving it None makes this runtime
        # allocate its own with bind(0)+close, which would make that lease
        # decorative: the lock would be held on ports nothing is listening on
        # while QEMU bound four unrelated ones.
        self.pinned_ports = ports
        self.accelerator = accelerator or self._pick_accelerator()
        self.timings: list[tuple[str, float]] = []
        self._process: subprocess.Popen | None = None
        self._ports: GuestPorts | None = None
        self._qmp: QmpClient | None = None
        self._qmp_path: str | None = None
        self._log_path: Path | None = None
        self._checkpoints: dict[str, Checkpoint] = {}
        self._children: list[QemuRuntime] = []

    @staticmethod
    def _default_qmp_dir() -> Path:
        # QMP is a unix socket: the path caps at 108 bytes, so it must be short
        # and must not live under a deep scratch directory.
        for candidate in ("/tmp", "/var/tmp"):
            if os.access(candidate, os.W_OK):
                return Path(candidate)
        return Path.cwd()

    def _pick_accelerator(self) -> str:
        """KVM when the node has it, TCG when it does not.

        Deliberately not an error.  A TCG guest is roughly an order of magnitude
        slower and is NOT valid for any timing or benchmark-parity measurement,
        but it is valid for proving that a harness talks to a guest at all, which
        is most of what a test suite needs.
        """
        if kvm_available():
            return "kvm"
        _LOG.warning(
            "/dev/kvm is unavailable; falling back to -accel tcg. This is ~10x "
            "slower and is not valid for timing or benchmark-parity numbers."
        )
        return "tcg"

    def _record(self, label: str, seconds: float) -> None:
        self.timings.append((label, seconds))
        _LOG.info("[qemu] %-28s = %6.2fs", label, seconds)

    def start(self, *, attempts: int = 3) -> RuntimeState:
        """Boot the VM, retrying on a lost host-port race.

        Ports come from ``bind(0)`` + ``close``, so N VMs booting at the same
        moment can be handed the same port and the loser's QEMU dies immediately
        with "Could not set up host forwarding". Retrying with fresh ports makes
        multi-VM-per-node startup reliable.
        """
        if not self.image.is_file():
            raise QemuError(f"VM image missing: {self.image}")
        # Retrying exists only to survive a lost bind(0) race. With pinned ports
        # there is no race to lose: a collision means something else holds the
        # port, and retrying the same four ports would just fail again more
        # slowly, hiding a real conflict behind a timeout.
        attempts = 1 if self.pinned_ports is not None else attempts
        last: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._start_once()
            except BaseException as exc:
                # EVERY unsuccessful exit must tear the process down, not just
                # the retry path: a boot timeout raises TimeoutError rather than
                # QemuError, and __exit__ never runs if __enter__ raised, so
                # anything that escapes here leaks a whole VM -- its host ports,
                # its full -m memory, and its QMP socket on disk.
                retryable = (
                    isinstance(exc, QemuError)
                    and "died early" in str(exc)
                    and attempt < attempts
                )
                if not retryable:
                    self._teardown_process()
                    raise
                last = exc
                _LOG.warning(
                    "VM boot attempt %d/%d failed (%s); retrying with new ports",
                    attempt,
                    attempts,
                    exc,
                )
                self._teardown_process()
                time.sleep(1.0 + attempt)
        raise QemuError(str(last))

    def _start_once(self) -> RuntimeState:
        if self._process is not None and self._process.poll() is None:
            return self.state()
        ports = self.pinned_ports or GuestPorts(
            server=free_port(),
            chromium=free_port(),
            vnc=free_port(),
            vlc=free_port(),
        )
        hostfwd = ",".join(
            [
                f"hostfwd=tcp::{ports.server}-:{GUEST_SERVER_PORT}",
                f"hostfwd=tcp::{ports.chromium}-:{GUEST_CHROMIUM_PORT}",
                f"hostfwd=tcp::{ports.vnc}-:{GUEST_VNC_PORT}",
                f"hostfwd=tcp::{ports.vlc}-:{GUEST_VLC_PORT}",
            ]
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"qemu_{ports.server}.log"
        qmp_path = str(self.qmp_dir / f"deqmp_{os.getpid()}_{uuid.uuid4().hex[:8]}.sock")
        if len(qmp_path.encode("utf-8")) >= 100:
            raise QemuError(f"QMP socket path is too long for AF_UNIX: {qmp_path}")
        command = [
            self.qemu_binary,
            "-cpu",
            "host" if self.accelerator == "kvm" else "max",
            "-smp",
            str(self.smp),
            "-m",
            self.memory,
            "-machine",
            f"type=q35,accel={self.accelerator}",
            "-drive",
            f"file={self.image},if=virtio,format=qcow2,snapshot=on",
            "-netdev",
            f"user,id=net0,{hostfwd}",
            "-device",
            "virtio-net-pci,netdev=net0",
            "-display",
            "none",
            # -nographic is what puts the guest's serial console on stdio, which
            # is why the log below is worth reading after a boot that never came
            # up.  It is also why stdin MUST be redirected: close_fds never
            # applies to fd 0, so without it QEMU reads whatever is piped into
            # the pool process and eats a supervisor's protocol.
            "-nographic",
            # A QMP monitor is what makes savevm/loadvm reachable at all.
            "-qmp",
            f"unix:{qmp_path},server=on,wait=off",
        ]
        if self.accelerator == "kvm":
            command.insert(1, "-enable-kvm")
        _LOG.info(
            "booting VM %s accel=%s server_port=%d qmp=%s",
            self.image,
            self.accelerator,
            ports.server,
            qmp_path,
        )
        started = time.time()
        handle = log_path.open("w")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        self._process = process
        self._ports = ports
        self._qmp_path = qmp_path
        self._log_path = log_path
        self._wait_ready(ports.server, process, timeout_s=self.boot_timeout_s)
        self._record("vm_boot_ready", time.time() - started)
        # Disk-space note: QEMU creates the `-snapshot` temp overlay under
        # TMPDIR and unlinks it IMMEDIATELY, so the fd target already reads
        # "(deleted)".  The overlay grows to ~2.7 GB per VM once it holds savevm
        # RAM state, but a SIGKILL cannot leak it -- the kernel frees the space
        # when QEMU dies.  No extra cleanup is needed.
        return self.state()

    def state(self) -> RuntimeState:
        if self._process is None or self._ports is None:
            raise QemuError("runtime is not started")
        return RuntimeState(
            runtime_id=self.runtime_id,
            ports=self._ports,
            base_url=f"http://127.0.0.1:{self._ports.server}",
            accelerator=self.accelerator,
            log_path=None if self._log_path is None else str(self._log_path),
            detail={
                "image": str(self.image),
                "pid": self._process.pid,
                "qmp_path": self._qmp_path,
                "smp": self.smp,
                "memory": self.memory,
                "owns_image": self.owns_image,
            },
        )

    def stop(self) -> None:
        """Idempotent teardown, including any CoW children this runtime made."""
        for child in list(self._children):
            try:
                child.stop()
            except Exception as exc:  # a child must not block the parent's stop
                _LOG.warning("child runtime stop failed: %r", exc)
        self._children.clear()
        self._teardown_process()
        if self.owns_image:
            try:
                self.image.unlink(missing_ok=True)
            except OSError as exc:
                _LOG.warning("could not remove forked overlay %s: %r", self.image, exc)

    def _teardown_process(self) -> None:
        if self._qmp is not None:
            self._qmp.close()
            self._qmp = None
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._qmp_path:
            try:
                os.unlink(self._qmp_path)
            except OSError:
                pass
        self._process = None
        self._ports = None
        self._qmp_path = None
        self._checkpoints.clear()

    def is_ready(self, *, timeout_s: float = 0.0) -> bool:
        if self._process is None or self._ports is None:
            return False
        try:
            self._wait_ready(
                self._ports.server, self._process, timeout_s=max(0.0, timeout_s) or 0.5
            )
        except (QemuError, TimeoutError):
            return False
        return True

    def _wait_ready(
        self, server_port: int, process: subprocess.Popen, *, timeout_s: float
    ) -> None:
        """Poll the guest agent's ``/screenshot`` until it answers 200."""
        url = f"http://127.0.0.1:{server_port}/screenshot"
        started = time.time()
        delay = 0.2
        while time.time() - started < timeout_s:
            if process.poll() is not None:
                raise QemuError(f"qemu died early (rc={process.returncode})")
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status == 200:
                        _LOG.info(
                            "guest agent ready after %.1fs (:%d)",
                            time.time() - started,
                            server_port,
                        )
                        return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(delay)
            delay = min(delay * 1.5, 3.0)
        raise TimeoutError(
            f"guest agent not ready after {timeout_s}s (:{server_port})"
        )

    def _monitor(self) -> QmpClient:
        if self._qmp_path is None:
            raise QemuError("runtime is not started")
        if self._qmp is None:
            self._qmp = QmpClient(self._qmp_path, io_timeout_s=self.snapshot_timeout_s)
        return self._qmp

    def suspend(self) -> None:
        """QMP ``stop``: freeze the vCPUs without discarding state."""
        self._monitor().execute("stop")

    def resume(self) -> None:
        """QMP ``cont``: unfreeze a suspended runtime."""
        self._monitor().execute("cont")

    def checkpoint(self, name: str) -> Checkpoint:
        """QEMU ``savevm`` -- full RAM + disk state into the running overlay."""
        started = time.time()
        output = self._monitor().hmp(f"savevm {name}")
        elapsed = time.time() - started
        if output.strip():
            raise QemuError(f"savevm {name} failed: {output.strip()}")
        record = Checkpoint(
            name=name,
            created_monotonic_ns=time.monotonic_ns(),
            kind="qemu_savevm",
            detail={"seconds": elapsed, "image": str(self.image)},
        )
        self._checkpoints[name] = record
        self._record(f"savevm[{name}]", elapsed)
        return record

    def restore(self, name: str) -> Checkpoint:
        """QEMU ``loadvm`` + wait until the in-VM agent answers again."""
        if self._process is None or self._ports is None:
            raise QemuError("runtime is not started")
        started = time.time()
        output = self._monitor().hmp(f"loadvm {name}")
        if output.strip():
            raise QemuError(f"loadvm {name} failed: {output.strip()}")
        loaded = time.time()
        self._wait_ready(
            self._ports.server, self._process, timeout_s=self.restore_timeout_s
        )
        ready = time.time()
        self._record(f"loadvm[{name}]", loaded - started)
        self._record("loadvm_guest_ready", ready - loaded)
        existing = self._checkpoints.get(name)
        return existing or Checkpoint(
            name=name, created_monotonic_ns=time.monotonic_ns(), kind="qemu_savevm"
        )

    def has_checkpoint(self, name: str) -> bool:
        return name in self._checkpoints

    def list_checkpoints(self) -> tuple[Checkpoint, ...]:
        return tuple(
            sorted(self._checkpoints.values(), key=lambda item: item.created_monotonic_ns)
        )

    def delete_checkpoint(self, name: str) -> None:
        """QEMU ``delvm`` -- drop a checkpoint so the overlay stays bounded."""
        if name not in self._checkpoints:
            return
        started = time.time()
        try:
            output = self._monitor().hmp(f"delvm {name}")
            if output.strip():
                _LOG.warning("delvm %s: %s", name, output.strip())
        except Exception as exc:  # cleanup must never fail a rollout
            _LOG.warning("delvm %s failed: %r", name, exc)
        self._checkpoints.pop(name, None)
        self._record(f"delvm[{name}]", time.time() - started)

    def ensure_base(self) -> Checkpoint:
        """Guarantee the clean base checkpoint exists; create it if it does not."""
        if self.base_checkpoint in self._checkpoints:
            return self._checkpoints[self.base_checkpoint]
        return self.checkpoint(self.base_checkpoint)

    def reset_to_base(self) -> Checkpoint:
        """Rewind to the clean base state.  ~4.5 s against ~15 s for a reboot."""
        self.ensure_base()
        return self.restore(self.base_checkpoint)

    def fork(self, *, name: str | None = None) -> "QemuRuntime":
        """A new runtime on a copy-on-write overlay of this one's image.

        ``qemu-img create -f qcow2 -b <base> -F qcow2 <child>``.  The child is a
        separate VM with its own ports, its own QMP socket, and its own
        checkpoints; it shares only the immutable backing file.  Explicit
        ``-F qcow2`` is required by modern ``qemu-img``, which refuses to probe
        the backing format.

        The parent's image must not be written to while children exist -- which
        holds here because the parent boots with ``snapshot=on``, so the backing
        file is never modified by either.
        """
        binary = shutil.which(self.qemu_img_binary) or self.qemu_img_binary
        child_name = name or f"fork-{uuid.uuid4().hex[:8]}"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        child_image = self.overlay_dir / f"{self.image.stem}.{child_name}.qcow2"
        if child_image.exists():
            raise QemuError(f"fork overlay already exists: {child_image}")
        command = [
            str(binary),
            "create",
            "-f",
            "qcow2",
            "-b",
            str(self.image.resolve()),
            "-F",
            "qcow2",
            str(child_image),
        ]
        started = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise QemuError(
                f"qemu-img create failed (rc={result.returncode}): "
                f"{(result.stderr or result.stdout).strip()}"
            )
        self._record(f"cow_fork[{child_name}]", time.time() - started)
        child = QemuRuntime(
            image=child_image,
            qemu_binary=self.qemu_binary,
            qemu_img_binary=self.qemu_img_binary,
            overlay_dir=self.overlay_dir,
            log_dir=self.log_dir,
            qmp_dir=self.qmp_dir,
            smp=self.smp,
            memory=self.memory,
            accelerator=self.accelerator,
            boot_timeout_s=self.boot_timeout_s,
            restore_timeout_s=self.restore_timeout_s,
            snapshot_timeout_s=self.snapshot_timeout_s,
            base_checkpoint=self.base_checkpoint,
            runtime_id=f"{self.runtime_id}.{child_name}",
            owns_image=True,
        )
        self._children.append(child)
        return child

    def __enter__(self) -> "QemuRuntime":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
