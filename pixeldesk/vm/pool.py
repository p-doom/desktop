"""A prewarming pool of desktop sessions, with leases and a status file.

The pool's entire requirement of a session is ``close() -> None``.  It is generic
over ``DesktopSessionEnv``, so ``DesktopSession``, a raw ``QemuRuntime``, or a
caller's own wrapper all satisfy it with no adapter.

A VM boot is 13-16 s and a first boot is worse, so a rollout that starts by
booting pays that per rollout.  The pool keeps a floor of ready sessions warm in
the background, hands them out under a lease with an activity timeout, retires
them after a bounded number of rollouts (bounding drift on a long-lived guest),
and reaps leases whose holder died without releasing.  The status file makes all
of that inspectable from another process, which is how a multi-node run is
debugged.

The port allocator is vendored here rather than imported, and its shape matters:
a *file-locked slot*, not ``bind(0)``.  ``bind(0)`` hands out a port that is then
closed before QEMU binds it, so two simultaneous starts can be handed the same
one; an ``flock`` on a slot file is held for the lifetime of the lease, and the
allocator additionally probes each port in the block before accepting the slot.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol, Self, TextIO, cast

SessionStatus = Literal["ready", "leased"]
RetireReason = Literal["retired", "failed"]


@dataclass(frozen=True)
class WorkerPorts:
    """One aligned block of host ports for a single desktop.

    Exactly the four ports ``QemuRuntime`` forwards, and no more.  There is no
    QEMU-side VNC port: the runtime boots ``-display none -nographic``, so
    exposing one would mean changing every VM's command line rather than
    allocating a port.

    The stride stays at 10 even though only four ports are used, so ``base + 4 ..
    base + 9`` is unused headroom and no existing slot's port numbers move.
    """

    server: int
    chromium: int
    vnc: int
    vlc: int

    def as_dict(self) -> dict[str, int]:
        return {
            "server": self.server,
            "chromium": self.chromium,
            "vnc": self.vnc,
            "vlc": self.vlc,
        }


@dataclass
class PortLease:
    """A held slot.  The advisory lock lives as long as the lease does."""

    ports: WorkerPorts
    slot: int
    workdir: Path
    _lock_file: TextIO
    logdir: Path | None = None
    _released: bool = False

    def release(self) -> None:
        """Drop the advisory lock and remove the lease's own working directory.

        Only the tree this lease created is removed, and only if it is empty.
        ``rmtree`` is deliberately NOT used, so a session that failed to clean up
        its own scratch leaves visible evidence instead of having it deleted
        underneath the diagnosis.
        """
        if self._released:
            return
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()
        self._released = True
        for directory in (self.workdir, self.logdir):
            if directory is None:
                continue
            with contextlib.suppress(OSError):
                directory.rmdir()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def ports_for_worker(base: int, worker_id: int, stride: int = 10) -> WorkerPorts:
    start = base + worker_id * stride
    ports = WorkerPorts(
        server=start,
        chromium=start + 1,
        vnc=start + 2,
        vlc=start + 3,
    )
    for port in (ports.server, ports.chromium, ports.vnc, ports.vlc):
        if port > 65535:
            raise ValueError(f"worker port {port} exceeds the TCP port range")
    return ports


def assert_ports_available(ports: WorkerPorts) -> None:
    for port in (ports.server, ports.chromium, ports.vnc, ports.vlc):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
            except OSError as exc:
                raise RuntimeError(f"port {port} is already in use") from exc


def get_port_base(job_id: str | None, *, configured_base: str | None = None) -> int:
    """Pick a per-job base so two jobs on one node do not collide."""
    if configured_base is not None:
        try:
            base = int(configured_base)
        except ValueError as exc:
            raise ValueError(
                f"DESKTOP_ENV_PORT_BASE must be an integer, got {configured_base!r}"
            ) from exc
        if base < 1024 or base > 65535:
            raise ValueError(
                f"DESKTOP_ENV_PORT_BASE must be between 1024 and 65535, got {base}"
            )
        return base
    if job_id and job_id.isdigit():
        return 20000 + int(job_id) % 10000
    return 20000


def allocate_worker_ports(
    *,
    lock_dir: str | Path,
    work_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    stride: int = 10,
    max_slots: int = 512,
) -> PortLease:
    """Reserve a per-session port block behind an advisory file lock."""
    configured_base = os.environ.get("DESKTOP_ENV_PORT_BASE")
    base = get_port_base(
        os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID")),
        configured_base=configured_base,
    )
    if configured_base is not None and base % stride:
        raise ValueError(
            f"DESKTOP_ENV_PORT_BASE {base} must be aligned to port stride {stride}"
        )
    root = Path(lock_dir)
    work_root = Path(work_dir) if work_dir is not None else root
    log_root = Path(log_dir) if log_dir is not None else None
    root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    if log_root is not None:
        log_root.mkdir(parents=True, exist_ok=True)
    for slot in range(max_slots):
        ports = ports_for_worker(base, slot, stride=stride)
        lock_file = (root / f"ports_{slot:04d}.lock").open("a+")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            continue
        try:
            assert_ports_available(ports)
        except Exception:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            continue
        lease_name = f"w{os.getpid()}_{slot:x}"
        workdir = work_root / lease_name
        logdir = log_root / lease_name if log_root is not None else None
        workdir.mkdir(parents=True, exist_ok=True)
        if logdir is not None:
            logdir.mkdir(parents=True, exist_ok=True)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} ports={ports}\n")
        lock_file.flush()
        return PortLease(
            ports=ports, slot=slot, workdir=workdir, _lock_file=lock_file, logdir=logdir
        )
    raise RuntimeError(f"no available port blocks under {root} from base {base}")


class DesktopSessionEnv(Protocol):
    """The pool's entire requirement of a session."""

    def close(self) -> None: ...


class PortAllocator(Protocol):
    def __call__(
        self,
        *,
        lock_dir: str | Path,
        work_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
    ) -> PortLease: ...


@dataclass(frozen=True)
class DesktopPoolConfig:
    min_ready_sessions: int = 1
    max_sessions: int = 5
    max_rollouts_per_session: int = 50
    checkout_timeout_s: float = 900.0
    lease_timeout_s: float = 300.0
    startup_timeout_s: float = 840.0
    startup_retry_backoff_s: float = 30.0
    startup_retry_backoff_max_s: float = 300.0
    status_heartbeat_interval_s: float = 10.0
    root_dir: Path | None = None
    status_dir: Path | None = None
    runtime_dir: Path | None = None
    log_runtime_dir: Path | None = None
    port_lock_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate pool sizing, timeout, and path fields after construction."""
        for name in (
            "root_dir",
            "status_dir",
            "runtime_dir",
            "log_runtime_dir",
            "port_lock_dir",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        if self.min_ready_sessions < 0:
            raise ValueError("min_ready_sessions must be non-negative")
        if self.max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if self.min_ready_sessions > self.max_sessions:
            raise ValueError("min_ready_sessions cannot exceed max_sessions")
        if self.max_rollouts_per_session < 1:
            raise ValueError("max_rollouts_per_session must be at least 1")
        if self.checkout_timeout_s <= 0:
            raise ValueError("checkout_timeout_s must be positive")
        if self.lease_timeout_s <= 0:
            raise ValueError("lease_timeout_s must be positive")
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.startup_retry_backoff_s < 0:
            raise ValueError("startup_retry_backoff_s must be non-negative")
        if self.startup_retry_backoff_max_s <= 0:
            raise ValueError("startup_retry_backoff_max_s must be positive")
        if self.status_heartbeat_interval_s < 0:
            raise ValueError("status_heartbeat_interval_s must be non-negative")


@dataclass
class DesktopPoolSession[DesktopEnvT: DesktopSessionEnv]:
    session_id: str
    env: DesktopEnvT
    lease: PortLease
    status: SessionStatus
    rollouts_completed: int
    created_at: float
    updated_at: float
    leased_at: float | None = None
    last_activity_at: float | None = None
    last_error: str | None = None
    closed: bool = False


@dataclass
class StartingDesktopSession:
    session_id: str
    created_at: float
    updated_at: float
    lease: PortLease | None = None
    last_error: str | None = None


class CheckedOutDesktopSession[DesktopEnvT: DesktopSessionEnv]:
    def __init__(
        self,
        pool: "DesktopSessionPool[DesktopEnvT]",
        session: DesktopPoolSession[DesktopEnvT],
    ) -> None:
        self._pool = pool
        self._session = session
        self._released = False

    @property
    def env(self) -> DesktopEnvT:
        return self._session.env

    @property
    def session_id(self) -> str:
        return self._session.session_id

    def tracked_env(self) -> DesktopEnvT:
        """An env proxy that refreshes lease activity around every method call."""
        return cast(
            DesktopEnvT,
            _ActivityTrackedDesktopEnv(
                env=self._session.env,
                touch=lambda: self._pool.touch(self._session.session_id),
            ),
        )

    def touch(self) -> None:
        self._pool.touch(self._session.session_id)

    def release(self, *, failed: bool = False, error: str | None = None) -> None:
        """Return this leased session to the pool exactly once."""
        if self._released:
            return
        self._released = True
        self._pool.release(self._session.session_id, failed=failed, error=error)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release(failed=exc is not None, error=None if exc is None else repr(exc))


class DesktopSessionPool[DesktopEnvT: DesktopSessionEnv]:
    """Prewarms desktop sessions inside one worker process."""

    def __init__(
        self,
        *,
        config: DesktopPoolConfig,
        root_dir: Path,
        session_factory: Callable[[PortLease], DesktopEnvT],
        port_allocator: PortAllocator = allocate_worker_ports,
        worker_name: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Create pool bookkeeping and dependency hooks without starting sessions."""
        self.config = config
        self.root_dir = Path(root_dir)
        self.status_dir = (
            Path(config.status_dir)
            if config.status_dir is not None
            else self.root_dir / "status"
        )
        self.port_lock_dir = (
            Path(config.port_lock_dir)
            if config.port_lock_dir is not None
            else self.root_dir / "port_locks"
        )
        self.runtime_dir = (
            Path(config.runtime_dir)
            if config.runtime_dir is not None
            else self.root_dir / "runtime"
        )
        self.log_dir = self.root_dir / "logs"
        self.log_write_dir = (
            Path(config.log_runtime_dir)
            if config.log_runtime_dir is not None
            else self.log_dir
        )
        self.artifact_dir = self.root_dir / "artifacts"
        self.status_path = (
            self.status_dir / f"{worker_name or _default_worker_name()}.json"
        )
        self._session_factory = session_factory
        self._port_allocator = port_allocator
        self._clock = clock
        self._condition = threading.Condition()
        self._sessions: dict[str, DesktopPoolSession[DesktopEnvT]] = {}
        self._starting_sessions: dict[str, StartingDesktopSession] = {}
        self._retiring_session_ids: set[str] = set()
        self._session_seq = 0
        self._closed = False
        self._started = False
        self._retry_scheduled = False
        self._total_started = 0
        self._total_failed = 0
        self._total_stale_leases_retired = 0
        self._last_error: str | None = None
        self._lease_watchdog_started = False
        self._status_heartbeat_started = False
        self._consecutive_start_failures = 0
        self._next_start_attempt_at: float | None = None

    def start(self) -> None:
        """Create pool directories and begin prewarming ready sessions."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.port_lock_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.log_write_dir != self.log_dir:
            _ensure_symlink_dir(self.log_write_dir, self.log_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self._condition:
            if self._started:
                return
            self._started = True
            self._ensure_min_ready_locked()
            self._start_lease_watchdog_locked()
            self._start_status_heartbeat_locked()
            self._write_status_locked()

    def checkout(
        self, *, timeout_s: float | None = None
    ) -> CheckedOutDesktopSession[DesktopEnvT]:
        """Block until a ready session is available or checkout times out."""
        if not self._started:
            self.start()
        effective_timeout_s = (
            self.config.checkout_timeout_s if timeout_s is None else timeout_s
        )
        if effective_timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = self._clock() + effective_timeout_s
        with self._condition:
            while True:
                self._raise_if_closed_locked()
                ready = self._ready_sessions_locked()
                if ready:
                    session = ready[0]
                    now = self._clock()
                    session.status = "leased"
                    session.leased_at = now
                    session.last_activity_at = now
                    session.updated_at = now
                    self._write_status_locked()
                    return CheckedOutDesktopSession(self, session)
                self._ensure_min_ready_locked()
                remaining_s = deadline - self._clock()
                if remaining_s <= 0:
                    raise TimeoutError(
                        "timed out waiting for a ready desktop session after "
                        f"{effective_timeout_s:.1f}s"
                    )
                self._condition.wait(timeout=min(1.0, remaining_s))

    def touch(self, session_id: str) -> None:
        """Record client activity for a checked-out session."""
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None or session.status != "leased":
                return
            now = self._clock()
            session.last_activity_at = now
            session.updated_at = now
            self._write_status_locked()

    def release(
        self, session_id: str, *, failed: bool = False, error: str | None = None
    ) -> None:
        """Return a leased session and retire or reuse it according to policy."""
        session: DesktopPoolSession[DesktopEnvT] | None = None
        close_reason: RetireReason = "retired"
        should_retire = False
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.rollouts_completed += 1
            session.updated_at = self._clock()
            session.last_activity_at = session.updated_at
            session.last_error = error
            should_retire = (
                failed
                or session.rollouts_completed >= self.config.max_rollouts_per_session
            )
            if should_retire:
                self._sessions.pop(session_id, None)
                self._retiring_session_ids.add(session_id)
                close_reason = "failed" if failed else "retired"
                if failed:
                    self._total_failed += 1
                    self._last_error = error
            else:
                session.status = "ready"
                session.leased_at = None
                session.last_error = None
            self._write_status_locked()
            self._condition.notify_all()

        if should_retire and session is not None:
            self._retire_session_async(session, reason=close_reason)
        else:
            with self._condition:
                self._ensure_min_ready_locked()
                self._write_status_locked()

    def close(self) -> None:
        """Stop the pool and close every session that is still tracked."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                self._retiring_session_ids.add(session.session_id)
            self._write_status_locked()
            self._condition.notify_all()

        # One session that will not close must not strand the others: aborting on
        # the first raising ``close()`` leaks every later session's VM, memory and
        # port block.  Failures are collected into the status file rather than
        # raised, matching ``_retire_session``.
        failures: list[str] = []
        for session in sessions:
            try:
                _close_session_resources(session)
            except Exception as exc:
                failures.append(f"close failed: {_exception_message(exc)}")

        with self._condition:
            for session in sessions:
                self._retiring_session_ids.discard(session.session_id)
            if failures:
                self._total_failed += len(failures)
                self._last_error = "; ".join(failures)
            self._write_status_locked()
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self._status_payload_locked()

    def reap_stale_leases(self) -> int:
        """Retire sessions leased longer than the configured activity timeout."""
        stale_sessions: list[DesktopPoolSession[DesktopEnvT]] = []
        with self._condition:
            now = self._clock()
            for session in list(self._sessions.values()):
                if not self._session_is_stale_locked(session, now=now):
                    continue
                self._sessions.pop(session.session_id, None)
                self._retiring_session_ids.add(session.session_id)
                session.updated_at = now
                session.last_activity_at = now
                session.last_error = (
                    "lease timed out after "
                    f"{self.config.lease_timeout_s:.1f}s without activity"
                )
                self._total_failed += 1
                self._total_stale_leases_retired += 1
                self._last_error = session.last_error
                stale_sessions.append(session)
            if stale_sessions:
                self._write_status_locked()
                self._condition.notify_all()

        for session in stale_sessions:
            self._retire_session_async(session, reason="failed")
        return len(stale_sessions)

    def _ensure_min_ready_locked(self) -> None:
        """Start background sessions until the ready-session floor is covered."""
        if self._closed or not self._started or self._startup_cooling_down_locked():
            return
        while (
            len(self._ready_sessions_locked()) + len(self._starting_sessions)
            < self.config.min_ready_sessions
            and self._active_session_count_locked() < self.config.max_sessions
        ):
            self._start_session_async_locked()

    def _start_session_async_locked(self) -> None:
        """Reserve a session id and launch its startup thread."""
        self._session_seq += 1
        session_id = f"session-{self._session_seq:06d}"
        now = self._clock()
        self._starting_sessions[session_id] = StartingDesktopSession(
            session_id=session_id, created_at=now, updated_at=now
        )
        self._write_status_locked()
        _start_daemon_thread(
            name=f"desktop-pool-start-{session_id}",
            target=self._start_session,
            session_id=session_id,
        )

    def _start_session(self, session_id: str) -> None:
        """Allocate ports and construct one desktop session for the pool."""
        lease: PortLease | None = None
        env: DesktopEnvT | None = None
        try:
            lease = self._port_allocator(
                lock_dir=self.port_lock_dir,
                work_dir=self.runtime_dir,
                log_dir=self.log_write_dir,
            )
            with self._condition:
                starting = self._starting_sessions.get(session_id)
                if starting is not None:
                    starting.lease = lease
                    starting.updated_at = self._clock()
                    self._write_status_locked()
                    self._condition.notify_all()
                if self._closed:
                    self._starting_sessions.pop(session_id, None)
                    self._write_status_locked()
                    self._condition.notify_all()
                    lease.release()
                    return
            env = self._session_factory(lease)
        except Exception as exc:
            if env is not None:
                _close_env(env)
            if lease is not None:
                lease.release()
            self._record_start_failure(session_id, exc)
            return

        with self._condition:
            starting = self._starting_sessions.get(session_id)
            created_at = self._clock() if starting is None else starting.created_at
        session = DesktopPoolSession[DesktopEnvT](
            session_id=session_id,
            env=env,
            lease=lease,
            status="ready",
            rollouts_completed=0,
            created_at=created_at,
            updated_at=self._clock(),
        )
        close_immediately = False
        with self._condition:
            self._starting_sessions.pop(session_id, None)
            if self._closed:
                close_immediately = True
            else:
                self._sessions[session_id] = session
                self._total_started += 1
                self._consecutive_start_failures = 0
                self._next_start_attempt_at = None
                self._ensure_min_ready_locked()
            self._write_status_locked()
            self._condition.notify_all()

        if close_immediately:
            # This runs on a daemon startup thread, so an unguarded raise would
            # surface only as an unhandled-thread traceback and the pool's own
            # error counters would never learn that a session failed to close.
            try:
                _close_session_resources(session)
            except Exception as exc:
                with self._condition:
                    self._total_failed += 1
                    self._last_error = f"close failed: {_exception_message(exc)}"
                    self._write_status_locked()
                    self._condition.notify_all()

    def _record_start_failure(self, session_id: str, exc: Exception) -> None:
        """Record a failed startup and schedule the next retry attempt."""
        message = _exception_message(exc)
        with self._condition:
            self._starting_sessions.pop(session_id, None)
            self._total_failed += 1
            self._last_error = message
            self._consecutive_start_failures += 1
            self._next_start_attempt_at = self._next_retry_deadline_locked()
            self._schedule_retry_locked()
            self._write_status_locked()
            self._condition.notify_all()

    def _schedule_retry_locked(self) -> None:
        """Schedule a delayed prewarm retry after a startup failure."""
        if self._closed or self._retry_scheduled:
            return
        if self.config.startup_retry_backoff_s <= 0:
            self._ensure_min_ready_locked()
            return
        self._retry_scheduled = True
        _start_daemon_thread(
            name="desktop-pool-retry", target=self._retry_after_backoff
        )

    def _retry_after_backoff(self) -> None:
        """Wait until the current retry deadline, then try to refill the pool."""
        with self._condition:
            while True:
                if self._closed:
                    self._retry_scheduled = False
                    self._write_status_locked()
                    self._condition.notify_all()
                    return
                remaining = self._startup_cooldown_remaining_locked()
                if remaining <= 0:
                    self._retry_scheduled = False
                    self._ensure_min_ready_locked()
                    self._write_status_locked()
                    self._condition.notify_all()
                    return
                self._condition.wait(timeout=remaining)

    def _startup_cooling_down_locked(self) -> bool:
        return self._startup_cooldown_remaining_locked() > 0

    def _startup_cooldown_remaining_locked(self) -> float:
        if self._next_start_attempt_at is None:
            return 0.0
        return self._next_start_attempt_at - self._clock()

    def _next_retry_deadline_locked(self) -> float | None:
        if self.config.startup_retry_backoff_s <= 0:
            return None
        failures = max(1, self._consecutive_start_failures)
        delay = min(
            self.config.startup_retry_backoff_s,
            self.config.startup_retry_backoff_max_s,
        )
        for _ in range(failures - 1):
            delay = min(delay * 2, self.config.startup_retry_backoff_max_s)
            if delay >= self.config.startup_retry_backoff_max_s:
                break
        return self._clock() + delay

    def _retire_session_async(
        self, session: DesktopPoolSession[DesktopEnvT], *, reason: RetireReason
    ) -> None:
        """Close a retired session on a background thread."""
        _start_daemon_thread(
            name=f"desktop-pool-retire-{session.session_id}",
            target=self._retire_session,
            session=session,
            reason=reason,
        )

    def _retire_session(
        self, session: DesktopPoolSession[DesktopEnvT], reason: RetireReason
    ) -> None:
        """Close one session, release its ports, and trigger replenishment."""
        try:
            _close_session_resources(session)
        except Exception as exc:
            with self._condition:
                self._total_failed += 1
                self._last_error = f"{reason} close failed: {_exception_message(exc)}"
        finally:
            with self._condition:
                self._retiring_session_ids.discard(session.session_id)
                self._ensure_min_ready_locked()
                self._write_status_locked()
                self._condition.notify_all()

    def _start_lease_watchdog_locked(self) -> None:
        """Start the background stale-lease reaper once per pool."""
        if self._lease_watchdog_started:
            return
        self._lease_watchdog_started = True
        _start_daemon_thread(
            name="desktop-pool-lease-watchdog", target=self._lease_watchdog_loop
        )

    def _lease_watchdog_loop(self) -> None:
        """Periodically retire abandoned checked-out sessions."""
        poll_s = min(10.0, max(1.0, self.config.lease_timeout_s / 10.0))
        while True:
            with self._condition:
                if self._closed:
                    return
                self._condition.wait(timeout=poll_s)
                if self._closed:
                    return
            self.reap_stale_leases()

    def _start_status_heartbeat_locked(self) -> None:
        """Start the background status heartbeat once per pool."""
        if self._status_heartbeat_started or self.config.status_heartbeat_interval_s <= 0:
            return
        self._status_heartbeat_started = True
        _start_daemon_thread(
            name="desktop-pool-status-heartbeat", target=self._status_heartbeat_loop
        )

    def _status_heartbeat_loop(self) -> None:
        """Refresh top-level status timestamps while the worker is alive."""
        interval_s = self.config.status_heartbeat_interval_s
        while True:
            with self._condition:
                if self._closed:
                    return
                self._condition.wait(timeout=interval_s)
                if self._closed:
                    return
                self._write_status_locked()

    def _session_is_stale_locked(
        self, session: DesktopPoolSession[DesktopEnvT], *, now: float
    ) -> bool:
        """Whether a leased session has exceeded the activity timeout."""
        if session.status != "leased":
            return False
        activity_at = session.last_activity_at or session.leased_at
        if activity_at is None:
            return False
        return now - activity_at >= self.config.lease_timeout_s

    def _ready_sessions_locked(self) -> list[DesktopPoolSession[DesktopEnvT]]:
        """Ready sessions ordered by age while the lock is held."""
        return sorted(
            (
                session
                for session in self._sessions.values()
                if session.status == "ready"
            ),
            key=lambda session: session.created_at,
        )

    def _active_session_count_locked(self) -> int:
        return (
            len(self._sessions)
            + len(self._starting_sessions)
            + len(self._retiring_session_ids)
        )

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise RuntimeError("DesktopSessionPool is closed")

    def _write_status_locked(self) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self.status_path, self._status_payload_locked())

    def _status_payload_locked(self) -> dict[str, Any]:
        """Build the JSON-serializable status payload for this worker pool."""
        now = self._clock()
        sessions = [
            _session_payload(session, now=now)
            for session in sorted(
                self._sessions.values(), key=lambda item: item.session_id
            )
        ]
        starting_sessions = [
            _starting_session_payload(session, now=now)
            for session in sorted(
                self._starting_sessions.values(), key=lambda item: item.session_id
            )
        ]
        starting_ages = [
            session["age_s"]
            for session in starting_sessions
            if isinstance(session.get("age_s"), int | float)
        ]
        return {
            "worker_name": self.status_path.stem,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "root_dir": str(self.root_dir),
            "port_lock_dir": str(self.port_lock_dir),
            "status_dir": str(self.status_dir),
            "runtime_dir": str(self.runtime_dir),
            "log_dir": str(self.log_dir),
            "log_write_dir": str(self.log_write_dir),
            "status_path": str(self.status_path),
            "updated_at": now,
            "closed": self._closed,
            "min_ready_sessions": self.config.min_ready_sessions,
            "max_sessions": self.config.max_sessions,
            "max_rollouts_per_session": self.config.max_rollouts_per_session,
            "checkout_timeout_s": self.config.checkout_timeout_s,
            "lease_timeout_s": self.config.lease_timeout_s,
            "startup_timeout_s": self.config.startup_timeout_s,
            "startup_retry_backoff_s": self.config.startup_retry_backoff_s,
            "startup_retry_backoff_max_s": self.config.startup_retry_backoff_max_s,
            "status_heartbeat_interval_s": self.config.status_heartbeat_interval_s,
            "ready": sum(1 for session in sessions if session["status"] == "ready"),
            "starting": len(self._starting_sessions),
            "leased": sum(1 for session in sessions if session["status"] == "leased"),
            "retiring": len(self._retiring_session_ids),
            "oldest_starting_age_s": max(starting_ages) if starting_ages else None,
            "total_started": self._total_started,
            "total_failed": self._total_failed,
            "stale_leases_retired": self._total_stale_leases_retired,
            "retry_scheduled": self._retry_scheduled,
            "consecutive_start_failures": self._consecutive_start_failures,
            "next_start_attempt_at": self._next_start_attempt_at,
            "startup_cooldown_remaining_s": max(
                0.0, self._startup_cooldown_remaining_locked()
            ),
            "last_error": self._last_error,
            "starting_sessions": starting_sessions,
            "sessions": sessions,
        }


def _default_worker_name() -> str:
    host = socket.gethostname().split(".")[0]
    return f"{host}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _session_payload[DesktopEnvT: DesktopSessionEnv](
    session: DesktopPoolSession[DesktopEnvT], *, now: float
) -> dict[str, Any]:
    """Serialize one session's state, port lease, and health details."""
    activity_at = session.last_activity_at or session.updated_at
    payload = {
        "session_id": session.session_id,
        "status": session.status,
        "rollouts_completed": session.rollouts_completed,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "leased_at": session.leased_at,
        "last_activity_at": session.last_activity_at,
        "lease_age_s": (
            None if session.leased_at is None else max(0.0, now - session.leased_at)
        ),
        "idle_s": max(0.0, now - activity_at),
        "last_error": session.last_error,
        "lease_slot": session.lease.slot,
        "workdir": str(session.lease.workdir),
        "ports": session.lease.ports.as_dict(),
        "health": _env_health(session.env),
    }
    if session.lease.logdir is not None:
        payload["logdir"] = str(session.lease.logdir)
        payload["persistent_logdir"] = str(session.lease.logdir.resolve())
    return payload


def _starting_session_payload(
    session: StartingDesktopSession, *, now: float
) -> dict[str, Any]:
    lease = session.lease
    payload: dict[str, Any] = {
        "session_id": session.session_id,
        "status": "starting",
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "age_s": max(0.0, now - session.created_at),
        "last_error": session.last_error,
    }
    if lease is not None:
        payload.update(
            {
                "lease_slot": lease.slot,
                "workdir": str(lease.workdir),
                "ports": lease.ports.as_dict(),
            }
        )
    return payload


class _ActivityTrackedDesktopEnv:
    """Lightweight proxy that touches a pool lease around env method calls."""

    def __init__(self, *, env: DesktopSessionEnv, touch: Callable[[], None]) -> None:
        self._env = env
        self._touch = touch

    def __getattr__(self, name: str) -> object:
        attr = getattr(self._env, name)
        if not callable(attr):
            return attr

        def tracked_call(*args: object, **kwargs: object) -> object:
            self._touch()
            try:
                return attr(*args, **kwargs)
            finally:
                self._touch()

        return tracked_call


def _env_health(env: DesktopSessionEnv) -> Mapping[str, object]:
    """Safely ask a session for health details."""
    health = getattr(env, "health", None)
    if not callable(health):
        return {}
    try:
        value = health()
    except Exception as exc:
        return {"health_error": repr(exc)}
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {"value": value}


def _close_session_resources[DesktopEnvT: DesktopSessionEnv](
    session: DesktopPoolSession[DesktopEnvT],
) -> None:
    if session.closed:
        return
    session.closed = True
    try:
        _close_env(session.env)
    finally:
        session.lease.release()


def _close_env(env: DesktopSessionEnv) -> None:
    env.close()


def _exception_message(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _ensure_symlink_dir(link_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_path.symlink_to(target_dir, target_is_directory=True)
    except FileExistsError:
        if not link_path.is_symlink() or link_path.resolve() != target_dir.resolve():
            raise RuntimeError(
                f"refusing to replace a non-symlink or wrong-target log path: {link_path}"
            ) from None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temp file before replacing the target."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _start_daemon_thread(
    *, name: str, target: Callable[..., None], **kwargs: object
) -> None:
    """Start a daemon thread with keyword arguments for background pool work."""
    threading.Thread(target=target, kwargs=kwargs, name=name, daemon=True).start()
