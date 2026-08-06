"""One desktop session: an isolated runtime plus a verified reset.

CONSOLIDATION (five implementations -> one).  The generic session logic in each
predecessor was reimplemented or re-forked; the task-specific halves were not
generic at all and stay with their owner:

  ``rung1/vm.py``            1,587  KvmFixtureSession: per-task isolation, port
                                    serialization, start, reset + attested
                                    receipt, close.  KEPT.  Its Chrome-fixture
                                    launch, CDP-over-raw-websocket client, and
                                    browser diagnostics (~900 LOC) are fixture
                                    content, not session lifecycle -- LEFT BEHIND.
  ``rung1b/vm.py``              619  ``from ..rung1.transport import
                                    HttpVmTransport`` -- a fork of rung1 whose own
                                    body is app fixtures (scroll HTML, drag
                                    setup, an in-guest HTTP server).  Only its
                                    guest-script marker protocol is generic; that
                                    is KEPT as ``GuestScript``.  Fixtures LEFT.
  ``rung2/vm.py``               451  the same fork lineage, the same split: Calc
                                    /Writer/Files/Chrome fixtures LEFT, the
                                    marker protocol already covered.
  ``desktop/proxy.py``      480+124  a JSON-line subprocess proxy to run a
                                    third-party env in another interpreter.  Its
                                    process-GROUP teardown discipline is the best
                                    of the five and is KEPT as
                                    ``ProcessGroupReaper``.  The proxy itself is
                                    LEFT: it exists to bridge to a foreign
                                    package, which this package has no business
                                    depending on.
  ``qemu_fast_reset.py``        549  now ``vm/qemu.py``; its tier-2 "snapshot the
                                    post-setup state and skip setup next time"
                                    idea survives as ``reset_to_checkpoint``.

The reset attestation is kept from rung1 and is the least obvious value here: a
reset that silently no-ops is indistinguishable from a working one unless
something proves the guest actually rewound.  This session plants a nonce file in
the guest *before* the reset and requires it to be gone *after*, and hashes the
runtime's own state on both sides so the generation must advance.  What was
dropped is the introspection of a specific provider object's ``timings`` list;
the ``Runtime`` protocol now reports its own checkpoint records.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import signal
import socket
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from ..execute.transport import HttpGuiTransport
from .osworld_client import OSWorldClient
from .runtime import Runtime

GUEST_JSON_MARKER = "DESKTOP_ENV_JSON="


class SessionError(RuntimeError):
    """A session could not be isolated, started, reset, or torn down."""


_ID_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = _ID_COMPONENT.sub("-", value).strip("-.")
    return (cleaned or fallback)[:40]


def task_unique_session_id() -> str:
    """An auditable id unique across jobs, tasks, and processes.

    Scheduler variables are read opportunistically; outside a scheduler the id is
    still unique because of the pid and the random suffix.
    """
    job = _safe_component(
        os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "local")),
        fallback="local",
    )
    task = _safe_component(os.environ.get("SLURM_PROCID", "0"), fallback="0")
    run = _safe_component(os.environ.get("RUN_ID", "no-run"), fallback="no-run")
    return f"{job}-{task}-{run[:16]}-{os.getpid()}-{uuid.uuid4().hex[:10]}"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write JSON through a private temp file, then rename onto the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, raw = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(handle_fd, 0o600)
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def node_port_allocation_lock() -> Iterator[None]:
    """Serialize ``bind(0)`` -> QEMU handoff across cooperating processes.

    The runtime retries a lost bind race on its own, but serializing allocation
    through the point where QEMU has actually bound closes the race between all
    jobs sharing a node instead of merely surviving it.
    """
    path = Path(f"/tmp/desktop-env-port-allocation-{os.getuid()}.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class GuestScript:
    """Run a Python program in the guest and read one JSON line back.

    The marker protocol is from ``rung1b``/``rung2``: the guest program prints
    ``<MARKER><json>`` on a line of its own, and the host reads only that line.
    It exists because the guest's stdout is shared with anything the program
    imports -- GTK warnings, X11 chatter, deprecation notices -- so "parse the
    whole of stdout as JSON" fails intermittently and unreproducibly.
    """

    def __init__(self, client: OSWorldClient, *, marker: str = GUEST_JSON_MARKER) -> None:
        self.client = client
        self.marker = marker

    def run_json(self, program: str, *, timeout_s: float | None = None) -> Any:
        result = self.client.execute(
            ["python3", "-c", program], check=True, timeout_s=timeout_s
        )
        return self.parse(result)

    def parse(self, result: dict[str, Any]) -> Any:
        output = result.get("output")
        if not isinstance(output, str):
            raise SessionError("guest script produced no stdout")
        lines = [line for line in output.splitlines() if line.startswith(self.marker)]
        if len(lines) != 1:
            raise SessionError(
                f"guest script emitted {len(lines)} result markers, expected 1"
            )
        try:
            return json.loads(lines[0][len(self.marker) :])
        except json.JSONDecodeError as exc:
            raise SessionError(f"guest script emitted invalid JSON: {exc}") from exc

    def resolve_guest_root(self, name: str) -> PurePosixPath:
        """A writable per-session directory inside the guest.

        Tries ``$HOME`` first and falls back to the guest's temp dir, because the
        pinned image's home is not writable under every login mode.
        """
        program = f"""
import json,os,pathlib,tempfile
name={name!r}
candidates=[]
home=os.environ.get('HOME')
if home: candidates.append(pathlib.Path(home)/name)
candidates.append(pathlib.Path(tempfile.gettempdir())/name)
chosen=None
for candidate in candidates:
    try:
        candidate.mkdir(parents=True,exist_ok=True)
        probe=candidate/'.writable'
        probe.write_text('ok',encoding='utf-8')
        probe.unlink()
        chosen=str(candidate)
        break
    except OSError:
        continue
if chosen is None: raise RuntimeError('no writable guest root')
print({self.marker!r}+json.dumps({{'root':chosen}}))
""".strip()
        payload = self.run_json(program)
        if not isinstance(payload, dict) or "root" not in payload:
            raise SessionError(f"guest root resolution failed: {payload!r}")
        return PurePosixPath(str(payload["root"]))


class ProcessGroupReaper:
    """Terminate whole process groups and confirm they are actually gone.

    Signalling a pid is not enough for a VM: QEMU spawns helpers, and a
    ``terminate`` that returns does not mean the group is dead.  This escalates
    SIGTERM -> SIGKILL and *verifies* by scanning ``/proc`` for live members,
    ignoring zombies, because ``killpg(pgid, 0)`` succeeds against a group whose
    only remaining member is a zombie.
    """

    def __init__(self, group_ids: tuple[int, ...]) -> None:
        self.group_ids = tuple(
            gid for gid in group_ids if gid > 0 and gid != os.getpgrp()
        )

    def alive(self) -> tuple[int, ...]:
        return tuple(gid for gid in self.group_ids if self._group_alive(gid))

    def signal(self, sig: int, *, group_ids: tuple[int, ...] | None = None) -> None:
        for gid in group_ids if group_ids is not None else self.group_ids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(gid, sig)

    def terminate_and_wait(
        self, *, terminate_timeout_s: float = 30.0, kill_timeout_s: float = 10.0
    ) -> bool:
        remaining = self.alive()
        if remaining:
            self.signal(signal.SIGTERM, group_ids=remaining)
        if self._wait(terminate_timeout_s):
            return True
        remaining = self.alive()
        if remaining:
            self.signal(signal.SIGKILL, group_ids=remaining)
        return self._wait(kill_timeout_s)

    def _wait(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            if not self.alive():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _group_alive(group_id: int) -> bool:
        status = _linux_process_group_has_live_members(group_id)
        if status is not None:
            return status
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def _linux_process_group_has_live_members(group_id: int) -> bool | None:
    """``None`` when ``/proc`` is unavailable; otherwise ignore zombies."""
    if not os.path.isdir("/proc"):
        return None
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join("/proc", entry, "stat"), encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        close = text.rfind(")")
        if close < 0:
            continue
        fields = text[close + 2 :].split()
        if len(fields) < 3:
            continue
        try:
            member_group = int(fields[2])
        except ValueError:
            continue
        if member_group == group_id and fields[0] != "Z":
            return True
    return False


def process_group_of(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


@dataclass(frozen=True)
class ResetReceipt:
    """Evidence that one reset really rewound the guest."""

    session_id: str
    reset_id: str
    reset_sequence: int
    checkpoint_name: str
    prior_generation_id: str
    new_generation_id: str
    reset_started_monotonic_ns: int
    reset_completed_monotonic_ns: int
    guest_sentinel_path_sha256: str
    guest_sentinel_nonce_sha256: str
    runtime_state_before_sha256: str
    runtime_state_after_sha256: str
    attestor_mac: str
    receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DesktopSession:
    """One isolated desktop, restored to a clean checkpoint per episode.

    Isolation is enforced, not documented: one live VM per scheduler task (an
    ``flock`` on a per-task lock file), a per-VM scratch directory below the
    task's own temp root, and a ``TMPDIR`` that points at it so QEMU's unlinked
    overlay is freed with the allocation even under SIGKILL.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        scratch_root: Path | None = None,
        metadata_path: Path | None = None,
        session_id: str | None = None,
        checkpoint_name: str | None = None,
        require_single_task: bool = True,
        forbid_gpu_visibility: bool = False,
        transport_timeout_s: float = 30.0,
    ) -> None:
        self.runtime = runtime
        self.session_id = session_id or task_unique_session_id()
        self.checkpoint_name = checkpoint_name or getattr(
            runtime, "base_checkpoint", "desktop_env_base"
        )
        self.require_single_task = require_single_task
        self.forbid_gpu_visibility = forbid_gpu_visibility
        self.transport_timeout_s = float(transport_timeout_s)
        self.client: OSWorldClient | None = None
        self.transport: HttpGuiTransport | None = None
        self.scratch_dir: Path | None = None
        self.scratch_source: str | None = None
        self.metadata_path = metadata_path
        self._requested_scratch_root = scratch_root
        self._scratch_root: Path | None = None
        self._scratch_root_owned = False
        self._task_lock_handle: Any | None = None
        self._task_lock_path: Path | None = None
        self._saved_environment: dict[str, str | None] = {}
        self._reset_sequence = 0
        self._attestor_secret = secrets.token_bytes(32)
        self._outstanding_receipt_sha256: str | None = None
        self._consumed_receipts: set[str] = set()
        self._started = False

    def _set_environment(self, key: str, value: str) -> None:
        if key not in self._saved_environment:
            self._saved_environment[key] = os.environ.get(key)
        os.environ[key] = value

    def _restore_environment(self) -> None:
        for key, value in self._saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._saved_environment.clear()

    def _prepare_isolation(self) -> None:
        if self.forbid_gpu_visibility and os.environ.get("CUDA_VISIBLE_DEVICES", ""):
            raise SessionError("GPU visibility is forbidden for this session")
        ntasks = os.environ.get("SLURM_NTASKS")
        if self.require_single_task and ntasks is not None and ntasks != "1":
            raise SessionError(f"exactly one scheduler task is required, got {ntasks}")
        root = self._requested_scratch_root
        if root is None:
            raw = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR_JOB")
            if raw:
                root = Path(raw)
                self.scratch_source = "scheduler_tmpdir"
            else:
                job = _safe_component(
                    os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "local")),
                    fallback="local",
                )
                task = _safe_component(os.environ.get("SLURM_PROCID", "0"), fallback="0")
                root = Path("/tmp") / f"desktop-env-job-{os.getuid()}-{job}-{task}"
                self.scratch_source = "job_unique_tmp_fallback"
                self._scratch_root_owned = True
        else:
            self.scratch_source = "explicit"
        root = root.resolve()
        scheduler_tmp = os.environ.get("SLURM_TMPDIR")
        if scheduler_tmp and not root.is_relative_to(Path(scheduler_tmp).resolve()):
            raise SessionError("VM scratch root must live below the scheduler TMPDIR")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.stat().st_uid != os.geteuid():
            raise SessionError("VM scratch root is not owned by this user")
        if self._scratch_root_owned:
            os.chmod(root, 0o700)
        self._scratch_root = root
        self.scratch_dir = root / f"desktop-env-{self.session_id}"
        self.scratch_dir.mkdir(mode=0o700, parents=False, exist_ok=False)

        # A task may own at most one live VM.  The lock is scoped to the scratch
        # root so separate array tasks do not contend with each other.
        lock_path = root / f"desktop-env-task-{os.environ.get('SLURM_PROCID', '0')}.lock"
        self._task_lock_path = lock_path
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise SessionError("another VM is already live in this task") from exc
        self._task_lock_handle = handle

        # QEMU's -snapshot overlay is created and unlinked immediately; TMPDIR is
        # the only reliable placement boundary for it.
        self._set_environment("TMPDIR", str(self.scratch_dir))

    def start(self) -> HttpGuiTransport:
        if self._started:
            return self._require_transport()
        try:
            self._prepare_isolation()
            with node_port_allocation_lock():
                state = self.runtime.start()
            self.client = OSWorldClient(state.base_url, timeout_s=self.transport_timeout_s)
            self.transport = HttpGuiTransport(
                state.base_url, timeout_s=self.transport_timeout_s
            )
            self.runtime.ensure_base()
            self._started = True
            self._write_metadata(state_detail=state.detail, ports=state.ports.as_dict())
        except Exception:
            self.close()
            raise
        return self._require_transport()

    def _require_transport(self) -> HttpGuiTransport:
        if self.transport is None:
            raise SessionError("session is not started")
        return self.transport

    def guest(self) -> GuestScript:
        if self.client is None:
            raise SessionError("session is not started")
        return GuestScript(self.client)

    def reset(self) -> HttpGuiTransport:
        """Reset to the clean checkpoint and consume the receipt for you."""
        transport, receipt = self.reset_with_receipt()
        self.consume_receipt(receipt)
        return transport

    def reset_with_receipt(self) -> tuple[HttpGuiTransport, ResetReceipt]:
        """Reset, and hand back proof that the guest actually rewound."""
        if not self._started or self.client is None:
            raise SessionError("session is not started")
        if self._outstanding_receipt_sha256 is not None:
            raise SessionError(
                "the previous reset receipt must be consumed before another reset"
            )
        sequence = self._reset_sequence + 1
        sentinel_path = (
            f"/tmp/desktop_env_reset_attestation_{self.session_id}_{sequence}.nonce"
        )
        nonce = secrets.token_hex(32)
        started = time.monotonic_ns()
        self._plant_sentinel(sentinel_path, nonce)
        before = self._runtime_observation()
        self.runtime.restore(self.checkpoint_name)
        after = self._runtime_observation()
        if before == after:
            raise SessionError("the reset did not change the runtime's observed state")
        prior_generation = hashlib.sha256(before).hexdigest()
        new_generation = hashlib.sha256(after).hexdigest()
        self._verify_sentinel_removed(sentinel_path)
        completed = time.monotonic_ns()
        self._reset_sequence = sequence
        # Held input state must never cross an episode boundary: a fresh transport
        # means a fresh audit, so a button believed held before the reset cannot
        # be believed held after it.
        base_url = self.transport.base_url if self.transport else ""
        self.transport = HttpGuiTransport(base_url, timeout_s=self.transport_timeout_s)
        payload = {
            "session_id": self.session_id,
            "reset_id": uuid.uuid4().hex,
            "reset_sequence": sequence,
            "checkpoint_name": self.checkpoint_name,
            "prior_generation_id": prior_generation,
            "new_generation_id": new_generation,
            "reset_started_monotonic_ns": started,
            "reset_completed_monotonic_ns": completed,
            "guest_sentinel_path_sha256": hashlib.sha256(
                sentinel_path.encode("utf-8")
            ).hexdigest(),
            "guest_sentinel_nonce_sha256": hashlib.sha256(
                nonce.encode("utf-8")
            ).hexdigest(),
            "runtime_state_before_sha256": prior_generation,
            "runtime_state_after_sha256": new_generation,
        }
        mac = hmac.new(
            self._attestor_secret, canonical_json(payload), hashlib.sha256
        ).hexdigest()
        receipt_sha256 = hashlib.sha256(
            canonical_json({**payload, "attestor_mac": mac})
        ).hexdigest()
        receipt = ResetReceipt(**payload, attestor_mac=mac, receipt_sha256=receipt_sha256)
        self._outstanding_receipt_sha256 = receipt.receipt_sha256
        return self._require_transport(), receipt

    def reset_to_checkpoint(self, name: str, *, setup: Any = None) -> HttpGuiTransport:
        """Restore ``name``, creating it from ``setup`` the first time.

        This is the second tier of the fast-reset idea: on the first call for a
        tag, run the (expensive) per-task setup once and snapshot the result; on
        every later call, restore that snapshot and skip both the reboot and the
        setup.
        """
        if not self._started:
            raise SessionError("session is not started")
        if self.runtime.has_checkpoint(name):
            self.runtime.restore(name)
            base_url = self.transport.base_url if self.transport else ""
            self.transport = HttpGuiTransport(
                base_url, timeout_s=self.transport_timeout_s
            )
            return self._require_transport()
        transport = self.reset()
        if setup is not None:
            setup(transport)
        self.runtime.checkpoint(name)
        return transport

    def _runtime_observation(self) -> bytes:
        """A stable hash input describing the runtime's externally visible state.

        Deliberately *not* an introspection of a provider's private dictionary,
        which is what the predecessor did and which made the receipt depend on one
        provider implementation's internals.
        """
        checkpoints = [
            {"name": item.name, "kind": item.kind, "created": item.created_monotonic_ns}
            for item in self.runtime.list_checkpoints()
        ]
        ready = self.runtime.is_ready()
        cursor: list[int] | None = None
        with contextlib.suppress(Exception):
            if self.client is not None:
                cursor = list(self.client.cursor_position())
        screenshot_sha256: str | None = None
        with contextlib.suppress(Exception):
            if self.client is not None:
                screenshot_sha256 = hashlib.sha256(self.client.screenshot()).hexdigest()
        return canonical_json(
            {
                "checkpoints": checkpoints,
                "ready": ready,
                "cursor": cursor,
                "screenshot_sha256": screenshot_sha256,
                "reset_sequence": self._reset_sequence,
            }
        )

    def _plant_sentinel(self, path: str, nonce: str) -> None:
        if self.client is None:
            raise SessionError("session is not started")
        program = (
            "from pathlib import Path;"
            f"p=Path({path!r});v={nonce!r};"
            "p.write_text(v,encoding='utf-8');"
            "assert p.read_text(encoding='utf-8')==v"
        )
        try:
            self.client.execute(["python3", "-c", program])
        except Exception as exc:
            raise SessionError(f"could not plant the pre-reset guest sentinel: {exc}") from exc

    def _verify_sentinel_removed(self, path: str) -> None:
        if self.client is None:
            raise SessionError("session is not started")
        program = f"from pathlib import Path;assert not Path({path!r}).exists()"
        try:
            self.client.execute(["python3", "-c", program])
        except Exception as exc:
            raise SessionError(
                "the reset did not rewind the pre-reset guest sentinel"
            ) from exc

    def consume_receipt(self, receipt: ResetReceipt) -> None:
        """Verify a receipt's MAC, ordering, and single use."""
        if not isinstance(receipt, ResetReceipt):
            raise SessionError("reset receipt type mismatch")
        payload = receipt.as_dict()
        receipt_sha256 = payload.pop("receipt_sha256")
        mac = payload.pop("attestor_mac")
        expected_mac = hmac.new(
            self._attestor_secret, canonical_json(payload), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_mac, mac):
            raise SessionError("reset receipt MAC does not verify")
        expected_sha256 = hashlib.sha256(
            canonical_json({**payload, "attestor_mac": mac})
        ).hexdigest()
        if not hmac.compare_digest(expected_sha256, receipt_sha256):
            raise SessionError("reset receipt digest does not verify")
        if receipt_sha256 in self._consumed_receipts:
            raise SessionError("reset receipt was already consumed")
        if receipt.session_id != self.session_id:
            raise SessionError("reset receipt belongs to another session")
        if receipt.reset_sequence != self._reset_sequence:
            raise SessionError(
                f"reset receipt is out of order: {receipt.reset_sequence} != "
                f"{self._reset_sequence}"
            )
        if self._outstanding_receipt_sha256 != receipt_sha256:
            raise SessionError("reset receipt does not match the outstanding reset")
        self._consumed_receipts.add(receipt_sha256)
        self._outstanding_receipt_sha256 = None

    def _write_metadata(
        self, *, state_detail: dict[str, Any], ports: dict[str, int]
    ) -> None:
        if self.metadata_path is None:
            return
        write_json_atomic(
            self.metadata_path,
            {
                "schema_version": "desktop_env_session_v1",
                "session_id": self.session_id,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "runtime": getattr(self.runtime, "name", type(self.runtime).__name__),
                "runtime_state": state_detail,
                "ports": ports,
                "checkpoint_name": self.checkpoint_name,
                "scratch": {
                    "directory": None if self.scratch_dir is None else str(self.scratch_dir),
                    "source": self.scratch_source,
                },
                "scheduler": {
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                    "proc_id": os.environ.get("SLURM_PROCID", "0"),
                    "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
                },
                "one_vm_per_task": self.require_single_task,
                "closed": False,
            },
        )

    def _finalize_metadata(self, *, closed: bool, errors: list[str]) -> None:
        if self.metadata_path is None or not self.metadata_path.is_file():
            return
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            metadata["closed"] = closed
            metadata["cleanup_errors"] = errors
            write_json_atomic(self.metadata_path, metadata)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"metadata finalization failed: {exc}")

    def close(self) -> None:
        """Tear down, collecting every failure instead of stopping at the first.

        Raises only if something failed AND no exception is already propagating,
        so a cleanup problem cannot mask the error that caused the cleanup.
        """
        errors: list[str] = []
        stopped = True
        pid = None
        with contextlib.suppress(Exception):
            pid = self.runtime.state().detail.get("pid")  # type: ignore[attr-defined]
        try:
            self.runtime.stop()
        except Exception as exc:
            stopped = False
            errors.append(f"runtime stop failed: {exc}")
        if isinstance(pid, int):
            group = process_group_of(pid)
            if group is not None:
                reaper = ProcessGroupReaper((group,))
                if reaper.alive() and not reaper.terminate_and_wait():
                    stopped = False
                    errors.append("runtime process group survived SIGKILL")
        self.transport = None
        self.client = None
        scratch = self.scratch_dir
        if scratch is not None:
            try:
                shutil.rmtree(scratch)
            except OSError as exc:
                errors.append(f"VM scratch cleanup failed: {exc}")
            self.scratch_dir = None
        if scratch is not None and scratch.exists():
            errors.append("VM scratch directory still exists after cleanup")
        if self._task_lock_handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._task_lock_handle.fileno(), fcntl.LOCK_UN)
            self._task_lock_handle.close()
            self._task_lock_handle = None
        if self._task_lock_path is not None:
            try:
                self._task_lock_path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"task lock cleanup failed: {exc}")
            self._task_lock_path = None
        if self._scratch_root_owned and self._scratch_root is not None:
            with contextlib.suppress(OSError):
                self._scratch_root.rmdir()
        self._scratch_root = None
        self._scratch_root_owned = False
        self._restore_environment()
        self._started = False
        self._finalize_metadata(closed=stopped, errors=errors)
        if errors and sys.exc_info()[0] is None:
            raise SessionError("; ".join(errors))

    def __enter__(self) -> "DesktopSession":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "DesktopSession",
    "GuestScript",
    "ProcessGroupReaper",
    "ResetReceipt",
    "SessionError",
    "canonical_json",
    "node_port_allocation_lock",
    "process_group_of",
    "sha256_file",
    "task_unique_session_id",
    "write_json_atomic",
]
